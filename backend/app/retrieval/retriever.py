from __future__ import annotations

import hashlib
import logging
from typing import Callable

from app.retrieval.errors import (
    ConfigError,
    EmbedFailureError,
    InvalidQueryError,
    StoreFailureError,
)
from app.retrieval.models import RetrievalHit, RetrievalRequest, RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BGE query prefix (§12 of the design doc).
# Applied to query strings ONLY — never to indexed passages.
# ---------------------------------------------------------------------------
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Limits enforced at validation time.
TOP_K_MIN = 1
TOP_K_MAX = 100
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0

# Log at most this many chars of a query (fingerprint logged separately).
_QUERY_LOG_CHARS = 60

# Known valid source IDs from the locked corpus registry.
_KNOWN_SOURCE_IDS = frozenset({"fastapi", "python", "react", "docker", "qdrant"})


class Retriever:
    """Embed a query → search Qdrant → rank → deduplicate → :class:`RetrievalResult`.

    Both *store* and *embedder* are injectable, so tests never need a live
    Qdrant server or a sentence-transformers model download.

    Security invariants
    -------------------
    - ``source_id`` is the only caller-supplied filter; collection name and
      ``is_active`` filtering are owned by the adapter, not by the caller.
    - Query text is never logged beyond its SHA-256 fingerprint at INFO level.
      The first ``_QUERY_LOG_CHARS`` characters are logged at DEBUG only.
    - ``IngestionCatalog`` is intentionally absent — this class cannot mutate
      SQLite state.
    """

    def __init__(
        self,
        store,  # VectorStoreClient — avoid circular import; duck-typed
        embedder,  # EmbedderProtocol
        collection_name: str | None = None,
    ) -> None:
        # Import COLLECTION_NAME lazily to avoid circular imports in tests
        # that construct a Retriever before the indexer module is loaded.
        if collection_name is None:
            from app.ingestion.indexer import COLLECTION_NAME  # lazy
            collection_name = COLLECTION_NAME
        self._store = store
        self._embedder = embedder
        self._collection = collection_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Run the full retrieval pipeline for *request*.

        Never raises :class:`RetrievalError` — errors are captured and returned
        as ``RetrievalResult(ok=False, ...)``.  This keeps the caller's control
        flow simple and allows a single failed retrieval to be surfaced without
        crashing the surrounding request handler.
        """
        try:
            self._validate(request)
        except InvalidQueryError as exc:
            return RetrievalResult(
                query=request.query,
                ok=False,
                error=str(exc),
                error_type="InvalidQuery",
            )

        try:
            hits = self._do_retrieve(request)
        except EmbedFailureError as exc:
            return RetrievalResult(
                query=request.query,
                ok=False,
                error=str(exc),
                error_type="EmbedFailure",
            )
        except StoreFailureError as exc:
            return RetrievalResult(
                query=request.query,
                ok=False,
                error=str(exc),
                error_type="StoreFailure",
            )
        except ConfigError as exc:
            return RetrievalResult(
                query=request.query,
                ok=False,
                error=str(exc),
                error_type="ConfigError",
            )

        return RetrievalResult(query=request.query, hits=hits)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, request: RetrievalRequest) -> None:
        if not request.query or not request.query.strip():
            raise InvalidQueryError("Query must not be empty or whitespace-only.")
        if not (TOP_K_MIN <= request.top_k <= TOP_K_MAX):
            raise InvalidQueryError(
                f"top_k must be in [{TOP_K_MIN}, {TOP_K_MAX}], got {request.top_k}."
            )
        if not (THRESHOLD_MIN <= request.score_threshold <= THRESHOLD_MAX):
            raise InvalidQueryError(
                f"score_threshold must be in [{THRESHOLD_MIN}, {THRESHOLD_MAX}], "
                f"got {request.score_threshold}."
            )

    def _do_retrieve(self, request: RetrievalRequest) -> list[RetrievalHit]:
        query_text = request.query
        fingerprint = hashlib.sha256(query_text.encode()).hexdigest()[:16]
        logger.info("Retrieval query fingerprint=%s top_k=%d", fingerprint, request.top_k)
        logger.debug(
            "Query text (first %d chars): %r",
            _QUERY_LOG_CHARS,
            query_text[:_QUERY_LOG_CHARS],
        )

        # Embed with BGE query prefix.
        prefixed = BGE_QUERY_PREFIX + query_text
        try:
            vectors = self._embedder.embed([prefixed])
        except Exception as exc:
            raise EmbedFailureError(f"Embedding failed: {exc}") from exc

        if not vectors or len(vectors[0]) == 0:
            raise EmbedFailureError("Embedder returned an empty vector.")

        query_vector = vectors[0]

        # Search through the Protocol — is_active + source_id filtering
        # is constructed *inside* the adapter, not here.
        try:
            raw_hits = self._store.search(
                self._collection,
                query_vector,
                request.top_k,
                request.source_id,
            )
        except Exception as exc:
            raise StoreFailureError(f"Vector store search failed: {exc}") from exc

        # Client-side score threshold filter.
        raw_hits = [h for h in raw_hits if h.score >= request.score_threshold]

        # Deterministic sort: score DESC, then source_id, document_id, chunk_id ASC.
        raw_hits.sort(
            key=lambda h: (
                -h.score,
                h.payload.get("source_id", ""),
                h.payload.get("document_id", ""),
                h.payload.get("chunk_id", ""),
            )
        )

        # Deduplicate by chunk_id — keep first occurrence.
        seen_chunk_ids: set[str] = set()
        deduplicated = []
        for hit in raw_hits:
            cid = hit.payload.get("chunk_id", hit.point_id)
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                deduplicated.append(hit)

        return [_hit_from_search_result(h) for h in deduplicated]


# ---------------------------------------------------------------------------
# Mapping helper
# ---------------------------------------------------------------------------


def _hit_from_search_result(hit) -> RetrievalHit:
    """Map a ``SearchHit`` (from the vector store) to a :class:`RetrievalHit`."""
    p = hit.payload
    return RetrievalHit(
        score=hit.score,
        source_id=p.get("source_id", ""),
        document_id=p.get("document_id", ""),
        chunk_id=p.get("chunk_id", hit.point_id),
        canonical_url=p.get("canonical_url", ""),
        title=p.get("title", ""),
        headings=p.get("headings", []),
        breadcrumb=p.get("breadcrumb", ""),
        text=p.get("text", ""),
        chunk_index=p.get("chunk_index", 0),
    )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_retriever(
    *,
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    embedder=None,
) -> Retriever:
    """Wire up a production Retriever with live Qdrant + BgeEmbedder.

    All heavy imports are deferred so the factory is importable without
    qdrant_client or sentence-transformers installed.
    """
    from qdrant_client import QdrantClient  # lazy

    from app.core.config import get_settings
    from app.ingestion.embedder import BgeEmbedder
    from app.ingestion.indexer import QdrantClientAdapter

    settings = get_settings()
    url = qdrant_url or settings.qdrant_url
    adapter = QdrantClientAdapter(QdrantClient(url=url))
    emb = embedder or BgeEmbedder()
    return Retriever(adapter, emb, collection_name)
