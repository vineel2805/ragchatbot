from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.ingestion.catalog import IngestionCatalog
from app.ingestion.document_models import DocumentChunk
from app.ingestion.embedder import EMBEDDING_DIM, EMBEDDING_MODEL, EmbedderProtocol
from app.ingestion.ids import make_point_id

logger = logging.getLogger(__name__)

COLLECTION_NAME = "devdocs_chunks"
EMBED_BATCH = 64   # texts per sentence-transformers call
UPSERT_BATCH = 64  # points per Qdrant upsert call


# ---------------------------------------------------------------------------
# Data types shared by the Protocol and the fake store used in tests
# ---------------------------------------------------------------------------


@dataclass
class PointData:
    """One Qdrant point: UUID string id, float vector, arbitrary payload dict."""

    id: str
    vector: list[float]
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# VectorStoreClient Protocol — injectable for tests
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStoreClient(Protocol):
    """Minimal Qdrant surface.  Real impl: QdrantClientAdapter.  Tests: FakeVectorStore."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(
        self, collection_name: str, vector_size: int, distance: str
    ) -> None: ...

    def collection_vector_size(self, collection_name: str) -> int: ...

    def upsert(self, collection_name: str, points: list[PointData]) -> None: ...

    def scroll_document(self, collection_name: str, document_id: str) -> list[str]:
        """Return all point-id strings whose payload.document_id == document_id."""
        ...

    def set_is_active(
        self, collection_name: str, point_ids: list[str], is_active: bool
    ) -> None: ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class IndexDocumentResult:
    document_id: str
    chunks_upserted: int = 0
    chunks_skipped: int = 0
    points_deactivated: int = 0
    ok: bool = True
    error: str | None = None


@dataclass
class IndexResult:
    source_id: str
    run_id: str
    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_upserted: int = 0
    chunks_skipped: int = 0
    points_deactivated: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# QdrantIndexer — the main class
# ---------------------------------------------------------------------------


class QdrantIndexer:
    """Embeds DocumentChunks and synchronises them into a Qdrant collection.

    Both *store* and *embedder* are injectable so tests never need a live Qdrant
    server or a sentence-transformers model download.

    Invariants
    ----------
    - A chunk's Qdrant point UUID is deterministic: ``make_point_id(chunk.chunk_id)``.
    - Upserting the same chunk_id twice is idempotent (same UUID → same point).
    - Stale points (present in Qdrant but absent from the new chunk set) are
      deactivated (``is_active=False``) rather than deleted, preserving history.
    - ``catalog.record_indexing()`` is called *only* after a successful upsert.
    """

    def __init__(
        self,
        store: VectorStoreClient,
        embedder: EmbedderProtocol,
        catalog: IngestionCatalog,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._catalog = catalog
        self._collection = collection_name

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection with dim=384, Cosine if absent; validate dim if present."""
        if not self._store.collection_exists(self._collection):
            self._store.create_collection(self._collection, EMBEDDING_DIM, "COSINE")
            logger.info(
                "Created Qdrant collection %r (dim=%d, Cosine)", self._collection, EMBEDDING_DIM
            )
        else:
            actual = self._store.collection_vector_size(self._collection)
            if actual != EMBEDDING_DIM:
                raise ValueError(
                    f"Collection {self._collection!r} has vector dim={actual}, "
                    f"expected {EMBEDDING_DIM}. "
                    "Drop and recreate the collection or run a full rebuild."
                )
            logger.debug("Collection %r validated (dim=%d)", self._collection, actual)

    # ------------------------------------------------------------------
    # Document indexing
    # ------------------------------------------------------------------

    def index_document(
        self,
        source_id: str,
        url: str,
        chunks: list[DocumentChunk],
        run_id: str,
    ) -> IndexDocumentResult:
        """Embed and upsert all chunks for one document; deactivate stale old points.

        Each document is handled atomically from the caller's perspective:
        errors are caught and returned in ``IndexDocumentResult.error`` rather
        than propagated, so one bad document does not abort its siblings.
        """
        if not chunks:
            # Empty document — deactivate any existing Qdrant points for it.
            record = self._catalog.get_url(source_id, url)
            if record is not None:
                n = self.deactivate_document(record.document_id)
                return IndexDocumentResult(document_id=record.document_id, points_deactivated=n)
            return IndexDocumentResult(document_id="")

        doc_id = chunks[0].document_id
        extracted_sha256 = chunks[0].extracted_sha256
        chunker_version = chunks[0].chunker_version

        # Skip if Qdrant already reflects the current extraction + chunker version.
        if not self._catalog.needs_indexing(
            source_id,
            url,
            extracted_sha256=extracted_sha256,
            chunker_version=chunker_version,
        ):
            return IndexDocumentResult(
                document_id=doc_id,
                chunks_skipped=len(chunks),
            )

        try:
            return self._do_index(
                source_id, url, doc_id, chunks, extracted_sha256, chunker_version, run_id
            )
        except Exception as exc:
            logger.error("Failed to index %s/%s: %s", source_id, url, exc, exc_info=True)
            return IndexDocumentResult(document_id=doc_id, ok=False, error=str(exc))

    # ------------------------------------------------------------------
    # Stale-point deactivation helpers
    # ------------------------------------------------------------------

    def deactivate_document(self, document_id: str) -> int:
        """Set ``is_active=False`` on all Qdrant points for *document_id*. Returns count."""
        point_ids = self._store.scroll_document(self._collection, document_id)
        if point_ids:
            self._store.set_is_active(self._collection, point_ids, False)
        return len(point_ids)

    def deactivate_stale_for_source(
        self, source_id: str, active_document_ids: set[str]
    ) -> int:
        """Deactivate Qdrant points for every document_id no longer in *active_document_ids*."""
        deactivated = 0
        for document_id in self._catalog.list_document_ids(source_id):
            if document_id not in active_document_ids:
                deactivated += self.deactivate_document(document_id)
        return deactivated

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _do_index(
        self,
        source_id: str,
        url: str,
        doc_id: str,
        chunks: list[DocumentChunk],
        extracted_sha256: str,
        chunker_version: str,
        run_id: str,
    ) -> IndexDocumentResult:
        """Embed → upsert → deactivate stale → record.  Raises on any failure."""
        # Snapshot existing point IDs for this document before we overwrite them.
        existing_ids: set[str] = set(
            self._store.scroll_document(self._collection, doc_id)
        )

        # Embed all chunk texts in batches.
        texts = [chunk.text for chunk in chunks]
        vectors = self._embed_batched(texts)

        # Build PointData objects.
        timestamp = self._catalog._stamp()
        new_points: list[PointData] = []
        new_point_ids: set[str] = set()
        for chunk, vector in zip(chunks, vectors):
            point_id = make_point_id(chunk.chunk_id)
            new_points.append(
                PointData(
                    id=point_id,
                    vector=vector,
                    payload=_build_payload(chunk, run_id, timestamp),
                )
            )
            new_point_ids.add(point_id)

        # Upsert to Qdrant in batches.
        self._upsert_batched(new_points)

        # Deactivate points that belonged to this document but are no longer present.
        stale_ids = list(existing_ids - new_point_ids)
        if stale_ids:
            self._store.set_is_active(self._collection, stale_ids, False)

        # Record success in SQLite *after* Qdrant upsert succeeds.
        self._catalog.record_indexing(
            source_id,
            url,
            extracted_sha256=extracted_sha256,
            chunker_version=chunker_version,
            run_id=run_id,
        )

        return IndexDocumentResult(
            document_id=doc_id,
            chunks_upserted=len(new_points),
            points_deactivated=len(stale_ids),
        )

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            result.extend(self._embedder.embed(texts[i : i + EMBED_BATCH]))
        return result

    def _upsert_batched(self, points: list[PointData]) -> None:
        for i in range(0, len(points), UPSERT_BATCH):
            self._store.upsert(self._collection, points[i : i + UPSERT_BATCH])


# ---------------------------------------------------------------------------
# QdrantClientAdapter — wraps real qdrant_client (all imports lazy)
# ---------------------------------------------------------------------------


class QdrantClientAdapter:
    """Adapts a real ``qdrant_client.QdrantClient`` to the ``VectorStoreClient`` Protocol.

    All imports of ``qdrant_client`` are deferred to method bodies so that the
    module can be imported safely even when qdrant_client is not installed
    (e.g. during unit tests that inject a fake store).
    """

    def __init__(self, qdrant_client: Any) -> None:
        self._client = qdrant_client

    def collection_exists(self, collection_name: str) -> bool:
        return bool(self._client.collection_exists(collection_name))

    def create_collection(
        self, collection_name: str, vector_size: int, distance: str
    ) -> None:
        from qdrant_client.models import Distance, VectorParams  # lazy

        dist = Distance[distance.upper()]
        self._client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=vector_size, distance=dist),
        )

    def collection_vector_size(self, collection_name: str) -> int:
        info = self._client.get_collection(collection_name)
        vectors = info.config.params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
        raise ValueError(
            f"Cannot determine vector size for collection {collection_name!r}. "
            "Named vectors are not supported."
        )

    def upsert(self, collection_name: str, points: list[PointData]) -> None:
        from qdrant_client.models import PointStruct  # lazy

        self._client.upsert(
            collection_name,
            points=[
                PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points
            ],
        )

    def scroll_document(self, collection_name: str, document_id: str) -> list[str]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # lazy

        results, _ = self._client.scroll(
            collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=10_000,
            with_payload=False,
            with_vectors=False,
        )
        return [str(r.id) for r in results]

    def set_is_active(
        self, collection_name: str, point_ids: list[str], is_active: bool
    ) -> None:
        if not point_ids:
            return
        from qdrant_client.models import PointIdsList  # lazy

        self._client.set_payload(
            collection_name,
            payload={"is_active": is_active},
            points=PointIdsList(points=point_ids),
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_qdrant_indexer(
    catalog: IngestionCatalog,
    *,
    qdrant_url: str | None = None,
    collection_name: str = COLLECTION_NAME,
    embedder: EmbedderProtocol | None = None,
) -> "QdrantIndexer":
    """Wire up a real QdrantIndexer with live qdrant_client + BgeEmbedder.

    All heavy imports are deferred so the factory itself is importable without
    qdrant_client or sentence-transformers being installed.
    """
    from qdrant_client import QdrantClient  # lazy

    from app.core.config import get_settings
    from app.ingestion.embedder import BgeEmbedder

    settings = get_settings()
    url = qdrant_url or settings.qdrant_url
    adapter = QdrantClientAdapter(QdrantClient(url=url))
    emb: EmbedderProtocol = embedder or BgeEmbedder()
    return QdrantIndexer(adapter, emb, catalog, collection_name)


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------


def _build_payload(chunk: DocumentChunk, run_id: str, created_at: str) -> dict[str, Any]:
    """Build the Qdrant point payload from a DocumentChunk."""
    return {
        # Identity
        "source_id": chunk.source_id,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "canonical_url": chunk.canonical_url,
        # Content
        "title": chunk.title,
        "headings": chunk.headings,
        "breadcrumb": chunk.breadcrumb,
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        # Versioning
        "chunker_version": chunk.chunker_version,
        "embedding_model": EMBEDDING_MODEL,
        # Run provenance
        "ingestion_run_id": run_id,
        # Lifecycle
        "is_active": True,
        "created_at": created_at,
    }
