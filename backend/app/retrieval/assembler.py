from __future__ import annotations

import logging
import unicodedata
from typing import Callable

from app.retrieval.models import AssembledContext, ChunkContext, RetrievalResult

logger = logging.getLogger(__name__)

# Sentinel used when no count_tokens function is injected.
_UNSET = object()
_MIN_REDUNDANT_CONTENT_CHARS = 256


class ContextAssembler:
    """Assemble a token-budget-bounded context from a :class:`RetrievalResult`.

    The *count_tokens_fn* parameter is injectable so that tests can pass a
    trivial function (e.g. ``lambda text: len(text.split())``) without
    triggering a HuggingFace model download.

    When ``count_tokens_fn`` is ``None`` (the default), the real
    ``count_tokens`` from ``app.ingestion.tokenize`` is imported lazily on the
    first ``assemble()`` call.

    Invariants
    ----------
    - Never calls an LLM.
    - Never reads or writes Qdrant or SQLite.
    - Never exceeds ``token_budget`` in ``AssembledContext.total_tokens``.
    - Chunks that would overflow the budget are skipped and their ``chunk_id``
      recorded in ``AssembledContext.truncated_chunk_ids``.
    - Output is deterministic: same ``RetrievalResult`` input always produces
      the same ``AssembledContext`` output.
    """

    def __init__(
        self,
        count_tokens_fn: Callable[[str], int] | None = None,
    ) -> None:
        self._count_tokens_fn = count_tokens_fn

    def assemble(
        self,
        result: RetrievalResult,
        *,
        max_chunks: int = 5,
        token_budget: int = 2000,
    ) -> AssembledContext:
        """Build an :class:`AssembledContext` from *result*.

        Parameters
        ----------
        result:
            A ``RetrievalResult`` (may have ``ok=False``; in that case,
            returns an empty context with the original query).
        max_chunks:
            Maximum number of chunks to include (must be >= 1).
        token_budget:
            Maximum total token count across all included chunks.
            Must be >= 1.

        Returns
        -------
        AssembledContext
            Always returned (never raises).
        """
        if max_chunks < 1:
            max_chunks = 1
        if token_budget < 1:
            token_budget = 1

        count = self._get_count_tokens()
        chunks: list[ChunkContext] = []
        truncated_ids: list[str] = []
        total_tokens = 0
        for hit in _unique_hits(result.hits):
            if len(chunks) >= max_chunks:
                truncated_ids.append(hit.chunk_id)
                continue

            try:
                tok = count(hit.text)
            except Exception as exc:
                logger.warning("Token count failed for chunk %s: %s", hit.chunk_id, exc)
                tok = 0

            if total_tokens + tok > token_budget:
                # Chunk would overflow — skip and record.
                truncated_ids.append(hit.chunk_id)
                logger.debug(
                    "Skipping chunk %s (tokens=%d, remaining=%d)",
                    hit.chunk_id,
                    tok,
                    token_budget - total_tokens,
                )
                continue

            chunks.append(
                ChunkContext(
                    chunk_id=hit.chunk_id,
                    source_id=hit.source_id,
                    canonical_url=hit.canonical_url,
                    title=hit.title,
                    headings=hit.headings,
                    breadcrumb=hit.breadcrumb,
                    text=hit.text,
                    score=hit.score,
                    token_count=tok,
                    truncated=False,
                )
            )
            total_tokens += tok

        return AssembledContext(
            chunks=chunks,
            total_tokens=total_tokens,
            token_budget=token_budget,
            truncated_chunk_ids=truncated_ids,
            query=result.query,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_count_tokens(self) -> Callable[[str], int]:
        if self._count_tokens_fn is not None:
            return self._count_tokens_fn
        # Lazy import — only triggers model download when the real tokenizer is used.
        from app.ingestion.tokenize import count_tokens  # lazy
        return count_tokens


def _unique_hits(hits):
    """Keep the first ranked hit for each substantial normalized document chunk."""
    seen: set[tuple[str, str, str, str]] = set()
    for hit in hits:
        normalized = _normalize_chunk_text(hit.text)
        if len(normalized) < _MIN_REDUNDANT_CONTENT_CHARS:
            yield hit
            continue
        key = (hit.source_id, hit.document_id, hit.canonical_url, normalized)
        if key in seen:
            continue
        seen.add(key)
        yield hit


def _normalize_chunk_text(text: str) -> str:
    """Normalize only formatting differences, preserving case and code content."""
    return " ".join(unicodedata.normalize("NFKC", text).split())
