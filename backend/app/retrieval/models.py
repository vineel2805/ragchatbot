from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalRequest:
    """Validated retrieval request.

    Callers construct this directly; ``Retriever.retrieve()`` re-validates
    before any expensive operation.

    Parameters
    ----------
    query:
        Raw user query string.  Must not be blank after stripping whitespace.
    top_k:
        Maximum number of Qdrant hits to request.  Clamped to [1, 100].
    score_threshold:
        Minimum cosine-similarity score to accept.  In [0.0, 1.0].
        Results below this value are discarded client-side after the search.
    source_id:
        When set, restrict results to this source (e.g. ``"fastapi"``).
        ``None`` means all sources.
    """

    query: str
    top_k: int = 10
    score_threshold: float = 0.0
    source_id: str | None = None


# ---------------------------------------------------------------------------
# Per-hit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalHit:
    """One matched chunk returned from the vector store."""

    score: float
    source_id: str
    document_id: str
    chunk_id: str
    canonical_url: str
    title: str
    headings: list[str]
    breadcrumb: str
    text: str
    chunk_index: int


# ---------------------------------------------------------------------------
# Retrieval result envelope
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Outcome of a single ``Retriever.retrieve()`` call.

    On success: ``ok=True``, ``hits`` is a deterministically-ordered,
    deduplicated list of :class:`RetrievalHit`.

    On failure: ``ok=False``, ``error`` contains a human-readable description,
    ``error_type`` is one of ``"InvalidQuery"``, ``"EmbedFailure"``,
    ``"StoreFailure"``, ``"ConfigError"``.
    """

    query: str
    hits: list[RetrievalHit] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    error_type: str | None = None  # "InvalidQuery"|"EmbedFailure"|"StoreFailure"|"ConfigError"


# ---------------------------------------------------------------------------
# Context assembly models
# ---------------------------------------------------------------------------


@dataclass
class ChunkContext:
    """A single chunk as it appears inside an :class:`AssembledContext`."""

    chunk_id: str
    source_id: str
    canonical_url: str
    title: str
    headings: list[str]
    breadcrumb: str
    text: str
    score: float
    token_count: int
    truncated: bool = False


@dataclass
class AssembledContext:
    """Result of :class:`ContextAssembler`.

    ``truncated_chunk_ids`` lists the ``chunk_id`` values of hits that were
    *skipped* because they would have exceeded the token budget.  The caller
    can surface this to the user (e.g. "context truncated") rather than
    silently hiding that content was dropped.
    """

    chunks: list[ChunkContext]
    total_tokens: int
    token_budget: int
    truncated_chunk_ids: list[str]
    query: str
