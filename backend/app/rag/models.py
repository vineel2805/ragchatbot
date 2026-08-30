#models.py
from __future__ import annotations

from dataclasses import dataclass, field

from app.generation.models import Citation


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RAGRequest:
    """End-to-end RAG request.

    Wraps all tuning knobs for a single query so callers don't need to
    import ``RetrievalRequest``, ``ContextAssembler``, or ``Generator``
    directly.

    Parameters
    ----------
    query:
        The user's natural-language question.
    source_id:
        Restrict retrieval to a single corpus source.  ``None`` searches
        across all five sources.
    top_k:
        Maximum number of vector hits to request from Qdrant.  [1, 100].
    score_threshold:
        Minimum cosine-similarity score to keep a retrieved chunk.  [0, 1].
    max_chunks:
        Maximum number of chunks to include in the assembled context.
    token_budget:
        Maximum total token count across assembled context chunks.
    """

    query: str
    source_id: str | None = None
    top_k: int = 10
    score_threshold: float = 0.0
    max_chunks: int = 5
    token_budget: int = 2000


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


@dataclass
class RAGResponse:
    """Unified result of a complete Retriever → Assembler → Generator pass.

    On success: ``ok=True``, ``answer`` contains the grounded answer,
    ``citations`` contains validated source references.

    On failure: ``ok=False``, ``error`` has a safe (no-secret) message,
    ``error_stage`` identifies which pipeline stage failed:

    - ``"retrieval"``  — Retriever returned ``ok=False``.
    - ``"generation"`` — Generator returned ``ok=False``.
      (Context assembly never fails; it always returns an ``AssembledContext``.)

    ``context_was_empty`` is ``True`` when retrieval succeeded but returned
    zero chunks, preventing the LLM from being called.
    """

    query: str
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    error_stage: str | None = None  # "retrieval" | "generation"
    context_was_truncated: bool = False
    context_was_empty: bool = False
    fabricated_url_count: int = 0
    chunks_retrieved: int = 0   # number of hits from the retriever
    chunks_in_context: int = 0  # number of chunks actually assembled
