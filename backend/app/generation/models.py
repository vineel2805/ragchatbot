from __future__ import annotations

from dataclasses import dataclass, field

from app.retrieval.models import AssembledContext


# ---------------------------------------------------------------------------
# Generation request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRequest:
    """Input to the generation layer.

    Parameters
    ----------
    query:
        The original user question (not pre-processed).
    context:
        Assembled documentation context produced by :class:`ContextAssembler`.
        The generator will refuse to call the LLM when ``context.chunks`` is
        empty.
    """

    query: str
    context: AssembledContext


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """A source reference validated against the retrieved context.

    Only URLs that appear in ``AssembledContext.chunks`` are ever surfaced
    as citations.  URLs mentioned by the LLM that are not in the retrieved
    context are counted as ``GenerationResult.fabricated_url_count`` and
    excluded from this list.
    """

    url: str
    title: str
    source_id: str


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Outcome of a single :meth:`Generator.generate` call.

    On success: ``ok=True``, ``answer`` contains the model's text,
    ``citations`` contains validated source references.

    On failure: ``ok=False``, ``error`` has a safe (no-secret) message,
    ``error_type`` is one of:

    - ``"EmptyContext"``    — context had no chunks; LLM was not called.
    - ``"InvalidRequest"``  — query blank or request otherwise malformed.
    - ``"ProviderError"``   — HTTP error, timeout, or 429 from the provider.
    - ``"MalformedResponse"`` — provider returned unparseable JSON / empty text.
    """

    query: str
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    error_type: str | None = None
    context_was_truncated: bool = False
    fabricated_url_count: int = 0  # URLs in answer not found in retrieved context
