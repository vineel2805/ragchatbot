from __future__ import annotations

import logging

from app.generation.errors import (
    EmptyContextError,
    GenerationError,
    InvalidRequestError,
    MalformedResponseError,
    ProviderError,
)
from app.generation.models import Citation, GenerationRequest, GenerationResult
from app.generation.prompt import build_prompt, extract_urls_from_text
from app.retrieval.models import AssembledContext

logger = logging.getLogger(__name__)

# Maximum query length enforced before calling any external service.
_MAX_QUERY_CHARS = 2_000


class Generator:
    """Orchestrate prompt building → LLM call → citation validation.

    *client* is injectable for tests (any object satisfying
    :class:`~app.generation.client.LLMClient`).

    Invariants
    ----------
    - Never calls the LLM when ``context.chunks`` is empty.
    - Never surfaces secrets in ``GenerationResult.error``.
    - Never modifies ingestion or retrieval state.
    - Citation URLs are validated against the retrieved context; fabricated
      URLs are counted but excluded from ``GenerationResult.citations``.
    - All errors are captured into ``GenerationResult(ok=False, …)`` so
      the caller always gets a typed result rather than a raw exception.
    """

    def __init__(self, client) -> None:  # client: LLMClient
        self._client = client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run the full generation pipeline.

        Returns a :class:`GenerationResult` — never raises.
        """
        # --- Validate request ---
        try:
            self._validate(request)
        except InvalidRequestError as exc:
            return GenerationResult(
                query=request.query,
                ok=False,
                error=str(exc),
                error_type="InvalidRequest",
            )
        except EmptyContextError as exc:
            return GenerationResult(
                query=request.query,
                ok=False,
                error=str(exc),
                error_type="EmptyContext",
                context_was_truncated=bool(request.context.truncated_chunk_ids),
            )

        # --- Build prompt ---
        system_prompt, user_prompt = build_prompt(request.query, request.context)

        # --- Call LLM ---
        try:
            raw_answer = self._client.complete(system_prompt, user_prompt)
        except MalformedResponseError as exc:
            return GenerationResult(
                query=request.query,
                ok=False,
                error=str(exc),
                error_type="MalformedResponse",
                context_was_truncated=bool(request.context.truncated_chunk_ids),
            )
        except ProviderError as exc:
            # ProviderError already has a safe (no-secret) message.
            return GenerationResult(
                query=request.query,
                ok=False,
                error=str(exc),
                error_type="ProviderError",
                context_was_truncated=bool(request.context.truncated_chunk_ids),
            )
        except GenerationError as exc:
            return GenerationResult(
                query=request.query,
                ok=False,
                error=str(exc),
                error_type="ProviderError",
                context_was_truncated=bool(request.context.truncated_chunk_ids),
            )

        # --- Validate citations ---
        citations, fabricated_count = _validate_citations(raw_answer, request.context)

        if fabricated_count:
            logger.warning(
                "Generation produced %d URL(s) not found in retrieved context "
                "(query fingerprint omitted). Fabricated URLs excluded from citations.",
                fabricated_count,
            )

        return GenerationResult(
            query=request.query,
            answer=raw_answer,
            citations=citations,
            ok=True,
            context_was_truncated=bool(request.context.truncated_chunk_ids),
            fabricated_url_count=fabricated_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, request: GenerationRequest) -> None:
        if not request.query or not request.query.strip():
            raise InvalidRequestError("Query must not be empty or whitespace-only.")
        if len(request.query) > _MAX_QUERY_CHARS:
            raise InvalidRequestError(
                f"Query exceeds maximum length of {_MAX_QUERY_CHARS} characters."
            )
        if not request.context.chunks:
            raise EmptyContextError(
                "No documentation context available for this query. "
                "Cannot generate a grounded answer."
            )


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------


def _validate_citations(
    answer_text: str,
    context: AssembledContext,
) -> tuple[list[Citation], int]:
    """Return ``(validated_citations, fabricated_url_count)``.

    A citation is *valid* iff its URL appears verbatim in the assembled context.
    URLs mentioned in the answer that are not in the context are counted as
    fabricated and excluded from the returned list.

    Only context chunks whose URL appears in the answer are included as
    citations — this avoids listing every retrieved chunk as a citation even
    when the model didn't actually reference it.
    """
    valid_urls: dict[str, tuple[str, str]] = {
        chunk.canonical_url: (chunk.title, chunk.source_id)
        for chunk in context.chunks
    }

    mentioned_urls = extract_urls_from_text(answer_text)

    fabricated = 0
    citations: list[Citation] = []
    seen: set[str] = set()

    for url in mentioned_urls:
        # Strip trailing punctuation that regex may have captured.
        url = url.rstrip(".,;:!?)")
        if url in valid_urls and url not in seen:
            title, source_id = valid_urls[url]
            citations.append(Citation(url=url, title=title, source_id=source_id))
            seen.add(url)
        elif url not in valid_urls:
            fabricated += 1
            logger.debug("Fabricated URL detected (not in context): [redacted]")

    return citations, fabricated


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_generator(
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 30.0,
) -> Generator:
    """Wire up a production Generator with a live OpenRouterClient."""
    from app.generation.client import make_openrouter_client

    client = make_openrouter_client(api_key=api_key, model=model, timeout=timeout)
    return Generator(client)
