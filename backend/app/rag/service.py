from __future__ import annotations

import logging

from app.generation.generator import Generator
from app.generation.models import GenerationRequest
from app.rag.models import RAGRequest, RAGResponse
from app.retrieval.assembler import ContextAssembler
from app.retrieval.models import RetrievalRequest
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGService:
    """Connect Retriever → ContextAssembler → Generator into a single call.

    All three components are injected so that:
    - Tests can substitute fakes for each stage independently.
    - Production wiring uses live Qdrant, BgeEmbedder, and OpenRouterClient.
    - This class owns no mutable state of its own.

    Invariants
    ----------
    - Never calls the LLM when retrieval returns zero chunks.
    - Never modifies ingestion, catalog, or Qdrant state.
    - Never raises — all errors are returned as ``RAGResponse(ok=False, …)``.
    - Secrets (API keys) are never included in ``RAGResponse.error``.
    """

    def __init__(
        self,
        retriever: Retriever,
        assembler: ContextAssembler,
        generator: Generator,
    ) -> None:
        self._retriever = retriever
        self._assembler = assembler
        self._generator = generator

    def answer(self, request: RAGRequest) -> RAGResponse:
        """Execute the full RAG pipeline for *request*.

        Pipeline
        --------
        1. Build :class:`RetrievalRequest` from *request* parameters.
        2. Call ``Retriever.retrieve()`` → :class:`RetrievalResult`.
        3. Short-circuit with ``error_stage="retrieval"`` if retrieval fails.
        4. Call ``ContextAssembler.assemble()`` → :class:`AssembledContext`.
        5. Short-circuit with ``context_was_empty=True`` (ok=False) when
           the assembled context has no chunks (retriever returned no hits).
        6. Build :class:`GenerationRequest` and call ``Generator.generate()``.
        7. Return a unified :class:`RAGResponse`.

        Returns
        -------
        RAGResponse
            Always returned, never raises.
        """
        # ----------------------------------------------------------------
        # Stage 1 — Retrieval
        # ----------------------------------------------------------------
        retrieval_req = RetrievalRequest(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            source_id=request.source_id,
        )
        retrieval_result = self._retriever.retrieve(retrieval_req)

        if not retrieval_result.ok:
            logger.warning(
                "RAG retrieval failed: error_type=%s", retrieval_result.error_type
            )
            return RAGResponse(
                query=request.query,
                ok=False,
                error=retrieval_result.error,
                error_stage="retrieval",
            )

        chunks_retrieved = len(retrieval_result.hits)
        logger.debug("RAG retrieved %d chunks", chunks_retrieved)

        # ----------------------------------------------------------------
        # Stage 2 — Context assembly
        # ----------------------------------------------------------------
        context = self._assembler.assemble(
            retrieval_result,
            max_chunks=request.max_chunks,
            token_budget=request.token_budget,
        )
        chunks_in_context = len(context.chunks)

        if chunks_in_context == 0:
            # Retrieval succeeded but produced no usable chunks (all filtered
            # out by score threshold, token budget, or there were simply none).
            logger.info("RAG context empty after assembly — skipping LLM call.")
            return RAGResponse(
                query=request.query,
                ok=False,
                error=(
                    "No relevant documentation found for this query. "
                    "Try broadening your question or removing the source filter."
                ),
                error_stage="retrieval",
                context_was_empty=True,
                chunks_retrieved=chunks_retrieved,
                chunks_in_context=0,
            )

        # ----------------------------------------------------------------
        # Stage 3 — Generation
        # ----------------------------------------------------------------
        gen_req = GenerationRequest(query=request.query, context=context)
        gen_result = self._generator.generate(gen_req)

        # Derive truncation flag from the assembled context — this is the
        # authoritative source; generator fakes and real implementations both
        # benefit from this single derivation point.
        context_was_truncated = bool(context.truncated_chunk_ids)

        if not gen_result.ok:
            logger.warning("RAG generation failed: error_type=%s", gen_result.error_type)
            return RAGResponse(
                query=request.query,
                ok=False,
                error=gen_result.error,
                error_stage="generation",
                context_was_truncated=context_was_truncated,
                chunks_retrieved=chunks_retrieved,
                chunks_in_context=chunks_in_context,
            )

        return RAGResponse(
            query=request.query,
            answer=gen_result.answer,
            citations=gen_result.citations,
            ok=True,
            context_was_truncated=context_was_truncated,
            context_was_empty=False,
            fabricated_url_count=gen_result.fabricated_url_count,
            chunks_retrieved=chunks_retrieved,
            chunks_in_context=chunks_in_context,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_rag_service(
    *,
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    openrouter_api_key: str | None = None,
    openrouter_model: str | None = None,
    embedder=None,
    count_tokens_fn=None,
) -> RAGService:
    """Wire up a production RAGService with live Qdrant + BgeEmbedder + OpenRouter.

    All heavy imports (qdrant_client, sentence_transformers, httpx) are
    deferred to the respective factory functions, so this module remains
    importable in test environments.

    Parameters are optional; when ``None``, they are read from
    :class:`~app.core.config.Settings` (which loads from .env).
    """
    from app.core.config import get_settings
    from app.generation.generator import make_generator
    from app.retrieval.retriever import make_retriever

    settings = get_settings()

    retriever = make_retriever(
        qdrant_url=qdrant_url or settings.qdrant_url,
        collection_name=collection_name or settings.qdrant_collection,
        embedder=embedder,
    )
    assembler = ContextAssembler(count_tokens_fn=count_tokens_fn)
    generator = make_generator(
        api_key=openrouter_api_key or settings.openrouter_api_key,
        model=openrouter_model or settings.openrouter_model,
    )
    return RAGService(retriever, assembler, generator)
