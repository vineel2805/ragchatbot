"""RAG query route: POST /api/v1/rag/query.

Design notes
------------
- HTTP status codes:
    200  — pipeline ran (check ``ok`` in body for success vs failure).
    422  — request body failed Pydantic validation or ``source_id`` unknown.
    500  — unexpected unhandled exception (should never happen in production
           because ``RAGService.answer()`` never raises, but kept for safety).

- Secret safety: the route never includes API keys or internal tracebacks in
  any response body.  ``RAGService`` already redacts secrets from error messages;
  the route adds no new surface area.

- The route accepts ``Union[RAGQueryResponse, RAGErrorResponse]`` as its
  response model so FastAPI generates a correct OpenAPI schema for both shapes.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.api.deps import RagServiceDep
from app.api.schemas.rag import (
    CitationOut,
    RAGErrorResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from app.rag.models import RAGRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


@router.post(
    "/rag/query",
    summary="Answer a question using the RAG pipeline",
    description=(
        "Retrieves relevant documentation chunks, assembles context, and calls "
        "the LLM to produce a grounded answer with citations.\n\n"
        "Always returns HTTP 200.  Inspect ``ok`` in the response body to "
        "distinguish a successful answer from a pipeline error."
    ),
    responses={
        200: {
            "description": "Pipeline executed (success or graceful failure). "
            "Check ``ok`` field.",
        },
        422: {"description": "Request body validation failed."},
    },
)
def query(
    body: RAGQueryRequest,
    svc: RagServiceDep,
) -> JSONResponse:
    """Execute the full RAG pipeline and return a structured response."""

    # -----------------------------------------------------------------
    # Extra validation not expressible as Pydantic field constraints.
    # Raises ValueError → caught below → 422 Unprocessable Entity.
    # -----------------------------------------------------------------
    try:
        body.validate_source_id()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    # -----------------------------------------------------------------
    # Build internal request and run the pipeline.
    # RAGService.answer() never raises — all errors are in the result.
    # -----------------------------------------------------------------
    rag_request = RAGRequest(
        query=body.query,
        source_id=body.source_id,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
        max_chunks=body.max_chunks,
        token_budget=body.token_budget,
    )

    try:
        result = svc.answer(rag_request)
    except Exception:
        # Defensive catch — RAGService.answer() is documented as never-raise,
        # but belt-and-suspenders for any future change.
        logger.exception("Unexpected exception in RAGService.answer()")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again later.",
        )

    # -----------------------------------------------------------------
    # Map RAGResponse → API response schema.
    # -----------------------------------------------------------------
    if not result.ok:
        body_out = RAGErrorResponse(
            ok=False,
            query=result.query,
            error=result.error or "An unknown pipeline error occurred.",
            error_stage=result.error_stage,
            context_was_empty=result.context_was_empty,
            chunks_retrieved=result.chunks_retrieved,
            chunks_in_context=result.chunks_in_context,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=body_out.model_dump(),
        )

    citations_out = [
        CitationOut(url=c.url, title=c.title, source_id=c.source_id)
        for c in result.citations
    ]
    body_out = RAGQueryResponse(
        query=result.query,
        answer=result.answer,
        citations=citations_out,
        context_was_truncated=result.context_was_truncated,
        fabricated_url_count=result.fabricated_url_count,
        chunks_retrieved=result.chunks_retrieved,
        chunks_in_context=result.chunks_in_context,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=body_out.model_dump(),
    )
