"""FastAPI dependency providers for the RAG layer.

The :func:`get_rag_service` dependency is the single seam through which tests
can inject a fake ``RAGService`` via ``app.dependency_overrides``, keeping
all test doubles completely out of production code.

Production wiring defers all heavy imports (qdrant_client,
sentence_transformers, httpx) to ``make_rag_service``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.rag.service import RAGService, make_rag_service


@lru_cache(maxsize=1)
def _build_rag_service() -> RAGService:
    """Construct the singleton production RAGService.

    ``lru_cache`` ensures the heavy setup (embedder load, Qdrant connection)
    happens at most once per process lifetime.  The cache is busted in tests
    by replacing the dependency entirely via ``app.dependency_overrides``.
    """
    return make_rag_service()


def get_rag_service() -> RAGService:
    """FastAPI dependency that returns the process-wide RAGService instance."""
    return _build_rag_service()


# Convenience type alias for use in route signatures.
RagServiceDep = Annotated[RAGService, Depends(get_rag_service)]
