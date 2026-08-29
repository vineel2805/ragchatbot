"""Public API for the retrieval package."""
from __future__ import annotations

from app.retrieval.assembler import ContextAssembler
from app.retrieval.errors import (
    ConfigError,
    EmbedFailureError,
    InvalidQueryError,
    RetrievalError,
    StoreFailureError,
)
from app.retrieval.models import (
    AssembledContext,
    ChunkContext,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResult,
)
from app.retrieval.retriever import BGE_QUERY_PREFIX, Retriever, make_retriever

__all__ = [
    # Models
    "RetrievalRequest",
    "RetrievalHit",
    "RetrievalResult",
    "ChunkContext",
    "AssembledContext",
    # Errors
    "RetrievalError",
    "InvalidQueryError",
    "EmbedFailureError",
    "StoreFailureError",
    "ConfigError",
    # Core classes
    "Retriever",
    "ContextAssembler",
    # Constants
    "BGE_QUERY_PREFIX",
    # Factory
    "make_retriever",
]
