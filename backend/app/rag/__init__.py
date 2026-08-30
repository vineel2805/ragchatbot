"""RAG (Retrieval-Augmented Generation) service layer.

Connects retrieval, context assembly, and generation into a single pipeline.
"""
from app.rag.models import RAGRequest, RAGResponse
from app.rag.service import RAGService, make_rag_service

__all__ = [
    "RAGRequest",
    "RAGResponse",
    "RAGService",
    "make_rag_service",
]
