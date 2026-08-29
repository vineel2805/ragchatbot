"""Pydantic v2 request/response schemas for the RAG API layer.

Kept separate from the internal dataclasses in ``app.rag.models`` so that:
- The pipeline internals are not coupled to HTTP serialisation concerns.
- Field aliases, JSON examples, and OpenAPI metadata live here, not in the core.
- Validation (Pydantic) is distinct from the pipeline's own checks (Retriever).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Known source IDs (mirrors _KNOWN_SOURCE_IDS in retriever.py).
# Validated here so the API returns HTTP 422 before the pipeline is invoked.
# ---------------------------------------------------------------------------
_VALID_SOURCE_IDS: frozenset[str] = frozenset(
    {"fastapi", "python", "react", "docker", "qdrant"}
)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class RAGQueryRequest(BaseModel):
    """Body for ``POST /api/v1/rag/query``."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="The user's natural-language question.",
        examples=["How do I define a path parameter in FastAPI?"],
    )
    source_id: str | None = Field(
        default=None,
        description=(
            "Restrict retrieval to one corpus source. "
            f"Allowed values: {sorted(_VALID_SOURCE_IDS)}. "
            "Omit to search across all sources."
        ),
        examples=["fastapi"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of vector hits to request from the store.",
    )
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine-similarity score to keep a retrieved chunk.",
    )
    max_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks to include in the assembled context.",
    )
    token_budget: int = Field(
        default=2_000,
        ge=1,
        le=8_000,
        description="Maximum total token count across all assembled context chunks.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How do I define a path parameter in FastAPI?",
                    "source_id": "fastapi",
                    "top_k": 10,
                    "score_threshold": 0.0,
                }
            ]
        }
    }

    def validate_source_id(self) -> "RAGQueryRequest":
        """Check that *source_id*, when provided, is a known corpus source.

        Returns self so callers can chain.  Raises ``ValueError`` on unknown IDs.
        Pydantic field validators cannot access a runtime constant easily, so
        this is a method called explicitly in the route handler — it keeps the
        test surface clean and avoids over-coupling to the set of sources.
        """
        if self.source_id is not None and self.source_id not in _VALID_SOURCE_IDS:
            raise ValueError(
                f"Unknown source_id {self.source_id!r}. "
                f"Valid values: {sorted(_VALID_SOURCE_IDS)}"
            )
        return self


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------


class CitationOut(BaseModel):
    """A validated source citation returned by the generator."""

    url: str
    title: str
    source_id: str


# ---------------------------------------------------------------------------
# Success response
# ---------------------------------------------------------------------------


class RAGQueryResponse(BaseModel):
    """Successful response from ``POST /api/v1/rag/query``.

    All fields are always present; clients should check ``ok`` first.
    """

    query: str = Field(description="The original query, echoed back.")
    answer: str = Field(description="The grounded answer from the LLM.")
    citations: list[CitationOut] = Field(
        default_factory=list,
        description="Source citations validated against the retrieved context.",
    )
    context_was_truncated: bool = Field(
        description="True when some retrieved chunks were dropped to fit the token budget."
    )
    fabricated_url_count: int = Field(
        description="Count of URLs mentioned by the LLM that were not in the retrieved context.",
    )
    chunks_retrieved: int = Field(
        description="Number of vector hits returned by the retriever."
    )
    chunks_in_context: int = Field(
        description="Number of chunks that made it into the assembled context."
    )


# ---------------------------------------------------------------------------
# Error response (pipeline failure)
# ---------------------------------------------------------------------------


class RAGErrorResponse(BaseModel):
    """Error body returned when the RAG pipeline cannot produce an answer.

    HTTP status code is always **200** (the request was valid; the pipeline
    just could not fulfil it).  Use ``ok=False`` + ``error_stage`` to
    distinguish failure kinds on the client side.
    """

    ok: bool = False
    query: str
    error: str = Field(description="Human-readable error message safe to show callers.")
    error_stage: str | None = Field(
        default=None,
        description='Pipeline stage that failed: "retrieval" or "generation".',
    )
    context_was_empty: bool = Field(
        default=False,
        description="True when retrieval succeeded but returned zero usable chunks.",
    )
    chunks_retrieved: int = 0
    chunks_in_context: int = 0
