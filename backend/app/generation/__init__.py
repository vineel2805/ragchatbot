"""Public API for the generation package."""
from __future__ import annotations

from app.generation.client import LLMClient, OpenRouterClient, make_openrouter_client
from app.generation.errors import (
    EmptyContextError,
    GenerationError,
    InvalidRequestError,
    MalformedResponseError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from app.generation.generator import Generator, make_generator
from app.generation.models import Citation, GenerationRequest, GenerationResult
from app.retrieval.models import AssembledContext

__all__ = [
    # Models
    "GenerationRequest",
    "GenerationResult",
    "Citation",
    "AssembledContext",
    # Errors
    "GenerationError",
    "EmptyContextError",
    "InvalidRequestError",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    "MalformedResponseError",
    # Client
    "LLMClient",
    "OpenRouterClient",
    "make_openrouter_client",
    # Generator
    "Generator",
    "make_generator",
]
