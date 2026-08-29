from __future__ import annotations


class GenerationError(Exception):
    """Base class for all generation errors."""


class EmptyContextError(GenerationError):
    """The assembled context contained no chunks; LLM was not called."""


class InvalidRequestError(GenerationError):
    """Query is blank or the request is otherwise malformed."""


class ProviderError(GenerationError):
    """The LLM provider returned an error (HTTP 4xx/5xx, timeout, 429).

    The message is safe to surface to callers — it never contains
    the API key or any other secret.
    """


class RateLimitError(ProviderError):
    """The provider returned HTTP 429 Too Many Requests."""


class TimeoutError(ProviderError):
    """The request to the provider timed out."""


class MalformedResponseError(GenerationError):
    """The provider returned a response that could not be parsed."""
