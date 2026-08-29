from __future__ import annotations


class RetrievalError(Exception):
    """Base class for all retrieval errors."""


class InvalidQueryError(RetrievalError):
    """Query is blank, or a retrieval parameter (top_k, score_threshold) is out of range."""


class EmbedFailureError(RetrievalError):
    """The embedder raised an exception while encoding the query."""


class StoreFailureError(RetrievalError):
    """The vector store raised an exception during the search call."""


class ConfigError(RetrievalError):
    """Misconfiguration detected at runtime (e.g. wrong collection vector dimension)."""
