from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from app.ingestion.tokenize import BGE_MODEL_NAME

logger = logging.getLogger(__name__)

# BGE bge-small-en-v1.5 produces 384-dimensional L2-normalised vectors.
EMBEDDING_DIM = 384
EMBEDDING_MODEL: str = BGE_MODEL_NAME  # "BAAI/bge-small-en-v1.5"


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Minimal interface for a batch text embedder.

    Implementations MUST return L2-normalised float vectors of exactly
    EMBEDDING_DIM (384) dimensions, one per input text, in the same order.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one normalised vector per text."""
        ...


class BgeEmbedder:
    """BGE bge-small-en-v1.5 embedder backed by sentence-transformers.

    The model is loaded on first ``embed()`` call so that:
    - importing this module is cheap (no heavy torch import at module load time);
    - tests that inject a fake embedder never trigger a model download.
    """

    def __init__(self) -> None:
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return L2-normalised embedding vectors for *texts*, same order."""
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return [v.tolist() for v in vectors]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Loaded embedding model %s (dim=%d)", EMBEDDING_MODEL, EMBEDDING_DIM)
        return self._model
