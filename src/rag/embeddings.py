"""
Multilingual embedding layer — kept separate from the LLM.

Default model: paraphrase-multilingual-MiniLM-L12-v2
Supports French ↔ English semantic similarity for cross-lingual retrieval.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Thin wrapper around sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        if not texts:
            return []
        vectors = self._model.encode(texts, show_progress_bar=len(texts) > 20, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Cached singleton for Streamlit / CLI reuse."""
    return EmbeddingModel()
