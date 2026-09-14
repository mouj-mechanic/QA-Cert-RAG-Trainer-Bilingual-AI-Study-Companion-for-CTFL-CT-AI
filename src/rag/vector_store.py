"""
Persistent ChromaDB vector store with metadata filtering support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_COLLECTION_NAME, VECTOR_DB_DIR

logger = logging.getLogger(__name__)


class VectorStore:
    """Simple ChromaDB wrapper — no LangChain."""

    def __init__(
        self,
        persist_directory: Path | str = VECTOR_DB_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection."""
        try:
            self._client.delete_collection(self.collection_name)
            logger.info("Deleted collection %s", self.collection_name)
        except Exception:  # noqa: BLE001 — collection may not exist
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """Add chunks in batches."""
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self._collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
        logger.info("Added %s vectors to collection %s", total, self.collection_name)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Similarity search with optional metadata filter.

        Chroma returns distances (lower = closer for cosine space when using
        their distance metric). We convert to a simple similarity score:
        similarity = 1 - distance (clamped).
        """
        if self.count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]], "scores": [[]]}

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)
        distances = result.get("distances") or [[]]
        scores = [[max(0.0, 1.0 - d) for d in row] for row in distances]
        result["scores"] = scores
        return result

    def is_ready(self) -> bool:
        return self.count > 0
