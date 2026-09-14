"""
Retriever with mandatory certification metadata filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import MIN_RELEVANCE_SCORE, TOP_K
from src.rag.embeddings import EmbeddingModel, get_embedding_model
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """One retrieved chunk with score and metadata."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def certification(self) -> str:
        return str(self.metadata.get("certification", ""))

    @property
    def citation_label(self) -> str:
        meta = self.metadata
        cert = meta.get("certification", "?")
        version = meta.get("version", "")
        parts = [f"{cert} {version}".strip()]
        if meta.get("chapter"):
            parts.append(f"Chapter {meta['chapter']}")
        if meta.get("section"):
            parts.append(f"Section {meta['section']}")
        if meta.get("page") not in (None, ""):
            parts.append(f"Page {meta['page']}")
        if meta.get("document"):
            parts.append(str(meta["document"]))
        return " — ".join(parts)


@dataclass
class RetrievalResult:
    """Full retrieval result for debug panel and pipeline."""

    query: str
    certification: str
    chunks: list[RetrievedChunk]
    top_k: int

    @property
    def count(self) -> int:
        return len(self.chunks)

    @property
    def has_evidence(self) -> bool:
        """Backward-compatible alias for the central grounding decision."""
        from src.rag.grounding import has_sufficient_evidence

        return has_sufficient_evidence(self)

    def context_text(self, max_chunks: int | None = None) -> str:
        """Build LLM context from retrieved chunks."""
        selected = self.chunks if max_chunks is None else self.chunks[:max_chunks]
        blocks: list[str] = []
        for i, chunk in enumerate(selected, start=1):
            blocks.append(
                f"[Source {i}] {chunk.citation_label}\n{chunk.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def unique_citations(self, limit: int = 3) -> list[dict[str, Any]]:
        """Deduplicate sources for UI display (up to `limit`)."""
        seen: set[str] = set()
        citations: list[dict[str, Any]] = []
        for chunk in self.chunks:
            key = (
                f"{chunk.metadata.get('document')}|"
                f"{chunk.metadata.get('chapter')}|"
                f"{chunk.metadata.get('section')}|"
                f"{chunk.metadata.get('page')}"
            )
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "certification": chunk.metadata.get("certification", ""),
                    "version": chunk.metadata.get("version", ""),
                    "chapter": chunk.metadata.get("chapter", ""),
                    "section": chunk.metadata.get("section", ""),
                    "page": chunk.metadata.get("page", ""),
                    "document": chunk.metadata.get("document", ""),
                    "source_file": chunk.metadata.get("source_file", ""),
                    "score": round(chunk.score, 4),
                }
            )
            if len(citations) >= limit:
                break
        return citations


class Retriever:
    """
    Top-k similarity retriever with hard certification filter.

    Selecting CTFL must never return CT-AI chunks (and vice versa).
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: EmbeddingModel | None = None,
        top_k: int = TOP_K,
        min_score: float = MIN_RELEVANCE_SCORE,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or get_embedding_model()
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        certification: str,
        top_k: int | None = None,
        chapter: str | None = None,
    ) -> RetrievalResult:
        """Retrieve chunks filtered by certification (and optional chapter)."""
        k = top_k or self.top_k
        where: dict[str, Any] = {"certification": certification}
        if chapter:
            where = {
                "$and": [
                    {"certification": certification},
                    {"chapter": str(chapter)},
                ]
            }

        logger.info(
            "Retrieving top_%s for certification=%s chapter=%s query=%r",
            k,
            certification,
            chapter,
            query[:80],
        )

        if not self.vector_store.is_ready():
            logger.warning("Vector store is empty — no retrieval possible")
            return RetrievalResult(query=query, certification=certification, chunks=[], top_k=k)

        embedding = self.embedder.embed_query(query)
        raw = self.vector_store.query(query_embedding=embedding, top_k=k, where=where)

        chunks: list[RetrievedChunk] = []
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        scores = (raw.get("scores") or [[]])[0]

        for text, meta, score in zip(documents, metadatas, scores):
            meta = meta or {}
            # Safety: never mix certifications even if filter failed
            if meta.get("certification") != certification:
                logger.warning(
                    "Dropped chunk with unexpected certification %s (wanted %s)",
                    meta.get("certification"),
                    certification,
                )
                continue
            if score < self.min_score:
                logger.debug("Dropped low-score chunk (%.3f < %.3f)", score, self.min_score)
                continue
            chunks.append(RetrievedChunk(text=text or "", score=float(score), metadata=dict(meta)))

        logger.info("Retrieved %s chunks after filtering", len(chunks))
        return RetrievalResult(query=query, certification=certification, chunks=chunks, top_k=k)
