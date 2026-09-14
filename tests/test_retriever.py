"""Retriever behaviour tests."""

from __future__ import annotations

from src.rag.retriever import Retriever


def test_retriever_returns_scores_and_metadata(indexed_store):
    store, embedder, _ = indexed_store
    retriever = Retriever(vector_store=store, embedder=embedder, top_k=3, min_score=0.0)
    result = retriever.retrieve("What is the pesticide paradox?", certification="CTFL")
    assert result.count >= 1
    chunk = result.chunks[0]
    assert isinstance(chunk.score, float)
    assert "document" in chunk.metadata
    citations = result.unique_citations(limit=3)
    assert len(citations) >= 1
