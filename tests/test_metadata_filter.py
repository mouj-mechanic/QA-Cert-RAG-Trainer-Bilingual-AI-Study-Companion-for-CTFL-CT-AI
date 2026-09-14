"""Certification metadata filtering must isolate CTFL from CT-AI."""

from __future__ import annotations

from src.rag.retriever import Retriever


def test_ctfl_filter_excludes_ctai(indexed_store):
    store, embedder, _ = indexed_store
    retriever = Retriever(vector_store=store, embedder=embedder, top_k=5, min_score=0.0)
    result = retriever.retrieve("pesticide paradox testing principles", certification="CTFL")
    assert result.chunks, "Expected CTFL hits for pesticide paradox"
    assert all(c.certification == "CTFL" for c in result.chunks)


def test_ctai_filter_excludes_ctfl(indexed_store):
    store, embedder, _ = indexed_store
    retriever = Retriever(vector_store=store, embedder=embedder, top_k=5, min_score=0.0)
    result = retriever.retrieve("precision recall F1-score training data", certification="CT-AI")
    assert result.chunks, "Expected CT-AI hits for ML metrics"
    assert all(c.certification == "CT-AI" for c in result.chunks)


def test_empty_store_returns_no_chunks(tmp_path):
    from src.rag.embeddings import EmbeddingModel
    from src.rag.vector_store import VectorStore

    store = VectorStore(persist_directory=tmp_path / "empty_db", collection_name="empty")
    store.reset()
    retriever = Retriever(vector_store=store, embedder=EmbeddingModel(), top_k=3, min_score=0.0)
    result = retriever.retrieve("anything", certification="CTFL")
    assert result.chunks == []
    assert result.has_evidence is False
