"""Shared pytest fixtures — no paid API calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import load_all_documents
from src.rag.vector_store import VectorStore


CTFL_TEXT = """1.2 Testing Principles

1.3 Testing shows the presence of defects
Testing can show that defects are present, but cannot prove that there are no defects.

1.3 The pesticide paradox
If the same tests are repeated over and over again, eventually these tests no longer find new defects.
To overcome this pesticide paradox, test cases need to be regularly reviewed and revised.
"""

CTAI_TEXT = """3.2 Quality metrics for ML models

Precision is the proportion of positive identifications that were actually correct.
Recall is the proportion of actual positives that were identified correctly.
The F1-score is the harmonic mean of precision and recall.

Training data is used to train the model. Test data is used to evaluate the trained model
and should be independent from the training data.
"""


@pytest.fixture
def sample_resources(tmp_path: Path) -> Path:
    """Create a temporary resources tree with synthetic study notes (not official syllabus)."""
    ctfl_en = tmp_path / "ctfl" / "en"
    ctai_en = tmp_path / "ct_ai" / "en"
    ctfl_fr = tmp_path / "ctfl" / "fr"
    ctai_fr = tmp_path / "ct_ai" / "fr"
    for folder in (ctfl_en, ctai_en, ctfl_fr, ctai_fr):
        folder.mkdir(parents=True)

    (ctfl_en / "CTFL_Sample_Notes.txt").write_text(CTFL_TEXT, encoding="utf-8")
    (ctai_en / "CTAI_Sample_Notes.txt").write_text(CTAI_TEXT, encoding="utf-8")
    (ctfl_fr / "CTFL_Notes_FR.txt").write_text(
        "1.3 Paradoxe du pesticide\n"
        "Si les memes tests sont repetes, ils finissent par ne plus trouver de nouveaux defauts.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def indexed_store(sample_resources: Path, tmp_path: Path):
    """Build an in-temp Chroma index from sample resources using real embeddings.

    Embedding model download may take time on first run; mocked in unit tests that
    do not need semantic search.
    """
    from src.ingestion.document_loader import load_all_documents
    from src.rag.embeddings import EmbeddingModel

    docs = load_all_documents(sample_resources)
    chunks = chunk_documents(docs, chunk_size=400, overlap=50)
    embedder = EmbeddingModel()
    store = VectorStore(persist_directory=tmp_path / "chroma_test", collection_name="test_istqb")
    store.reset()
    texts = [c.text for c in chunks]
    metas = [c.to_chroma()[1] for c in chunks]
    ids = [f"t{i}" for i in range(len(chunks))]
    embeddings = embedder.embed_documents(texts)
    store.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
    return store, embedder, chunks
