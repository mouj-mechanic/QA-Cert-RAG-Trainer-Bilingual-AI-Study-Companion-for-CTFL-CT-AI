"""Chunking and metadata tests."""

from __future__ import annotations

from src.ingestion.chunker import chunk_document, chunk_documents
from src.ingestion.document_loader import DocumentPage, LoadedDocument, load_all_documents


def test_chunks_preserve_certification_metadata(sample_resources):
    docs = load_all_documents(sample_resources)
    chunks = chunk_documents(docs, chunk_size=300, overlap=40)
    assert chunks, "Expected at least one chunk"

    required = {"certification", "version", "language", "document", "source_file"}
    for chunk in chunks:
        assert required.issubset(chunk.metadata.keys())
        assert chunk.metadata["certification"] in {"CTFL", "CT-AI"}
        assert chunk.metadata["version"] in {"4.0.1", "2.0"}
        assert chunk.metadata["language"] in {"en", "fr"}
        assert chunk.text.strip()


def test_chunk_overlap_produces_multiple_chunks():
    long_text = ("Testing principle paragraph. " * 80).strip()
    doc = LoadedDocument(
        source_file="ctfl/en/long.txt",
        certification="CTFL",
        version="4.0.1",
        language="en",
        document="long",
        pages=[DocumentPage(text=long_text, page=1, chapter="1", section="1.1")],
    )
    chunks = chunk_document(doc, chunk_size=200, overlap=40)
    assert len(chunks) >= 2
    assert all(c.metadata["page"] == 1 for c in chunks)
    assert all(c.metadata["chapter"] == "1" for c in chunks)


def test_empty_resources_fail_gracefully(tmp_path):
    empty = tmp_path / "resources"
    (empty / "ctfl" / "en").mkdir(parents=True)
    docs = load_all_documents(empty)
    assert docs == []
