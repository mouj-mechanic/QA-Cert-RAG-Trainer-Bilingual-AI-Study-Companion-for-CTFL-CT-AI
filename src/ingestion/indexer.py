"""
Build or rebuild the local ChromaDB index from /resources.

Usage:
    python -m src.ingestion.indexer
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict

from src.config import RESOURCES_DIR, VECTOR_DB_DIR
from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import load_all_documents
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_index(reset: bool = True) -> dict[str, dict[str, int]]:
    """
    Scan resources, chunk, embed, and persist to ChromaDB.

    Returns nested stats: {certification: {language: chunk_count, "documents": n}}
    """
    logger.info("Scanning resources at %s", RESOURCES_DIR)
    documents = load_all_documents(RESOURCES_DIR)

    if not documents:
        print()
        print("No supported documents found under /resources.")
        print("Place official PDF/TXT/MD files in:")
        print("  resources/ctfl/en/")
        print("  resources/ctfl/fr/")
        print("  resources/ct_ai/en/")
        print("  resources/ct_ai/fr/")
        print()
        print("Do not commit copyrighted PDFs to a public repository.")
        return {}

    chunks = chunk_documents(documents)
    logger.info("Created %s chunks from %s documents", len(chunks), len(documents))

    embedder = EmbeddingModel()
    store = VectorStore(persist_directory=VECTOR_DB_DIR)
    if reset:
        store.reset()

    texts = [c.text for c in chunks]
    metadatas = [c.to_chroma()[1] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    logger.info("Embedding %s chunks with %s …", len(texts), embedder.model_name)
    embeddings = embedder.embed_documents(texts)

    store.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    logger.info("Index written to %s", VECTOR_DB_DIR)

    # Statistics
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    doc_counts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for chunk in chunks:
        cert = chunk.metadata["certification"]
        lang = chunk.metadata["language"].upper()
        stats[cert][lang] += 1
        doc_counts[cert][lang].add(chunk.metadata["document"])

    print()
    print("=" * 40)
    print("Indexing complete")
    print("=" * 40)
    for cert in sorted(stats.keys()):
        for lang in sorted(stats[cert].keys()):
            n_docs = len(doc_counts[cert][lang])
            n_chunks = stats[cert][lang]
            print(f"{cert} / {lang}")
            print(f"  {n_docs} document{'s' if n_docs != 1 else ''}")
            print(f"  {n_chunks} chunks")
            print()
    print(f"Total chunks: {len(chunks)}")
    print(f"Vector DB: {VECTOR_DB_DIR}")
    return {c: dict(langs) for c, langs in stats.items()}


def main() -> int:
    build_index(reset=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
