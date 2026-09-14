"""
Retrieval evaluation CLI.

Usage:
    python -m evaluation.evaluate_retrieval

Computes Top-1 / Top-3 / Top-5 hit rates against evaluation/questions.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVALUATION_DIR, VECTOR_DB_DIR
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _is_doc_match(doc: str, expected: str) -> bool:
    if not expected or expected.startswith("REPLACE_WITH_"):
        return False
    return expected.lower() in doc.lower() or doc.lower() == expected.lower()


def hit_at_k(chunks: list, expected_document: str, expected_chapter: str, k: int) -> bool:
    """True when expected document (and optional chapter) appears in top-k."""
    for chunk in chunks[:k]:
        doc = str(chunk.metadata.get("document", ""))
        chapter = str(chunk.metadata.get("chapter", ""))
        if not _is_doc_match(doc, expected_document):
            continue
        if expected_chapter and chapter and str(chapter) != str(expected_chapter):
            continue
        return True
    return False


def main() -> int:
    questions_path = EVALUATION_DIR / "questions.json"
    if not questions_path.exists():
        print(f"Missing {questions_path}")
        return 1

    store = VectorStore(persist_directory=VECTOR_DB_DIR)
    if not store.is_ready():
        print("Vector index is empty. Run: python -m src.ingestion.indexer")
        return 1

    items = json.loads(questions_path.read_text(encoding="utf-8"))
    placeholders = [
        q for q in items if str(q.get("expected_document", "")).startswith("REPLACE_WITH_")
    ]
    if placeholders:
        print(
            f"Note: {len(placeholders)}/{len(items)} questions still use "
            "PLACEHOLDER expected_document values."
        )
        print("Update evaluation/questions.json after indexing your official resources.\n")

    retriever = Retriever(vector_store=store, top_k=5, min_score=0.0)

    top1 = top3 = top5 = 0
    evaluated = 0
    cert_leaks = 0

    print("=" * 50)
    print("Chat_ISTQB — Retrieval evaluation")
    print("=" * 50)

    for item in items:
        qid = item["id"]
        certification = item["certification"]
        question = item["question"]
        expected_doc = item.get("expected_document", "")
        expected_chapter = item.get("expected_chapter") or ""

        result = retriever.retrieve(query=question, certification=certification, top_k=5)

        for chunk in result.chunks:
            if chunk.metadata.get("certification") != certification:
                cert_leaks += 1

        if expected_doc.startswith("REPLACE_WITH_"):
            print(f"[{qid}] SKIPPED (placeholder expected_document)")
            continue

        evaluated += 1
        h1 = hit_at_k(result.chunks, expected_doc, expected_chapter, 1)
        h3 = hit_at_k(result.chunks, expected_doc, expected_chapter, 3)
        h5 = hit_at_k(result.chunks, expected_doc, expected_chapter, 5)
        top1 += int(h1)
        top3 += int(h3)
        top5 += int(h5)
        status = "HIT" if h5 else "MISS"
        print(
            f"[{qid}] {status}  top1={h1} top3={h3} top5={h5}  retrieved={result.count}"
        )

    print()
    print(f"Questions in file     : {len(items)}")
    print(f"Questions evaluated   : {evaluated}")
    if evaluated:
        print(f"Top-1 hit rate        : {top1 / evaluated:.0%} ({top1}/{evaluated})")
        print(f"Top-3 hit rate        : {top3 / evaluated:.0%} ({top3}/{evaluated})")
        print(f"Top-5 hit rate        : {top5 / evaluated:.0%} ({top5}/{evaluated})")
    else:
        print("No questions evaluated yet — fill in expected_document placeholders.")
    print(f"Certification leaks   : {cert_leaks} (must be 0)")
    print()
    print("Future metrics (not computed in V1): answer correctness, groundedness,")
    print("citation correctness, hallucination rate.")
    return 0 if cert_leaks == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
