"""CT-AI baseline retrieval + E2E validation harness (observe only)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import has_openai_key
from src.rag.rag_pipeline import RAGPipeline
from src.rag.retriever import Retriever
from src.trainer.evaluator import AnswerEvaluator
from src.trainer.trainer import Trainer


def show_retrieval(label: str, query: str, certification: str = "CT-AI") -> dict:
    retriever = Retriever(top_k=4)
    result = retriever.retrieve(query=query, certification=certification, top_k=4)
    print("=" * 70)
    print(f"[{label}] QUESTION: {query}")
    print(f"retrieved_count={result.count} certification_filter={certification}")
    for i, chunk in enumerate(result.chunks, 1):
        m = chunk.metadata
        preview = chunk.text.replace("\n", " ")[:220]
        print(f"\nTOP {i}: score={chunk.score:.4f}")
        print(f"  page={m.get('page')!r} chapter={m.get('chapter')!r} section={m.get('section')!r}")
        print(f"  document={m.get('document')!r}")
        print(f"  cert={m.get('certification')!r} version={m.get('version')!r} lang={m.get('language')!r}")
        print(f"  preview: {preview}")
    payload = {
        "query": query,
        "count": result.count,
        "chunks": [
            {
                "score": c.score,
                "page": c.metadata.get("page"),
                "chapter": c.metadata.get("chapter"),
                "section": c.metadata.get("section"),
                "preview": c.text[:300],
                "certification": c.metadata.get("certification"),
            }
            for c in result.chunks
        ],
    }
    return payload


def main() -> None:
    out: dict = {"english": [], "french": [], "filter_check": {}, "e2e": [], "trainer": {}, "hallucination": {}}

    english_qs = [
        (
            "EN1",
            "What is the difference between narrow AI and general AI?",
            # Expected area: ~p15 chapter 1.1
        ),
        (
            "EN2",
            "What is overfitting in machine learning?",
            # Expected: glossary/overfitting sections
        ),
        (
            "EN3",
            "Explain precision, recall and F1-score for classifiers.",
            # Expected: ~p33-34 metrics
        ),
        (
            "EN4",
            "What are the input layer, hidden layers and output layer in a deep neural network?",
            # Expected: ~p36
        ),
        (
            "EN5",
            "How does back-to-back testing help with the test oracle problem?",
            # Expected: ~p62
        ),
    ]

    print("\n######## STEP 4 — ENGLISH RETRIEVAL ########\n")
    for qid, q in english_qs:
        out["english"].append({"id": qid, **show_retrieval(qid, q)})

    print("\n######## STEP 5 — FRENCH → ENGLISH RETRIEVAL ########\n")
    french_qs = [
        ("FR1", "Quelle est la différence entre narrow AI et general AI ?", "EN1"),
        ("FR2", "Qu'est-ce que le overfitting en apprentissage automatique ?", "EN2"),
    ]
    for qid, q, linked in french_qs:
        payload = show_retrieval(qid, q)
        payload["linked_english"] = linked
        out["french"].append(payload)

    print("\n######## STEP 6 — CERTIFICATION FILTER ########\n")
    retriever = Retriever(top_k=4)
    # Programmatic: where filter must request CT-AI only; also attempt CTFL
    ctfl = retriever.retrieve("narrow AI overfitting neural network", certification="CTFL", top_k=4)
    ctai = retriever.retrieve("narrow AI overfitting neural network", certification="CT-AI", top_k=4)
    print(f"CTFL retrieved={ctfl.count} (expected 0 if no CTFL docs)")
    print(f"CT-AI retrieved={ctai.count}")
    print(f"All CT-AI certs? {all(c.certification=='CT-AI' for c in ctai.chunks)}")
    # Inspect where clause construction via source: all returned metas
    out["filter_check"] = {
        "ctfl_count": ctfl.count,
        "ctai_count": ctai.count,
        "ctai_all_ctai": all(c.certification == "CT-AI" for c in ctai.chunks),
        "ctai_certs": [c.certification for c in ctai.chunks],
    }

    # Save intermediate JSON for report writing
    Path("evaluation/_baseline_run.json").write_text(
        json.dumps({k: out[k] for k in ("english", "french", "filter_check")}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nSaved evaluation/_baseline_run.json (retrieval parts)")

    if not has_openai_key():
        print("\nOPENAI_API_KEY missing — skipping E2E/trainer/hallucination LLM steps.")
        print("has_openai_key=False")
        return

    print("\n######## STEP 7 — END-TO-END RAG ########\n")
    pipeline = RAGPipeline()
    e2e_qs = [
        ("E2E1", "What is the difference between narrow AI and general AI?", "en"),
        ("E2E2", "Explain precision, recall and F1-score for classifiers.", "en"),
        ("E2E3", "How does back-to-back testing help with the test oracle problem?", "en"),
    ]
    for qid, q, lang in e2e_qs:
        ans = pipeline.ask(question=q, certification="CT-AI", response_language=lang)
        print("=" * 70)
        print(f"{qid} Q: {q}")
        print(f"found={ans.found} retrieved={ans.retrieval.count}")
        for i, c in enumerate(ans.retrieval.chunks, 1):
            print(f"  chunk{i}: score={c.score:.4f} page={c.metadata.get('page')} preview={c.text[:120]!r}")
        print("--- CONTEXT (truncated) ---")
        print(ans.retrieval.context_text()[:1200])
        print("--- ANSWER ---")
        print(ans.answer)
        print("--- CITATIONS ---")
        print(ans.citations)
        out["e2e"].append(
            {
                "id": qid,
                "question": q,
                "found": ans.found,
                "answer": ans.answer,
                "citations": ans.citations,
                "pages": [c.metadata.get("page") for c in ans.retrieval.chunks],
                "scores": [c.score for c in ans.retrieval.chunks],
            }
        )

    print("\n######## STEP 8 — FRENCH RESPONSE FROM ENGLISH SOURCE ########\n")
    fr = pipeline.ask(
        question="Quelle est la différence entre narrow AI et general AI ?",
        certification="CT-AI",
        response_language="fr",
    )
    print("found=", fr.found)
    print("retrieved pages=", [c.metadata.get("page") for c in fr.retrieval.chunks])
    print("ANSWER:\n", fr.answer)
    print("CITATIONS:", fr.citations)
    out["french_response"] = {
        "answer": fr.answer,
        "citations": fr.citations,
        "pages": [c.metadata.get("page") for c in fr.retrieval.chunks],
        "found": fr.found,
    }

    print("\n######## STEP 9 — TRAINER MODE ########\n")
    trainer = Trainer()
    turn = trainer.explain(
        topic="Explain overfitting in machine learning in the context of AI testing.",
        certification="CT-AI",
        response_language="en",
    )
    print("found=", turn.found)
    print("EXPLANATION:\n", turn.explanation)
    print("COMPREHENSION Q:\n", turn.comprehension_question)
    print("CITATIONS:", turn.citations)
    evaluator = AnswerEvaluator()
    # Partially correct learner answer
    partial = (
        "Overfitting means the model performs well on training data but I am not sure why it happens "
        "or what testers should do about it."
    )
    evaluation = evaluator.evaluate(
        original_topic="overfitting in machine learning",
        comprehension_question=turn.comprehension_question or "What is overfitting?",
        student_answer=partial,
        certification="CT-AI",
        response_language="en",
    )
    print("VERDICT:", evaluation.verdict)
    print("FEEDBACK:\n", evaluation.feedback)
    out["trainer"] = {
        "explanation": turn.explanation,
        "comprehension_question": turn.comprehension_question,
        "citations": turn.citations,
        "verdict": evaluation.verdict,
        "feedback": evaluation.feedback,
        "eval_pages": [c.metadata.get("page") for c in evaluation.retrieval.chunks],
    }

    print("\n######## STEP 10 — HALLUCINATION CONTROL ########\n")
    # Deliberately outside CT-AI syllabus scope / not in document
    hq = "According to the ISTQB CT-AI syllabus, what is the official CTFL pesticide paradox definition?"
    hall = pipeline.ask(question=hq, certification="CT-AI", response_language="en")
    print("Q:", hq)
    print("retrieved=", hall.retrieval.count)
    for i, c in enumerate(hall.retrieval.chunks, 1):
        print(f" TOP{i} score={c.score:.4f} page={c.metadata.get('page')} preview={c.text[:180]!r}")
    print("found=", hall.found)
    print("ANSWER:\n", hall.answer)
    out["hallucination"] = {
        "question": hq,
        "found": hall.found,
        "answer": hall.answer,
        "count": hall.retrieval.count,
        "scores": [c.score for c in hall.retrieval.chunks],
        "pages": [c.metadata.get("page") for c in hall.retrieval.chunks],
        "previews": [c.text[:200] for c in hall.retrieval.chunks],
    }

    Path("evaluation/_baseline_run.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved full evaluation/_baseline_run.json")


if __name__ == "__main__":
    main()
