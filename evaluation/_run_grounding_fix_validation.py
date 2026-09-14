"""Post bug-fix grounding validation (LLM steps). No RAG architecture changes."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MIN_RELEVANCE_SCORE, has_openai_key
from src.rag.grounding import has_sufficient_evidence, not_found_message
from src.rag.rag_pipeline import RAGPipeline
from src.rag.retriever import Retriever
from src.trainer.evaluator import AnswerEvaluator
from src.trainer.trainer import Trainer

FORBIDDEN_WEAK_AI = re.compile(r"\bweak\s+AI\b", re.I)


def dump_chunks(retrieval) -> list[dict]:
    return [
        {
            "score": round(c.score, 4),
            "page": c.metadata.get("page"),
            "chapter": c.metadata.get("chapter"),
            "section": c.metadata.get("section"),
            "preview": c.text[:220].replace("\n", " "),
        }
        for c in retrieval.chunks
    ]


def main() -> int:
    assert has_openai_key(), "OPENAI_API_KEY required"
    print("MIN_RELEVANCE_SCORE=", MIN_RELEVANCE_SCORE)

    # Score table (raw retrieve with min_score=0 to observe bands)
    raw = Retriever(top_k=4, min_score=0.0)
    gated = Retriever(top_k=4)  # uses config floor
    table = []
    probes = [
        ("EN1", "What is the difference between narrow AI and general AI?", True),
        ("EN2", "What is overfitting in machine learning?", True),
        ("EN3", "Explain precision, recall and F1-score for classifiers.", True),
        ("EN4", "What are the input layer, hidden layers and output layer in a deep neural network?", True),
        ("EN5", "How does back-to-back testing help with the test oracle problem?", True),
        ("FR1", "Quelle est la différence entre narrow AI et general AI ?", True),
        ("FR2", "Qu'est-ce que le overfitting en apprentissage automatique ?", True),
        (
            "OOS",
            "According to the ISTQB CT-AI syllabus, what is the official CTFL pesticide paradox definition?",
            False,
        ),
    ]
    print("\n## Score comparison (raw min_score=0)")
    print("id | relevant? | top1 | top2 | top3 | top4 | gated_count | sufficient?")
    for qid, q, relevant in probes:
        r = raw.retrieve(q, certification="CT-AI", top_k=4)
        scores = [round(c.score, 4) for c in r.chunks] + [None] * (4 - len(r.chunks))
        g = gated.retrieve(q, certification="CT-AI", top_k=4)
        ok = has_sufficient_evidence(g)
        row = {
            "id": qid,
            "relevant": relevant,
            "scores": scores[:4],
            "gated_count": g.count,
            "sufficient": ok,
        }
        table.append(row)
        print(
            f"{qid} | {relevant} | {scores[0]} | {scores[1]} | {scores[2]} | {scores[3]} | "
            f"{g.count} | {ok}"
        )

    pipeline = RAGPipeline()
    out: dict = {"score_table": table, "e2e": [], "french": {}, "trainer": {}, "hallucination": {}}

    e2e_qs = [
        ("E2E1", "What is the difference between narrow AI and general AI?", "en"),
        ("E2E2", "Explain precision, recall and F1-score for classifiers.", "en"),
        ("E2E3", "How does back-to-back testing help with the test oracle problem?", "en"),
    ]
    for qid, q, lang in e2e_qs:
        ans = pipeline.ask(q, certification="CT-AI", response_language=lang)
        print("\n" + "=" * 70)
        print(qid, "found=", ans.found)
        print("CHUNKS:", dump_chunks(ans.retrieval))
        print("ANSWER:\n", ans.answer)
        print("CITATIONS:", ans.citations)
        weak = bool(FORBIDDEN_WEAK_AI.search(ans.answer))
        print("CONTAINS_WEAK_AI=", weak)
        out["e2e"].append(
            {
                "id": qid,
                "question": q,
                "found": ans.found,
                "answer": ans.answer,
                "citations": ans.citations,
                "chunks": dump_chunks(ans.retrieval),
                "contains_weak_ai": weak,
                "llm_skipped": not ans.found and ans.answer == not_found_message(lang),
            }
        )

    fr = pipeline.ask(
        "Quelle est la différence entre narrow AI et general AI ?",
        certification="CT-AI",
        response_language="fr",
    )
    print("\n## FRENCH")
    print("found=", fr.found, "pages=", [c.metadata.get("page") for c in fr.retrieval.chunks])
    print(fr.answer)
    print(fr.citations)
    out["french"] = {
        "found": fr.found,
        "answer": fr.answer,
        "citations": fr.citations,
        "chunks": dump_chunks(fr.retrieval),
        "contains_weak_ai": bool(FORBIDDEN_WEAK_AI.search(fr.answer)),
    }

    trainer = Trainer()
    turn = trainer.explain(
        topic="Explain overfitting in machine learning in the context of AI testing.",
        certification="CT-AI",
        response_language="en",
    )
    print("\n## TRAINER")
    print("found=", turn.found)
    print(turn.explanation)
    print("Q:", turn.comprehension_question)
    evaluator = AnswerEvaluator()
    evaluation = evaluator.evaluate(
        original_topic="overfitting in machine learning",
        comprehension_question=turn.comprehension_question or "What is overfitting?",
        student_answer=(
            "Overfitting means the model performs well on training data but I am not sure "
            "why it happens or what testers should do about it."
        ),
        certification="CT-AI",
        response_language="en",
    )
    print("VERDICT:", evaluation.verdict)
    print(evaluation.feedback)
    out["trainer"] = {
        "found": turn.found,
        "explanation": turn.explanation,
        "comprehension_question": turn.comprehension_question,
        "citations": turn.citations,
        "verdict": evaluation.verdict,
        "feedback": evaluation.feedback,
        "eval_pages": [c.metadata.get("page") for c in evaluation.retrieval.chunks],
        "eval_sufficient": has_sufficient_evidence(evaluation.retrieval),
    }

    hq = "According to the ISTQB CT-AI syllabus, what is the official CTFL pesticide paradox definition?"
    hall = pipeline.ask(hq, certification="CT-AI", response_language="en")
    print("\n## HALLUCINATION")
    print("found=", hall.found, "count=", hall.retrieval.count)
    print("chunks=", dump_chunks(hall.retrieval))
    print("ANSWER:", hall.answer)
    print("deterministic=", hall.answer == not_found_message("en"))
    out["hallucination"] = {
        "found": hall.found,
        "answer": hall.answer,
        "chunks": dump_chunks(hall.retrieval),
        "deterministic_refusal": hall.answer == not_found_message("en"),
        "sufficient": has_sufficient_evidence(hall.retrieval),
    }

    Path("evaluation/_grounding_fix_run.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSaved evaluation/_grounding_fix_run.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
