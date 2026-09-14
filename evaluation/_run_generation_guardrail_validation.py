"""Live validation of generation grounding guardrail — exact prior E2E cases."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import OPENAI_MODEL, has_openai_key
from src.rag.grounding import CERTIFICATION_TEMPERATURE, generation_settings, not_found_message
from src.rag.rag_pipeline import RAGPipeline
from src.trainer.evaluator import AnswerEvaluator
from src.trainer.trainer import Trainer

AGI_FLAGS = re.compile(
    r"human intelligence|intelligence humaine|any intellectual|tâche intellectuelle|"
    r"intellectual task|human cognitive|capacités cognitives|reconnaissance d['’]images|"
    r"image recognition|capturing noise",
    re.I,
)


def report_case(label: str, ans) -> dict:
    print("\n" + "=" * 70)
    print(label)
    print("found=", ans.found, "grounding=", ans.debug.get("grounding"))
    print("repair_attempted=", ans.debug.get("repair_attempted"))
    print("unsupported_count=", ans.debug.get("unsupported_claims_count"))
    print("unsupported=", ans.debug.get("unsupported_claims"))
    print("DRAFT:\n", (ans.debug.get("draft_answer") or "")[:800])
    if ans.debug.get("repaired_answer"):
        print("REPAIRED:\n", str(ans.debug.get("repaired_answer"))[:800])
    print("FINAL:\n", ans.answer)
    print("CITATIONS:", ans.citations)
    flags = []
    if AGI_FLAGS.search(ans.answer or ""):
        flags.append("agi_embellishment")
    if ans.found and ans.debug.get("grounding") != "PASS":
        flags.append("found_without_pass")
    print("FLAGS", flags or "NONE")
    return {
        "label": label,
        "found": ans.found,
        "grounding": ans.debug.get("grounding"),
        "repair_attempted": ans.debug.get("repair_attempted"),
        "unsupported_claims": ans.debug.get("unsupported_claims"),
        "draft_answer": ans.debug.get("draft_answer"),
        "repaired_answer": ans.debug.get("repaired_answer"),
        "final_answer": ans.answer,
        "citations": ans.citations,
        "flags": flags,
        "chunks": ans.debug.get("chunks"),
    }


def main() -> int:
    assert has_openai_key()
    print("model=", OPENAI_MODEL)
    print("settings=", generation_settings())
    print("temperature=", CERTIFICATION_TEMPERATURE)

    pipeline = RAGPipeline()
    out: dict = {"cases": [], "repairs_triggered": 0}

    e2e1_en = pipeline.ask(
        "What is the difference between narrow AI and general AI?",
        certification="CT-AI",
        response_language="en",
    )
    out["cases"].append(report_case("E2E1 EN", e2e1_en))

    e2e1_fr = pipeline.ask(
        "Quelle est la différence entre narrow AI et general AI ?",
        certification="CT-AI",
        response_language="fr",
    )
    out["cases"].append(report_case("E2E1 FR", e2e1_fr))

    e2e2 = pipeline.ask(
        "Explain precision, recall and F1-score for classifiers.",
        certification="CT-AI",
        response_language="en",
    )
    out["cases"].append(report_case("E2E2", e2e2))

    e2e3 = pipeline.ask(
        "How does back-to-back testing help with the test oracle problem?",
        certification="CT-AI",
        response_language="en",
    )
    out["cases"].append(report_case("E2E3", e2e3))

    # French question → English retrieval → French answer (same as prior bilingual check)
    fr_en_fr = pipeline.ask(
        "Quelle est la différence entre narrow AI et general AI ?",
        certification="CT-AI",
        response_language="fr",
    )
    cites = fr_en_fr.citations or []
    en_source = all(
        (c.get("document") or "").startswith("ISTQB") or "CTAI" in (c.get("document") or "")
        or c.get("source_file", "").endswith(".pdf")
        for c in cites
    ) if cites else False
    print("\n## French→English→French")
    print("found=", fr_en_fr.found, "grounding=", fr_en_fr.debug.get("grounding"))
    print("citations_en_corpus=", bool(cites), "sample=", cites[:1])
    out["french_en_fr"] = {
        "found": fr_en_fr.found,
        "grounding": fr_en_fr.debug.get("grounding"),
        "repair_attempted": fr_en_fr.debug.get("repair_attempted"),
        "final_answer": fr_en_fr.answer,
        "citations": cites,
        "flags": report_case("FR→EN→FR (detail)", fr_en_fr)["flags"],
    }

    trainer = Trainer()
    turn = trainer.explain(
        topic="Explain overfitting in machine learning in the context of AI testing.",
        certification="CT-AI",
        response_language="en",
    )
    print("\n## TRAINER")
    print("found=", turn.found, "grounding=", turn.debug.get("grounding"))
    print("repair=", turn.debug.get("repair_attempted"))
    print(turn.explanation[:600])
    print("Q:", turn.comprehension_question)
    evaluation = AnswerEvaluator().evaluate(
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
    print(evaluation.feedback[:500])
    out["trainer"] = {
        "found": turn.found,
        "grounding": turn.debug.get("grounding"),
        "repair_attempted": turn.debug.get("repair_attempted"),
        "explanation": turn.explanation,
        "comprehension_question": turn.comprehension_question,
        "verdict": evaluation.verdict,
        "eval_grounding": evaluation.debug.get("grounding"),
        "eval_repair": evaluation.debug.get("repair_attempted"),
    }

    hall = pipeline.ask(
        "According to the ISTQB CT-AI syllabus, what is the official CTFL pesticide paradox definition?",
        certification="CT-AI",
        response_language="en",
    )
    print("\n## OOS")
    print("found=", hall.found, "answer=", hall.answer)
    print("deterministic=", hall.answer == not_found_message("en"))
    out["hallucination"] = {
        "found": hall.found,
        "answer": hall.answer,
        "deterministic": hall.answer == not_found_message("en"),
        "llm_bypassed": hall.debug.get("grounding") == "N/A",
    }

    out["repairs_triggered"] = sum(
        1 for c in out["cases"] if c.get("repair_attempted")
    ) + int(bool(out["trainer"].get("repair_attempted"))) + int(
        bool(out["trainer"].get("eval_repair"))
    )

    Path("evaluation/_generation_guardrail_run.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nrepairs_triggered=", out["repairs_triggered"])
    print("Saved evaluation/_generation_guardrail_run.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
