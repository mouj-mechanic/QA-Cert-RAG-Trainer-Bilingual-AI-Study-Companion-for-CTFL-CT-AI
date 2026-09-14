"""
Student answer evaluation against retrieved official context.

Uses retrieval re-fetch + generation grounding guardrail for feedback text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from src.llm.provider import LLMProvider, get_llm_provider
from src.rag.grounding import (
    generate_with_grounding_guardrail,
    generation_settings,
    has_sufficient_evidence,
    not_found_message,
)
from src.rag.retriever import RetrievalResult, Retriever

logger = logging.getLogger(__name__)

Verdict = Literal["correct", "partial", "incorrect", "unknown"]


@dataclass
class EvaluationResult:
    """Pedagogical feedback for a student answer."""

    verdict: Verdict
    feedback: str
    retrieval: RetrievalResult
    citations: list[dict[str, Any]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def _parse_verdict(text: str) -> Verdict:
    lower = text.lower()
    if "🟡" in text or re.search(r"partially correct|partiellement", lower):
        return "partial"
    if "❌" in text or re.search(r"\bincorrect\b", lower):
        return "incorrect"
    if "✅" in text or re.search(r"\bcorrect\b", lower):
        return "correct"
    return "unknown"


class AnswerEvaluator:
    """Evaluate learner answers using retrieved ISTQB context + grounding guardrail."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or get_llm_provider()

    def evaluate(
        self,
        original_topic: str,
        comprehension_question: str,
        student_answer: str,
        certification: str,
        response_language: str,
        chapter: str | None = None,
    ) -> EvaluationResult:
        """Retrieve relevant source again, then evaluate with guarded feedback."""
        query = f"{original_topic}\n{comprehension_question}"
        retrieval = self.retriever.retrieve(
            query=query,
            certification=certification,
            chapter=chapter,
        )
        debug: dict[str, Any] = {
            "query": query,
            "certification": certification,
            "response_language": response_language,
            "retrieved_count": retrieval.count,
            "student_answer": student_answer,
            "sufficient_evidence": has_sufficient_evidence(retrieval),
            "generation_settings": generation_settings(),
            "chunks": [
                {
                    "score": c.score,
                    "metadata": c.metadata,
                    "preview": c.text[:240],
                    "chunk_id": f"S{i}",
                }
                for i, c in enumerate(retrieval.chunks, start=1)
            ],
        }

        if not has_sufficient_evidence(retrieval):
            logger.info("Insufficient evidence — skipping LLM for trainer evaluation")
            debug["grounding"] = "N/A"
            debug["repair_attempted"] = False
            debug["unsupported_claims_count"] = 0
            return EvaluationResult(
                verdict="unknown",
                feedback=not_found_message(response_language),
                retrieval=retrieval,
                citations=[],
                debug=debug,
            )

        task = (
            "Evaluate the student's answer against the retrieved official context only.\n"
            "Classify as exactly one of: ✅ Correct / 🟡 Partially correct / ❌ Incorrect.\n"
            "Explain what was correct, incomplete, and incorrect.\n"
            "Provide a short corrected explanation grounded only in the retrieved chunks.\n"
            f"Original topic: {original_topic}\n"
            f"Comprehension question: {comprehension_question}\n"
            f"Student's answer: {student_answer}"
        )
        guarded = generate_with_grounding_guardrail(
            llm=self.llm,
            task_instruction=task,
            retrieval=retrieval,
            language=response_language,
            certification=certification,
            include_comprehension_question=False,
        )
        debug.update(guarded.debug)

        if not guarded.found or not guarded.grounded:
            return EvaluationResult(
                verdict="unknown",
                feedback=guarded.answer,
                retrieval=retrieval,
                citations=[],
                debug=debug,
            )

        return EvaluationResult(
            verdict=_parse_verdict(guarded.answer),
            feedback=guarded.answer,
            retrieval=retrieval,
            citations=retrieval.unique_citations(limit=3),
            debug=debug,
        )
