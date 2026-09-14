"""
Trainer mode: explain → wait for student answer → evaluate.

Generation uses the shared grounding guardrail (draft → validate → one repair).
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

TrainerStage = Literal["idle", "awaiting_comprehension", "evaluated"]


@dataclass
class TrainerTurn:
    """One trainer explanation turn ending with a comprehension question."""

    explanation: str
    comprehension_question: str
    found: bool
    retrieval: RetrievalResult
    citations: list[dict[str, Any]] = field(default_factory=list)
    topic: str = ""
    debug: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""


def _fallback_comprehension_question(language: str) -> str:
    if language == "fr":
        return "Pouvez-vous reformuler ce concept avec vos propres mots à partir de l'explication ?"
    return "Can you restate this concept in your own words based on the explanation?"


def _split_comprehension_question(full_text: str, language: str) -> tuple[str, str]:
    """Split trainer output into explanation + final comprehension question if combined."""
    markers = [
        r"🧠\s*Check your understanding\s*:?\s*",
        r"Check your understanding\s*:?\s*",
        r"🧠\s*Vérifie(?:z)? ta compréhension\s*:?\s*",
        r"Vérifie(?:z)? (?:ta|votre) compréhension\s*:?\s*",
        r"Question de compréhension\s*:?\s*",
        r"Comprehension question\s*:?\s*",
    ]
    for marker in markers:
        match = re.search(marker, full_text, flags=re.IGNORECASE)
        if match:
            explanation = full_text[: match.start()].strip()
            question = full_text[match.end() :].strip()
            if question:
                return explanation, question

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]
    if len(paragraphs) >= 2 and paragraphs[-1].endswith("?"):
        return "\n\n".join(paragraphs[:-1]), paragraphs[-1]
    return full_text.strip(), _fallback_comprehension_question(language)


class Trainer:
    """Patient, rigorous ISTQB trainer backed by retrieval + grounding guardrail."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or get_llm_provider()

    def explain(
        self,
        topic: str,
        certification: str,
        response_language: str,
        chapter: str | None = None,
    ) -> TrainerTurn:
        """Retrieve, explain (guarded), and produce one comprehension question."""
        retrieval = self.retriever.retrieve(
            query=topic,
            certification=certification,
            chapter=chapter,
        )
        debug: dict[str, Any] = {
            "query": topic,
            "certification": certification,
            "response_language": response_language,
            "retrieved_count": retrieval.count,
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
            logger.info("Insufficient evidence — skipping LLM for trainer explain")
            msg = not_found_message(response_language)
            debug["grounding"] = "N/A"
            debug["repair_attempted"] = False
            debug["unsupported_claims_count"] = 0
            return TrainerTurn(
                explanation=msg,
                comprehension_question="",
                found=False,
                retrieval=retrieval,
                citations=[],
                topic=topic,
                debug=debug,
                raw_response=msg,
            )

        task = (
            "You are a patient ISTQB trainer. Explain the learner topic using ONLY the "
            "retrieved chunks. Preserve official terminology from the chunks. "
            "Cite certification/chapter/section/page briefly when present in chunk headers. "
            f"Learner topic: {topic}"
        )
        guarded = generate_with_grounding_guardrail(
            llm=self.llm,
            task_instruction=task,
            retrieval=retrieval,
            language=response_language,
            certification=certification,
            include_comprehension_question=True,
        )
        debug.update(guarded.debug)

        if not guarded.found or not guarded.grounded:
            return TrainerTurn(
                explanation=guarded.answer,
                comprehension_question="",
                found=False,
                retrieval=retrieval,
                citations=[],
                topic=topic,
                debug=debug,
                raw_response=guarded.draft_answer,
            )

        explanation = guarded.answer
        question = guarded.comprehension_question.strip()
        if not question:
            explanation, question = _split_comprehension_question(explanation, response_language)
        if not question:
            question = _fallback_comprehension_question(response_language)

        return TrainerTurn(
            explanation=explanation,
            comprehension_question=question,
            found=True,
            retrieval=retrieval,
            citations=retrieval.unique_citations(limit=3),
            topic=topic,
            debug=debug,
            raw_response=guarded.answer,
        )
