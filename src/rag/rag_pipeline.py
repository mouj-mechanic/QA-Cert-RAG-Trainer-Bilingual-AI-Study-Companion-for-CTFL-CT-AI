"""
End-to-end RAG pipeline: retrieve → evidence gate → grounded generate.

Interviewable flow:
retrieval → enough evidence?
  NO  → deterministic refusal (no LLM)
  YES → structured draft → validate → one repair → fallback if still unsafe
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.llm.provider import LLMProvider, get_llm_provider
from src.rag.grounding import (
    generate_with_grounding_guardrail,
    generation_settings,
    has_sufficient_evidence,
    not_found_message,
)
from src.rag.retriever import RetrievalResult, Retriever

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """Grounded answer package for the UI."""

    answer: str
    found: bool
    retrieval: RetrievalResult
    citations: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "ask"
    debug: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Retrieve-then-generate with retrieval + generation grounding gates."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or get_llm_provider()

    def ask(
        self,
        question: str,
        certification: str,
        response_language: str,
        chapter: str | None = None,
        top_k: int | None = None,
    ) -> RAGAnswer:
        """Ask mode: grounded answer with citations."""
        retrieval = self.retriever.retrieve(
            query=question,
            certification=certification,
            top_k=top_k,
            chapter=chapter,
        )

        sufficient = has_sufficient_evidence(retrieval)
        debug: dict[str, Any] = {
            "query": question,
            "certification": certification,
            "response_language": response_language,
            "retrieved_count": retrieval.count,
            "sufficient_evidence": sufficient,
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

        if not sufficient:
            logger.info("Insufficient evidence — skipping LLM for ask mode")
            debug["grounding"] = "N/A"
            debug["repair_attempted"] = False
            debug["unsupported_claims_count"] = 0
            return RAGAnswer(
                answer=not_found_message(response_language),
                found=False,
                retrieval=retrieval,
                citations=[],
                mode="ask",
                debug=debug,
            )

        task = (
            "Answer the learner's certification question using ONLY the retrieved chunks.\n"
            f"Learner question: {question}"
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

        return RAGAnswer(
            answer=guarded.answer,
            found=guarded.found and guarded.grounded,
            retrieval=retrieval,
            citations=retrieval.unique_citations(limit=3) if guarded.found else [],
            mode="ask",
            debug=debug,
        )
