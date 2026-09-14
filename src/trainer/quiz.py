"""
Quiz mode: AI-generated practice MCQs grounded in retrieved syllabus chunks.

Questions are NOT official ISTQB exam items.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from src.llm.provider import LLMProvider, get_llm_provider
from src.rag.grounding import (
    CERTIFICATION_TEMPERATURE,
    has_sufficient_evidence,
    not_found_message,
    numbered_chunks,
    validate_grounding,
)
from src.rag.retriever import Retriever
from src.trainer.prompts import (
    QUIZ_SYSTEM_PROMPT,
    build_quiz_generation_prompt,
)

logger = logging.getLogger(__name__)

Difficulty = Literal["Easy", "Medium", "Exam-like"]


@dataclass
class QuizQuestion:
    """One practice MCQ."""

    question: str
    options: dict[str, str]
    correct: str
    explanation: str
    source_hint: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    label: str = "AI-generated practice question based on the indexed syllabus."


@dataclass
class QuizSession:
    """In-memory quiz progress (stored in Streamlit session_state)."""

    certification: str
    language: str
    difficulty: Difficulty
    chapter: str | None
    total: int
    questions: list[QuizQuestion] = field(default_factory=list)
    current_index: int = 0
    answers: list[str] = field(default_factory=list)
    correct_count: int = 0
    finished: bool = False

    @property
    def score_percent(self) -> float:
        if not self.answers:
            return 0.0
        return round(100.0 * self.correct_count / len(self.answers), 1)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from model output, tolerating accidental fences."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


class QuizEngine:
    """Generate and grade practice questions from RAG context."""

    SEED_QUERIES = {
        "CTFL": [
            "fundamental testing principles",
            "test process testware",
            "static testing reviews",
            "black-box white-box testing techniques",
            "test management risk-based testing",
            "test tools automation",
            "defect management",
        ],
        "CT-AI": [
            "AI system characteristics",
            "machine learning training data test data",
            "precision recall F1-score",
            "bias fairness AI testing",
            "neural networks testing",
            "self-learning systems",
            "AI quality characteristics",
        ],
    }

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or get_llm_provider()

    def _seed_query(self, certification: str, index: int, chapter: str | None) -> str:
        seeds = self.SEED_QUERIES.get(certification, self.SEED_QUERIES["CTFL"])
        base = seeds[index % len(seeds)]
        if chapter:
            return f"chapter {chapter} {base}"
        return base

    def generate_question(
        self,
        certification: str,
        language: str,
        difficulty: Difficulty,
        index: int,
        chapter: str | None = None,
    ) -> QuizQuestion | None:
        """Generate one grounded MCQ or return None if no evidence."""
        query = self._seed_query(certification, index, chapter)
        retrieval = self.retriever.retrieve(
            query=query,
            certification=certification,
            chapter=chapter,
            top_k=4,
        )
        if not has_sufficient_evidence(retrieval):
            return None

        user_prompt = build_quiz_generation_prompt(
            context=retrieval.context_text(),
            language=language,
            certification=certification,
            difficulty=difficulty,
            chapter=chapter,
        )
        raw = self.llm.generate(
            system_prompt=QUIZ_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=CERTIFICATION_TEMPERATURE,
        )
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse quiz JSON: %s | raw=%s", exc, raw[:300])
            return None

        options = data.get("options") or {}
        correct = str(data.get("correct", "A")).upper().strip()
        if correct not in options and options:
            correct = next(iter(options.keys()))

        explanation = str(data.get("explanation", "")).strip()
        chunks = numbered_chunks(retrieval)
        validation = validate_grounding(
            explanation,
            evidence=[],
            retrieved_chunks=chunks,
            llm=self.llm,
            require_evidence_links=False,
        )
        if not validation.grounded:
            # One repair of explanation only
            repair_raw = self.llm.generate(
                system_prompt=(
                    "Rewrite ONLY the quiz explanation so every fact is supported by the "
                    "provided chunks. Remove unsupported claims. Return plain text only."
                ),
                user_prompt=(
                    f"Explanation:\n{explanation}\n\n"
                    f"Unsupported:\n{validation.unsupported_claims}\n\n"
                    f"Chunks:\n{retrieval.context_text()}"
                ),
                temperature=CERTIFICATION_TEMPERATURE,
                max_tokens=400,
            )
            validation2 = validate_grounding(
                repair_raw,
                evidence=[],
                retrieved_chunks=chunks,
                llm=self.llm,
                require_evidence_links=False,
            )
            explanation = repair_raw.strip() if validation2.grounded else ""

        return QuizQuestion(
            question=str(data.get("question", "")).strip(),
            options={str(k): str(v) for k, v in options.items()},
            correct=correct,
            explanation=explanation,
            source_hint=str(data.get("source_hint", "")).strip(),
            citations=retrieval.unique_citations(limit=3),
        )

    def start_quiz(
        self,
        certification: str,
        language: str,
        difficulty: Difficulty,
        num_questions: int,
        chapter: str | None = None,
    ) -> QuizSession:
        """Generate a full quiz session (best effort)."""
        session = QuizSession(
            certification=certification,
            language=language,
            difficulty=difficulty,
            chapter=chapter,
            total=num_questions,
        )
        attempts = 0
        while len(session.questions) < num_questions and attempts < num_questions * 3:
            q = self.generate_question(
                certification=certification,
                language=language,
                difficulty=difficulty,
                index=attempts,
                chapter=chapter,
            )
            attempts += 1
            if q and q.question and len(q.options) >= 2:
                session.questions.append(q)

        session.total = len(session.questions)
        return session

    def grade_current(self, session: QuizSession, choice: str) -> tuple[bool, str]:
        """Grade the current question and advance. Returns (is_correct, feedback)."""
        if session.finished or session.current_index >= len(session.questions):
            return False, "Quiz already finished."

        question = session.questions[session.current_index]
        choice = choice.upper().strip()
        is_correct = choice == question.correct.upper()
        session.answers.append(choice)
        if is_correct:
            session.correct_count += 1

        if session.language == "fr":
            status = "✅ Bonne réponse" if is_correct else "❌ Mauvaise réponse"
            correct_line = f"Bonne option : {question.correct}"
            label = question.label
        else:
            status = "✅ Correct" if is_correct else "❌ Incorrect"
            correct_line = f"Correct option: {question.correct}"
            label = question.label

        feedback = (
            f"{status}\n\n"
            f"{correct_line}\n\n"
            f"{question.explanation}\n\n"
            f"📚 {question.source_hint}\n\n"
            f"_{label}_"
        )

        session.current_index += 1
        if session.current_index >= len(session.questions):
            session.finished = True
        return is_correct, feedback

    def empty_resources_message(self, language: str) -> str:
        return not_found_message(language)
