"""Regression tests for grounding / evidence-gate bug-fixes."""

from __future__ import annotations

from typing import Any

from src.config import MIN_RELEVANCE_SCORE
from src.llm.provider import LLMProvider
from src.rag.grounding import has_sufficient_evidence, not_found_message
from src.rag.rag_pipeline import RAGPipeline
from src.rag.retriever import RetrievedChunk, RetrievalResult, Retriever
from src.trainer.prompts import (
    ASK_SYSTEM_PROMPT,
    BASE_SYSTEM_PROMPT,
    STRICT_CONTEXT_ONLY_RULE,
    build_ask_user_prompt,
)


class CountingLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        self.calls += 1
        return "LLM SHOULD NOT BE CALLED"


class FixedRetriever:
    """Stub retriever returning a pre-built RetrievalResult."""

    def __init__(self, result: RetrievalResult) -> None:
        self._result = result

    def retrieve(self, *args: Any, **kwargs: Any) -> RetrievalResult:
        return self._result


def _result_with_scores(scores: list[float], certification: str = "CT-AI") -> RetrievalResult:
    chunks = [
        RetrievedChunk(
            text=f"chunk text {i}",
            score=s,
            metadata={"certification": certification, "page": i, "document": "doc"},
        )
        for i, s in enumerate(scores, start=1)
    ]
    return RetrievalResult(
        query="q",
        certification=certification,
        chunks=chunks,
        top_k=len(scores) or 4,
    )


def test_min_relevance_score_is_evidence_based_floor():
    """Bug-fix raised the floor from 0.25; keep it configurable but not regress to 0.25."""
    assert MIN_RELEVANCE_SCORE >= 0.50


def test_has_sufficient_evidence_false_when_empty():
    empty = RetrievalResult(query="q", certification="CT-AI", chunks=[], top_k=4)
    assert has_sufficient_evidence(empty) is False


def test_has_sufficient_evidence_false_when_top1_below_floor():
    # Mimics the out-of-scope baseline band (~0.41)
    weak = _result_with_scores([0.41, 0.40, 0.38, 0.34])
    assert has_sufficient_evidence(weak) is False


def test_has_sufficient_evidence_true_when_top1_meets_floor():
    strong = _result_with_scores([0.75, 0.66, 0.61, 0.57])
    assert has_sufficient_evidence(strong) is True


def test_insufficient_evidence_does_not_call_llm():
    llm = CountingLLM()
    weak = _result_with_scores([0.41, 0.40, 0.38, 0.34])
    pipeline = RAGPipeline(retriever=FixedRetriever(weak), llm=llm)  # type: ignore[arg-type]
    answer = pipeline.ask(
        question="What is the CTFL pesticide paradox according to CT-AI?",
        certification="CT-AI",
        response_language="en",
    )
    assert llm.calls == 0
    assert answer.found is False
    assert answer.answer == not_found_message("en")


def test_deterministic_refusal_french():
    llm = CountingLLM()
    weak = _result_with_scores([0.30])
    pipeline = RAGPipeline(retriever=FixedRetriever(weak), llm=llm)  # type: ignore[arg-type]
    answer = pipeline.ask(
        question="Question hors corpus",
        certification="CT-AI",
        response_language="fr",
    )
    assert llm.calls == 0
    assert answer.answer == not_found_message("fr")
    assert "ressources ISTQB actuellement indexées" in answer.answer


def test_grounding_prompt_contains_strict_context_only_rule():
    assert STRICT_CONTEXT_ONLY_RULE in BASE_SYSTEM_PROMPT
    assert STRICT_CONTEXT_ONLY_RULE in ASK_SYSTEM_PROMPT
    assert "model memory" in BASE_SYSTEM_PROMPT.lower() or "model memory" in ASK_SYSTEM_PROMPT.lower()
    user = build_ask_user_prompt(
        question="q",
        context="ctx",
        language="en",
        certification="CT-AI",
    )
    assert STRICT_CONTEXT_ONLY_RULE in user
    assert "do not mention it" in STRICT_CONTEXT_ONLY_RULE.lower()


def test_certification_filter_still_isolates_with_new_floor(indexed_store):
    store, embedder, _ = indexed_store
    retriever = Retriever(
        vector_store=store,
        embedder=embedder,
        top_k=5,
        min_score=MIN_RELEVANCE_SCORE,
    )
    ctfl = retriever.retrieve("pesticide paradox testing principles", certification="CTFL")
    ctai = retriever.retrieve("precision recall F1-score training data", certification="CT-AI")
    assert all(c.certification == "CTFL" for c in ctfl.chunks)
    assert all(c.certification == "CT-AI" for c in ctai.chunks)
