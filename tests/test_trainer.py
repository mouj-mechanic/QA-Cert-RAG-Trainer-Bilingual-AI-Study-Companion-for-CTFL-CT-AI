"""Trainer behaviour with mocked LLM (no paid API calls)."""

from __future__ import annotations

import json

from src.config import has_openai_key
from src.llm.provider import LLMProvider, MissingAPIKeyError
from src.rag.retriever import Retriever
from src.trainer.evaluator import AnswerEvaluator, _parse_verdict
from src.trainer.trainer import Trainer, _split_comprehension_question


class ScriptedLLM(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            raise AssertionError("Unexpected extra LLM call")
        return self.responses.pop(0)


def test_split_comprehension_question():
    text = (
        "The pesticide paradox means repeated identical tests lose effectiveness.\n\n"
        "🧠 Check your understanding:\n"
        "Why must tests be revised over time?"
    )
    explanation, question = _split_comprehension_question(text, "en")
    assert "pesticide" in explanation.lower()
    assert "revised" in question.lower() or "Why" in question


def test_trainer_produces_comprehension_stage(indexed_store):
    store, embedder, _ = indexed_store
    retriever = Retriever(vector_store=store, embedder=embedder, top_k=3, min_score=0.0)
    # Wording aligned with fixture syllabus so lexical grounding passes.
    draft = json.dumps(
        {
            "answer": (
                "If the same tests are repeated over and over again, eventually these "
                "tests no longer find new defects. To overcome this pesticide paradox, "
                "test cases need to be regularly reviewed and revised."
            ),
            "evidence": [
                {
                    "claim": "same tests repeated eventually no longer find new defects",
                    "source_chunk_id": "S1",
                }
            ],
            "comprehension_question": (
                "Why must test cases be regularly reviewed and revised?"
            ),
        }
    )
    llm = ScriptedLLM([draft, json.dumps({"grounded": True, "unsupported_claims": []})])
    trainer = Trainer(retriever=retriever, llm=llm)
    turn = trainer.explain(
        topic="Explain the pesticide paradox",
        certification="CTFL",
        response_language="en",
    )
    assert turn.found is True
    assert turn.comprehension_question
    assert "defect" in turn.comprehension_question.lower() or "?" in turn.comprehension_question
    assert turn.citations


def test_evaluator_uses_retrieval_and_classifies(indexed_store):
    store, embedder, _ = indexed_store
    retriever = Retriever(vector_store=store, embedder=embedder, top_k=3, min_score=0.0)
    draft = json.dumps(
        {
            "answer": (
                "✅ Correct\n\nYou correctly identified that if the same tests are "
                "repeated over and over again, eventually these tests no longer find "
                "new defects."
            ),
            "evidence": [
                {
                    "claim": "same tests repeated eventually no longer find new defects",
                    "source_chunk_id": "S1",
                }
            ],
            "comprehension_question": "",
        }
    )
    llm = ScriptedLLM([draft, json.dumps({"grounded": True, "unsupported_claims": []})])
    evaluator = AnswerEvaluator(retriever=retriever, llm=llm)
    result = evaluator.evaluate(
        original_topic="pesticide paradox",
        comprehension_question="Why revise tests?",
        student_answer="Because the same tests stop finding new bugs.",
        certification="CTFL",
        response_language="en",
    )
    assert result.verdict == "correct"
    assert result.retrieval.has_evidence


def test_parse_verdict_variants():
    assert _parse_verdict("✅ Correct\nWell done") == "correct"
    assert _parse_verdict("🟡 Partially correct\nMissing detail") == "partial"
    assert _parse_verdict("❌ Incorrect\nNot grounded") == "incorrect"


def test_missing_api_key_helper_does_not_crash():
    assert isinstance(has_openai_key(), bool)


def test_openai_provider_raises_without_key():
    from src.llm.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="")
    assert provider.is_available() is False
    try:
        provider.generate("sys", "user")
        assert False, "Expected MissingAPIKeyError"
    except MissingAPIKeyError:
        pass
