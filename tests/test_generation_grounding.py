"""Regression tests for generation grounding guardrail (mocked LLM)."""

from __future__ import annotations

import json
from typing import Any

from src.llm.provider import LLMProvider
from src.rag.grounding import (
    MAX_REPAIR_ATTEMPTS,
    EvidenceLink,
    GroundingValidation,
    generate_with_grounding_guardrail,
    grounded_fallback_message,
    not_found_message,
    validate_grounding,
)
from src.rag.rag_pipeline import RAGPipeline
from src.rag.retriever import RetrievedChunk, RetrievalResult


class ScriptedLLM(LLMProvider):
    """Returns scripted responses in order; counts calls."""

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


def _chunks() -> list[tuple[str, RetrievedChunk]]:
    return [
        (
            "S1",
            RetrievedChunk(
                text=(
                    "Narrow AI, also known as weak AI, is designed to perform specific tasks "
                    "and represents all deployed AI systems in use today. Narrow AI-based systems "
                    "operate within a limited domain."
                ),
                score=0.75,
                metadata={"certification": "CT-AI", "page": 15, "document": "doc"},
            ),
        ),
        (
            "S2",
            RetrievedChunk(
                text=(
                    "Frontier AI systems remain task-specific and have not yet achieved the "
                    "versatility of general AI."
                ),
                score=0.66,
                metadata={"certification": "CT-AI", "page": 15, "document": "doc"},
            ),
        ),
    ]


def _retrieval() -> RetrievalResult:
    return RetrievalResult(
        query="narrow vs general",
        certification="CT-AI",
        chunks=[c for _, c in _chunks()],
        top_k=4,
    )


def _draft_json(answer: str, claims: list[tuple[str, str]], question: str = "") -> str:
    return json.dumps(
        {
            "answer": answer,
            "evidence": [{"claim": c, "source_chunk_id": s} for c, s in claims],
            "comprehension_question": question,
        }
    )


def _validation_json(grounded: bool, unsupported: list[str] | None = None) -> str:
    return json.dumps({"grounded": grounded, "unsupported_claims": unsupported or []})


def test_a_fully_grounded_answer_passes_without_repair():
    answer = (
        "Narrow AI operates within a limited domain and represents deployed AI systems today. "
        "Systems remain task-specific and have not yet achieved the versatility of general AI."
    )
    draft = _draft_json(
        answer,
        [
            ("Narrow AI operates within a limited domain", "S1"),
            ("have not yet achieved the versatility of general AI", "S2"),
        ],
    )
    llm = ScriptedLLM([draft, _validation_json(True)])
    result = generate_with_grounding_guardrail(
        llm=llm,
        task_instruction="Explain narrow vs general AI",
        retrieval=_retrieval(),
        language="en",
        certification="CT-AI",
    )
    assert result.grounded is True
    assert result.found is True
    assert result.repair_attempted is False
    assert result.answer == answer
    assert len(llm.calls) == 2  # draft + validate


def test_b_unsupported_claim_is_rejected_by_validator():
    chunks = _chunks()
    validation = validate_grounding(
        answer="Narrow AI is also called weak AI. General AI equals human intelligence.",
        evidence=[
            EvidenceLink("Narrow AI is also called weak AI", "S1"),
            EvidenceLink("General AI equals human intelligence", "S2"),
        ],
        retrieved_chunks=chunks,
        llm=ScriptedLLM(
            [
                _validation_json(
                    False,
                    ["General AI equals human intelligence is not supported by the chunks"],
                )
            ]
        ),
    )
    assert validation.grounded is False
    assert validation.unsupported_claims


def test_c_repair_succeeds_and_returns_safe_answer():
    bad = _draft_json(
        "Narrow AI is limited-domain. General AI can do any intellectual task a human can do.",
        [
            ("Narrow AI is limited-domain", "S1"),
            ("General AI can do any intellectual task a human can do", "S2"),
        ],
    )
    good_answer = (
        "Narrow AI operates within a limited domain. "
        "Systems have not yet achieved the versatility of general AI."
    )
    good = _draft_json(
        good_answer,
        [
            ("Narrow AI operates within a limited domain", "S1"),
            ("have not yet achieved the versatility of general AI", "S2"),
        ],
    )
    llm = ScriptedLLM(
        [
            bad,
            _validation_json(False, ["human intellectual task claim unsupported"]),
            good,
            _validation_json(True),
        ]
    )
    result = generate_with_grounding_guardrail(
        llm=llm,
        task_instruction="Explain narrow vs general AI",
        retrieval=_retrieval(),
        language="en",
        certification="CT-AI",
    )
    assert result.repair_attempted is True
    assert result.grounded is True
    assert result.answer == good_answer
    assert "intellectual task" not in result.answer.lower()


def test_d_failed_repair_does_not_return_unsafe_answer():
    bad = _draft_json(
        "General AI matches human cognitive abilities across all domains.",
        [("General AI matches human cognitive abilities", "S2")],
    )
    still_bad = _draft_json(
        "General AI is human-level intelligence.",
        [("General AI is human-level intelligence", "S2")],
    )
    llm = ScriptedLLM(
        [
            bad,
            _validation_json(False, ["unsupported AGI claim"]),
            still_bad,
            _validation_json(False, ["still unsupported"]),
        ]
    )
    result = generate_with_grounding_guardrail(
        llm=llm,
        task_instruction="Explain narrow vs general AI",
        retrieval=_retrieval(),
        language="en",
        certification="CT-AI",
    )
    assert result.grounded is False
    assert result.found is False
    assert result.answer == grounded_fallback_message("en")
    assert "human" not in result.answer.lower() or "indexed ISTQB" in result.answer


def test_e_maximum_repair_attempts_is_one():
    assert MAX_REPAIR_ATTEMPTS == 1
    bad = _draft_json(
        "General AI matches human cognitive abilities across all domains without limits.",
        [("General AI matches human cognitive abilities", "S1")],
    )
    still_bad = _draft_json(
        "General AI equals human-level intelligence in every professional domain.",
        [("General AI equals human-level intelligence", "S1")],
    )
    llm = ScriptedLLM(
        [
            bad,
            _validation_json(False, ["u1"]),
            still_bad,
            _validation_json(False, ["u2"]),
        ]
    )
    result = generate_with_grounding_guardrail(
        llm=llm,
        task_instruction="x",
        retrieval=_retrieval(),
        language="en",
        certification="CT-AI",
    )
    # draft + validate + repair + validate2 = 4 calls; no second repair
    assert len(llm.calls) == 4
    assert result.repair_attempted is True
    assert result.grounded is False
    assert result.answer == grounded_fallback_message("en")


def test_deterministic_strip_recovers_when_repair_empty():
    """If repair returns empty JSON answer, strip supported sentences from the draft."""
    draft = _draft_json(
        (
            "Narrow AI, also known as weak AI, is designed to perform specific tasks "
            "and operates within a limited domain. "
            "General AI is similar to human intelligence across every domain."
        ),
        [
            ("Narrow AI is also known as weak AI", "S1"),
            ("General AI is similar to human intelligence", "S2"),
        ],
    )
    empty_repair = _draft_json("", [])
    llm = ScriptedLLM(
        [
            draft,
            _validation_json(False, ["human intelligence unsupported"]),
            empty_repair,
        ]
    )
    result = generate_with_grounding_guardrail(
        llm=llm,
        task_instruction="x",
        retrieval=_retrieval(),
        language="en",
        certification="CT-AI",
    )
    assert result.repair_attempted is True
    assert result.grounded is True
    assert "human intelligence" not in result.answer.lower()
    assert "narrow ai" in result.answer.lower()
    assert result.debug.get("deterministic_strip") is True


def test_f_oos_deterministic_refusal_bypasses_generation():
    class CountingLLM(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0

        def is_available(self) -> bool:
            return True

        def generate(self, *args: Any, **kwargs: Any) -> str:
            self.calls += 1
            return "should not run"

    class WeakRetriever:
        def retrieve(self, *args: Any, **kwargs: Any) -> RetrievalResult:
            return RetrievalResult(
                query="oos",
                certification="CT-AI",
                chunks=[
                    RetrievedChunk(
                        text="unrelated release notes",
                        score=0.41,
                        metadata={"certification": "CT-AI", "page": 91},
                    )
                ],
                top_k=4,
            )

    llm = CountingLLM()
    pipeline = RAGPipeline(retriever=WeakRetriever(), llm=llm)  # type: ignore[arg-type]
    answer = pipeline.ask(
        question="What is the official CTFL pesticide paradox definition?",
        certification="CT-AI",
        response_language="en",
    )
    assert llm.calls == 0
    assert answer.found is False
    assert answer.answer == not_found_message("en")


def test_validate_rejects_unknown_chunk_id_without_llm():
    validation = validate_grounding(
        answer="Some claim",
        evidence=[EvidenceLink("Some claim", "S99")],
        retrieved_chunks=_chunks(),
        llm=None,
    )
    assert validation.grounded is False
    assert any("S99" in c for c in validation.unsupported_claims)


def test_lexical_rejects_human_intelligence_embellishment_without_llm():
    """Genuine E2E1-style embellishment must fail even if no LLM judge is used."""
    from src.rag.grounding import lexical_unsupported_sentences

    chunk_texts = [c.text for _, c in _chunks()]
    answer = (
        "Narrow AI, also known as weak AI, is designed to perform specific tasks "
        "and operates within a limited domain. In contrast, general AI would possess "
        "the ability to understand, learn, and apply knowledge across a wide range of "
        "tasks, similar to human intelligence. However, general AI has not yet been achieved."
    )
    unsupported = lexical_unsupported_sentences(answer, chunk_texts)
    assert unsupported
    assert any("human intelligence" in s.lower() for s in unsupported)

    validation = validate_grounding(
        answer=answer,
        evidence=[
            EvidenceLink("Narrow AI is also known as weak AI", "S1"),
            EvidenceLink("general AI has not yet been achieved", "S2"),
        ],
        retrieved_chunks=_chunks(),
        llm=None,
    )
    assert validation.grounded is False
