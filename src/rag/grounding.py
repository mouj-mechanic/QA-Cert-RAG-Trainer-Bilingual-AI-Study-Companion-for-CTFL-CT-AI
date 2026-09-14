"""
Central grounding for Chat_ISTQB.

Two layers (keep distinct in interviews):

1) Retrieval grounding — `has_sufficient_evidence`
   Enough chunks above the score floor? If not → deterministic refusal, no LLM.

2) Generation grounding — `generate_with_grounding_guardrail`
   Draft with claim→chunk evidence → validate → one repair → safe fallback.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.config import MIN_RELEVANCE_SCORE, OPENAI_MODEL
from src.llm.provider import LLMProvider
from src.rag.retriever import RetrievedChunk, RetrievalResult
from src.trainer.prompts import NOT_FOUND_EN, NOT_FOUND_FR

logger = logging.getLogger(__name__)

# Certification generation settings (documented for portfolio baseline)
CERTIFICATION_TEMPERATURE = 0.0
CERTIFICATION_MAX_TOKENS = 1400
MAX_REPAIR_ATTEMPTS = 1

GROUNDED_FALLBACK_EN = (
    "I cannot provide a sufficiently source-grounded answer from the currently "
    "indexed ISTQB resources."
)
GROUNDED_FALLBACK_FR = (
    "Je ne peux pas fournir une réponse suffisamment étayée par les ressources "
    "ISTQB actuellement indexées."
)

STRUCTURED_DRAFT_SYSTEM = """You produce a grounded certification study answer as STRICT JSON only.
No markdown fences. No prose outside JSON.

Schema:
{
  "answer": "user-visible answer text in the requested language",
  "evidence": [
    {"claim": "one substantive factual claim from the answer", "source_chunk_id": "S1"}
  ],
  "comprehension_question": "optional; one question grounded in the same context, or empty string"
}

Rules:
- Use ONLY the retrieved chunks. Every factual ISTQB / AI-testing claim in "answer" must appear in "evidence" with a real source_chunk_id from the provided list.
- Do not invent chunk IDs.
- Do not add facts from model memory.
- Pedagogical connectors are allowed only if they introduce no new factual content.
- Prefer a shorter grounded answer over a richer unsupported one.
"""

VALIDATION_SYSTEM = """You are a RAG grounding judge.
You receive ONLY: an answer, declared evidence links, and retrieved chunk texts.
Decide whether each substantive factual claim in the answer is DIRECTLY supported by the supplied chunk texts.

Critical:
- A statement that is generally true but ABSENT from the supplied chunks is UNSUPPORTED.
- Do NOT judge general world knowledge truth.
- Do NOT accept examples, analogies, or definitions that the chunks do not state
  (e.g. "image recognition", "human intelligence") unless those exact ideas appear in the chunks.
- Do NOT reattribute a property the chunks state about concept A onto concept B.
  Example: if chunks say Narrow AI cannot generalize, an answer must not present
  "ability to generalize" as the definition of General AI unless chunks say so.
- Pedagogical connectors without new facts are OK.
- Ignore empty/boilerplate lines.

Return STRICT JSON only:
{"grounded": true/false, "unsupported_claims": ["..."]}
No markdown fences.
"""

REPAIR_SYSTEM = """You repair a study answer so it is fully supported by retrieved ISTQB chunks.
Rules:
- Remove unsupported claims listed by the validator.
- Do NOT replace them with other external knowledge.
- Stay on the SAME learner topic. Do not switch to a neighboring concept
  (e.g. do not replace an overfitting explanation with underfitting).
- Preserve supported content, language, and official terminology from the chunks.
- Use ONLY the retrieved chunks.
- Never return an empty answer if any supported sentences remain.
- Always include a non-empty "evidence" array: every remaining substantive claim must
  map to a real source_chunk_id from the allowed list.
- Return STRICT JSON with the same schema: answer, evidence, comprehension_question.
No markdown fences.
"""


def has_sufficient_evidence(retrieval: RetrievalResult) -> bool:
    """
    Retrieval-layer gate: enough evidence to attempt generation?

    When False, callers MUST return `not_found_message` and MUST NOT call the LLM.
    """
    if not retrieval.chunks:
        return False
    return float(retrieval.chunks[0].score) >= MIN_RELEVANCE_SCORE


def not_found_message(language: str) -> str:
    """Deterministic bilingual refusal when retrieval evidence is insufficient."""
    return NOT_FOUND_FR if language == "fr" else NOT_FOUND_EN


def grounded_fallback_message(language: str) -> str:
    """Deterministic fallback when generation fails the grounding guardrail."""
    return GROUNDED_FALLBACK_FR if language == "fr" else GROUNDED_FALLBACK_EN


def generation_settings() -> dict[str, Any]:
    """Documented generation settings for certification answers."""
    return {
        "model": OPENAI_MODEL,
        "temperature": CERTIFICATION_TEMPERATURE,
        "max_tokens": CERTIFICATION_MAX_TOKENS,
        "max_repair_attempts": MAX_REPAIR_ATTEMPTS,
    }


@dataclass
class EvidenceLink:
    claim: str
    source_chunk_id: str


@dataclass
class StructuredDraft:
    answer: str
    evidence: list[EvidenceLink] = field(default_factory=list)
    comprehension_question: str = ""


@dataclass
class GroundingValidation:
    grounded: bool
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass
class GuardrailResult:
    """Result of draft → validate → (optional) repair → fallback."""

    answer: str
    found: bool
    grounded: bool
    repair_attempted: bool
    unsupported_claims: list[str] = field(default_factory=list)
    draft_answer: str = ""
    repaired_answer: str | None = None
    comprehension_question: str = ""
    evidence: list[EvidenceLink] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def chunk_id_for_index(index: int) -> str:
    """Stable IDs: S1, S2, ... matching Source N labels."""
    return f"S{index}"


def numbered_chunks(retrieval: RetrievalResult) -> list[tuple[str, RetrievedChunk]]:
    return [(chunk_id_for_index(i), chunk) for i, chunk in enumerate(retrieval.chunks, start=1)]


def numbered_context(retrieval: RetrievalResult) -> str:
    """Context block with explicit chunk IDs for the grounded contract."""
    blocks: list[str] = []
    for cid, chunk in numbered_chunks(retrieval):
        blocks.append(f"[{cid}] {chunk.citation_label}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def parse_structured_draft(raw: str) -> StructuredDraft:
    data = _extract_json_object(raw)
    evidence_raw = data.get("evidence") or []
    evidence: list[EvidenceLink] = []
    for item in evidence_raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        sid = str(item.get("source_chunk_id", "")).strip()
        if claim and sid:
            evidence.append(EvidenceLink(claim=claim, source_chunk_id=sid))
    return StructuredDraft(
        answer=str(data.get("answer", "")).strip(),
        evidence=evidence,
        comprehension_question=str(data.get("comprehension_question", "")).strip(),
    )


_STOPWORDS = {
    "that", "this", "with", "from", "have", "been", "were", "where", "which",
    "while", "about", "their", "there", "would", "could", "should", "into",
    "also", "such", "than", "then", "them", "they", "does", "doing", "make",
    "made", "over", "under", "after", "before", "between", "being", "because",
    "dans", "pour", "avec", "une", "des", "les", "est", "sont", "plus", "comme",
    "cette", "cela", "aussi", "mais", "pas", "sur", "par", "que", "qui", "dont",
}


def _content_words(text: str) -> set[str]:
    words = set(re.findall(r"[A-Za-zÀ-ÿ]{4,}", text.lower()))
    return {w for w in words if w not in _STOPWORDS}


def lexical_unsupported_sentences(answer: str, chunk_texts: list[str]) -> list[str]:
    """
    Deterministic support check: flag answer sentences whose content words
    barely overlap the retrieved chunk corpus.

    True-but-absent embellishments often fail this check even when an LLM
    judge is overly permissive.
    """
    corpus = "\n".join(chunk_texts).lower()
    corpus_words = _content_words(corpus)
    unsupported: list[str] = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    for sentence in sentences:
        words = _content_words(sentence)
        if len(words) < 3:
            continue
        overlap = len(words & corpus_words) / max(len(words), 1)
        # Short claim-like sentences: require full content-word coverage.
        if len(words) < 5 and not words <= corpus_words:
            unsupported.append(sentence)
            continue
        if overlap < 0.55:
            unsupported.append(sentence)
    return unsupported


# Phrases that are common model embellishments; flag when absent from chunks.
_EMBELLISHMENT_MARKERS = (
    "human intelligence",
    "intelligence humaine",
    "similar to human",
    "comme un humain",
    "image recognition",
    "reconnaissance d'images",
    "reconnaissance d’images",
    "intellectual task",
    "tâche intellectuelle",
    "tache intellectuelle",
    "cognitive abilities",
    "capacités cognitives",
    "capacites cognitives",
    "capturing noise",
    "capture le bruit",
    "including its noise",
    "learned the noise",
    "noise and outliers",
    "learning curves",
)


def absent_embellishment_markers(answer: str, chunk_texts: list[str]) -> list[str]:
    """Flag known embellishment phrases present in the answer but not in chunks."""
    corpus = "\n".join(chunk_texts).lower()
    answer_l = answer.lower()
    hits: list[str] = []
    for marker in _EMBELLISHMENT_MARKERS:
        if marker in answer_l and marker not in corpus:
            hits.append(marker)
    return hits


def validate_grounding(
    answer: str,
    evidence: list[EvidenceLink],
    retrieved_chunks: list[tuple[str, RetrievedChunk]],
    llm: LLMProvider | None = None,
    require_evidence_links: bool = True,
    language: str = "en",
) -> GroundingValidation:
    """
    Validate that substantive claims are supported by supplied chunks only.

    Programmatic checks first (chunk IDs + lexical sentence support);
    optional LLM semantic check when provider given.

    Lexical overlap is English-oriented (indexed syllabus text is EN). For French
    answers, skip lexical and rely on evidence IDs + the LLM support judge.
    """
    unsupported: list[str] = []
    if not answer.strip():
        return GroundingValidation(grounded=False, unsupported_claims=["empty answer"])

    valid_ids = {cid for cid, _ in retrieved_chunks}
    chunk_map = {cid: chunk.text for cid, chunk in retrieved_chunks}
    chunk_texts = [text for text in chunk_map.values()]

    if require_evidence_links and not evidence:
        unsupported.append("no evidence links declared for the answer")
    for link in evidence:
        if link.source_chunk_id not in valid_ids:
            unsupported.append(
                f"fabricated or unknown chunk id {link.source_chunk_id!r} for claim: {link.claim}"
            )

    # Lexical gate catches EN embellishments a permissive judge may miss.
    # Skip for FR: French wording cannot lexically overlap English chunks fairly.
    if language != "fr":
        lexical = lexical_unsupported_sentences(answer, chunk_texts)
        unsupported.extend(f"low lexical support vs chunks: {s}" for s in lexical)

    # Language-agnostic: known embellishment phrases absent from retrieved text.
    for marker in absent_embellishment_markers(answer, chunk_texts):
        unsupported.append(f"embellishment marker absent from chunks: {marker!r}")

    if llm is None:
        return GroundingValidation(grounded=not unsupported, unsupported_claims=unsupported)

    evidence_payload = [
        {"claim": e.claim, "source_chunk_id": e.source_chunk_id} for e in evidence
    ]
    chunks_payload = [
        {"id": cid, "text": text[:2000]} for cid, text in chunk_map.items()
    ]
    user_prompt = (
        "Answer to judge:\n"
        f"{answer}\n\n"
        f"Declared evidence:\n{json.dumps(evidence_payload, ensure_ascii=False)}\n\n"
        f"Retrieved chunks (ONLY source of truth):\n{json.dumps(chunks_payload, ensure_ascii=False)}\n\n"
        "Is each substantive claim in the answer directly supported by the supplied chunks?\n"
        "Remember: generally-true-but-absent statements are UNSUPPORTED."
    )
    raw = llm.generate(
        system_prompt=VALIDATION_SYSTEM,
        user_prompt=user_prompt,
        temperature=CERTIFICATION_TEMPERATURE,
        max_tokens=800,
    )
    try:
        data = _extract_json_object(raw)
        grounded = bool(data.get("grounded"))
        llm_unsupported = [str(x) for x in (data.get("unsupported_claims") or [])]
        all_unsupported = unsupported + llm_unsupported
        # Programmatic failures always win over a permissive LLM judge
        if unsupported:
            grounded = False
        elif not grounded and not llm_unsupported:
            all_unsupported = ["validator marked ungrounded without listing claims"]
        return GroundingValidation(
            grounded=grounded and not unsupported,
            unsupported_claims=all_unsupported,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Grounding validator JSON parse failed: %s", exc)
        return GroundingValidation(
            grounded=False,
            unsupported_claims=unsupported + [f"validator parse error: {exc}"],
        )


def _draft_user_prompt(
    task_instruction: str,
    language: str,
    certification: str,
    context: str,
    allowed_ids: list[str],
) -> str:
    lang = "French" if language == "fr" else "English"
    return (
        f"Certification filter: {certification}\n"
        f"Response language: {lang}\n"
        f"Allowed source_chunk_id values: {', '.join(allowed_ids)}\n\n"
        f"Retrieved chunks:\n{context}\n\n"
        f"Task:\n{task_instruction}\n"
    )


def generate_with_grounding_guardrail(
    llm: LLMProvider,
    task_instruction: str,
    retrieval: RetrievalResult,
    language: str,
    certification: str,
    include_comprehension_question: bool = False,
) -> GuardrailResult:
    """
    Draft (structured) → validate → at most one repair → fallback.

    Interviewable flow; no infinite retries.
    """
    chunks = numbered_chunks(retrieval)
    allowed_ids = [cid for cid, _ in chunks]
    context = numbered_context(retrieval)
    settings = generation_settings()

    base_debug: dict[str, Any] = {
        "generation_settings": settings,
        "grounding": "PENDING",
        "repair_attempted": False,
        "unsupported_claims_count": 0,
        "unsupported_claims": [],
    }

    if include_comprehension_question:
        task_instruction = (
            f"{task_instruction}\n"
            "Also set comprehension_question to exactly ONE comprehension check "
            "grounded only in the retrieved chunks. Do not reveal its answer."
        )
    else:
        task_instruction = f"{task_instruction}\nSet comprehension_question to an empty string."

    user_prompt = _draft_user_prompt(
        task_instruction=task_instruction,
        language=language,
        certification=certification,
        context=context,
        allowed_ids=allowed_ids,
    )

    raw_draft = llm.generate(
        system_prompt=STRUCTURED_DRAFT_SYSTEM,
        user_prompt=user_prompt,
        temperature=CERTIFICATION_TEMPERATURE,
        max_tokens=CERTIFICATION_MAX_TOKENS,
    )
    try:
        draft = parse_structured_draft(raw_draft)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Structured draft parse failed: %s", exc)
        base_debug.update(
            {
                "grounding": "FAIL",
                "unsupported_claims_count": 1,
                "unsupported_claims": [f"draft parse error: {exc}"],
                "draft_answer": raw_draft[:500],
            }
        )
        return GuardrailResult(
            answer=grounded_fallback_message(language),
            found=False,
            grounded=False,
            repair_attempted=False,
            unsupported_claims=[f"draft parse error: {exc}"],
            draft_answer=raw_draft[:2000],
            debug=base_debug,
        )

    validation = validate_grounding(
        draft.answer,
        draft.evidence,
        chunks,
        llm=llm,
        require_evidence_links=True,
        language=language,
    )
    base_debug["draft_answer"] = draft.answer
    base_debug["unsupported_claims"] = validation.unsupported_claims
    base_debug["unsupported_claims_count"] = len(validation.unsupported_claims)

    if validation.grounded:
        base_debug["grounding"] = "PASS"
        return GuardrailResult(
            answer=draft.answer,
            found=True,
            grounded=True,
            repair_attempted=False,
            unsupported_claims=[],
            draft_answer=draft.answer,
            comprehension_question=draft.comprehension_question,
            evidence=draft.evidence,
            debug=base_debug,
        )

    # Exactly one repair attempt
    base_debug["repair_attempted"] = True
    repair_user = (
        f"Response language: {'French' if language == 'fr' else 'English'}\n"
        f"Allowed source_chunk_id values: {', '.join(allowed_ids)}\n\n"
        f"Retrieved chunks:\n{context}\n\n"
        f"Previous answer:\n{draft.answer}\n\n"
        f"Previous evidence JSON:\n"
        f"{json.dumps([{'claim': e.claim, 'source_chunk_id': e.source_chunk_id} for e in draft.evidence], ensure_ascii=False)}\n\n"
        f"Unsupported claims to remove:\n"
        f"{json.dumps(validation.unsupported_claims, ensure_ascii=False)}\n\n"
        "Rewrite the JSON answer removing those unsupported claims. "
        "Keep a non-empty evidence array with only supported claims. "
        "Stay on the same topic as the previous answer. "
        "Do not add new external facts."
    )
    raw_repair = llm.generate(
        system_prompt=REPAIR_SYSTEM,
        user_prompt=repair_user,
        temperature=CERTIFICATION_TEMPERATURE,
        max_tokens=CERTIFICATION_MAX_TOKENS,
    )
    try:
        repaired = parse_structured_draft(raw_repair)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Repair draft parse failed: %s", exc)
        repaired = None
        base_debug["repair_parse_error"] = str(exc)

    candidate_for_strip = draft.answer
    last_unsupported = list(validation.unsupported_claims)

    if repaired is not None and repaired.answer.strip():
        validation2 = validate_grounding(
            repaired.answer,
            repaired.evidence,
            chunks,
            llm=llm,
            require_evidence_links=False,
            language=language,
        )
        base_debug["repaired_answer"] = repaired.answer
        base_debug["unsupported_claims"] = validation2.unsupported_claims
        base_debug["unsupported_claims_count"] = len(validation2.unsupported_claims)

        if validation2.grounded:
            base_debug["grounding"] = "PASS"
            return GuardrailResult(
                answer=repaired.answer,
                found=True,
                grounded=True,
                repair_attempted=True,
                unsupported_claims=[],
                draft_answer=draft.answer,
                repaired_answer=repaired.answer,
                comprehension_question=repaired.comprehension_question
                or draft.comprehension_question,
                evidence=repaired.evidence,
                debug=base_debug,
            )
        candidate_for_strip = repaired.answer
        last_unsupported = validation2.unsupported_claims
    else:
        last_unsupported = validation.unsupported_claims + ["empty or unparseable repair"]
        base_debug["repaired_answer"] = (raw_repair or "")[:2000]
        base_debug["unsupported_claims"] = last_unsupported
        base_debug["unsupported_claims_count"] = len(last_unsupported)

    # Deterministic last resort (EN): strip unsupported sentences from repair then draft.
    chunk_texts = [chunk.text for _, chunk in chunks]
    if language != "fr":
        for label, text in (
            ("repaired", candidate_for_strip),
            ("draft", draft.answer),
        ):
            supported_only = _strip_to_supported(text, chunk_texts)
            if (
                supported_only
                and not lexical_unsupported_sentences(supported_only, chunk_texts)
                and not absent_embellishment_markers(supported_only, chunk_texts)
                and _has_strong_lexical_sentence(supported_only, chunk_texts)
            ):
                base_debug["grounding"] = "PASS"
                base_debug["deterministic_strip"] = True
                base_debug["deterministic_strip_source"] = label
                base_debug["unsupported_claims"] = []
                base_debug["unsupported_claims_count"] = 0
                return GuardrailResult(
                    answer=supported_only,
                    found=True,
                    grounded=True,
                    repair_attempted=True,
                    unsupported_claims=[],
                    draft_answer=draft.answer,
                    repaired_answer=base_debug.get("repaired_answer"),
                    comprehension_question=(
                        (repaired.comprehension_question if repaired else "")
                        or draft.comprehension_question
                    ),
                    evidence=(repaired.evidence if repaired else draft.evidence),
                    debug=base_debug,
                )

    # Still unsafe — never return ungrounded answer
    base_debug["grounding"] = "FAIL"
    base_debug["unsupported_claims"] = last_unsupported
    base_debug["unsupported_claims_count"] = len(last_unsupported)
    return GuardrailResult(
        answer=grounded_fallback_message(language),
        found=False,
        grounded=False,
        repair_attempted=True,
        unsupported_claims=last_unsupported,
        draft_answer=draft.answer,
        repaired_answer=base_debug.get("repaired_answer"),
        debug=base_debug,
    )


def _keep_lexically_supported_sentences(answer: str, chunk_texts: list[str]) -> str:
    """Drop sentences the lexical gate flags; return remaining text or empty."""
    bad = set(lexical_unsupported_sentences(answer, chunk_texts))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    kept = [s for s in sentences if s not in bad]
    return " ".join(kept).strip()


def _strip_to_supported(answer: str, chunk_texts: list[str]) -> str:
    """Remove lexically weak sentences and sentences with absent embellishment markers."""
    markers = absent_embellishment_markers(answer, chunk_texts)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    lexical_bad = set(lexical_unsupported_sentences(answer, chunk_texts))
    kept: list[str] = []
    for sentence in sentences:
        if sentence in lexical_bad:
            continue
        lower = sentence.lower()
        if any(m in lower for m in markers):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def _has_strong_lexical_sentence(answer: str, chunk_texts: list[str]) -> bool:
    """True if at least one sentence has high content-word overlap with chunks."""
    corpus_words = _content_words("\n".join(chunk_texts))
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        words = _content_words(sentence)
        if len(words) >= 4 and (len(words & corpus_words) / len(words)) >= 0.55:
            return True
    return False
