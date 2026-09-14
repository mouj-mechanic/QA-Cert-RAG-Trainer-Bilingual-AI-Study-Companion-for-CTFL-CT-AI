"""
Chat ISTQB — Unofficial AI Study Companion

Streamlit entry point. Session memory only (no user accounts).
UI/UX layer only — RAG / grounding / trainer backends are unchanged.
"""

from __future__ import annotations

# Streamlit Community Cloud often ships an older system SQLite than Chroma needs.
try:
    __import__("pysqlite3")
    import sys as _sys

    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import logging
from typing import Any

import streamlit as st

from src.config import CERTIFICATIONS, has_openai_key
from src.llm.provider import MissingAPIKeyError
from src.rag.rag_pipeline import RAGPipeline
from src.rag.vector_store import VectorStore
from src.trainer.evaluator import AnswerEvaluator
from src.trainer.quiz import QuizEngine, QuizSession
from src.trainer.trainer import Trainer
from src.ui.components import (
    inject_styles,
    render_assistant_answer,
    render_footer,
    render_header,
    render_rag_debug,
    render_sources,
    render_welcome,
    thinking_status,
)
from src.ui.translations import t

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Chat ISTQB — AI Study Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "trainer_stage": "idle",  # idle | awaiting_comprehension | evaluated
        "trainer_topic": "",
        "trainer_comprehension_q": "",
        "trainer_last_debug": None,
        "trainer_last_citations": [],
        "quiz_session": None,
        "quiz_feedback": None,
        "last_debug": None,
        "show_sources": True,
        # Fresh-session UI defaults (CT-AI + Français + Formateur)
        "ui_language": "fr",
        "ui_certification": "CT-AI",
        "ui_mode": "trainer",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_session() -> None:
    st.session_state.messages = []
    st.session_state.trainer_stage = "idle"
    st.session_state.trainer_topic = ""
    st.session_state.trainer_comprehension_q = ""
    st.session_state.trainer_last_debug = None
    st.session_state.trainer_last_citations = []
    st.session_state.quiz_session = None
    st.session_state.quiz_feedback = None
    st.session_state.last_debug = None


@st.cache_resource(show_spinner=False)
def _get_vector_store() -> VectorStore:
    return VectorStore()


@st.cache_resource(show_spinner=False)
def _get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def _get_trainer() -> Trainer:
    return Trainer()


@st.cache_resource(show_spinner=False)
def _get_evaluator() -> AnswerEvaluator:
    return AnswerEvaluator()


@st.cache_resource(show_spinner=False)
def _get_quiz_engine() -> QuizEngine:
    return QuizEngine()


def _append(role: str, content: str, **meta: Any) -> None:
    st.session_state.messages.append({"role": role, "content": content, **meta})


def _sidebar() -> tuple[str, str, str, str | None]:
    language = st.session_state.ui_language

    with st.sidebar:
        st.markdown(
            f'<div class="ci-sidebar-brand">{t(language, "app_title")}</div>',
            unsafe_allow_html=True,
        )
        st.caption(t(language, "app_tagline"))
        st.markdown(
            f'<div class="ci-section-label">{t(language, "settings")}</div>',
            unsafe_allow_html=True,
        )

        # Certification — store key, display label
        cert_keys = list(CERTIFICATIONS.keys())
        cert_labels = [CERTIFICATIONS[k]["label"] for k in cert_keys]
        try:
            cert_index = cert_keys.index(st.session_state.ui_certification)
        except ValueError:
            cert_index = cert_keys.index("CT-AI") if "CT-AI" in cert_keys else 0
        selected_label = st.selectbox(
            t(language, "certification"),
            options=cert_labels,
            index=cert_index,
        )
        certification = next(
            k for k, v in CERTIFICATIONS.items() if v["label"] == selected_label
        )
        st.session_state.ui_certification = certification

        # Language — labels follow active UI language
        lang_code_by_label = {
            t(language, "lang_option_fr"): "fr",
            t(language, "lang_option_en"): "en",
        }
        lang_labels = list(lang_code_by_label.keys())
        current_lang_label = (
            t(language, "lang_option_fr")
            if st.session_state.ui_language == "fr"
            else t(language, "lang_option_en")
        )
        chosen_lang_label = st.selectbox(
            t(language, "language"),
            options=lang_labels,
            index=lang_labels.index(current_lang_label),
        )
        new_language = lang_code_by_label[chosen_lang_label]
        if new_language != st.session_state.ui_language:
            st.session_state.ui_language = new_language
            st.rerun()
        language = st.session_state.ui_language

        # Mode — store internal key
        mode_options = [
            ("trainer", t(language, "mode_trainer")),
            ("ask", t(language, "mode_ask")),
            ("quiz", t(language, "mode_quiz")),
        ]
        mode_labels = [label for _, label in mode_options]
        mode_by_label = {label: key for key, label in mode_options}
        try:
            mode_index = [key for key, _ in mode_options].index(st.session_state.ui_mode)
        except ValueError:
            mode_index = 0
        chosen_mode_label = st.selectbox(
            t(language, "mode"),
            options=mode_labels,
            index=mode_index,
        )
        mode = mode_by_label[chosen_mode_label]
        st.session_state.ui_mode = mode

        chapter_raw = st.text_input(
            t(language, "chapter_optional"),
            value="",
            placeholder=t(language, "chapter_placeholder"),
        )
        chapter = chapter_raw.strip() or None

        st.checkbox(t(language, "show_sources"), key="show_sources")

        st.divider()
        if st.button(t(language, "new_question"), use_container_width=True):
            st.session_state.trainer_stage = "idle"
            st.session_state.trainer_topic = ""
            st.session_state.trainer_comprehension_q = ""
            st.session_state.quiz_feedback = None
        if st.button(t(language, "reset_session"), use_container_width=True):
            _reset_session()
            st.rerun()

        store = _get_vector_store()
        st.caption(t(language, "indexed_chunks", count=store.count))

    return certification, language, mode, chapter


def _guardrails(language: str) -> bool:
    """Show warnings; return False if generation should be blocked."""
    ok = True
    if not has_openai_key():
        st.warning(t(language, "missing_api_key"))
        ok = False
    store = _get_vector_store()
    if not store.is_ready():
        st.warning(t(language, "empty_index"))
        ok = False
    return ok


def _render_history(language: str, mode: str) -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_answer(
                    language,
                    msg["content"],
                    citations=msg.get("citations") or [],
                    comprehension_question=msg.get("comprehension_question"),
                    show_sources=st.session_state.show_sources,
                    show_explanation_label=mode == "trainer",
                )
            else:
                st.markdown(msg["content"])


def _handle_ask(certification: str, language: str, chapter: str | None, prompt: str) -> None:
    _append("user", prompt)
    try:
        with thinking_status(language) as status:
            status.update(label=t(language, "status_searching"), state="running")
            result = _get_pipeline().ask(
                question=prompt,
                certification=certification,
                response_language=language,
                chapter=chapter,
            )
    except MissingAPIKeyError as exc:
        _append("assistant", str(exc))
        return

    st.session_state.last_debug = result.debug
    _append(
        "assistant",
        result.answer,
        citations=result.citations if result.found else [],
        debug=result.debug,
    )


def _handle_trainer(certification: str, language: str, chapter: str | None, prompt: str) -> None:
    stage = st.session_state.trainer_stage

    if stage == "awaiting_comprehension":
        _append("user", prompt)
        try:
            with thinking_status(language, evaluating=True) as status:
                status.update(label=t(language, "status_evaluating"), state="running")
                evaluation = _get_evaluator().evaluate(
                    original_topic=st.session_state.trainer_topic,
                    comprehension_question=st.session_state.trainer_comprehension_q,
                    student_answer=prompt,
                    certification=certification,
                    response_language=language,
                    chapter=chapter,
                )
        except MissingAPIKeyError as exc:
            _append("assistant", str(exc))
            return

        st.session_state.last_debug = evaluation.debug
        st.session_state.trainer_stage = "evaluated"
        _append(
            "assistant",
            evaluation.feedback,
            citations=evaluation.citations,
            debug=evaluation.debug,
        )
        return

    _append("user", prompt)
    try:
        with thinking_status(language) as status:
            status.update(label=t(language, "status_searching"), state="running")
            turn = _get_trainer().explain(
                topic=prompt,
                certification=certification,
                response_language=language,
                chapter=chapter,
            )
    except MissingAPIKeyError as exc:
        _append("assistant", str(exc))
        return

    st.session_state.last_debug = turn.debug
    st.session_state.trainer_last_citations = turn.citations
    st.session_state.trainer_topic = turn.topic
    st.session_state.trainer_comprehension_q = turn.comprehension_question

    if turn.found and turn.comprehension_question:
        st.session_state.trainer_stage = "awaiting_comprehension"
        _append(
            "assistant",
            turn.explanation,
            citations=turn.citations,
            comprehension_question=turn.comprehension_question,
            debug=turn.debug,
        )
    else:
        st.session_state.trainer_stage = "idle"
        _append("assistant", turn.explanation, citations=[], debug=turn.debug)


def _render_quiz_ui(certification: str, language: str, chapter: str | None) -> None:
    st.subheader(t(language, "mode_quiz"))
    st.caption(t(language, "quiz_label"))

    session: QuizSession | None = st.session_state.quiz_session

    if session is None:
        diff_labels = {
            t(language, "easy"): "Easy",
            t(language, "medium"): "Medium",
            t(language, "exam_like"): "Exam-like",
        }
        diff_ui = st.selectbox(t(language, "difficulty"), list(diff_labels.keys()))
        difficulty = diff_labels[diff_ui]
        n = st.selectbox(t(language, "num_questions"), [5, 10])

        if st.button(t(language, "start_quiz"), type="primary"):
            if not _guardrails(language):
                st.stop()
            with thinking_status(language) as status:
                status.update(label=t(language, "status_generating"), state="running")
                try:
                    session = _get_quiz_engine().start_quiz(
                        certification=certification,
                        language=language,
                        difficulty=difficulty,  # type: ignore[arg-type]
                        num_questions=int(n),
                        chapter=chapter,
                    )
                except MissingAPIKeyError as exc:
                    st.error(str(exc))
                    return
            if not session.questions:
                st.warning(t(language, "quiz_no_questions"))
                return
            st.session_state.quiz_session = session
            st.session_state.quiz_feedback = None
            st.rerun()
        return

    if session.finished and not st.session_state.quiz_feedback:
        total = len(session.answers)
        st.success(
            f"{t(language, 'quiz_finished')} — "
            f"{t(language, 'score_result', correct=session.correct_count, total=total)} "
            f"— {session.score_percent}%"
        )
        st.progress(min(1.0, session.score_percent / 100.0))
        st.markdown(f"**{t(language, 'breakdown')}**")
        for i, (q, ans) in enumerate(zip(session.questions, session.answers), start=1):
            mark = "✅" if ans == q.correct else "❌"
            st.markdown(
                f"{mark} Q{i}: {t(language, 'chose')} **{ans}** "
                f"({t(language, 'correct_label')} **{q.correct}**)"
            )
        if st.button(t(language, "start_quiz")):
            st.session_state.quiz_session = None
            st.session_state.quiz_feedback = None
            st.rerun()
        return

    if st.session_state.quiz_feedback:
        answered_idx = max(0, session.current_index - 1)
        question = session.questions[answered_idx]
        st.markdown(st.session_state.quiz_feedback)
        if st.session_state.show_sources:
            render_sources(question.citations, language)
        if st.button(t(language, "next_question"), type="primary"):
            st.session_state.quiz_feedback = None
            st.rerun()
        return

    idx = session.current_index
    question = session.questions[idx]
    st.markdown(
        f"**{t(language, 'quiz_progress', current=idx + 1, total=len(session.questions))}**"
    )
    st.progress((idx) / max(len(session.questions), 1))
    st.caption(question.label)
    st.markdown(question.question)

    option_keys = list(question.options.keys())
    labels = [f"{k}. {question.options[k]}" for k in option_keys]
    choice_label = st.radio(t(language, "select_option"), labels, index=None)

    if st.button(t(language, "submit_choice"), type="primary", disabled=choice_label is None):
        chosen = choice_label.split(".", 1)[0].strip() if choice_label else ""
        _, feedback = _get_quiz_engine().grade_current(session, chosen)
        st.session_state.quiz_feedback = feedback
        st.session_state.quiz_session = session
        st.rerun()


def main() -> None:
    _init_state()
    inject_styles()
    certification, language, mode, chapter = _sidebar()

    render_header(language)

    if mode == "trainer":
        st.caption(t(language, "trainer_mode_badge"))

    can_generate = _guardrails(language)

    if mode == "quiz":
        _render_quiz_ui(certification, language, chapter)
        render_rag_debug(st.session_state.last_debug, language)
        render_footer(language)
        return

    if not st.session_state.messages:
        render_welcome(language, certification)

    _render_history(language, mode)

    if st.session_state.trainer_stage == "awaiting_comprehension" and mode == "trainer":
        st.info(t(language, "awaiting_answer"))

    placeholder = (
        t(language, "trainer_placeholder")
        if mode == "trainer"
        else t(language, "ask_placeholder")
    )
    prompt = st.chat_input(placeholder, disabled=not can_generate)

    if prompt:
        if mode == "ask":
            _handle_ask(certification, language, chapter, prompt)
        else:
            _handle_trainer(certification, language, chapter, prompt)
        st.rerun()

    render_rag_debug(st.session_state.last_debug, language)
    render_footer(language)


if __name__ == "__main__":
    main()
