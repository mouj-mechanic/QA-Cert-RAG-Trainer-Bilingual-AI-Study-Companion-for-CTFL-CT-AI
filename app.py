"""
Chat_ISTQB — Unofficial AI Study Companion

Streamlit entry point. Session memory only (no user accounts).
"""

from __future__ import annotations

# Streamlit Community Cloud often ships an older system SQLite than Chroma needs.
# pysqlite3-binary is optional locally (esp. on Windows); required on Cloud.
try:
    __import__("pysqlite3")
    import sys as _sys

    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import logging
from typing import Any

import streamlit as st

from src.config import CERTIFICATIONS, UNOFFICIAL_BANNER, has_openai_key
from src.llm.provider import MissingAPIKeyError
from src.rag.rag_pipeline import RAGPipeline
from src.rag.vector_store import VectorStore
from src.trainer.evaluator import AnswerEvaluator
from src.trainer.quiz import QuizEngine, QuizSession
from src.trainer.trainer import Trainer
from src.ui.components import render_footer, render_rag_debug, render_sources
from src.ui.translations import t

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Chat_ISTQB",
    page_icon="📘",
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
    with st.sidebar:
        st.title("Chat_ISTQB")
        st.caption(t(st.session_state.get("_ui_lang", "en"), "app_subtitle"))
        st.info(UNOFFICIAL_BANNER)

        cert_labels = {v["label"]: k for k, v in CERTIFICATIONS.items()}
        cert_label = st.selectbox(
            t("en", "certification") + " / " + t("fr", "certification"),
            options=list(cert_labels.keys()),
        )
        certification = cert_labels[cert_label]

        lang_label = st.selectbox(
            t("en", "language") + " / Langue",
            options=["English", "Français"],
        )
        language = "en" if lang_label == "English" else "fr"
        st.session_state._ui_lang = language

        mode_map = {
            t(language, "mode_trainer"): "trainer",
            t(language, "mode_ask"): "ask",
            t(language, "mode_quiz"): "quiz",
        }
        mode_label = st.selectbox(t(language, "mode"), options=list(mode_map.keys()))
        mode = mode_map[mode_label]

        chapter_raw = st.text_input(t(language, "chapter_optional"), value="")
        chapter = chapter_raw.strip() or None

        st.checkbox(t(language, "show_sources"), key="show_sources")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(t(language, "new_question")):
                st.session_state.trainer_stage = "idle"
                st.session_state.trainer_topic = ""
                st.session_state.trainer_comprehension_q = ""
                st.session_state.quiz_feedback = None
        with col2:
            if st.button(t(language, "reset_session")):
                _reset_session()
                st.rerun()

        store = _get_vector_store()
        st.caption(f"Indexed chunks: {store.count}")

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


def _render_history(language: str) -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations") and st.session_state.show_sources:
                render_sources(msg["citations"], language)
            if msg.get("comprehension_question"):
                st.markdown(f"**{t(language, 'check_understanding')}**")
                st.markdown(msg["comprehension_question"])


def _handle_ask(certification: str, language: str, chapter: str | None, prompt: str) -> None:
    _append("user", prompt)
    try:
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
        # Student is answering the comprehension question
        _append("user", prompt)
        try:
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

    # New topic explanation
    _append("user", prompt)
    try:
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

    # Setup screen
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
            with st.spinner("…"):
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

    # Final score screen
    if session.finished and not st.session_state.quiz_feedback:
        st.success(
            f"{t(language, 'quiz_finished')} — "
            f"{t(language, 'score')}: {session.correct_count}/{len(session.answers)} "
            f"— {session.score_percent}%"
        )
        st.markdown(f"**{t(language, 'breakdown')}**")
        for i, (q, ans) in enumerate(zip(session.questions, session.answers), start=1):
            mark = "✅" if ans == q.correct else "❌"
            st.markdown(f"{mark} Q{i}: chose **{ans}** (correct **{q.correct}**)")
        if st.button(t(language, "start_quiz")):
            st.session_state.quiz_session = None
            st.session_state.quiz_feedback = None
            st.rerun()
        return

    # Feedback after an answer
    if st.session_state.quiz_feedback:
        # Show the question that was just answered (index already advanced)
        answered_idx = max(0, session.current_index - 1)
        question = session.questions[answered_idx]
        st.markdown(st.session_state.quiz_feedback)
        if st.session_state.show_sources:
            render_sources(question.citations, language)
        if st.button(t(language, "next_question"), type="primary"):
            st.session_state.quiz_feedback = None
            st.rerun()
        return

    # Active question
    idx = session.current_index
    question = session.questions[idx]
    st.markdown(f"**Question {idx + 1}/{len(session.questions)}**")
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
    certification, language, mode, chapter = _sidebar()

    st.title("Chat_ISTQB")
    st.caption(t(language, "app_subtitle"))
    st.info(UNOFFICIAL_BANNER)

    can_generate = _guardrails(language)

    if mode == "quiz":
        _render_quiz_ui(certification, language, chapter)
        render_rag_debug(st.session_state.last_debug, language)
        render_footer()
        return

    _render_history(language)

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
    render_footer()


if __name__ == "__main__":
    main()
