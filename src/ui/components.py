"""Reusable Streamlit UI helpers for Chat ISTQB (presentation only)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import streamlit as st

from src.ui.translations import t


def inject_styles() -> None:
    """Modern learning-product look — limited, Streamlit-safe CSS."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

.stApp {
  background: linear-gradient(165deg, #f4f7ff 0%, #f8fafc 42%, #eef6ff 100%);
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f5f8ff 100%);
  border-right: 1px solid #e2e8f0;
}

.ci-brand {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-weight: 700;
  font-size: 2rem;
  letter-spacing: -0.03em;
  color: #1e3a8a;
  margin: 0 0 0.15rem 0;
  line-height: 1.15;
}

.ci-tagline {
  color: #334155;
  font-size: 1.02rem;
  margin: 0 0 0.65rem 0;
}

.ci-badge {
  display: inline-block;
  background: linear-gradient(90deg, #4f46e5, #0ea5e9);
  color: white;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  margin-bottom: 0.75rem;
}

.ci-disclaimer {
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 0.55rem 0.85rem;
  color: #64748b;
  font-size: 0.82rem;
  margin: 0.4rem 0 1rem 0;
}

.ci-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 1rem 1.1rem;
  margin: 0.55rem 0;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

.ci-card-title {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-weight: 600;
  color: #1e3a8a;
  margin-bottom: 0.45rem;
}

.ci-welcome {
  background: #ffffff;
  border: 1px solid #dbeafe;
  border-radius: 16px;
  padding: 1.25rem 1.35rem;
  margin: 0.5rem 0 1.1rem 0;
  box-shadow: 0 8px 24px rgba(37, 99, 235, 0.06);
}

.ci-sidebar-brand {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-weight: 700;
  font-size: 1.35rem;
  color: #1e3a8a;
  margin-bottom: 0.15rem;
}

.ci-section-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #64748b;
  margin: 0.85rem 0 0.35rem 0;
}

div[data-testid="stChatInput"] textarea {
  border-radius: 14px !important;
}

@media (max-width: 768px) {
  .ci-brand { font-size: 1.55rem; }
  .ci-welcome { padding: 1rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header(language: str) -> None:
    st.markdown(
        f"""
<div class="ci-badge">{t(language, "app_badge")}</div>
<div class="ci-brand">{t(language, "app_title")}</div>
<p class="ci-tagline">{t(language, "app_tagline")}</p>
<div class="ci-disclaimer">{t(language, "disclaimer_short")}</div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome(language: str, certification: str) -> None:
    if certification == "CTFL":
        examples = [
            t(language, "welcome_ex_ctfl_1"),
            t(language, "welcome_ex_ctfl_2"),
            t(language, "welcome_ex_ctfl_3"),
        ]
    else:
        examples = [
            t(language, "welcome_ex_ctai_1"),
            t(language, "welcome_ex_ctai_2"),
            t(language, "welcome_ex_ctai_3"),
        ]
    examples_html = "<br/>".join(examples)
    st.markdown(
        f"""
<div class="ci-welcome">
  <div class="ci-card-title">{t(language, "welcome_title")}</div>
  <p>{t(language, "welcome_body")}</p>
  <p><strong>{t(language, "welcome_examples_title")}</strong></p>
  <p>{examples_html}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(language: str) -> None:
    st.divider()
    st.caption(t(language, "disclaimer_footer"))


def render_sources(citations: list[dict[str, Any]], language: str) -> None:
    if not citations:
        return
    with st.expander(t(language, "sources"), expanded=False):
        for cite in citations:
            cert = cite.get("certification", "")
            version = cite.get("version", "")
            bits = [f"**{cert} {version}**".strip()]
            if cite.get("chapter"):
                bits.append(f"{t(language, 'sources_chapter')} {cite['chapter']}")
            if cite.get("section"):
                bits.append(f"{t(language, 'sources_section')} {cite['section']}")
            if cite.get("page") not in (None, ""):
                bits.append(f"{t(language, 'sources_page')} {cite['page']}")
            st.markdown(" · ".join(bits))


def render_assistant_answer(
    language: str,
    content: str,
    *,
    citations: list[dict[str, Any]] | None = None,
    comprehension_question: str | None = None,
    show_sources: bool = True,
    show_explanation_label: bool = True,
) -> None:
    """Render an assistant turn with light visual structure."""
    if show_explanation_label:
        st.markdown(f"**{t(language, 'explanation')}**")
        st.markdown(content)
    else:
        st.markdown(content)

    if show_sources and citations:
        render_sources(citations, language)

    if comprehension_question:
        st.markdown(f"**{t(language, 'check_understanding')}**")
        st.info(f"**{t(language, 'your_turn')}**\n\n{comprehension_question}")


def render_rag_debug(debug: dict[str, Any] | None, language: str) -> None:
    """Collapsed developer panel — not part of the normal student UX."""
    if not debug:
        return
    with st.expander(t(language, "rag_debug"), expanded=False):
        st.write(f"**{t(language, 'debug_query')}:**", debug.get("query"))
        st.write(f"**{t(language, 'debug_certification')}:**", debug.get("certification"))
        st.write(f"**{t(language, 'debug_language')}:**", debug.get("response_language"))
        st.write(f"**{t(language, 'debug_chunks')}:**", debug.get("retrieved_count", 0))
        if "sufficient_evidence" in debug:
            st.write(f"**{t(language, 'debug_evidence')}:**", debug.get("sufficient_evidence"))
        if debug.get("generation_settings"):
            st.write(f"**{t(language, 'debug_settings')}:**", debug.get("generation_settings"))
        st.write(f"**{t(language, 'debug_grounding')}:**", debug.get("grounding", "N/A"))
        repair = debug.get("repair_attempted")
        if repair is not None:
            label = t(language, "debug_yes") if repair else t(language, "debug_no")
            st.write(f"**{t(language, 'debug_repair')}:**", label)
        if "unsupported_claims_count" in debug:
            st.write(
                f"**{t(language, 'debug_unsupported')}:**",
                debug.get("unsupported_claims_count", 0),
            )
        chunks = debug.get("chunks") or []
        for i, chunk in enumerate(chunks, start=1):
            cid = chunk.get("chunk_id") or f"S{i}"
            st.markdown(f"**Chunk {cid}** — score: `{chunk.get('score')}`")
            st.json(chunk.get("metadata") or {})
            st.code(chunk.get("preview") or "", language=None)


@contextmanager
def thinking_status(language: str, evaluating: bool = False) -> Iterator[Any]:
    """Safe high-level waiting UX while RAG/LLM work runs."""
    first = t(language, "status_evaluating" if evaluating else "status_searching")
    with st.status(first, expanded=True) as status:
        try:
            yield status
        finally:
            status.update(label=t(language, "status_done"), state="complete")
