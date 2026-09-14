"""Reusable Streamlit UI helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import DISCLAIMER, UNOFFICIAL_BANNER
from src.ui.translations import t


def render_banner() -> None:
    st.info(UNOFFICIAL_BANNER)


def render_footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)


def render_sources(citations: list[dict[str, Any]], language: str) -> None:
    if not citations:
        return
    with st.expander(t(language, "sources"), expanded=True):
        for cite in citations:
            cert = cite.get("certification", "")
            version = cite.get("version", "")
            lines = [f"**{cert} {version}**".strip()]
            if cite.get("chapter"):
                lines.append(f"Chapter {cite['chapter']}")
            if cite.get("section"):
                lines.append(f"Section {cite['section']}")
            if cite.get("page") not in (None, ""):
                lines.append(f"Page {cite['page']}")
            if cite.get("document"):
                lines.append(f"Document: `{cite['document']}`")
            st.markdown("  \n".join(lines))
            st.markdown("---")


def render_rag_debug(debug: dict[str, Any] | None, language: str) -> None:
    """Collapsed developer panel — not part of the normal student UX."""
    if not debug:
        return
    with st.expander(t(language, "rag_debug"), expanded=False):
        st.write("**User query:**", debug.get("query"))
        st.write("**Certification:**", debug.get("certification"))
        st.write("**Response language:**", debug.get("response_language"))
        st.write("**Retrieved chunks:**", debug.get("retrieved_count", 0))
        if "sufficient_evidence" in debug:
            st.write("**Sufficient evidence:**", debug.get("sufficient_evidence"))
        if debug.get("generation_settings"):
            st.write("**Generation settings:**", debug.get("generation_settings"))
        st.write("**Grounding:**", debug.get("grounding", "N/A"))
        repair = debug.get("repair_attempted")
        if repair is not None:
            st.write("**Repair attempted:**", "YES" if repair else "NO")
        if "unsupported_claims_count" in debug:
            st.write("**Unsupported claims:**", debug.get("unsupported_claims_count", 0))
        chunks = debug.get("chunks") or []
        for i, chunk in enumerate(chunks, start=1):
            cid = chunk.get("chunk_id") or f"S{i}"
            st.markdown(f"**Chunk {cid}** — score: `{chunk.get('score')}`")
            st.json(chunk.get("metadata") or {})
            st.code(chunk.get("preview") or "", language=None)


def render_chat_message(role: str, content: str) -> None:
    with st.chat_message(role):
        st.markdown(content)
