"""
Central configuration for Chat_ISTQB.

Baseline RAG choices (V1 — measurable, intentionally simple):
- Embedding model: paraphrase-multilingual-MiniLM-L12-v2
  (FR/EN capable, small enough for a laptop portfolio demo)
- Vector DB: ChromaDB (persistent, metadata filtering built-in)
- Chunk size: 1000 characters, overlap: 200
- Retrieval: top_k = 4, cosine similarity, no reranker
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths (always relative to the project root — safe for Streamlit Cloud)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# ---------------------------------------------------------------------------
# Embedding / chunking / retrieval baselines
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))
# Evidence floor for keeping a retrieved chunk / answering.
# Original V1 baseline used 0.25. CT-AI English real-corpus validation showed:
#   in-syllabus top-1 scores ≈ 0.61–0.80
#   out-of-scope top-1 ≈ 0.41 (still passed 0.25 → found=True + LLM called)
# Smallest evidence-based correction that separates those bands: 0.50
# (lowest observed relevant top-4 ≈ 0.56; highest OOS top-1 ≈ 0.41).
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.50"))

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Certification answers use temperature=0 via src.rag.grounding.CERTIFICATION_TEMPERATURE


def get_openai_api_key() -> str:
    """
    Resolve the OpenAI API key without exposing it.

    Preference order:
    1) Streamlit secrets (Community Cloud / local secrets.toml)
    2) Environment variable / .env via python-dotenv
    """
    try:
        import streamlit as st

        # st.secrets may raise if no secrets file exists locally
        if "OPENAI_API_KEY" in st.secrets:
            value = st.secrets["OPENAI_API_KEY"]
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY", "").strip()


# Backward-compatible module attribute (may be empty until secrets are available).
# Prefer get_openai_api_key() at call time for Streamlit Cloud.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Certification catalogue
# ---------------------------------------------------------------------------
CERTIFICATIONS = {
    "CTFL": {
        "label": "CTFL 4.0.1",
        "version": "4.0.1",
        "resource_folder": "ctfl",
    },
    "CT-AI": {
        "label": "CT-AI 2.0",
        "version": "2.0",
        "resource_folder": "ct_ai",
    },
}

SUPPORTED_LANGUAGES = ("en", "fr")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

CHROMA_COLLECTION_NAME = "istqb_chunks"

DISCLAIMER = (
    "Chat_ISTQB is an independent educational and portfolio project. "
    "It is not affiliated with, sponsored by, or endorsed by ISTQB®. "
    "ISTQB® is a registered trademark of the International Software Testing "
    "Qualifications Board. Users are responsible for obtaining study materials "
    "from their official sources."
)

UNOFFICIAL_BANNER = (
    "Unofficial educational project. Not affiliated with or endorsed by ISTQB®."
)


def has_openai_key() -> bool:
    """Return True when an API key is configured (secrets or environment)."""
    key = get_openai_api_key()
    return bool(key) and key != "sk-your-key-here"
