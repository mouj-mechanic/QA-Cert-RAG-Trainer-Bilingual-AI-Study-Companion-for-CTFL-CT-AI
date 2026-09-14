"""Language acceptance and bilingual UI helpers."""

from __future__ import annotations

from src.rag.retriever import Retriever
from src.ui.translations import t


def test_french_and_english_queries_accepted(indexed_store):
    store, embedder, _ = indexed_store
    retriever = Retriever(vector_store=store, embedder=embedder, top_k=4, min_score=0.0)

    en = retriever.retrieve("Explain the pesticide paradox", certification="CTFL")
    fr = retriever.retrieve("Explique le paradoxe du pesticide", certification="CTFL")

    # Both should be accepted (no crash) and return some CTFL evidence from fixtures
    assert en.certification == "CTFL"
    assert fr.certification == "CTFL"
    assert isinstance(en.chunks, list)
    assert isinstance(fr.chunks, list)


def test_translations_cover_both_languages():
    assert "Trainer" in t("en", "mode_trainer")
    assert "Formateur" in t("fr", "mode_trainer")
    assert t("en", "app_title") == "Chat ISTQB"
    assert t("fr", "app_title") == "Chat ISTQB"
    assert "_" not in t("en", "app_title")
    assert "cannot find" in t("en", "empty_index").lower() or "index" in t("en", "empty_index").lower()
    # Full localization: shared keys exist in both languages
    from src.ui.translations import TRANSLATIONS

    assert set(TRANSLATIONS["en"]) == set(TRANSLATIONS["fr"])
