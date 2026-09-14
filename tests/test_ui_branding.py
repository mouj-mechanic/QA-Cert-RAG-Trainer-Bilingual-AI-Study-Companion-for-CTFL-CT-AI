"""UI branding / localization smoke checks (no Streamlit runtime)."""

from __future__ import annotations

from pathlib import Path

from src.ui.translations import TRANSLATIONS, t


def test_visible_product_name_is_chat_istqb():
    assert t("en", "app_title") == "Chat ISTQB"
    assert t("fr", "app_title") == "Chat ISTQB"


def test_no_underscore_brand_in_translations():
    for lang, table in TRANSLATIONS.items():
        for key, value in table.items():
            if "app_" in key or key in {
                "welcome_title",
                "disclaimer_footer",
                "status_generating",
            }:
                assert "Chat_ISTQB" not in value, f"{lang}.{key} still uses underscore brand"


def test_app_py_defaults_and_branding_literals():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert 'page_title="Chat ISTQB — AI Study Companion"' in source
    assert '"ui_language": "fr"' in source
    assert '"ui_certification": "CT-AI"' in source
    assert '"ui_mode": "trainer"' in source
    assert " / Langue" not in source
    assert 't("en", "certification") +' not in source


def test_translation_helper_formatting():
    text = t("fr", "score_result", correct=8, total=10)
    assert "8" in text and "10" in text
