"""Bilingual UI strings for Chat_ISTQB."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "app_title": "Chat_ISTQB",
        "app_subtitle": "Unofficial AI Study Companion",
        "certification": "Certification",
        "language": "Language",
        "mode": "Mode",
        "chapter_optional": "Chapter (optional)",
        "chapter_any": "Any",
        "mode_trainer": "Trainer",
        "mode_ask": "Ask a Question",
        "mode_quiz": "Quiz",
        "ask_placeholder": "Ask an ISTQB-related question…",
        "trainer_placeholder": "What would you like to study? e.g. Explain the pesticide paradox",
        "send": "Send",
        "submit_answer": "Submit answer",
        "your_answer": "Your answer",
        "new_question": "New question",
        "reset_session": "Reset session",
        "show_sources": "Show sources",
        "sources": "📚 Sources",
        "check_understanding": "🧠 Check your understanding",
        "awaiting_answer": "Waiting for your answer to the comprehension question…",
        "rag_debug": "🔬 RAG Debug",
        "difficulty": "Difficulty",
        "num_questions": "Number of questions",
        "start_quiz": "Start quiz",
        "quiz_label": "AI-generated practice question based on the indexed syllabus.",
        "score": "Score",
        "missing_api_key": (
            "⚠️ OpenAI API key not configured. Copy `.env.example` to `.env` "
            "and set `OPENAI_API_KEY`. Indexing and retrieval can still work; "
            "answer generation requires the key."
        ),
        "empty_index": (
            "⚠️ The vector index is empty. Add resources under `/resources` "
            "and run: `python -m src.ingestion.indexer`"
        ),
        "quiz_no_questions": "Could not generate practice questions from indexed resources.",
        "quiz_finished": "Quiz finished",
        "select_option": "Select an option",
        "submit_choice": "Submit answer",
        "next_question": "Continue",
        "breakdown": "Breakdown",
        "easy": "Easy",
        "medium": "Medium",
        "exam_like": "Exam-like",
    },
    "fr": {
        "app_title": "Chat_ISTQB",
        "app_subtitle": "Compagnon d'étude IA non officiel",
        "certification": "Certification",
        "language": "Langue",
        "mode": "Mode",
        "chapter_optional": "Chapitre (optionnel)",
        "chapter_any": "Tous",
        "mode_trainer": "Formateur",
        "mode_ask": "Poser une question",
        "mode_quiz": "Quiz",
        "ask_placeholder": "Posez une question liée à l'ISTQB…",
        "trainer_placeholder": "Que souhaitez-vous étudier ? ex. Explique-moi le paradoxe du pesticide",
        "send": "Envoyer",
        "submit_answer": "Soumettre la réponse",
        "your_answer": "Votre réponse",
        "new_question": "Nouvelle question",
        "reset_session": "Réinitialiser la session",
        "show_sources": "Afficher les sources",
        "sources": "📚 Sources",
        "check_understanding": "🧠 Vérifiez votre compréhension",
        "awaiting_answer": "En attente de votre réponse à la question de compréhension…",
        "rag_debug": "🔬 RAG Debug",
        "difficulty": "Difficulté",
        "num_questions": "Nombre de questions",
        "start_quiz": "Démarrer le quiz",
        "quiz_label": "Question d'entraînement générée par IA à partir du syllabus indexé.",
        "score": "Score",
        "missing_api_key": (
            "⚠️ Clé API OpenAI manquante. Copiez `.env.example` vers `.env` "
            "et définissez `OPENAI_API_KEY`. L'indexation et la recherche "
            "fonctionnent sans clé ; la génération de réponses en a besoin."
        ),
        "empty_index": (
            "⚠️ L'index vectoriel est vide. Ajoutez des ressources sous `/resources` "
            "puis exécutez : `python -m src.ingestion.indexer`"
        ),
        "quiz_no_questions": "Impossible de générer des questions à partir des ressources indexées.",
        "quiz_finished": "Quiz terminé",
        "select_option": "Choisissez une option",
        "submit_choice": "Valider",
        "next_question": "Continuer",
        "breakdown": "Détail",
        "easy": "Facile",
        "medium": "Moyen",
        "exam_like": "Type examen",
    },
}


def t(language: str, key: str) -> str:
    """Translate a UI key; fall back to English then to the key itself."""
    lang = language if language in TRANSLATIONS else "en"
    return TRANSLATIONS[lang].get(key) or TRANSLATIONS["en"].get(key, key)
