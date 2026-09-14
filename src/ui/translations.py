"""Bilingual UI strings for Chat ISTQB (user-facing branding)."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "app_title": "Chat ISTQB",
        "app_tagline": "Your AI study coach for CTFL & CT-AI",
        "app_badge": "AI • RAG • QA",
        "settings": "Settings",
        "certification": "Certification",
        "language": "Language",
        "mode": "Mode",
        "chapter_optional": "Chapter (optional)",
        "chapter_placeholder": "e.g. 2 or 6.1",
        "chapter_any": "Any",
        "mode_trainer": "🎓 Trainer",
        "mode_ask": "💬 Ask a Question",
        "mode_quiz": "🧠 Quiz",
        "lang_option_en": "English",
        "lang_option_fr": "French",
        "ask_placeholder": (
            "What would you like to study? e.g. Explain the difference between precision and recall."
        ),
        "trainer_placeholder": (
            "What would you like to study? e.g. Explain model drift in AI testing."
        ),
        "send": "Send",
        "submit_answer": "Submit answer",
        "your_answer": "Your answer",
        "new_question": "New question",
        "reset_session": "Reset session",
        "show_sources": "Show sources",
        "sources": "📚 Sources used",
        "sources_chapter": "Chapter",
        "sources_section": "Section",
        "sources_page": "Page",
        "sources_document": "Document",
        "explanation": "✨ Explanation",
        "check_understanding": "🧠 Let's check your understanding",
        "your_turn": "🧠 Your turn!",
        "trainer_mode_badge": "🎓 Trainer Mode",
        "awaiting_answer": "Waiting for your answer to the comprehension question…",
        "rag_debug": "🔬 RAG Debug",
        "debug_query": "User query",
        "debug_certification": "Certification",
        "debug_language": "Response language",
        "debug_chunks": "Retrieved chunks",
        "debug_evidence": "Sufficient evidence",
        "debug_settings": "Generation settings",
        "debug_grounding": "Grounding",
        "debug_repair": "Repair attempted",
        "debug_unsupported": "Unsupported claims",
        "debug_yes": "YES",
        "debug_no": "NO",
        "difficulty": "Difficulty",
        "num_questions": "Number of questions",
        "start_quiz": "Start quiz",
        "quiz_label": "AI-generated practice question based on the indexed syllabus.",
        "quiz_progress": "Question {current} / {total}",
        "score": "Score",
        "score_result": "Score: {correct} / {total}",
        "missing_api_key": (
            "⚠️ OpenAI API key not configured. On Streamlit Cloud, set "
            "`OPENAI_API_KEY` in App secrets. Locally, copy `.env.example` to "
            "`.env`. Indexing and retrieval can still work; answer generation "
            "requires the key."
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
        "disclaimer_short": (
            "Independent educational project — not affiliated with or endorsed by ISTQB®."
        ),
        "disclaimer_footer": (
            "Chat ISTQB is an independent educational project and is not affiliated with, "
            "sponsored by, or endorsed by ISTQB®. ISTQB® is a registered trademark of the "
            "International Software Testing Qualifications Board."
        ),
        "welcome_title": "👋 Welcome to Chat ISTQB",
        "welcome_body": (
            "Your AI coach for revising **CTFL** and **CT-AI** from indexed reference resources."
        ),
        "welcome_examples_title": "You can for example ask:",
        "welcome_ex_ctai_1": "• Explain the concept of model drift.",
        "welcome_ex_ctai_2": "• What is the difference between precision and recall?",
        "welcome_ex_ctai_3": "• Ask me a comprehension question about overfitting.",
        "welcome_ex_ctfl_1": "• Explain the pesticide paradox.",
        "welcome_ex_ctfl_2": "• What does testing show about defects?",
        "welcome_ex_ctfl_3": "• Ask me a question about testing principles.",
        "status_searching": "🔎 Searching the syllabus…",
        "status_preparing": "🧠 Preparing the explanation…",
        "status_generating": "✨ Chat ISTQB is preparing your answer…",
        "status_done": "✅ Answer ready",
        "status_evaluating": "📝 Evaluating your answer…",
        "indexed_chunks": "Indexed chunks: {count}",
        "chose": "chose",
        "correct_label": "correct",
        "verdict_correct": "✅ Correct",
        "verdict_partial": "🟡 Partially correct",
        "verdict_incorrect": "❌ Incorrect",
    },
    "fr": {
        "app_title": "Chat ISTQB",
        "app_tagline": "Votre coach IA pour réviser CTFL & CT-AI",
        "app_badge": "IA • RAG • QA",
        "settings": "Paramètres",
        "certification": "Certification",
        "language": "Langue",
        "mode": "Mode",
        "chapter_optional": "Chapitre (optionnel)",
        "chapter_placeholder": "ex. 2 ou 6.1",
        "chapter_any": "Tous",
        "mode_trainer": "🎓 Formateur",
        "mode_ask": "💬 Poser une question",
        "mode_quiz": "🧠 Quiz",
        "lang_option_en": "Anglais",
        "lang_option_fr": "Français",
        "ask_placeholder": (
            "Que souhaitez-vous réviser ? Ex. : explique-moi la différence entre précision et rappel."
        ),
        "trainer_placeholder": (
            "Que souhaitez-vous réviser ? Ex. : explique-moi la dérive de modèle."
        ),
        "send": "Envoyer",
        "submit_answer": "Soumettre la réponse",
        "your_answer": "Votre réponse",
        "new_question": "Nouvelle question",
        "reset_session": "Réinitialiser",
        "show_sources": "Afficher les sources",
        "sources": "📚 Sources utilisées",
        "sources_chapter": "Chapitre",
        "sources_section": "Section",
        "sources_page": "Page",
        "sources_document": "Document",
        "explanation": "✨ Explication",
        "check_understanding": "🧠 Vérifions votre compréhension",
        "your_turn": "🧠 À vous !",
        "trainer_mode_badge": "🎓 Mode Formateur",
        "awaiting_answer": "En attente de votre réponse à la question de compréhension…",
        "rag_debug": "🔬 Diagnostic RAG",
        "debug_query": "Requête utilisateur",
        "debug_certification": "Certification",
        "debug_language": "Langue de réponse",
        "debug_chunks": "Chunks récupérés",
        "debug_evidence": "Preuve suffisante",
        "debug_settings": "Paramètres de génération",
        "debug_grounding": "Ancrage",
        "debug_repair": "Réparation tentée",
        "debug_unsupported": "Affirmations non étayées",
        "debug_yes": "OUI",
        "debug_no": "NON",
        "difficulty": "Difficulté",
        "num_questions": "Nombre de questions",
        "start_quiz": "Démarrer le quiz",
        "quiz_label": "Question d'entraînement générée par IA à partir du syllabus indexé.",
        "quiz_progress": "Question {current} / {total}",
        "score": "Score",
        "score_result": "Résultat : {correct} / {total}",
        "missing_api_key": (
            "⚠️ Clé API OpenAI manquante. Sur Streamlit Cloud, définissez "
            "`OPENAI_API_KEY` dans les Secrets de l'app. En local, copiez "
            "`.env.example` vers `.env`. L'indexation et la recherche "
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
        "disclaimer_short": (
            "Projet éducatif indépendant — non affilié ni approuvé par ISTQB®."
        ),
        "disclaimer_footer": (
            "Chat ISTQB est un projet éducatif indépendant et n'est ni affilié, ni sponsorisé, "
            "ni approuvé par ISTQB®. ISTQB® est une marque déposée de l'International "
            "Software Testing Qualifications Board."
        ),
        "welcome_title": "👋 Bienvenue dans Chat ISTQB",
        "welcome_body": (
            "Votre coach IA pour réviser **CTFL** et **CT-AI** à partir de ressources de référence indexées."
        ),
        "welcome_examples_title": "Vous pouvez par exemple demander :",
        "welcome_ex_ctai_1": "• Explique-moi le concept de dérive de modèle.",
        "welcome_ex_ctai_2": "• Quelle est la différence entre précision et rappel ?",
        "welcome_ex_ctai_3": "• Pose-moi une question sur le surapprentissage (overfitting).",
        "welcome_ex_ctfl_1": "• Explique-moi le paradoxe du pesticide.",
        "welcome_ex_ctfl_2": "• Que montre le test concernant les défauts ?",
        "welcome_ex_ctfl_3": "• Pose-moi une question sur les principes de test.",
        "status_searching": "🔎 Recherche dans le syllabus…",
        "status_preparing": "🧠 Préparation de l'explication…",
        "status_generating": "✨ Chat ISTQB prépare votre réponse…",
        "status_done": "✅ Réponse prête",
        "status_evaluating": "📝 Évaluation de votre réponse…",
        "indexed_chunks": "Chunks indexés : {count}",
        "chose": "choix",
        "correct_label": "correct",
        "verdict_correct": "✅ Bonne réponse",
        "verdict_partial": "🟡 Réponse partiellement correcte",
        "verdict_incorrect": "❌ Réponse incorrecte",
    },
}


def t(language: str, key: str, **fmt: object) -> str:
    """Translate a UI key; fall back to English then to the key itself."""
    lang = language if language in TRANSLATIONS else "en"
    text = TRANSLATIONS[lang].get(key) or TRANSLATIONS["en"].get(key, key)
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, ValueError):
            return text
    return text
