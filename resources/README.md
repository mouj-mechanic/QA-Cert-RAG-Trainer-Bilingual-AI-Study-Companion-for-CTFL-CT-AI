# Study resources (not committed as PDFs)

Place **your own** officially obtained ISTQB study materials here.

## Layout

```
resources/
  ctfl/
    en/          ← CTFL 4.0.1 English PDFs / TXT / MD
    fr/          ← CTFL 4.0.1 French PDFs / TXT / MD
  ct_ai/
    en/          ← CT-AI 2.0 English
    fr/          ← CT-AI 2.0 French
```

## Supported formats

- `.pdf`
- `.txt`
- `.md` / `.markdown`

## Important

- This repository does **not** redistribute official ISTQB syllabus PDFs.
- Do **not** commit copyrighted PDFs to a public GitHub repository (`.gitignore` already excludes `resources/**/*.pdf`).
- Obtain materials from official ISTQB / national board sources yourself.
- After adding files, rebuild the index:

```bash
python -m src.ingestion.indexer
```

Empty folders are expected in a fresh clone.
