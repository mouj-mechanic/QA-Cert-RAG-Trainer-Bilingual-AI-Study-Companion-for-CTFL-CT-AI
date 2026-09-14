# RAG evaluation

This folder measures **retrieval quality** for Chat_ISTQB.

## Philosophy

A portfolio RAG project should prove that retrieval can be tested — not only that an LLM can answer fluently.

V1 focuses on **retrieval hit rate**:

> Was an expected relevant source present in Top-K?

Generation metrics (correctness, groundedness, citation correctness, hallucination rate) are left as a clear extension point.

## questions.json

Complete this file **manually** after you index your official documents.

Do not invent a large automatic dataset. A few carefully labelled questions are more useful than hundreds of noisy ones.

Each item:

```json
{
  "id": "CTFL_001",
  "certification": "CTFL",
  "language": "en",
  "question": "...",
  "expected_document": "filename_stem_without_extension",
  "expected_chapter": "1"
}
```

`expected_document` should match the `document` metadata field (usually the file stem).
`expected_chapter` is optional but recommended when your PDFs expose chapter structure.

## Run

```bash
python -m evaluation.evaluate_retrieval
```

Requires a built vector index (`python -m src.ingestion.indexer`).

## Metrics reported

- Number of evaluation questions
- Top-1 hit rate
- Top-3 hit rate
- Top-5 hit rate
- Overall hit (expected doc present in Top-K used)
