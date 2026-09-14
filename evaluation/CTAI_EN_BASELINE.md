# CT-AI 2.0 English — RAG Baseline

Observation-only validation of the Chat_ISTQB V1 baseline against one real syllabus PDF.  
**No RAG parameters were changed** (embedding model, chunk size/overlap, top-k, ChromaDB, architecture).

Date: 2026-09-14

---

## Corpus

| Field | Value |
|---|---|
| Source file | `resources/ct_ai/en/ISTQB-_CTAI_Syllabus_v2.0_Release.pdf` |
| Document stem | `ISTQB-_CTAI_Syllabus_v2.0_Release` |
| Certification (from folder + config) | `CT-AI` |
| Version (from `src.config`) | `2.0` |
| Source language (from folder) | `en` |
| PDF pages (pypdf) | **93** |
| Documents discovered / loaded | **1 / 1** |
| Chunks created | **285** |
| Vectors stored | **285** |
| Chroma collection | `istqb_chunks` |
| Approx. indexed text size | ~245,073 characters across chunks |
| Printed vs physical page | Checked pages 1, 15, 36, 47, 93: printed `Page N of 93` matches physical PDF page index N |

### Chunk metadata checks (begin / middle / end)

| Sample | page | certification | version | language | chapter | section |
|---|---|---|---|---|---|---|
| Beginning | 1 | CT-AI | 2.0 | en | empty | empty |
| Middle | 40 | CT-AI | 2.0 | en | empty | empty |
| End | 93 | CT-AI | 2.0 | en | empty | empty |

Fill rates across 285 chunks:

- `chapter` present: 121 / 285 (heuristic from page text; empty when not found)
- `section` present: 82 / 285
- Unique pages covered: 93 / 93

**Page numbering observation:** physical PDF page index and syllabus-printed page number appear aligned in this release (`Page N of 93`). No page-offset bug observed.

**Metadata observation:** chapter/section are best-effort regex heuristics. Some TOC-like pages may get values such as `chapter=0` / `section=0.10`. Empty is preferred when unsure; no fabricated pages.

---

## Configuration (unchanged baseline)

| Setting | Value |
|---|---|
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | ChromaDB (`vector_db/`, cosine) |
| Chunk size | 1000 characters |
| Chunk overlap | 200 characters |
| Top-k | 4 |
| Min relevance score | 0.25 |

### Indexing warnings

- Hugging Face Hub: unauthenticated request warning (rate limits); model still loaded.
- Some HF HEAD requests returned 404 for optional files (`adapter_config.json`, `processor_config.json`, etc.) — normal for this model; not an extraction failure.
- PDF text extraction shows occasional broken words/spaces (e.g. `de ployed`, `processi ng`) — OCR/layout artifact from pypdf, not invented content.

---

## English retrieval tests

Questions were chosen after inspecting this PDF (not from general model memory).

### EN1 — Narrow AI vs general AI

**Question:** What is the difference between narrow AI and general AI?

| Rank | Score | Page | Ch/Sec | Preview focus |
|---|---|---|---|---|
| 1 | 0.7577 | 15 | 1 / 1.1 | Narrow AI limited domain… |
| 2 | 0.6673 | 15 | 1 / 1.1 | Conventional vs AI-based systems… |
| 3 | 0.6139 | 12 | 0 / 0.10 | TOC objectives: narrow AI to super AI |
| 4 | 0.5727 | 15 | 1 / 1.1 | Transparency / explainability nearby |

**Assessment: GOOD** — Top-1 and most of Top-4 land on the real §1.1 discussion (page 15).

### EN2 — Overfitting

**Question:** What is overfitting in machine learning?

| Rank | Score | Page | Preview focus |
|---|---|---|---|
| 1 | 0.6302 | 74 | Glossary definition of overfitting |
| 2 | 0.6043 | 61 | Underfitting / learning curves (related) |
| 3 | 0.6019 | 60 | Drift / statistical tests (weaker) |
| 4 | 0.5725 | 54 | Chapter 6 LO list mentioning related topics |

**Assessment: GOOD** — Top-1 is the official glossary definition; ranks 2–4 are thematically adjacent but noisier.

### EN3 — Precision, recall, F1-score

**Question:** Explain precision, recall and F1-score for classifiers.

| Rank | Score | Page | Preview focus |
|---|---|---|---|
| 1 | 0.6261 | 33 | Introduces accuracy, precision, recall, F1 |
| 2 | 0.5905 | 34 | Formulas: Precision/Recall = TP/(…) |
| 3 | 0.5631 | 69 | Terms chapter start |
| 4 | 0.5583 | 75 | Glossary entry for precision |

**Assessment: GOOD** — Core metric pages 33–34 retrieved in Top-2.

### EN4 — DNN layers

**Question:** What are the input layer, hidden layers and output layer in a deep neural network?

| Rank | Score | Page | Ch/Sec | Preview focus |
|---|---|---|---|---|
| 1 | 0.6509 | 35 | 3 / 3.3.3 | Perceptron / ANN intro |
| 2 | 0.6469 | 74 | — | Glossary: neural network |
| 3 | 0.5817 | 36 | 3 / 3.4.1 | **Structure of a deep neural network** |
| 4 | 0.5695 | 26 | 3 | LO list for neural networks |

**Assessment: PARTIAL** — Relevant page 36 appears at Top-3; Top-1 is adjacent but not the best chunk.

### EN5 — Back-to-back testing / test oracle

**Question:** How does back-to-back testing help with the test oracle problem?

| Rank | Score | Page | Ch/Sec | Preview focus |
|---|---|---|---|---|
| 1 | 0.7998 | 62 | 6 / 6.1.10 | Back-to-back testing / pseudo-oracle |
| 2 | 0.7361 | 39 | 4 / 4.1 | Test oracle problem explained |
| 3 | 0.6756 | 12 | 0 / 0.10 | TOC mentioning test oracles |
| 4 | 0.6752 | 39 | 4 / 4.1 | Oracle solutions |

**Assessment: GOOD** — Excellent Top-1 on the exact section.

### English summary

**Satisfactory: 4/5** (EN1, EN2, EN3, EN5 GOOD; EN4 PARTIAL)

---

## French → English retrieval tests

Same English corpus; French queries only.

### FR1 (linked to EN1)

**Question:** Quelle est la différence entre narrow AI et general AI ?

Top-4 pages: **15, 15, 12, 15** — scores 0.75 / 0.68 / 0.62 / 0.59  
Nearly identical to EN1.

**Assessment: GOOD**

### FR2 (linked to EN2)

**Question:** Qu'est-ce que le overfitting en apprentissage automatique ?

Top-4 pages: **74, 54, 60, 41** — Top-1 still glossary overfitting (0.61); ranks 2–4 differ slightly from EN2.

**Assessment: GOOD** (Top-1 match preserved; secondary ranks noisier)

### French→English summary

**Satisfactory: 2/2**

Multilingual MiniLM retrieves the same primary English sections for these two concepts without translating the source.

---

## Certification filtering

| Check | Result |
|---|---|
| Query with `certification=CT-AI` | Only `CT-AI` chunks returned |
| Query with `certification=CTFL` | **0** chunks (no CTFL docs indexed) |
| Chroma `where={"certification":"CTFL"}` | empty id list |
| Chroma `where={"certification":"CT-AI"}` | all returned metas = CT-AI |
| Automated `tests/test_metadata_filter.py` | **3 passed** |

**Result: PASS** (filter logic verified programmatically, not only by empty CTFL corpus)

---

## End-to-end RAG

Re-run with configured `OPENAI_API_KEY` (model `gpt-4o-mini`). No RAG knobs changed.

### E2E1 — Narrow AI vs general AI

- **Retrieved Top-4:** pages 15, 15, 12, 15 (scores ~0.76–0.57)
- **Citations displayed:** CT-AI 2.0 — Ch.1 §1.1 p.15; TOC p.12 — document `ISTQB-_CTAI_Syllabus_v2.0_Release`
- **Grounding:** Narrow-AI claims (limited domain, specialized tasks, cannot generalize without retraining) **supported** by page 15.
- **Flagged unsupported / weakly supported:** synonym “weak AI” **not present** in retrieved chunks; detailed “human cognitive abilities” characterization of general AI goes beyond the retrieved evidence (model partly hedges that general AI is not explicitly defined in context).

### E2E2 — Precision, recall, F1-score

- **Retrieved Top-4:** pages 33, 34, 69, 75
- **Citations:** p.33, p.34, p.69 — correct syllabus PDF
- **Grounding:** Formulas and definitions for precision/recall/F1 **supported** by pages 33–34. Numerical worked examples labeled as examples (B) — acceptable as pedagogical, not claimed as syllabus figures.

### E2E3 — Back-to-back testing / test oracle

- **Retrieved Top-4:** pages 62, 39, 12, 39
- **Citations:** Ch.6 §6.1.10 p.62; Ch.4 §4.1 p.39; TOC p.12
- **Grounding:** Pseudo-oracle explanation **supported** by page 62. Example (B) labeled as example.

**End-to-end grounding (overall): PASS** for core syllabus claims on E2E2/E2E3; E2E1 has **flagged** unsupported embellishment (“weak AI”).  
**Citation accuracy: PASS** — citations point to the indexed CT-AI PDF with relevant pages.

---

## French response from English source

- **Question (FR):** Quelle est la différence entre narrow AI et general AI ?
- **Retrieved English pages:** 15, 15, 12, 15
- **Answer language:** French
- **Terminology:** keeps **narrow AI** / **general AI** (with French glosses)
- **Citations:** CT-AI 2.0 English PDF, page 15 / 12

**Result: PASS** (French Q → English retrieval → French answer → English source citation)

Similar mild expansion risk on general AI as in E2E1 — recorded, not tuned.

---

## Trainer Mode

Topic: overfitting (CT-AI)

1. Explanation produced and source-cited (pages including 61 / 54).
2. Exactly **one** comprehension question: *What are some performance metrics that can indicate overfitting…?*
3. Learner answer (intentionally incomplete / off-focus): acknowledged training-data performance but did not answer the metrics question.
4. Evaluator **retrieved again** (eval pages: 61, 60, 74, 6).
5. Verdict: **❌ Incorrect** (not Partially correct — reasonable because the comprehension question was unanswered).
6. Feedback explained what was missing (metrics / learning curves) with a corrected explanation.

**Result: PASS** for the trainer workflow (explain → one Q → wait → evaluate against retrieved evidence).

Note: a “partially correct” verdict was not produced on this particular learner answer because the answer missed the asked question; the classification path itself works.

---

## Hallucination Control

**Question:** According to the ISTQB CT-AI syllabus, what is the official CTFL pesticide paradox definition?

**Retrieval evidence (weak / unrelated):**

| Rank | Score | Page | Content |
|---|---|---|---|
| 1 | 0.4136 | 91 | Release notes |
| 2 | 0.4043 | 58 | Adversarial testing |
| 3 | 0.3802 | 43 | Unrelated |
| 4 | 0.3422 | 76 | Glossary noise |

All scores > `MIN_RELEVANCE_SCORE` (0.25) → pipeline set `found=True` and still called the LLM.

**Generation:**
- (A) Correctly stated the official pesticide-paradox definition is **not found** in indexed CT-AI resources.
- (B) Then provided a **general** pesticide-paradox explanation **not grounded** in retrieved CT-AI chunks, without a strong “not from indexed ISTQB material” barrier matching V1 preference to simply decline.

**Result: FAIL** — partial refusal in (A), but (B) still invents content; retrieval threshold alone does not block the call.

---

## Problems observed

1. ~~Missing API key~~ — resolved; LLM steps completed.
2. **EN4 ranking imperfect:** best DNN structure chunk (page 36) at Top-3 — unchanged observation.
3. **Chapter/section heuristics incomplete / noisy** (occasional `0.x` TOC artifacts).
4. **PDF extraction spacing artifacts** (broken words).
5. **Out-of-scope queries still retrieve Top-4 above 0.25** — min-score gate insufficient for refusal.
6. **Hallucination control incomplete:** model refuses “official” claim then still narrates general knowledge in part (B).
7. **E2E1 grounding flag:** “weak AI” synonym not in retrieved evidence.
8. HF unauthenticated download warning during embedding model load.

No parameter changes were made in response to these observations.

---

## pytest regression

```text
15 passed
```

(Re-run after LLM validation phase — still 15/15. No new tests added.)

---

## Baseline conclusion

Retrieval baseline remains usable (EN 4/5, FR→EN 2/2). With the LLM connected:

- End-to-end answers and citations work for in-syllabus CT-AI questions.
- French response-from-English-source flow works.
- Trainer explain → one comprehension Q → evaluate-with-retrieval works.
- Hallucination control is **not fully reliable** yet when weak chunks still pass the min-score gate.

**Do not change chunk size, overlap, top-k, or embedding model yet.**  
Decide next experiments only after reviewing this updated report.

---

## Grounding bug-fix validation

Date: 2026-09-14 (after API-key E2E baseline)

**No changes** to embedding model, chunk size, overlap, top-k, ChromaDB, or architecture.

### Original defects (preserved)

1. Out-of-scope questions could retrieve chunks above `MIN_RELEVANCE_SCORE=0.25` and set `found=True`, then call the LLM.
2. After a soft refusal, the LLM could still append general knowledge (pesticide paradox part B).
3. E2E1 answers included “weak AI”; initially flagged as unsupported because truncated debug context omitted Top-4.

### Score-gate investigation (baseline scores, raw `min_score=0`)

| query | relevant? | top-1 | top-2 | top-3 | top-4 |
|---|---|---:|---:|---:|---:|
| EN1 narrow vs general AI | yes | 0.7577 | 0.6673 | 0.6139 | 0.5727 |
| EN2 overfitting | yes | 0.6302 | 0.6043 | 0.6019 | 0.5725 |
| EN3 precision/recall/F1 | yes | 0.6261 | 0.5905 | 0.5631 | 0.5583 |
| EN4 DNN layers | yes | 0.6509 | 0.6469 | 0.5817 | 0.5695 |
| EN5 back-to-back testing | yes | 0.7998 | 0.7361 | 0.6756 | 0.6752 |
| FR1 narrow vs general | yes | 0.7539 | 0.6813 | 0.6241 | 0.5882 |
| FR2 overfitting | yes | 0.6082 | 0.5854 | 0.5851 | 0.5773 |
| OOS CTFL pesticide paradox via CT-AI | no | 0.4136 | 0.4043 | 0.3802 | 0.3422 |

**Conclusion:** `0.25` does **not** separate relevant from irrelevant. Smallest evidence-based floor that keeps all in-syllabus Top-4 (≥0.56) and drops OOS (≤0.41):

**`MIN_RELEVANCE_SCORE` changed `0.25` → `0.50`** (documented in `src/config.py`).

### Code-level fixes

1. **`src/rag/grounding.py`** — central `has_sufficient_evidence()` + `not_found_message()`.
2. **Ask / Trainer / Evaluator / Quiz** — if evidence insufficient, return deterministic EN/FR not-found text and **do not call the LLM**.
3. **Stricter prompts** (`STRICT_CONTEXT_ONLY_RULE`, anti-synonym / no invented numbers / no fuller-than-context definitions).
4. **Ask generation temperature `0.0`** for more deterministic grounding.
5. **Regression tests** in `tests/test_grounding.py` (8 new tests; suite now 23).

### Clarification on defect #3 (“weak AI”)

Re-inspection after bug-fix: Top-4 **does** include a page-15 chunk containing the verbatim syllabus phrase:

> “Narrow AI, also known as weak AI…”

So “weak AI” can be **supported by retrieved evidence** when that chunk is present. The original flag was partly caused by truncated context dumps. Remaining E2E1 risk is **invented general-AI elaborations** (e.g. human-intelligence definitions) that are **not** in the retrieved text.

### Hallucination-control — before / after

| | Before | After |
|---|---|---|
| OOS gated chunk count | 4 (scores 0.34–0.41) | **0** |
| `found` | True | **False** |
| LLM called? | Yes | **No** |
| Answer | Soft refuse + invented general pesticide paradox | Deterministic: `I cannot find this information in the currently indexed ISTQB resources.` |

### E2E1 — before / after (narrow vs general AI)

| | Before | After |
|---|---|---|
| “weak AI” | Present (and actually available in Top-4 text) | Prompt discourages invented synonyms; phrase may still appear when copied from context |
| LLM on OOS path | Could invent after refuse | N/A for OOS (no LLM) |
| Residual issue | Human-level / “any intellectual task” AGI wording | **Still observed** on some EN/FR runs despite stricter prompts |

### Post-fix validation snapshot

- **E2E2 precision/recall/F1:** grounded formulas — PASS claims check
- **E2E3 back-to-back:** grounded pseudo-oracle — PASS claims check
- **E2E1 / FR narrow vs general:** residual unsupported AGI embellishment and occasional false “not mentioned” — **FAIL** under zero-unsupported rule
- **Trainer workflow:** still PASS (explain → one Q → evaluate with re-retrieval)
- **pytest:** 23 passed

### Bug-fix conclusion

Defects **#1** and **#2** are fixed by the evidence gate + deterministic refusal.  
Defect **#3** is partially clarified (weak AI is in syllabus Top-4) but **strict end-to-end grounding is not yet a full PASS** because the LLM can still add unsupported general-AI elaborations on the narrow/general contrast question.

**Still do not change chunk size / embedding model / top-k** based on this alone; next experiment (if any) should target generation grounding or chunk-boundary coverage for §1.1.2, not blind retrieval tuning.

---

## Generation grounding guardrail

Date: 2026-09-14 (after retrieval / OOS gate was already PASS)

This section is **generation grounding**, distinct from **retrieval grounding**.

| Layer | Question answered | Mechanism |
|---|---|---|
| Retrieval grounding | Is there enough indexed evidence to answer? | `has_sufficient_evidence` + score floor `0.50` → else deterministic refusal, **no LLM** |
| Generation grounding | Is the LLM answer supported by those chunks only? | Structured draft → validate → one repair → deterministic strip / fallback |

### Root cause (preserved failure)

Even with sufficient Top-4 evidence, `gpt-4o-mini` still embellished certification answers with plausible general-AI knowledge absent from the retrieved ISTQB text.

**Original E2E1 EN failure (before this guardrail):**

- Draft included: *“similar to human intelligence”* / broad AGI elaborations.
- An LLM-as-judge alone sometimes marked this as grounded (generally true ≠ supported).
- Strict end-to-end grounding remained **FAIL**.

### Implementation (no retrieval tuning)

Unchanged: embedding model, ChromaDB, chunk size/overlap, top-k, metadata filter, `MIN_RELEVANCE_SCORE`.

Added in `src/rag/grounding.py`:

1. **Generation settings:** model `gpt-4o-mini`, `temperature = 0.0`, `max_tokens = 1400`, `max_repair_attempts = 1`.
2. **Structured draft contract** (internal JSON): `answer` + `evidence[{claim, source_chunk_id}]` (user sees prose only).
3. **Validator:** programmatic checks (chunk IDs, EN lexical overlap, known embellishment markers) + LLM support judge that must not accept “true but absent”.
4. **Exactly one repair**, then optional **deterministic sentence strip** (EN) keeping only chunk-supported sentences.
5. **Safe fallback** EN/FR if still ungrounded — never return the unsafe draft.
6. Wired into Ask, Trainer explain, learner feedback, Quiz explanations.
7. RAG Debug panel: Grounding PASS/FAIL, Repair YES/NO, Unsupported claims count.

### E2E1 — first draft → repair → final

| | E2E1 EN | E2E1 FR |
|---|---|---|
| Draft | Contained unsupported AGI embellishment (*human intelligence* / *reconnaissance d'images* / generalize-as-AGI definition) | Same class of embellishment |
| Grounding (draft) | FAIL | FAIL |
| Repair attempted | YES | YES |
| Final | Narrow/weak AI limited-domain claims only (supported); unsafe AGI lines removed | Supported narrow/weak AI claims only |
| Final grounding | **PASS** | **PASS** |

### Regression tests

- Prior suite retained.
- Added generation-grounding tests A–F (+ lexical + deterministic-strip): mocked LLM only.
- **pytest: 32 passed / 0 failed** (after this section).

### Validation snapshot (post-guardrail)

- Retrieval / OOS gate: still PASS (OOS bypasses generation).
- E2E2 / E2E3: PASS without repair on this run.
- French→English→French: PASS (French answer, English CT-AI citations).
- Trainer Mode: PASS (explain grounded; repair/strip as needed).
- Hallucination / OOS: deterministic not-found, no LLM.

Repairs triggered on the full validation script run: **4** (E2E1 EN, E2E1 FR, FR→EN→FR detail, Trainer).

### Limitations (genuine)

- Repair may **over-delete** supported contrast sentences (e.g. “general AI has not yet been achieved”), yielding a shorter but safer answer.
- EN lexical overlap is approximate; FR relies more on markers + LLM judge (chunks are English).
- Embellishment markers are a finite denylist — novel unsupported phrases can still require the LLM judge.
- Temperature 0 reduces but does not eliminate repair/JSON flakiness; deterministic strip is the safety net.
- Comprehension questions are not themselves re-validated as certification claims (by design); explanations/feedback are.

**Strict end-to-end grounding for the validation set: PASS** under the zero-unsupported final-answer rule, with unsafe drafts never shown to the user.
