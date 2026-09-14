"""One-off corpus inspection for CT-AI baseline validation — not part of product."""
from __future__ import annotations

import re
from pypdf import PdfReader

r = PdfReader("resources/ct_ai/en/ISTQB-_CTAI_Syllabus_v2.0_Release.pdf")
keys = [
    "precision",
    "recall",
    "F1",
    "training data",
    "test data",
    "overfitting",
    "underfitting",
    "bias",
    "fairness",
    "neural network",
    "self-learning",
    "test oracle",
    "non-determin",
    "adversarial",
    "data poisoning",
    "concept drift",
    "explainab",
    "narrow AI",
    "general AI",
]
for k in keys:
    hits = []
    for i, p in enumerate(r.pages, 1):
        t = p.extract_text() or ""
        if re.search(k, t, re.I):
            hits.append(i)
    if hits:
        suffix = "..." if len(hits) > 8 else ""
        print(f"{k!r}: pages {hits[:8]}{suffix} (n={len(hits)})")

for pnum in [15, 24, 30, 36, 38, 53, 60, 62, 69]:
    t = r.pages[pnum - 1].extract_text() or ""
    print(f"\n==== PAGE {pnum} ====")
    print(t[:1000].replace("\n", " | "))
