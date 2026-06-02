"""M-2: merge-classification quality on a labeled claim-pair gold set.

The four-way decision procedure (kappa/sigma/chi + Context gate -> MERGE / RELATE /
CONFLICT / REJECT) is the paper's Contribution 2. M-2 measures it: per-class
precision/recall/F1 and the load-bearing FALSE-MERGE RATE (the dangerous error a
merge protocol must avoid, Section 4.3) -- the fraction of pairs that should NOT
merge (RELATE/CONFLICT/REJECT gold) that the procedure nonetheless classified MERGE.

The gold set is a small, deliberately CLEAR set of same-frame claim pairs spanning
the four outcomes (the full M-2 set should expand it). All pairs have distinct
content-addressed claim keys (kappa=0) so the SIGMA/CHI path is what is tested;
sigma uses all-MiniLM-L6-v2, chi uses the cross-encoder NLI (lexical chi cannot
adjudicate contradictions reliably). Reproduced by tests/test_m2_eval.py.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from smp.config import Thresholds
from smp.merge.classifier import classify
from smp.merge.signals import EmbeddingSimilarity, chi

DECISIONS = ("MERGE", "RELATE", "CONFLICT", "REJECT")

# (gold_decision, claim_a, claim_b) -- same frame (ctx_ok=True), distinct keys.
M2_GOLD: List[Tuple[str, str, str]] = [
    # MERGE: genuine paraphrases (same claim, reworded)
    ("MERGE", "Paris is the capital of France", "Paris is the capital city of France"),
    ("MERGE", "Mount Everest is the tallest mountain on Earth",
     "The tallest mountain on Earth is Mount Everest"),
    ("MERGE", "The novel 1984 was written by George Orwell",
     "George Orwell wrote the novel 1984"),
    ("MERGE", "Water boils at 100 degrees Celsius at sea level",
     "At sea level, water boils at 100 degrees Celsius"),
    # RELATE: related but distinct (overlap, not identity), non-contradicting
    ("RELATE", "Paris is the capital of France",
     "France is a country in western Europe"),
    ("RELATE", "Mount Everest is the tallest mountain",
     "Mount Everest is located in the Himalayas"),
    ("RELATE", "George Orwell wrote the book 1984",
     "The book 1984 is a dystopian novel"),
    ("RELATE", "The Eiffel Tower is in Paris",
     "Paris attracts millions of tourists each year"),
    # CONFLICT: same subject, contradictory value
    ("CONFLICT", "Paris is the capital of France", "Paris is the capital of Italy"),
    ("CONFLICT", "Mount Everest is the tallest mountain on Earth",
     "K2 is the tallest mountain on Earth"),
    ("CONFLICT", "The novel 1984 was written by George Orwell",
     "The novel 1984 was written by Aldous Huxley"),
    ("CONFLICT", "Water boils at 100 degrees Celsius at sea level",
     "Water boils at 90 degrees Celsius at sea level"),
    # REJECT: unrelated (no actionable relation)
    ("REJECT", "Paris is the capital of France",
     "The cat slept on the warm windowsill"),
    ("REJECT", "Mount Everest is the tallest mountain",
     "I enjoy a strong cup of coffee in the morning"),
    ("REJECT", "George Orwell wrote the book 1984",
     "Photosynthesis converts sunlight into chemical energy"),
    ("REJECT", "Water boils at 100 degrees Celsius",
     "The stock market closed higher today"),
]


def classification_metrics(gold: List[str], pred: List[str]) -> Dict:
    """Per-class precision/recall/F1 + macro-F1 + false-merge rate + confusion.

    false_merge_rate = fraction of NON-MERGE-gold pairs predicted MERGE (the
    safety error). Pure function over (gold, pred) label lists."""
    assert len(gold) == len(pred)
    per_class: Dict[str, Dict[str, float]] = {}
    for c in DECISIONS:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": sum(1 for g in gold if g == c)}
    non_merge = [(g, p) for g, p in zip(gold, pred) if g != "MERGE"]
    false_merges = sum(1 for g, p in non_merge if p == "MERGE")
    fmr = false_merges / len(non_merge) if non_merge else 0.0
    macro_f1 = sum(per_class[c]["f1"] for c in DECISIONS) / len(DECISIONS)
    confusion = {g: {p: 0 for p in DECISIONS} for g in DECISIONS}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1
    return {"per_class": per_class, "macro_f1": macro_f1,
            "false_merge_rate": fmr, "n_false_merges": false_merges,
            "confusion": confusion, "n": len(gold)}


def evaluate_m2(pairs, *, encoder: str, nli: str, cfg: Thresholds) -> Dict:
    """Run the four-way classifier on labeled pairs (ctx_ok=True, kappa=0) and
    return classification_metrics. sigma re-embeds each text with ``encoder``."""
    es = EmbeddingSimilarity(encoder=encoder)
    gold, pred = [], []
    for gold_dec, a, b in pairs:
        sigma = es.sigma(a, b)
        chi_val = chi(a, b, model=nli)
        dec = classify(cfg=cfg, kappa=0, sigma=sigma, chi=chi_val, ctx_ok=True)
        gold.append(gold_dec)
        pred.append(dec)
    m = classification_metrics(gold, pred)
    m["predictions"] = list(zip([g for g, _, _ in pairs], pred))
    return m


__all__ = ["M2_GOLD", "DECISIONS", "classification_metrics", "evaluate_m2"]
