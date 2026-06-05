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


# --------------------------------------------------------------------------- #
# E1: scoring, ROC over theta_merge, baselines, bootstrap CIs
# --------------------------------------------------------------------------- #
import random as _random
import re as _re

_WS = _re.compile(r"\s+")


def _kappa_textual(a: str, b: str) -> int:
    """Exact-dedup key match: 1 iff the two texts normalize identically.

    Content claim-keys are hashes of the normalized text; distinct texts get
    distinct keys (kappa=0). This is the key-only dedup baseline's whole signal.
    """
    na = _WS.sub(" ", a.strip().lower())
    nb = _WS.sub(" ", b.strip().lower())
    return int(na == nb)


def score_pairs(pairs, *, encoder: str, nli: str):
    """Encode once: return list of (gold_decision, sigma, chi) for the pairs."""
    es = EmbeddingSimilarity(encoder=encoder)
    return [(g, es.sigma(a, b), chi(a, b, model=nli)) for g, a, b in pairs]


def decisions_full(scored, *, theta: float, sigma_lo: float) -> List[str]:
    """Full four-way procedure decisions for pre-scored (gold, sigma, chi) rows
    (kappa=0, ctx_ok=True). r_min/tau are immaterial here (gate not exercised)."""
    cfg = Thresholds(theta_merge=theta, sigma_lo=sigma_lo, r_min=0.0, tau=0.0)
    return [classify(cfg=cfg, kappa=0, sigma=s, chi=c, ctx_ok=True)
            for _, s, c in scored]


def decisions_embedding_only(scored, *, theta: float) -> List[str]:
    """Embedding-only baseline: MERGE iff sigma>=theta, with NO contradiction
    (chi/NLI) guard -- so high-similarity contradictions leak through as merges."""
    return ["MERGE" if s >= theta else "REJECT" for _, s, _ in scored]


def roc_over_theta(scored, *, sigma_lo: float, n_points: int = 51):
    """ROC of the MERGE-vs-rest decision as theta_merge sweeps the open interval
    (sigma_lo, 1]. Uses the full MERGE rule (sigma>=theta AND chi!=contradicts).

    Returns (points, auc) where points = [(theta, fpr, tpr)] sorted by ascending
    theta. Sweeping the OPEN interval above sigma_lo fixes the m-E-1 crash
    (Thresholds forbids theta_merge<=sigma_lo, in particular theta_merge=0)."""
    gold = [g for g, _, _ in scored]
    n_pos = sum(1 for g in gold if g == "MERGE")
    n_neg = len(gold) - n_pos
    pts = []
    for i in range(n_points):
        theta = sigma_lo + (1.0 - sigma_lo) * (i + 1) / n_points  # in (sigma_lo, 1]
        tp = fp = 0
        for g, s, c in scored:
            is_merge = (s >= theta) and (c != "contradicts")
            if is_merge and g == "MERGE":
                tp += 1
            elif is_merge and g != "MERGE":
                fp += 1
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        pts.append((theta, fpr, tpr))
    # AUC via trapezoid over (fpr, tpr), anchored at (0,0) and (1,1).
    roc = sorted(set([(0.0, 0.0)] + [(f, t) for _, f, t in pts] + [(1.0, 1.0)]))
    auc = 0.0
    for (f0, t0), (f1, t1) in zip(roc, roc[1:]):
        auc += (f1 - f0) * (t0 + t1) / 2.0
    return pts, auc


def bootstrap_metric_ci(gold, pred, *, metric: str = "macro_f1",
                        n_boot: int = 2000, seed: int = 0):
    """Percentile [2.5, 97.5] bootstrap CI for a scalar metric of
    classification_metrics, resampling (gold, pred) pairs with replacement.
    Deterministic given seed. Returns (lo, hi)."""
    assert len(gold) == len(pred) and len(gold) > 0
    rng = _random.Random(seed)
    n = len(gold)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        p = [pred[i] for i in idx]
        vals.append(classification_metrics(g, p)[metric])
    vals.sort()
    lo = vals[max(0, int(0.025 * n_boot) - 1)]
    hi = vals[min(n_boot - 1, int(0.975 * n_boot))]
    return lo, hi


def evaluate_baseline(pairs, *, mode: str, encoder: str = "minilm",
                      nli: str = "mnli-cross-encoder", theta: float = 0.65,
                      sigma_lo: float = 0.05) -> Dict:
    """Baseline classifiers over labeled pairs.

    mode="key_only"      -> MERGE iff exact normalized-text key match (kappa);
                            needs no encoder. Distinct-key paraphrases never merge.
    mode="embedding_only"-> MERGE iff sigma>=theta, no NLI guard (encoder needed).
    mode="full"          -> the full four-way procedure (encoder + nli needed).
    """
    gold = [g for g, _, _ in pairs]
    if mode == "key_only":
        pred = ["MERGE" if _kappa_textual(a, b) else "REJECT"
                for _, a, b in pairs]
    elif mode == "embedding_only":
        scored = score_pairs(pairs, encoder=encoder, nli="lexical")
        pred = decisions_embedding_only(scored, theta=theta)
    elif mode == "full":
        scored = score_pairs(pairs, encoder=encoder, nli=nli)
        pred = decisions_full(scored, theta=theta, sigma_lo=sigma_lo)
    else:
        raise ValueError(f"unknown baseline mode {mode!r}")
    m = classification_metrics(gold, pred)
    m["predictions"] = list(zip(gold, pred))
    m["mode"] = mode
    return m


__all__ = ["M2_GOLD", "DECISIONS", "classification_metrics", "evaluate_m2",
           "score_pairs", "decisions_full", "decisions_embedding_only",
           "roc_over_theta", "bootstrap_metric_ci", "evaluate_baseline",
           "_kappa_textual"]
