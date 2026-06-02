"""M-1 DELTA metrics: paired per-question EM/F1 and bootstrap CIs on DIFFERENCES.

The headline is NOT an absolute agent F1; it is a DELTA between memory conditions
measured on the SAME held-out questions:

  * ``F1(distributed_merge) - F1(naive_union)``  -- does the merge protocol beat
    dumping the deltas (the load-bearing SMP claim);
  * ``F1(distributed_merge) - F1(centralized)``  -- is the merge LOSSLESS relative
    to a single store (should be ~0, a non-inferiority read);
  * ``F1(full_context) - F1(none)``              -- the measurable HEADROOM (the
    "gate" reframed as a competence FLOOR: a positive headroom means a delta is
    measurable AT ALL on this slice/model).

Because the conditions are run on the SAME questions, the delta is PAIRED: we
bootstrap by resampling QUESTIONS (rows), recomputing each condition's mean on the
resampled rows, and taking the difference -- a paired bootstrap that respects the
per-question correlation between conditions (tighter, correct CI vs an unpaired
two-sample bootstrap).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from smp.agent.hotpot_scorer import em_score, f1_score


@dataclass
class DeltaCI:
    a: str                     # condition A name
    b: str                     # condition B name (delta = mean_A - mean_B)
    metric: str                # "f1" | "em"
    point: float               # observed mean_A - mean_B
    lo: float                  # CI lower
    hi: float                  # CI upper
    mean_a: float
    mean_b: float
    n: int

    def as_dict(self) -> Dict:
        return {"a": self.a, "b": self.b, "metric": self.metric,
                "delta": self.point, "ci_lo": self.lo, "ci_hi": self.hi,
                "mean_a": self.mean_a, "mean_b": self.mean_b, "n": self.n}


def score_one(prediction: Optional[str], gold: str) -> Dict[str, float]:
    return {"em": em_score(prediction or "", gold),
            "f1": f1_score(prediction or "", gold)}


def _row_metric(r: Dict, metric: str) -> Optional[float]:
    """The metric for one row, or None if the row cannot supply it. A precomputed
    value is used as-is; em/f1 are recomputed from (prediction, gold) when absent;
    any OTHER metric absent from the row (e.g. recall_at_k on a row written by an
    older harness) returns None -> the caller SKIPS it rather than fabricating or
    crashing (the CSC resume-mixing KeyError)."""
    if metric in r:
        return float(r[metric])
    if metric in ("em", "f1"):
        return score_one(r.get("prediction"), r["gold"])[metric]
    return None


def per_question_scores(rows: Sequence[Dict], condition: str,
                        metric: str = "f1") -> List[float]:
    """Extract the per-question metric for ``condition`` from scored rows.

    ``rows``: list of {question_id, condition, prediction, gold} (or rows carrying
    a precomputed 'em'/'f1'/'recall_at_k'/...). Rows that cannot supply ``metric``
    are skipped. Returns the metric in stable row order.
    """
    out: List[float] = []
    for r in rows:
        if r.get("condition") != condition:
            continue
        v = _row_metric(r, metric)
        if v is not None:
            out.append(v)
    return out


def _aligned(rows: Sequence[Dict], a: str, b: str,
             metric: str) -> Tuple[List[float], List[float]]:
    """Return per-question (score_a, score_b) aligned by question_id, keeping only
    questions present in BOTH conditions (the paired set)."""
    by_q_a: Dict[str, float] = {}
    by_q_b: Dict[str, float] = {}
    for r in rows:
        cond = r.get("condition")
        if cond not in (a, b):
            continue
        v = _row_metric(r, metric)
        if v is None:
            continue
        (by_q_a if cond == a else by_q_b)[r["question_id"]] = v
    qs = [q for q in by_q_a if q in by_q_b]
    qs.sort()
    return [by_q_a[q] for q in qs], [by_q_b[q] for q in qs]


def _mean(xs: Sequence[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def bootstrap_delta(rows: Sequence[Dict], a: str, b: str, *, metric: str = "f1",
                    n_boot: int = 2000, alpha: float = 0.05,
                    seed: int = 0) -> DeltaCI:
    """Paired bootstrap CI for ``mean(metric|a) - mean(metric|b)``.

    Resamples QUESTIONS with replacement; for each resample recomputes both
    condition means on the SAME resampled questions and takes the difference. The
    CI is the empirical [alpha/2, 1-alpha/2] percentile interval of the bootstrap
    delta distribution. Deterministic given ``seed`` (reproducible).
    """
    sa, sb = _aligned(rows, a, b, metric)
    n = len(sa)
    point = _mean(sa) - _mean(sb)
    if n == 0:
        return DeltaCI(a, b, metric, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    deltas: List[float] = []
    idx = range(n)
    for _ in range(n_boot):
        pick = [rng.choice(idx) for _ in range(n)]
        ma = _mean([sa[i] for i in pick])
        mb = _mean([sb[i] for i in pick])
        deltas.append(ma - mb)
    deltas.sort()
    lo = deltas[max(0, int((alpha / 2) * n_boot) - 1)]
    hi = deltas[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return DeltaCI(a, b, metric, point, lo, hi, _mean(sa), _mean(sb), n)


# the three headline deltas (per mode x model), reframing the gate as a FLOOR.
HEADLINE_DELTAS = [
    ("distributed_merge", "naive_union", "merge beats dumping"),
    ("distributed_merge", "centralized", "merge is lossless vs single store"),
    ("full_context", "none", "measurable headroom (competence FLOOR)"),
]


def headline_table(rows: Sequence[Dict], *, metric: str = "f1",
                   n_boot: int = 2000, seed: int = 0) -> List[Dict]:
    """Compute the three headline delta CIs from scored rows (one mode+model)."""
    table = []
    for a, b, label in HEADLINE_DELTAS:
        ci = bootstrap_delta(rows, a, b, metric=metric, n_boot=n_boot, seed=seed)
        d = ci.as_dict()
        d["label"] = label
        table.append(d)
    return table


def tost_equivalent(rows: Sequence[Dict], a: str, b: str, *, margin: float,
                    metric: str = "f1", n_boot: int = 2000, alpha: float = 0.05,
                    seed: int = 0) -> Dict:
    """Two one-sided tests (TOST) for EQUIVALENCE / non-inferiority within +/-margin.

    A null-hypothesis CI straddling 0 does NOT show equivalence (it is "fail to
    reject"); TOST does. We declare equivalence iff the (1-2*alpha) CI of the paired
    delta (the 90% CI for alpha=0.05) lies entirely within [-margin, +margin]. The
    margin must be PRE-REGISTERED, not chosen after seeing the data.
    """
    ci = bootstrap_delta(rows, a, b, metric=metric, n_boot=n_boot,
                         alpha=2 * alpha, seed=seed)
    equivalent = (ci.n > 0) and (ci.lo > -margin) and (ci.hi < margin)
    return {"equivalent": equivalent, "margin": margin, "delta": ci.point,
            "ci_lo": ci.lo, "ci_hi": ci.hi, "n": ci.n, "alpha": alpha}


def holm_correct(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm-Bonferroni step-down: control the family-wise error rate across the
    simultaneously-tested deltas. Returns label -> survives-correction. Once a test
    fails its threshold alpha/(m-i), it and all larger p-values are rejected."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, bool] = {}
    still_significant = True
    for i, (label, p) in enumerate(items):
        thresh = alpha / (m - i)
        if still_significant and p <= thresh:
            out[label] = True
        else:
            out[label] = False
            still_significant = False
    return out


def condition_means(rows: Sequence[Dict], conditions: Sequence[str],
                    metric: str = "f1") -> Dict[str, Dict[str, float]]:
    """Per-condition mean + n for a quick floor/ceiling readout."""
    out: Dict[str, Dict[str, float]] = {}
    for c in conditions:
        xs = per_question_scores(rows, c, metric=metric)
        out[c] = {"mean": _mean(xs), "n": len(xs)}
    return out


__all__ = ["DeltaCI", "bootstrap_delta", "headline_table", "condition_means",
           "tost_equivalent", "holm_correct",
           "per_question_scores", "score_one", "HEADLINE_DELTAS"]
