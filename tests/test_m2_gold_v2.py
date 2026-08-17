"""E1: the enlarged, balanced M-2 gold set + ROC + baselines.

RED-first tests for the enlarged benchmark: a >=100-pair balanced gold set with
an ROC and CIs, replacing a smaller hand-built smoke set. This suite specifies
the enlarged set and the new evaluation surface; it must FAIL before m2_gold.py
/ the m2_eval additions exist, then pass.

These tests are encoder-free (structure + the pure metric/ROC math on synthetic
predictions); the real dense-rung numbers come from the smoke ladder + the
run_ablation_metrics driver, not from here.
"""
from __future__ import annotations

import pytest

from meld.merge.m2_eval import (
    DECISIONS,
    classification_metrics,
    roc_over_theta,
    bootstrap_metric_ci,
)


# --------------------------------------------------------------------------- #
# Gold set: size, balance, validity, mixed difficulty
# --------------------------------------------------------------------------- #
def test_gold_set_at_least_100_pairs():
    from meld.merge.m2_gold import M2_GOLD_V2
    assert len(M2_GOLD_V2) >= 100, f"need >=100 pairs, got {len(M2_GOLD_V2)}"


def test_gold_set_balanced_across_four_classes():
    from meld.merge.m2_gold import M2_GOLD_V2
    counts = {c: sum(1 for g, _, _ in M2_GOLD_V2 if g == c) for c in DECISIONS}
    assert set(counts) == set(DECISIONS)
    # balanced: every class within 20% of the mean class size
    mean = len(M2_GOLD_V2) / 4
    for c, n in counts.items():
        assert abs(n - mean) <= 0.2 * mean, f"class {c} unbalanced: {counts}"


def test_gold_labels_valid_and_pairs_unique():
    from meld.merge.m2_gold import M2_GOLD_V2
    for g, a, b in M2_GOLD_V2:
        assert g in DECISIONS
        assert isinstance(a, str) and isinstance(b, str) and a and b
        assert a != b, f"degenerate identical pair: {a!r}"
    pairs = [(a, b) for _, a, b in M2_GOLD_V2]
    assert len(set(pairs)) == len(pairs), "duplicate claim pairs in gold set"


def test_gold_set_has_easy_and_hard_strata():
    # The set must include HARD cases (e.g. lexically-divergent contradictions,
    # low-overlap paraphrases) -- the hardest case for a similarity gate --
    # not just clear ones.
    from meld.merge.m2_gold import build_gold_set
    rows = build_gold_set()
    diffs = {r["difficulty"] for r in rows}
    assert {"easy", "hard"} <= diffs, f"need easy+hard strata, got {diffs}"
    hard = [r for r in rows if r["difficulty"] == "hard"]
    assert sum(1 for r in hard if r["label"] == "CONFLICT") >= 5, \
        "need >=5 hard (lexically-divergent) CONFLICT cases to exercise the gate"


def test_tagged_and_flat_views_agree():
    from meld.merge.m2_gold import M2_GOLD_V2, build_gold_set
    rows = build_gold_set()
    assert len(rows) == len(M2_GOLD_V2)
    assert [(r["label"], r["a"], r["b"]) for r in rows] == list(M2_GOLD_V2)


# --------------------------------------------------------------------------- #
# ROC over theta_merge (fixes m-E-1: must not crash at the low end)
# --------------------------------------------------------------------------- #
def _synthetic_scores():
    # (gold, sigma, chi): a separable-ish synthetic set so ROC math is testable
    # without an encoder. MERGE pairs have high sigma; non-MERGE lower.
    rows = []
    for i in range(20):
        rows.append(("MERGE", 0.80 + 0.01 * (i % 5), "entails"))
    for i in range(20):
        rows.append(("RELATE", 0.50 + 0.01 * (i % 5), "neutral"))
    for i in range(20):
        rows.append(("CONFLICT", 0.70 + 0.01 * (i % 5), "contradicts"))
    for i in range(20):
        rows.append(("NO_RELATION", 0.10 + 0.01 * (i % 5), "neutral"))
    return rows


def test_roc_runs_without_crash_at_low_theta():
    # m-E-1 regression: the sweep previously crashed constructing
    # Thresholds(theta_merge=0.0). It must sweep the valid open interval.
    pts, auc = roc_over_theta(_synthetic_scores(), sigma_lo=0.05, n_points=21)
    assert len(pts) >= 10
    for theta, fpr, tpr in pts:
        assert 0.0 < theta <= 1.0
        assert 0.0 <= fpr <= 1.0 and 0.0 <= tpr <= 1.0


def test_roc_auc_in_unit_interval_and_separable_is_strong():
    pts, auc = roc_over_theta(_synthetic_scores(), sigma_lo=0.05, n_points=51)
    assert 0.0 <= auc <= 1.0
    # the synthetic set is separable on sigma for MERGE-vs-rest -> AUC well above chance
    assert auc >= 0.75, f"separable synthetic should score high, got {auc}"


def test_roc_fpr_monotone_nonincreasing_in_theta():
    pts, _ = roc_over_theta(_synthetic_scores(), sigma_lo=0.05, n_points=51)
    fprs = [fpr for _, fpr, _ in pts]  # pts sorted by increasing theta
    for lo, hi in zip(fprs, fprs[1:]):
        assert hi <= lo + 1e-9, "raising theta cannot increase MERGE false-positive rate"


# --------------------------------------------------------------------------- #
# Bootstrap CIs on the metrics (AE asked for CIs)
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_point_estimate():
    gold = ["MERGE", "MERGE", "RELATE", "CONFLICT", "NO_RELATION"] * 8
    pred = list(gold)
    pred[0] = "RELATE"  # one error
    lo, hi = bootstrap_metric_ci(gold, pred, metric="macro_f1",
                                 n_boot=500, seed=0)
    point = classification_metrics(gold, pred)["macro_f1"]
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0


# --------------------------------------------------------------------------- #
# Baselines: embedding-only (no NLI) and claim-key-only
# --------------------------------------------------------------------------- #
def test_keyonly_baseline_never_merges_distinct_key_paraphrases():
    # gold MERGE pairs are paraphrases with DISTINCT content keys (kappa=0);
    # a key-only dedup baseline must therefore miss all of them -> recall 0.
    from meld.merge.m2_eval import evaluate_baseline
    from meld.merge.m2_gold import M2_GOLD_V2
    m = evaluate_baseline(M2_GOLD_V2, mode="key_only")
    assert m["per_class"]["MERGE"]["recall"] == 0.0


def test_embedding_only_baseline_has_more_false_merges_than_full_on_conflicts():
    # Without the chi/NLI contradiction guard, high-sigma CONTRADICTIONS merge.
    # The embedding-only baseline must have a >= false-merge rate vs the full
    # procedure on the same (synthetic) scored set.
    from meld.merge.m2_eval import decisions_embedding_only, decisions_full
    rows = _synthetic_scores()
    gold = [g for g, _, _ in rows]
    full = classification_metrics(gold, decisions_full(rows, theta=0.65, sigma_lo=0.05))
    emb = classification_metrics(gold, decisions_embedding_only(rows, theta=0.65))
    assert emb["false_merge_rate"] >= full["false_merge_rate"]
    assert emb["false_merge_rate"] > 0.0  # the CONFLICT pairs leak through
