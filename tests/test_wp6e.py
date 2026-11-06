"""WP6e evaluation helpers (meld.eval.wp6e): card burden criterion, single-target outcomes, cost summary."""
from __future__ import annotations

from types import SimpleNamespace as NS

from meld.eval import wp6e as W


def _p(s, mergeable, bin=0, blocked=False):
    return {"bin": bin, "s_q": s, "mergeable": mergeable, "blocked": blocked}


def test_macro_f1_is_the_mean_of_both_class_f1s():
    assert W.macro_f1(1, 0, 0, 1) == 1.0
    # merge class: p = 1/2, r = 1 -> 2/3; non-merge class: tp' = 1, fp' = 0, fn' = 1 -> 2/3
    assert abs(W.macro_f1(1, 1, 0, 1) - 2 / 3) < 1e-12
    assert W.macro_f1(0, 0, 0, 3) == 0.5          # an empty class scores 0, the other 1


def test_replay_counts_merges_strictly_above_theta_never_blocked_or_disabled():
    pairs = [_p(9, True), _p(5, True), _p(9, False, blocked=True), _p(7, False), _p(9, True, bin=1)]
    # bin 0 theta 6: merges s 9 (mergeable) and 7 (non-mergeable); 5 not; blocked never; bin 1 disabled
    assert W.replay_counts(pairs, {0: 6}) == (1, 1, 2, 1)


def test_oracle_finds_the_per_bin_thresholds_that_maximise_macro_f1():
    pairs = [_p(9, True), _p(8, True), _p(3, False), _p(2, False),          # bin 0: separable
             _p(9, False, bin=1), _p(4, True, bin=1), _p(1, False, bin=1)]  # bin 1: not separable
    th, counts, f1 = W.oracle(pairs)
    assert counts == W.replay_counts(pairs, th) and abs(f1 - W.macro_f1(*counts)) < 1e-12
    # brute force over every per-bin cut
    cuts = {g: [-1] + sorted({p["s_q"] for p in pairs if p["bin"] == g}) for g in (0, 1)}
    best = max(W.macro_f1(*W.replay_counts(pairs, {0: a, 1: b})) for a in cuts[0] for b in cuts[1])
    assert abs(f1 - best) < 1e-12


def test_oracle_respects_a_false_merge_cap():
    pairs = [_p(9, True), _p(8, False), _p(7, True), _p(1, False)]
    th, (tp, fp, fn, tn), _ = W.oracle(pairs, max_fm=0.0)
    assert fp == 0 and tp == 1                    # only the cut above 8 keeps FM at 0


def test_card_mstar_is_the_smallest_m_with_enough_adequate_draws():
    # oracle macro-F1 0.80, FM 0.02: adequate iff f1 >= 0.78 and fm <= 0.03
    draws = {10: [(0.70, 0.0)] * 10,
             50: [(0.79, 0.01)] * 8 + [(0.79, 0.05)] * 2,     # 80 % adequate
             100: [(0.79, 0.01)] * 9 + [(0.5, 0.0)],          # 90 %
             200: [(0.80, 0.02)] * 10}
    out = W.card_mstar(draws, oracle_f1=0.80, oracle_fm=0.02)
    assert out["m_star"] == 100
    assert out["share"] == {10: 0.0, 50: 0.8, 100: 0.9, 200: 1.0}
    assert W.card_mstar({10: [(0.1, 0.0)]}, oracle_f1=0.8, oracle_fm=0.0)["m_star"] is None


def _d(b, decision, K=(), tested=None, links=()):
    return NS(b=b, decision=decision, K=tuple(K), tested=tested, links=tuple(links))


def test_target_outcomes_classify_each_labelled_pair_by_where_the_target_was():
    ds = [_d("x", "INSERT"),
          _d("y", "MERGE", K=("x",), tested=("x", "y")),                 # merged with its target
          _d("z", "RELATE", K=("y", "x"), tested=("y", "z")),            # target x in K, not tested
          _d("w", "RELATE", K=("y",), tested=("y", "w")),                # target x outside K
          _d("v", "MERGE", K=("y",), tested=("y", "v")),                 # target x outside K, merged via y's group
          _d("u", "CONFLICT", K=("x",), tested=("x", "u"), links=(("CONTRADICTS", "u", "x", 1.0),)),
          _d("t", "RELATE", K=("x", "y"), tested=("x", "t"))]
    pairs = [("x", "y"), ("z", "x"), ("x", "w"), ("x", "v"), ("x", "t")]
    out = W.target_outcomes(ds, pairs)
    assert out == [{"merged": True, "where": "tested"}, {"merged": False, "where": "in_K"},
                   {"merged": False, "where": "outside_K"}, {"merged": True, "where": "outside_K"},
                   {"merged": False, "where": "tested"}]
    assert W.linked(ds, [("x", "u"), ("x", "t")]) == [True, False]


def test_summary_ms_gives_median_and_p95():
    s = W.summary_ms(list(range(1, 101)))
    assert s["n"] == 100 and s["median"] == 50.5 and abs(s["p95"] - 95.05) < 1e-9
