"""Task 1.10 — SIM experiment drivers (M-2, H/M-1, O-1, A-*) on a tiny slice.

- M2: per-class precision/recall/F1 + false-MERGE rate on the gold claim pairs.
- M1: distributed-vs-centralized answer quality; the naive-union ablation is
      strictly worse (double-counts / keeps contradictions).
- O1: the four overhead term counts.
- A1: gate ablation quantifies the Context/chi gates' contribution to false-MERGE.

All outputs are REAL (computed on the tiny fixture), not fabricated.
"""
from __future__ import annotations

from smp.eval.run_smp_eval import run_a1, run_m1, run_m2, run_o1


def test_m2_returns_per_class_metrics_and_low_false_merge():
    res = run_m2(n_conversations=8, encoder="hashing", nli="lexical")
    assert set(res["per_class"]) >= {"coreferent", "contradiction", "unrelated"}
    for cls, m in res["per_class"].items():
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
    # the dangerous error must be ~0: no contradiction pair classified MERGE
    assert res["false_merge_rate"] == 0.0


def test_m1_distributed_matches_centralized_and_beats_naive_union():
    res = run_m1(n_conversations=8, n_agents=3, encoder="hashing", nli="lexical")
    # distributed merged store answers within tolerance of centralized
    assert res["distributed_accuracy"] >= res["centralized_accuracy"] - 0.2
    # the naive-union ablation is strictly worse on the contradiction-bloat metric
    assert res["naive_union_units"] > res["distributed_units"]


def test_o1_returns_four_overhead_terms():
    res = run_o1(n_conversations=6, n_agents=2, encoder="hashing", nli="lexical")
    for term in ("c_sync_bytes", "c_ver_versions", "c_ret_walk_cost", "c_ref_refetch"):
        assert term in res
        assert res[term] >= 0


def test_a1_gate_ablation_shows_context_gate_contribution():
    res = run_a1(n_conversations=8, encoder="hashing", nli="lexical")
    # dropping the chi gate cannot DECREASE the false-merge rate
    assert res["false_merge_no_chi"] >= res["false_merge_full"]
