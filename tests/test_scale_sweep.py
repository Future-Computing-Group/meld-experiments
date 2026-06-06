"""S-3: node-scaling of sync overhead (TDD).

Measures how delivery overhead scales as the cluster grows (N brains), the axis the
manuscript currently argues structurally ("the routing advantage widens with N") and only
sweeps to N=4. Per published claim, broadcast delivers to all N brains (O(N)); semantic
routing delivers only to the brains that need it (O(needed), flat in N) at maintained recall.
This pins the pure aggregation (no encoder) and the workload construction; the real-encoder
routing run is the smoke step.

Run:  PYTHONPATH=src python3 -m pytest tests/test_scale_sweep.py -q
"""
from smp.eval.scale_sweep import make_brains, make_workload, summarize_scale


def test_make_brains_are_distinct_and_sized():
    b = make_brains(8)
    assert len(b) == 8
    assert len(set(b)) == 8                       # distinct ids
    assert all(isinstance(v, list) and v for v in b.values())  # non-empty interests


def test_make_workload_each_claim_needed_by_one_real_brain():
    brains = make_brains(4)
    wl = make_workload(4, per_domain=2)
    assert len(wl) == 8                           # 4 domains x 2 claims
    needed = [nb for _, nb in wl]
    assert all(nb in brains for nb in needed)     # gold brain exists
    # each brain is the needed target for exactly per_domain claims
    from collections import Counter
    assert set(Counter(needed).values()) == {2}


def test_summarize_scale_overhead_ratio_grows_with_n():
    # n=2 cluster, 4 claims; semantic routes top-1 perfectly, broadcast hits both
    needed = [{"b0"}, {"b0"}, {"b1"}, {"b1"}]
    sem = [{"b0"}, {"b0"}, {"b1"}, {"b1"}]
    bc = [{"b0", "b1"}] * 4
    r = summarize_scale(2, sem, bc, needed)
    assert r["n_brains"] == 2 and r["n_claims"] == 4
    assert r["recall_semantic"] == 1.0
    assert r["deliv_per_claim_semantic"] == 1.0   # flat (needed-count)
    assert r["deliv_per_claim_broadcast"] == 2.0  # == N
    assert r["waste_semantic"] == 0.0
    assert abs(r["waste_broadcast"] - 0.5) < 1e-9
    assert r["broadcast_over_semantic"] == 2.0     # grows with N
