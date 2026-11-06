"""E2: perturbation sweeps that exercise the adaptive mechanisms.

These tests sweep the partition-duration / reorder-rate axes and stress the
merge decision procedure under noisy embeddings or degraded NLI, instead of
evaluating only on clean inputs. They specify the two sweeps:

  E2(a) partition_reorder_sweep -- CRDT vs LWW reconvergence over
        (partition-duration x reorder-rate). The CRDT must stay order-independent
        (modal agreement 1.0) in EVERY cell; LWW degrades with reordering.
  E2(b) robustness_sweep -- merge-procedure macro-F1 / false-merge-rate under
        injected embedding noise (perturbed sigma) and degraded NLI (flipped chi).

Encoder-free: E2(b) uses synthetic scored rows so the math is testable without
a model; the real dense-rung numbers come from the runner.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# E2(a): partition-duration x reorder-rate sweep
# --------------------------------------------------------------------------- #
def test_partial_shuffle_endpoints():
    import random
    from meld.eval.crdt_convergence import _partial_shuffle
    order = list(range(20))
    rng = random.Random(0)
    assert _partial_shuffle(order, 0.0, rng) == order            # identity
    out = _partial_shuffle(order, 1.0, rng)
    assert sorted(out) == order and len(out) == len(order)        # a permutation


def test_crdt_modal_agreement_is_one_in_every_cell():
    from meld.eval.crdt_convergence import partition_reorder_sweep
    cells = partition_reorder_sweep(durations=[2, 4, 8], reorder_rates=[0.0, 0.5, 1.0],
                                    n_claims=4, n_orderings=40, seed=0)
    assert len(cells) == 9
    for c in cells:
        assert c["crdt_modal_agreement"] == 1.0, f"CRDT not order-independent: {c}"
        assert c["crdt_n_distinct"] == 1


def test_lww_degrades_with_reorder_and_duration():
    from meld.eval.crdt_convergence import partition_reorder_sweep
    cells = partition_reorder_sweep(durations=[8], reorder_rates=[0.0, 1.0],
                                    n_claims=6, n_orderings=60, seed=1)
    by_r = {c["reorder_rate"]: c for c in cells}
    # in-order (reorder 0): every ordering identical -> both converge
    assert by_r[0.0]["lww_modal_agreement"] == 1.0
    # full reorder: LWW must lose order-independence (and stay <= CRDT)
    assert by_r[1.0]["lww_modal_agreement"] < 1.0
    assert by_r[1.0]["lww_modal_agreement"] <= by_r[1.0]["crdt_modal_agreement"]


def test_sweep_is_deterministic():
    from meld.eval.crdt_convergence import partition_reorder_sweep
    a = partition_reorder_sweep(durations=[4], reorder_rates=[1.0], n_claims=4,
                                n_orderings=30, seed=7)
    b = partition_reorder_sweep(durations=[4], reorder_rates=[1.0], n_claims=4,
                                n_orderings=30, seed=7)
    assert a == b


# --------------------------------------------------------------------------- #
# E2(b): decision-procedure robustness to noisy sigma / degraded chi
# --------------------------------------------------------------------------- #
def _synthetic_scored():
    rows = []
    for i in range(25):
        rows.append(("MERGE", 0.92, "entails"))
    for i in range(25):
        rows.append(("RELATE", 0.65, "neutral"))
    for i in range(25):
        rows.append(("CONFLICT", 0.70, "contradicts"))
    for i in range(25):
        rows.append(("NO_RELATION", 0.10, "neutral"))
    return rows


