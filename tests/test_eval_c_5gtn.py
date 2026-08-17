"""Unit tests for the multi-host partition-heal convergence PURE core.

The over-the-wire orchestrator replays the status-CRDT partition-heal protocol over real
brains + a real, healable partition, but its decision logic is the scrubbed, host-agnostic
core in meld.sync.eval_c_partition (stdlib-only, no HTTP/tc/docker), so it is testable here:

  * seeded_delivery_orders  -- per-brain permuted (link, effect) delivery order; same seed
                               must give the same order (order-independence demo is seeded).
  * per_seed_result         -- (during-statuses, delivery order, effects) -> the three
                               metrics. CRDT join is order-independent (reconverges to 1.0);
                               the last-writer-wins value is an ANALYTICAL counterfactual
                               read from the SAME delivery order actually used.
  * aggregate_results       -- mean + seed-as-unit bootstrap CI + reconvergence fractions.
"""
from __future__ import annotations

from meld.sync import eval_c_partition as M


# --- seeded delivery order (determinism) -----------------------------------------------

def test_seeded_delivery_orders_are_deterministic():
    brains = ["A", "B", "C", "D"]
    links = [("La", "DEPRECATED"), ("Lb", "OVERRULED")]
    o1 = M.seeded_delivery_orders(brains, links, seed=7)
    o2 = M.seeded_delivery_orders(brains, links, seed=7)
    assert o1 == o2                                   # same seed -> identical order everywhere
    assert set(o1) == set(brains)
    for b in brains:
        assert sorted(o1[b]) == sorted(links)         # a permutation of the same links


def test_seeded_delivery_orders_differ_across_seeds():
    brains = ["A", "B", "C", "D"]
    links = [("La", "x"), ("Lb", "y")]
    # over several seeds at least one per-brain order must flip (otherwise "permuted" is a lie)
    flips = any(M.seeded_delivery_orders(brains, links, seed=s)
                != M.seeded_delivery_orders(brains, links, seed=0) for s in range(1, 20))
    assert flips


# --- per-seed metric computation -------------------------------------------------------

def test_per_seed_differing_effects_crdt_reconverges_lww_does_not():
    # side_a (A,B) saw effect_a during the partition, side_b (C,D) saw effect_b.
    during = {"A": "DEPRECATED", "B": "DEPRECATED", "C": "OVERRULED", "D": "OVERRULED"}
    # a delivery order where each brain's LAST link differs by side -> LWW stays split.
    orders = {"A": [("Lb", "OVERRULED"), ("La", "DEPRECATED")],   # A's last = DEPRECATED
              "B": [("Lb", "OVERRULED"), ("La", "DEPRECATED")],   # B's last = DEPRECATED
              "C": [("La", "DEPRECATED"), ("Lb", "OVERRULED")],   # C's last = OVERRULED
              "D": [("La", "DEPRECATED"), ("Lb", "OVERRULED")]}   # D's last = OVERRULED
    r = M.per_seed_result(during_statuses=during, delivery_orders=orders,
                          effect_a="DEPRECATED", effect_b="OVERRULED")
    assert r["agreement_during"] < 1.0                # 2|2 split while partitioned
    assert r["crdt_healed"] == 1.0                    # join of both links identical everywhere
    assert r["lww_healed"] < 1.0                      # last-writer split -> not unanimous
    # the LWW value is the actual last-writer split: 2 DEPRECATED, 2 OVERRULED -> 0.5
    assert r["lww_healed"] == 0.5


def test_per_seed_lww_reads_the_actual_permuted_order():
    # a KNOWN permutation: every brain's LAST link is La (effect_a) -> LWW unanimous on a.
    during = {"A": "a", "B": "b"}
    orders = {"A": [("Lb", "b"), ("La", "a")],     # last = a
              "B": [("Lb", "b"), ("La", "a")]}     # last = a
    r = M.per_seed_result(during_statuses=during, delivery_orders=orders,
                          effect_a="a", effect_b="b")
    assert r["crdt_healed"] == 1.0
    assert r["lww_healed"] == 1.0                  # both brains' last-writer is a -> agree


def test_per_seed_identical_effects_never_diverge():
    during = {"A": "SAME", "B": "SAME", "C": "SAME", "D": "SAME"}
    orders = {b: [("La", "SAME"), ("Lb", "SAME")] for b in during}
    r = M.per_seed_result(during_statuses=during, delivery_orders=orders,
                          effect_a="SAME", effect_b="SAME")
    assert r["agreement_during"] == 1.0
    assert r["crdt_healed"] == 1.0 and r["lww_healed"] == 1.0


# --- partition composition (the 2|2 split plan) ----------------------------------------

def test_partition_plan_every_host_drops_all_opposite_side_ips():
    # the 2|2 split: each host on one side must drop egress to EVERY host on the other side
    # (one multi-peer tc tree per host), never to its own side. A per-PAIR plan would
    # del-root-wipe earlier peers; this per-HOST plan lists all opposite IPs at once.
    side_a = [("A", "10.0.0.1"), ("B", "10.0.0.2")]
    side_b = [("C", "10.0.0.3"), ("D", "10.0.0.4")]
    plan = dict(M.partition_plan(side_a, side_b))
    assert set(plan) == {"A", "B", "C", "D"}          # one entry per host
    assert plan["A"] == ["10.0.0.3", "10.0.0.4"]      # side_a hosts drop both side_b IPs
    assert plan["B"] == ["10.0.0.3", "10.0.0.4"]
    assert plan["C"] == ["10.0.0.1", "10.0.0.2"]      # side_b hosts drop both side_a IPs
    assert plan["D"] == ["10.0.0.1", "10.0.0.2"]
    # never shape a same-side peer
    assert "10.0.0.2" not in plan["A"] and "10.0.0.4" not in plan["C"]


# --- aggregation -----------------------------------------------------------------------

def test_aggregate_shape_and_full_reconvergence():
    per_seed = [
        {"seed": 1, "agreement_during": 0.5, "crdt_healed": 1.0, "lww_healed": 0.5,
         "partition_verified": True},
        {"seed": 2, "agreement_during": 0.5, "crdt_healed": 1.0, "lww_healed": 0.5,
         "partition_verified": True},
        {"seed": 3, "agreement_during": 0.5, "crdt_healed": 1.0, "lww_healed": 0.5,
         "partition_verified": True},
    ]
    agg = M.aggregate_results(per_seed)
    for metric in ("agreement_during", "crdt_healed", "lww_healed"):
        assert metric in agg
        pt, lo, hi = agg[metric]["mean"], agg[metric]["lo"], agg[metric]["hi"]
        assert lo <= pt <= hi
    assert agg["crdt_healed"]["mean"] == 1.0
    assert agg["crdt_reconverges"] == 1.0          # all seeds crdt_healed == 1.0
    assert agg["lww_reconverges"] == 0.0           # none of the LWW seeds hit 1.0
    assert agg["n_seeds"] == 3


def test_aggregate_partial_reconvergence_fraction():
    per_seed = [
        {"seed": 1, "agreement_during": 0.5, "crdt_healed": 1.0, "lww_healed": 0.5,
         "partition_verified": True},
        {"seed": 2, "agreement_during": 0.5, "crdt_healed": 1.0, "lww_healed": 1.0,
         "partition_verified": True},
    ]
    agg = M.aggregate_results(per_seed)
    assert agg["crdt_reconverges"] == 1.0
    assert agg["lww_reconverges"] == 0.5           # one of two LWW seeds reconverged
