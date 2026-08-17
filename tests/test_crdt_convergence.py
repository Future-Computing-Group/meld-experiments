"""C-1: status-CRDT convergence under concurrent edits (strong eventual consistency).

The one theorem-grade claim (Section 5.1): the per-claim status is a state-based
CRDT, so replicas that receive the SAME set of overrule/supersede links compute the
SAME effective status regardless of delivery ORDER. C-1 demonstrates it empirically:
apply the same event set in many random orderings and assert all orderings agree.
"""
from __future__ import annotations


def test_status_crdt_converges_under_all_orderings():
    from meld.eval.crdt_convergence import converge_across_orderings
    # escalating effects on c1 (+ one on c2); the lattice-join is order-independent
    events = [("c1", "L1", "DEPRECATED"), ("c1", "L2", "OVERRULED"),
              ("c1", "L3", "ACTIVE"), ("c2", "e4", "REVOKED")]
    res = converge_across_orderings(events, n_orderings=24, seed=0)
    assert res["n_orderings"] == 24
    assert res["agreement_fraction"] == 1.0          # every ordering -> same statuses
    assert res["status"]["c1"] == "OVERRULED"        # join(ACTIVE,DEPRECATED,OVERRULED)
    assert res["status"]["c2"] == "REVOKED"


def test_idempotent_redelivery_does_not_change_status():
    # re-delivering the same link id is idempotent (OR-set add): no status change.
    from meld.eval.crdt_convergence import converge_across_orderings
    events = [("c1", "L1", "DEPRECATED"), ("c1", "L1", "DEPRECATED"),
              ("c1", "L1", "DEPRECATED")]
    res = converge_across_orderings(events, n_orderings=8, seed=1)
    assert res["agreement_fraction"] == 1.0
    assert res["status"]["c1"] == "DEPRECATED"


def test_agreement_over_time_diverges_in_partition_reconverges_at_heal():
    # 2 nodes, one claim. A applies DEPRECATED at t=1; B (partitioned) only receives
    # the same link at heal t=3. CRDT: agree before (both ACTIVE), diverge during the
    # partition, reconverge after heal.
    from meld.eval.crdt_convergence import agreement_over_time
    node_logs = {"A": [(1.0, "L1", "DEPRECATED")],
                 "B": [(3.0, "L1", "DEPRECATED")]}
    pts = agreement_over_time(node_logs, checkpoints=[0.5, 2.0, 3.5], mode="crdt")
    assert pts[0]["agreement"] == 1.0    # both ACTIVE
    assert pts[1]["agreement"] == 0.5    # A DEPRECATED, B ACTIVE
    assert pts[2]["agreement"] == 1.0    # both DEPRECATED after heal


def test_crdt_reconverges_after_reordered_heal_where_lww_stays_diverged():
    # both nodes eventually receive BOTH conflicting links, but in different order
    # (partition delivered them out of order). CRDT joins to the same status; LWW's
    # last-writer differs by order, so it stays diverged after heal.
    from meld.eval.crdt_convergence import agreement_over_time
    node_logs = {"A": [(1.0, "L1", "DEPRECATED"), (4.0, "L2", "OVERRULED")],
                 "B": [(2.0, "L2", "OVERRULED"), (5.0, "L1", "DEPRECATED")]}
    crdt = agreement_over_time(node_logs, checkpoints=[6.0], mode="crdt")
    lww = agreement_over_time(node_logs, checkpoints=[6.0], mode="lww")
    assert crdt[0]["agreement"] == 1.0   # join({DEPRECATED,OVERRULED}) = OVERRULED both
    assert lww[0]["agreement"] == 0.5    # A last=OVERRULED, B last=DEPRECATED


def test_lww_baseline_diverges_where_crdt_converges():
    # the baseline contrast: last-writer-wins is order-dependent, so the same event
    # set under reordering yields DIFFERENT final statuses (agreement < 1), whereas
    # the status-CRDT converges (agreement == 1). Section 5.1's claim, made concrete.
    from meld.eval.crdt_convergence import converge_across_orderings
    events = [("c1", "L1", "DEPRECATED"), ("c1", "L2", "OVERRULED"),
              ("c1", "L3", "ACTIVE")]
    crdt = converge_across_orderings(events, n_orderings=24, seed=0, mode="crdt")
    lww = converge_across_orderings(events, n_orderings=24, seed=0, mode="lww")
    assert crdt["agreement_fraction"] == 1.0
    assert lww["agreement_fraction"] < 1.0


# --- modal-agreement + n_distinct (symmetric order-independence measure) ----------

def test_modal_agreement_and_n_distinct_reported():
    # The symmetric measure backing the manuscript (Section 7-c, in-process): the
    # agreement fraction is the size of the LARGEST cluster of identical per-claim
    # status maps over all orderings, divided by n_orderings (does not depend on
    # which ordering happens to be first). CRDT: one cluster (n_distinct==1,
    # modal==1.0); LWW: many clusters (n_distinct>1, modal<1.0).
    from meld.eval.crdt_convergence import converge_across_orderings
    events = [("c1", "lk1", "DEPRECATED"), ("c1", "lk2", "OVERRULED"),
              ("c2", "lk3", "DEPRECATED"), ("c2", "lk4", "REVOKED")]
    crdt = converge_across_orderings(events, n_orderings=200, seed=0, mode="crdt")
    lww = converge_across_orderings(events, n_orderings=200, seed=0, mode="lww")
    assert crdt["n_distinct"] == 1
    assert crdt["modal_agreement"] == 1.0
    assert lww["n_distinct"] >= 2          # reordering reaches multiple outcomes
    assert lww["modal_agreement"] < 1.0    # no single outcome dominates all orders


def test_canonical_c1_scenario_crdt_order_independent_lww_not():
    # The fixed canonical contested-lifecycle scenario the manuscript reports, run
    # at the manuscript's n_orderings=200, seed=0. CRDT is order-independent by
    # construction (one outcome); LWW diverges across orders.
    from meld.eval.crdt_convergence import run_inprocess_c1, CANONICAL_C1_SCENARIO
    rep = run_inprocess_c1(n_orderings=200, seed=0)
    assert rep["n_orderings"] == 200
    # CRDT: every ordering computes the SAME per-claim statuses
    assert rep["crdt"]["modal_agreement"] == 1.0
    assert rep["crdt"]["n_distinct"] == 1
    # the converged statuses are the lattice joins of each claim's effects
    assert rep["crdt"]["status"] == CANONICAL_C1_SCENARIO["expected_crdt_status"]
    # LWW: order picks the survivor -> diverges across orderings
    assert rep["lww"]["n_distinct"] >= 2
    assert rep["lww"]["modal_agreement"] < 1.0


def test_canonical_c1_scenario_includes_idempotent_redelivery():
    # the canonical event list re-delivers at least one link id (OR-set idempotence);
    # the CRDT result must be invariant to that re-delivery (n_distinct stays 1).
    from meld.eval.crdt_convergence import CANONICAL_C1_SCENARIO
    events = CANONICAL_C1_SCENARIO["events"]
    seen = {}
    duplicated = False
    for ck, lid, eff in events:
        if (ck, lid) in seen:
            assert seen[(ck, lid)] == eff   # same effect re-delivered (idempotent)
            duplicated = True
        seen[(ck, lid)] = eff
    assert duplicated, "canonical scenario must exercise idempotent re-delivery"


def test_run_inprocess_c1_is_reproducible():
    # same seed -> identical report (deterministic manuscript number).
    from meld.eval.crdt_convergence import run_inprocess_c1
    a = run_inprocess_c1(n_orderings=200, seed=0)
    b = run_inprocess_c1(n_orderings=200, seed=0)
    assert a["lww"]["modal_agreement"] == b["lww"]["modal_agreement"]
    assert a["lww"]["n_distinct"] == b["lww"]["n_distinct"]
