"""E3: the reject gate must not break status-CRDT convergence.

The concern: if the staleness/overrule reject gate filters G-Set ADMISSION
based on receipt-time local state, two replicas that receive the same links in
different orders can admit different subsets and diverge -- violating the
convergence premise the security argument rests on.

The resolution SMP adopts (and these RED-first tests pin down): admit every link
unconditionally into the grow-only set, and apply the staleness/overrule gate at
MATERIALIZATION-READ (effective_status_gated) -- a pure function of the converged
set, hence order-independent. The NAIVE alternative (gate at admission, dropping
a link that arrives once the target is already non-active) is shown to DIVERGE on
the same scenario, which is exactly why SMP gates at read, not at admission.
"""
from __future__ import annotations

from smp.merge.status_crdt import StatusState, effective_status, effective_status_gated


# --------------------------------------------------------------------------- #
# Materialization-read gate is a pure function of the converged set
# --------------------------------------------------------------------------- #
def test_gated_drops_stale_links_at_read():
    st = StatusState().add("L1", "OVERRULED").add("L2", "REVOKED")
    # ungated: join(OVERRULED, REVOKED) = REVOKED
    assert effective_status(st) == "REVOKED"
    # gate L2 (stale) at read -> only L1 counts -> OVERRULED
    assert effective_status_gated(st, stale=frozenset({"L2"})) == "OVERRULED"
    # gate everything -> ACTIVE (empty effective set)
    assert effective_status_gated(st, stale=frozenset({"L1", "L2"})) == "ACTIVE"


def test_gated_is_pure_function_of_set_not_insertion_order():
    a = StatusState().add("L1", "DEPRECATED").add("L2", "REVOKED").add("L3", "OVERRULED")
    b = StatusState().add("L3", "OVERRULED").add("L1", "DEPRECATED").add("L2", "REVOKED")
    stale = frozenset({"L1"})
    assert effective_status_gated(a, stale=stale) == effective_status_gated(b, stale=stale)


# --------------------------------------------------------------------------- #
# The reject-race scenario: materialization converges, admission can diverge
# --------------------------------------------------------------------------- #
def test_materialization_read_gate_converges_across_orderings():
    from smp.eval.crdt_convergence import reject_race_convergence
    out = reject_race_convergence(n_orderings=200, seed=0)
    assert out["materialization"]["n_distinct"] == 1
    assert out["materialization"]["modal_agreement"] == 1.0


def test_admission_time_gate_diverges_on_same_scenario():
    # The whole point: the naive admission-time gate is NOT order-independent.
    from smp.eval.crdt_convergence import reject_race_convergence
    out = reject_race_convergence(n_orderings=200, seed=0)
    assert out["admission"]["n_distinct"] > 1
    assert out["admission"]["modal_agreement"] < 1.0


def test_materialization_dominates_admission():
    from smp.eval.crdt_convergence import reject_race_convergence
    out = reject_race_convergence(n_orderings=200, seed=0)
    assert out["materialization"]["modal_agreement"] >= out["admission"]["modal_agreement"]


def test_reject_race_deterministic():
    from smp.eval.crdt_convergence import reject_race_convergence
    assert reject_race_convergence(n_orderings=100, seed=1) == \
           reject_race_convergence(n_orderings=100, seed=1)
