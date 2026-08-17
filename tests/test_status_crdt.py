"""Task 1.4 — status CRDT (OR-Set of append-only overrule/supersede links +
lattice-join recompute). REUSE+EXTEND: the status enum/lifecycle is inherited;
the OR-Set merge + recompute is NEW.

The load-bearing test (status-CRDT-convergence): per-claim effective status is a
state-based CRDT -- merge is commutative, associative, idempotent, so all
replicas that have received the same link set compute the SAME status regardless
of delivery order (strong eventual consistency, Prop. 5.2 -- DEMONSTRATED here,
not re-proved). The status lattice is ACTIVE <= DEPRECATED <= OVERRULED <= REVOKED.
"""
from __future__ import annotations

import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

from meld.merge.status_crdt import StatusState, effective_status, join, merge

# overrule links carry an effect status; identity by a stable link id.
_EFFECTS = ["DEPRECATED", "OVERRULED", "REVOKED"]


def _link(i, eff):
    return (f"link{i}", eff)


def _state(links):
    s = StatusState()
    for lid, eff in links:
        s = s.add(lid, eff)
    return s


# ---- lattice ----
def test_lattice_join_monotone():
    assert join("ACTIVE", "OVERRULED") == "OVERRULED"
    assert join("DEPRECATED", "ACTIVE") == "DEPRECATED"
    assert join("OVERRULED", "REVOKED") == "REVOKED"
    assert join("ACTIVE", "ACTIVE") == "ACTIVE"


# ---- SEC / CRDT properties ----
link_sets = st.lists(
    st.tuples(st.integers(min_value=0, max_value=6),
              st.sampled_from(_EFFECTS)),
    max_size=8,
)


@given(a=link_sets, b=link_sets)
@settings(max_examples=100)
def test_merge_commutative(a, b):
    sa, sb = _state(a), _state(b)
    assert effective_status(merge(sa, sb)) == effective_status(merge(sb, sa))


@given(a=link_sets, b=link_sets, c=link_sets)
@settings(max_examples=100)
def test_merge_associative(a, b, c):
    sa, sb, sc = _state(a), _state(b), _state(c)
    left = merge(merge(sa, sb), sc)
    right = merge(sa, merge(sb, sc))
    assert effective_status(left) == effective_status(right)


@given(a=link_sets)
@settings(max_examples=100)
def test_merge_idempotent(a):
    sa = _state(a)
    assert effective_status(merge(sa, sa)) == effective_status(sa)


def test_same_linkset_same_status_regardless_of_order():
    links = [_link(1, "DEPRECATED"), _link(2, "OVERRULED"), _link(3, "DEPRECATED")]
    statuses = {effective_status(_state(list(p))) for p in itertools.permutations(links)}
    assert len(statuses) == 1            # SEC: order-independent
    assert statuses == {"OVERRULED"}     # join of effects


def test_empty_state_is_active():
    assert effective_status(StatusState()) == "ACTIVE"


def test_append_only_no_destroy():
    s = StatusState().add("l1", "DEPRECATED")
    s2 = s.add("l1", "DEPRECATED")       # re-add same link-version is idempotent
    assert effective_status(s2) == "DEPRECATED"
    assert len(s2.links) == 1
