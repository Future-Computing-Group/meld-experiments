"""C-1: empirical status-CRDT convergence under concurrent (re)ordered delivery.

Applies a fixed event set (overrule/supersede links) in many random delivery
orderings and checks every ordering computes the SAME per-claim effective status --
the strong-eventual-consistency demonstration for the state-based status CRDT
(Section 5.1). Order-independence is structural (the merge is a join-semilattice);
this exercises it across orderings and reports the agreement fraction.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Tuple

from smp.merge.status_crdt import StatusState, effective_status

Event = Tuple[str, str, str]  # (claim_key, link_id, effect)
TimedEvent = Tuple[float, str, str]  # (time, link_id, effect) for one claim


def _status_at(applied: List[TimedEvent], mode: str) -> str:
    """Status of one claim from the events a node has applied (time-filtered)."""
    if not applied:
        return "ACTIVE"
    if mode == "crdt":
        st = StatusState()
        for _t, lid, eff in applied:
            st = st.add(lid, eff)
        return effective_status(st)
    if mode == "lww":
        return max(applied, key=lambda e: e[0])[2]   # latest-by-time wins (order-dependent)
    raise ValueError(f"unknown mode {mode!r}; choose 'crdt' or 'lww'")


def agreement_over_time(node_logs: Dict[str, List[TimedEvent]], *,
                        checkpoints: List[float], mode: str = "crdt") -> List[Dict]:
    """Status-agreement fraction across nodes at each checkpoint time, for ONE claim.

    ``node_logs`` maps node -> timed events that node has RECEIVED (a partition is
    modelled by a node receiving an event only at heal-time). At each checkpoint, each
    node's status is computed from its events with time <= checkpoint under ``mode``
    ('crdt' join, order-independent; or the 'lww' baseline, order-dependent). agreement
    = (largest set of nodes sharing one status) / n_nodes. CRDT reconverges to 1.0 once
    all nodes have all events; LWW can stay < 1.0 if delivery order differs."""
    n = len(node_logs) or 1
    out: List[Dict] = []
    for t in checkpoints:
        statuses = {node: _status_at([e for e in evs if e[0] <= t], mode)
                    for node, evs in node_logs.items()}
        agreement = max(Counter(statuses.values()).values()) / n
        out.append({"t": t, "agreement": agreement, "statuses": statuses})
    return out


def _statuses_for_order(events: List[Event], order: List[int],
                        mode: str = "crdt") -> Dict[str, str]:
    if mode == "crdt":
        # state-based CRDT: OR-set add + lattice-join recompute (order-independent)
        states: Dict[str, StatusState] = {}
        for i in order:
            ck, lid, eff = events[i]
            states[ck] = states.get(ck, StatusState()).add(lid, eff)
        return {ck: effective_status(st) for ck, st in states.items()}
    if mode == "lww":
        # last-writer-wins baseline: per claim, the LAST-applied effect wins
        # (order-dependent -> diverges under reordering)
        last: Dict[str, str] = {}
        for i in order:
            ck, _lid, eff = events[i]
            last[ck] = eff
        return last
    raise ValueError(f"unknown mode {mode!r}; choose 'crdt' or 'lww'")


def converge_across_orderings(events: List[Event], *, n_orderings: int = 24,
                              seed: int = 0, mode: str = "crdt") -> Dict:
    """Apply ``events`` in ``n_orderings`` random delivery orders under ``mode``
    ('crdt' or the 'lww' baseline); report whether all orders converge to the same
    per-claim status (agreement_fraction == 1.0 iff order-independent) and the
    converged status map (the first order's, when they agree)."""
    rng = random.Random(seed)
    n = len(events)
    results: List[Dict[str, str]] = []
    for _ in range(n_orderings):
        order = list(range(n))
        rng.shuffle(order)
        results.append(_statuses_for_order(events, order, mode=mode))
    ref = results[0]
    agree = sum(1 for r in results if r == ref)
    return {"n_orderings": n_orderings,
            "agreement_fraction": agree / n_orderings,
            "status": ref}


__all__ = ["converge_across_orderings", "agreement_over_time"]
