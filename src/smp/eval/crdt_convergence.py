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


def _status_key(status: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """Hashable canonical key for a per-claim status map (for clustering)."""
    return tuple(sorted(status.items()))


def converge_across_orderings(events: List[Event], *, n_orderings: int = 24,
                              seed: int = 0, mode: str = "crdt") -> Dict:
    """Apply ``events`` in ``n_orderings`` random delivery orders under ``mode``
    ('crdt' or the 'lww' baseline); report order-independence.

    Returns:
      ``agreement_fraction`` -- fraction of orderings equal to the FIRST ordering
        (kept for back-compat).
      ``modal_agreement`` -- size of the LARGEST cluster of identical per-claim
        status maps / n_orderings (symmetric: independent of which order is first).
        1.0 iff every ordering computes the same statuses (order-independent).
      ``n_distinct`` -- number of distinct per-claim status maps observed across
        the orderings (1 iff order-independent).
      ``status`` -- the MODAL status map (the first ordering's, when all agree).
    """
    rng = random.Random(seed)
    n = len(events)
    results: List[Dict[str, str]] = []
    for _ in range(n_orderings):
        order = list(range(n))
        rng.shuffle(order)
        results.append(_statuses_for_order(events, order, mode=mode))
    ref = results[0]
    agree = sum(1 for r in results if r == ref)
    clusters = Counter(_status_key(r) for r in results)
    modal_key, modal_count = clusters.most_common(1)[0]
    modal_status = dict(modal_key)
    return {"n_orderings": n_orderings,
            "agreement_fraction": agree / n_orderings,
            "modal_agreement": modal_count / n_orderings,
            "n_distinct": len(clusters),
            "status": modal_status}


# ---------------------------------------------------------------------------
# Canonical in-process C-1 scenario (Section 7-c, paragraph 1)
# ---------------------------------------------------------------------------
# A FIXED, principled contested-lifecycle event set: four claims, each overruled
# or superseded by MORE THAN ONE peer issuing a DIFFERENT status effect (the exact
# situation the status-CRDT exists to resolve), plus an idempotent re-delivery of
# one link id (OR-set add). Under the lattice-join (DEPRECATED<OVERRULED<REVOKED)
# every claim converges to the join of its effects regardless of delivery order;
# under last-writer-wins the surviving status is whichever link was applied LAST,
# so it changes with the ordering. This is the in-process demonstration of the
# order-independence Section 5.1 establishes by construction; the multi-host
# version (real partition heal) is the strong form (Section 7-c, paragraph 2).
CANONICAL_C1_SCENARIO: Dict = {
    "events": [
        # c1: deprecated by one peer, overruled by another (+ idempotent re-delivery)
        ("c1", "c1.deprecate", "DEPRECATED"),
        ("c1", "c1.overrule", "OVERRULED"),
        ("c1", "c1.overrule", "OVERRULED"),      # idempotent re-delivery (OR-set add)
        # c2: deprecated vs revoked
        ("c2", "c2.deprecate", "DEPRECATED"),
        ("c2", "c2.revoke", "REVOKED"),
        # c3: overruled vs revoked
        ("c3", "c3.overrule", "OVERRULED"),
        ("c3", "c3.revoke", "REVOKED"),
        # c4: three competing effects (deprecated / overruled / revoked)
        ("c4", "c4.deprecate", "DEPRECATED"),
        ("c4", "c4.overrule", "OVERRULED"),
        ("c4", "c4.revoke", "REVOKED"),
    ],
    # lattice-join of each claim's effects (DEPRECATED<OVERRULED<REVOKED):
    "expected_crdt_status": {
        "c1": "OVERRULED",   # join(DEPRECATED, OVERRULED)
        "c2": "REVOKED",     # join(DEPRECATED, REVOKED)
        "c3": "REVOKED",     # join(OVERRULED, REVOKED)
        "c4": "REVOKED",     # join(DEPRECATED, OVERRULED, REVOKED)
    },
}


def run_inprocess_c1(*, n_orderings: int = 200, seed: int = 0) -> Dict:
    """Run the canonical contested-lifecycle scenario in ``n_orderings`` random
    delivery orders under both the status-CRDT and the last-writer-wins baseline,
    and report each mode's order-independence (modal agreement + distinct outcomes).
    Deterministic in ``seed`` (reproducible manuscript number)."""
    events = CANONICAL_C1_SCENARIO["events"]
    crdt = converge_across_orderings(events, n_orderings=n_orderings, seed=seed,
                                     mode="crdt")
    lww = converge_across_orderings(events, n_orderings=n_orderings, seed=seed,
                                    mode="lww")
    return {"n_orderings": n_orderings, "seed": seed,
            "n_events": len(events), "n_claims": len(CANONICAL_C1_SCENARIO["expected_crdt_status"]),
            "crdt": crdt, "lww": lww}


# ---------------------------------------------------------------------------
# E2(a) (AE #4): partition-duration x reorder-rate perturbation sweep
# ---------------------------------------------------------------------------
_EFFECTS = ("DEPRECATED", "OVERRULED", "REVOKED")  # the status join order


def _partial_shuffle(order: List[int], reorder_rate: float,
                     rng: random.Random) -> List[int]:
    """Interpolate between identity (reorder_rate=0) and a full shuffle (=1):
    permute a reorder_rate fraction of positions among themselves, leave the rest."""
    order = list(order)
    n = len(order)
    k = int(round(reorder_rate * n))
    if k <= 1:
        return order
    idx = rng.sample(range(n), k)
    vals = [order[i] for i in idx]
    rng.shuffle(vals)
    for i, v in zip(idx, vals):
        order[i] = v
    return order


def _make_partition_scenario(n_claims: int, duration: int,
                             rng: random.Random) -> List[Event]:
    """A partition of ``duration`` queued status updates per claim (the partition
    duration): each claim receives ``duration`` overrule/supersede links carrying
    randomly chosen effects. CRDT joins them (order-independent); LWW keeps the
    last-applied (order-dependent)."""
    events: List[Event] = []
    for c in range(n_claims):
        ck = f"c{c}"
        for j in range(duration):
            eff = _EFFECTS[rng.randrange(len(_EFFECTS))]
            events.append((ck, f"{ck}.l{j}", eff))
    return events


def _converge_reorder(events: List[Event], *, n_orderings: int, seed: int,
                      mode: str, reorder_rate: float) -> Tuple[float, int]:
    """(modal_agreement, n_distinct) for ``events`` applied under ``mode`` across
    ``n_orderings`` deliveries, each partially shuffled at ``reorder_rate``."""
    rng = random.Random(seed)
    n = len(events)
    results: List[Dict[str, str]] = []
    for _ in range(n_orderings):
        order = _partial_shuffle(list(range(n)), reorder_rate, rng)
        results.append(_statuses_for_order(events, order, mode=mode))
    clusters = Counter(_status_key(r) for r in results)
    modal = clusters.most_common(1)[0][1]
    return modal / n_orderings, len(clusters)


def partition_reorder_sweep(*, durations: List[int], reorder_rates: List[float],
                            n_claims: int = 4, n_orderings: int = 50,
                            seed: int = 0) -> List[Dict]:
    """Sweep the partition-duration x reorder-rate grid; for each cell report the
    status-CRDT and last-writer-wins modal agreement + distinct-outcome counts.

    The CRDT is order-independent by construction, so its modal agreement is 1.0
    in every cell regardless of how long the partition ran or how reordered the
    heal delivery is; LWW loses agreement as reordering grows. Deterministic in
    ``seed`` (the per-cell scenario is generated from a cell-specific sub-seed so
    cells are independent and reproducible)."""
    cells: List[Dict] = []
    for di, dur in enumerate(durations):
        for ri, r in enumerate(reorder_rates):
            scen_rng = random.Random((seed << 8) ^ (dur << 4) ^ di)
            events = _make_partition_scenario(n_claims, dur, scen_rng)
            cell_seed = (seed * 1000) + di * 10 + ri
            c_ma, c_nd = _converge_reorder(events, n_orderings=n_orderings,
                                           seed=cell_seed, mode="crdt", reorder_rate=r)
            l_ma, l_nd = _converge_reorder(events, n_orderings=n_orderings,
                                           seed=cell_seed, mode="lww", reorder_rate=r)
            cells.append({"duration": dur, "reorder_rate": r, "n_claims": n_claims,
                          "n_events": len(events), "n_orderings": n_orderings,
                          "crdt_modal_agreement": c_ma, "crdt_n_distinct": c_nd,
                          "lww_modal_agreement": l_ma, "lww_n_distinct": l_nd})
    return cells


__all__ = ["converge_across_orderings", "agreement_over_time",
           "run_inprocess_c1", "CANONICAL_C1_SCENARIO",
           "partition_reorder_sweep", "_partial_shuffle"]
