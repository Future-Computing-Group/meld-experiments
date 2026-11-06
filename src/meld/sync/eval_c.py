"""eval-c multi-host dispatch (in-process): concurrent conflicting status updates under a
partition, then heal + resync.

A 2|2 split of brains all hold the same claim. While partitioned, the A-side applies one
overrule link (effect_a) and the B-side another (effect_b) -- each side blind to the other.
The status-CRDT makes each side's status the lattice-join of the links it has seen, so the
two sides DIVERGE while partitioned and RECONVERGE to the same join once heal+resync delivers
every link to every brain (agreement -> 1.0), order-independently. The last-writer-wins
baseline keeps only the latest-applied effect, so after a reordered heal the two sides keep
different "latest" effects and stay diverged. Built on BrainNode.apply_status/status_of
(signed authored effects joined by the authored-status lattice). The testbed run
(edge/run_eval_c.py) replays it over the wire: signed effects PUBLISHed to /patch, vis read
from /metrics.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from meld.sync.node import BrainNode


def _agreement(statuses: List[str]) -> float:
    """Fraction of brains agreeing on the modal status (1.0 == unanimous)."""
    return max(Counter(statuses).values()) / len(statuses) if statuses else 1.0


def eval_c_partition_heal(*, n_nodes: int, claim: str, encoder: str, nli: str,
                          effect_a: str, effect_b: str) -> Dict[str, float]:
    nodes = [BrainNode(f"n{i}", encoder=encoder, nli=nli) for i in range(n_nodes)]
    for n in nodes:
        n.ingest(claim)
    half = n_nodes // 2
    side_a, side_b = nodes[:half], nodes[half:]

    # PARTITION: each side applies only its own overrule link (cross-delivery blocked).
    for n in side_a:
        n.apply_status(claim, link_id="La", effect=effect_a)
    for n in side_b:
        n.apply_status(claim, link_id="Lb", effect=effect_b)
    crdt_during = _agreement([n.status_of(claim) for n in nodes])

    # HEAL + resync: every brain now sees BOTH links -> the status-CRDT joins them.
    for n in nodes:
        n.apply_status(claim, link_id="La", effect=effect_a)
        n.apply_status(claim, link_id="Lb", effect=effect_b)
    crdt_healed = _agreement([n.status_of(claim) for n in nodes])

    # LWW baseline: keep only the LATEST-applied effect. Heal delivers the OTHER side's link
    # last to each side, so side A's latest is effect_b and side B's latest is effect_a ->
    # they stay split iff the effects differ.
    lww = [effect_b] * len(side_a) + [effect_a] * len(side_b)
    lww_healed = _agreement(lww)

    return {"crdt_during": crdt_during, "crdt_healed": crdt_healed, "lww_healed": lww_healed}


__all__ = ["eval_c_partition_heal"]
