"""Per-claim status CRDT (architecture A.3 / restructure note Prop. 5.2).

A claim's local state is an add-biased OR-Set of incoming append-only
overrule/supersede links (each carrying an effect status). The effective status
is the lattice-join of the effects over the monotone status lattice
    ACTIVE  <=  DEPRECATED  <=  OVERRULED  <=  REVOKED
Merge of two replicas is set-union of the link sets followed by recomputation of
the join. This makes the per-claim status a state-based CRDT: merge is
commutative, associative, idempotent, so replicas with the same received link
set compute the SAME status regardless of delivery order (strong eventual
consistency).

REUSE+EXTEND: the status enum + ACTIVE<=DEPRECATED<=OVERRULED<=REVOKED ordering
is inherited from the precedent lifecycle; the OR-Set merge + lattice-join
recompute is NEW. The theorem is inherited (Prop. 5.2) -- this is the
demonstration vehicle (experiment C-1), not a re-proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# Monotone lattice rank: higher == more restricted (top = REVOKED).
_RANK = {"ACTIVE": 0, "DEPRECATED": 1, "OVERRULED": 2, "REVOKED": 3}
_BY_RANK = {v: k for k, v in _RANK.items()}


def join(a: str, b: str) -> str:
    """Lattice-join (least upper bound) of two statuses: the more-restricted."""
    return _BY_RANK[max(_RANK[a], _RANK[b])]


@dataclass(frozen=True)
class StatusState:
    """Add-biased OR-Set of overrule/supersede links.

    ``links`` maps a stable link id -> its effect status. Append-only: re-adding
    the same link id is idempotent (never destroys). Immutable; ``add``/``merge``
    return new states.
    """

    links: Dict[str, str] = field(default_factory=dict)

    def add(self, link_id: str, effect: str) -> "StatusState":
        new = dict(self.links)
        # On id collision, take the lattice-join of effects. A content-addressed
        # link id should map to a single effect; joining on collision keeps the
        # merge a true join-semilattice (order-independent) even if an id is
        # observed with differing effects across replicas.
        new[link_id] = join(new[link_id], effect) if link_id in new else effect
        return StatusState(links=new)


def merge(a: StatusState, b: StatusState) -> StatusState:
    """Set-union of the two OR-Sets, lattice-joining effects on id collision.
    Commutative, associative, idempotent (a join-semilattice on the powerset)."""
    merged = dict(a.links)
    for lid, eff in b.links.items():
        merged[lid] = join(merged[lid], eff) if lid in merged else eff
    return StatusState(links=merged)


def effective_status(state: StatusState) -> str:
    """Lattice-join of all incoming link effects. Empty -> ACTIVE."""
    status = "ACTIVE"
    for eff in state.links.values():
        status = join(status, eff)
    return status


__all__ = ["StatusState", "join", "merge", "effective_status"]
