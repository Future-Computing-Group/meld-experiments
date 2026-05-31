"""Wire admission (REJECT) gate (Cor. 2.1).

A graph delta is admitted to the merge decision procedure only if it is fresh
AND active. Stale (staleness-augmented retention R-tilde below the floor R_min)
or non-ACTIVE (DEPRECATED/OVERRULED/REVOKED) deltas are rejected at the wire and
never propagate -- the gate that turns the status-CRDT's convergence from a
liability into a guarantee.

REUSE: the retention status semantics from the firmament retention law
(OVERRULED/REVOKED -> zero retention; only ACTIVE propagates). NEW: placement of
the freshness/status check as a wire-level admission gate on PUBLISH/UPDATE.
"""
from __future__ import annotations

from dataclasses import dataclass

# Statuses that may propagate. Mirrors the retention status multiplier: only
# ACTIVE has a non-zero multiplier in the inherited law; DEPRECATED is retained
# locally but is not admitted across the wire as a fresh assertion.
_ADMISSIBLE_STATUSES = {"ACTIVE"}


@dataclass(frozen=True)
class Delta:
    retention: float          # staleness-augmented retention R-tilde of the unit/link
    r_min: float              # usefulness floor
    status: str = "ACTIVE"


def admit(delta: Delta) -> bool:
    """True iff the delta is fresh (R-tilde >= R_min) AND its status is
    admissible (ACTIVE). Strictly-below-floor is stale."""
    if delta.status not in _ADMISSIBLE_STATUSES:
        return False
    if delta.retention < delta.r_min:
        return False
    return True


__all__ = ["Delta", "admit"]
