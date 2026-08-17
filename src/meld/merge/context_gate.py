"""The Context gate (architecture A.2).

Scopes ``same-enough``: two claims that are ``same-enough`` in ONE Context may
only ``overlap`` across Contexts. The gate is what stops a high-sigma pair like
"Paris (the city)" and "Paris (of Greek myth)" from merging: they live in
different frames (cells), so they are Context-incompatible and the classifier
RELATEs them instead of MERGEing.

A Context is the frame in which a claim is valid: its cell/scope, its validity
interval, and freshness. Derived from the vendored proto Endpoint (cell/ACL
scope); freshness is the status field (is_fresh == status == "ACTIVE"). REUSE
the proto frame fields; NEW = the compatibility predicate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Context:
    cell: str                          # cell/scope frame, e.g. "agent.public"
    valid_from: float = 0.0
    valid_until: Optional[float] = None  # None = open-ended
    is_fresh: bool = True


def _intervals_overlap(a: Context, b: Context) -> bool:
    a_hi = float("inf") if a.valid_until is None else a.valid_until
    b_hi = float("inf") if b.valid_until is None else b.valid_until
    return a.valid_from <= b_hi and b.valid_from <= a_hi


def context_compatible(a: Context, b: Context) -> bool:
    """True iff the two claims share a frame and may legitimately be merged:
    same cell AND overlapping validity intervals. Different cells (frames) are
    incompatible regardless of textual similarity (the false-merge guard)."""
    if a.cell != b.cell:
        return False
    if not _intervals_overlap(a, b):
        return False
    return True


__all__ = ["Context", "context_compatible"]
