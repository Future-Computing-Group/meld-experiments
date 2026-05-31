"""Task 1.5 — the wire admission (REJECT) gate (Cor. 2.1).

Stale or overruled deltas never propagate: a delta whose staleness-augmented
retention falls below R_min, or whose status is not ACTIVE, is rejected at the
wire before it can reach the merge decision procedure. REUSE the retention law
for R-tilde; NEW = its placement as a wire-level admission gate.
"""
from __future__ import annotations

from smp.merge.reject_gate import Delta, admit


def test_stale_delta_rejected():
    d = Delta(retention=0.1, r_min=0.5, status="ACTIVE")
    assert admit(d) is False


def test_overruled_claim_rejected():
    d = Delta(retention=0.9, r_min=0.5, status="OVERRULED")
    assert admit(d) is False


def test_revoked_claim_rejected():
    d = Delta(retention=0.9, r_min=0.5, status="REVOKED")
    assert admit(d) is False


def test_active_fresh_admitted():
    d = Delta(retention=0.9, r_min=0.5, status="ACTIVE")
    assert admit(d) is True


def test_boundary_below_r_min_rejected():
    # strictly-below-floor is stale (float boundary discipline)
    assert admit(Delta(retention=0.4999, r_min=0.5, status="ACTIVE")) is False
