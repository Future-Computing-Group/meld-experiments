"""Skew-free sync-latency measurement for the coordinator (§7-s).

The interval Patch-publish -> global-consistency is stamped with a SINGLE clock (the
coordinator's), so cross-node clock skew cannot corrupt it (the NPS review C2 bug).
``mark_globally_consistent`` is called when the coordinator has observed all peers
apply the Patch (their acks returned); the measured interval therefore includes the
ack round-trip -- a small, honest overcount of true convergence -- but never subtracts
a remote node's timestamp. There is deliberately no API that accepts a foreign
timestamp.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional


class SyncLatencyMeter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._published: Dict[str, float] = {}
        self._consistent: Dict[str, float] = {}

    def reset(self) -> None:
        """Clear all records (per-cell isolation: each campaign cell measures fresh)."""
        self._published.clear()
        self._consistent.clear()

    def mark_published(self, patch_id: str) -> None:
        self._published[patch_id] = self._clock()

    def mark_globally_consistent(self, patch_id: str) -> None:
        # stamped with the coordinator's own clock; no remote timestamp accepted.
        self._consistent[patch_id] = self._clock()

    def latency_ms(self, patch_id: str) -> Optional[float]:
        t0 = self._published.get(patch_id)
        t1 = self._consistent.get(patch_id)
        if t0 is None or t1 is None:
            return None
        return (t1 - t0) * 1000.0

    def _completed_ms(self) -> List[float]:
        out = []
        for pid, t0 in self._published.items():
            t1 = self._consistent.get(pid)
            if t1 is not None:
                out.append((t1 - t0) * 1000.0)
        return sorted(out)

    def summary(self) -> Dict[str, float]:
        """n + p50/p95 + max over completed (globally-consistent) Patches."""
        xs = self._completed_ms()
        if not xs:
            return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

        def pct(p: float) -> float:
            if len(xs) == 1:
                return xs[0]
            idx = p / 100.0 * (len(xs) - 1)
            lo = int(idx)
            frac = idx - lo
            hi = min(lo + 1, len(xs) - 1)
            return xs[lo] + frac * (xs[hi] - xs[lo])

        return {"n": len(xs), "p50_ms": pct(50), "p95_ms": pct(95), "max_ms": xs[-1]}


__all__ = ["SyncLatencyMeter"]
