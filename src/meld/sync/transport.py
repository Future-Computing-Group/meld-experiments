"""Reliable Patch transport with per-peer health + skew-free latency integration.

Adapts the sound NPS propagation design (periodic health-tracked push) for a CRDT
payload, fixing its review criticals: ONLY HTTP 2xx is delivery (3xx/4xx/5xx and send
exceptions are non-delivery -- NPS treated 3xx as success, silently dropping Patches);
a publish is globally-consistent only when EVERY peer delivered, which is when the
SyncLatencyMeter's consistency mark fires (so a partial publish leaves latency
honestly pending). Payload-agnostic: ``send(peer, payload_bytes) -> response`` with a
``.status_code`` is the exec seam (real = pooled httpx client; mocked in tests).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional


class PatchTransport:
    def __init__(self, send: Callable, *, latency_meter=None, fail_threshold: int = 3):
        self._send = send
        self._meter = latency_meter
        self._fail_threshold = fail_threshold
        self._consec_fail: Dict[str, int] = {}

    def _deliver(self, peer: str, payload: bytes) -> bool:
        try:
            resp = self._send(peer, payload)
        except Exception:
            self._consec_fail[peer] = self._consec_fail.get(peer, 0) + 1
            return False
        status = getattr(resp, "status_code", 0)
        delivered = 200 <= status < 300              # 2xx ONLY (NPS H3 fix)
        self._consec_fail[peer] = 0 if delivered else self._consec_fail.get(peer, 0) + 1
        return delivered

    def publish(self, patch_id: str, payload: bytes,
                peers: List[str]) -> Dict[str, bool]:
        """Push the Patch to every peer; return per-peer delivery (2xx only). Marks the
        latency meter published-now, and globally-consistent only if all delivered."""
        if self._meter is not None:
            self._meter.mark_published(patch_id)
        delivered = {peer: self._deliver(peer, payload) for peer in peers}
        if self._meter is not None and delivered and all(delivered.values()):
            self._meter.mark_globally_consistent(patch_id)
        return delivered

    def peer_healthy(self, peer: str) -> bool:
        return self._consec_fail.get(peer, 0) < self._fail_threshold


__all__ = ["PatchTransport"]
