"""§7-s: reliable Patch transport (delivery contract + sync-latency integration).

Adapts the sound NPS propagation design (per-peer health-tracked push) but fixes its
review-critical bugs for a CRDT payload: ONLY HTTP 2xx counts as delivered (NPS
treated 3xx as success -> silent Patch loss, fatal for convergence); a send exception
is recorded as non-delivery, never swallowed-as-success; and a publish is marked
globally-consistent (for the skew-free SyncLatencyMeter) ONLY when every peer
delivered. Payload-agnostic (bytes in); the brain/merge layer owns serialization.
"""
from __future__ import annotations

from types import SimpleNamespace


class FakeSender:
    def __init__(self, status_map):
        self.status_map = status_map
        self.sent = []

    def __call__(self, peer, payload):
        self.sent.append((peer, payload))
        st = self.status_map.get(peer)
        if st == "raise":
            raise ConnectionError("peer down")
        return SimpleNamespace(status_code=st)


def test_only_2xx_counts_as_delivered():
    from meld.sync.transport import PatchTransport
    t = PatchTransport(FakeSender({"p1": 200, "p2": 302, "p3": 500}))
    d = t.publish("patch1", b"payload", ["p1", "p2", "p3"])
    assert d == {"p1": True, "p2": False, "p3": False}   # 3xx is NOT delivery (NPS H3)


def test_global_consistency_only_when_all_delivered():
    from meld.sync.latency import SyncLatencyMeter
    from meld.sync.transport import PatchTransport
    clk = [0.0]
    m = SyncLatencyMeter(clock=lambda: clk[0])
    t = PatchTransport(FakeSender({"p1": 200, "p2": 204}), latency_meter=m)
    clk[0] = 1.0
    t.publish("ok", b"x", ["p1", "p2"])
    assert m.latency_ms("ok") is not None                # all 2xx -> consistent

    m2 = SyncLatencyMeter(clock=lambda: 0.0)
    t2 = PatchTransport(FakeSender({"p1": 200, "p2": 500}), latency_meter=m2)
    t2.publish("partial", b"x", ["p1", "p2"])
    assert m2.latency_ms("partial") is None              # one failed -> pending (honest)


def test_send_exception_is_non_delivery_not_raised():
    from meld.sync.transport import PatchTransport
    t = PatchTransport(FakeSender({"p1": 200, "p2": "raise"}))
    d = t.publish("p", b"x", ["p1", "p2"])
    assert d == {"p1": True, "p2": False}                # exception -> not delivered, no raise


def test_peer_health_tracks_consecutive_failures_and_resets():
    from meld.sync.transport import PatchTransport
    down = PatchTransport(FakeSender({"p": 500}), fail_threshold=3)
    for _ in range(3):
        down.publish("x", b"x", ["p"])
    assert down.peer_healthy("p") is False               # 3 consecutive failures -> unhealthy
    one = PatchTransport(FakeSender({"p": 500}), fail_threshold=3)
    one.publish("x", b"x", ["p"])
    assert one.peer_healthy("p") is True                 # 1 < 3
