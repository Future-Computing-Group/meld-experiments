"""§7-s: skew-free sync-latency measurement.

The NPS review's CRITICAL finding (C2): latency measured by subtracting one node's
clock from another's is corrupted by cross-node clock skew (~tens of ms, the same
order as the metric). SMP sync latency (Patch publish -> global consistency) is
inherently cross-node, so we measure it ENTIRELY ON THE COORDINATOR'S CLOCK: t0 when
the coordinator publishes, t1 when the coordinator observes all peers have applied it
(acks returned). The interval includes the ack round-trip (a small, honest overcount
of true convergence), but it never subtracts a remote timestamp -> skew-free.
"""
from __future__ import annotations


def test_latency_measured_on_one_clock_skew_free():
    from smp.sync.latency import SyncLatencyMeter
    t = [0.0]
    meter = SyncLatencyMeter(clock=lambda: t[0])   # single injected (coordinator) clock
    t[0] = 10.0
    meter.mark_published("p1")
    t[0] = 10.250
    meter.mark_globally_consistent("p1")
    assert abs(meter.latency_ms("p1") - 250.0) < 1e-6


def test_latency_pending_until_globally_consistent():
    from smp.sync.latency import SyncLatencyMeter
    meter = SyncLatencyMeter(clock=lambda: 0.0)
    meter.mark_published("p2")
    assert meter.latency_ms("p2") is None        # not yet consistent


def test_latency_summary_percentiles():
    from smp.sync.latency import SyncLatencyMeter
    t = [0.0]
    meter = SyncLatencyMeter(clock=lambda: t[0])
    for i, d in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        t[0] = float(i)
        meter.mark_published(f"p{i}")
        t[0] = i + d
        meter.mark_globally_consistent(f"p{i}")
    s = meter.summary()
    assert s["n"] == 5
    assert abs(s["p50_ms"] - 300.0) < 1.0        # median of [100,200,300,400,500]
    assert s["max_ms"] >= s["p95_ms"] >= s["p50_ms"]


def test_meter_takes_no_remote_timestamp():
    # structural skew-safety: the API marks events by id and stamps them with the
    # meter's own clock -- there is no entry point that accepts a foreign timestamp.
    from smp.sync.latency import SyncLatencyMeter
    import inspect
    sig = inspect.signature(SyncLatencyMeter.mark_globally_consistent)
    assert list(sig.parameters) == ["self", "patch_id"]   # id only, no timestamp arg
