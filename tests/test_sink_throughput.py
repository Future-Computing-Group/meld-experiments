"""M1 sink-saturation harness (TDD).

Tests the saturation behaviour with a FakeMergeStep of known service time, so the
harness is verified independent of the real (embed+NLI) merge cost: below the sink's
capacity (1/service) achieved throughput tracks the offered rate; above it, throughput
saturates near 1/service while latency climbs. Run:
  PYTHONPATH=src python3 -m pytest tests/test_sink_throughput.py -q
"""
import time

import pytest

from meld.eval.sink_throughput import run_saturation, FakeMergeStep, saturation_knee


class _Timed(FakeMergeStep):
    """FakeMergeStep that records its actual service times, so the capacity bound is the one
    measured in this run (sleep overshoot under machine load moves it) rather than a constant."""

    def __init__(self, service_s):
        super().__init__(service_s)
        self.actual = []

    def __call__(self, payload):  # noqa: ANN001
        t = time.perf_counter()
        super().__call__(payload)
        self.actual.append(time.perf_counter() - t)

    @property
    def capacity(self):
        return len(self.actual) / sum(self.actual)


def test_throughput_tracks_offered_below_capacity_and_saturates_above():
    step = _Timed(service_s=0.01)                 # single-worker capacity ~100 merges/s unloaded
    res = run_saturation(step, offered_rates=[20, 500], n_per_rate=60)
    # every offered claim is eventually merged
    assert res[20]["n"] == 60 and res[500]["n"] == 60
    cap = step.capacity                           # 1/(mean actual service time), this run
    # above capacity: the sink is never idle, so throughput saturates at the measured 1/service
    assert 0.7 * cap <= res[500]["merged_per_s"] <= 1.1 * cap, (cap, res[500])
    assert res[500]["merged_per_s"] < 0.5 * 500, res[500]
    # below capacity: achieved throughput tracks the offered rate (only meaningful if the
    # machine is not so loaded that 20/s is itself near the measured capacity)
    if cap < 3 * 20:
        pytest.skip(f"machine too loaded: measured capacity {cap:.0f}/s")
    assert 0.6 * 20 <= res[20]["merged_per_s"] <= 1.25 * 20, res[20]
    # saturation manifests as climbing queueing latency
    assert res[500]["p95_ms"] > res[20]["p95_ms"], (res[20], res[500])


def test_knee_is_max_achieved_throughput():
    step = _Timed(service_s=0.01)
    res = run_saturation(step, offered_rates=[10, 50, 300], n_per_rate=40)
    knee = saturation_knee(res)
    assert 0.7 * step.capacity <= knee <= 1.1 * step.capacity, (step.capacity, res)


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, "->", repr(e)[:200])
    sys.exit(1 if fails else 0)
