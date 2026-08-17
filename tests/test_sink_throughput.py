"""M1 sink-saturation harness (TDD).

Tests the saturation behaviour with a FakeMergeStep of known service time, so the
harness is verified independent of the real (embed+NLI) merge cost: below the sink's
capacity (1/service) achieved throughput tracks the offered rate; above it, throughput
saturates near 1/service while latency climbs. Run:
  PYTHONPATH=src python3 -m pytest tests/test_sink_throughput.py -q
"""
from meld.eval.sink_throughput import run_saturation, FakeMergeStep, saturation_knee


def test_throughput_tracks_offered_below_capacity_and_saturates_above():
    step = FakeMergeStep(service_s=0.01)          # single-worker capacity ~100 merges/s
    res = run_saturation(step, offered_rates=[20, 500], n_per_rate=60)
    # every offered claim is eventually merged
    assert res[20]["n"] == 60 and res[500]["n"] == 60
    # below capacity: achieved throughput tracks the offered rate (~20/s)
    assert 12 <= res[20]["merged_per_s"] <= 30, res[20]
    # above capacity: throughput saturates near 1/service (~100/s), nowhere near offered 500
    assert 60 <= res[500]["merged_per_s"] <= 150, res[500]
    assert res[500]["merged_per_s"] < 250, res[500]
    # saturation manifests as climbing queueing latency
    assert res[500]["p95_ms"] > res[20]["p95_ms"], (res[20], res[500])


def test_knee_is_max_achieved_throughput():
    step = FakeMergeStep(service_s=0.01)
    res = run_saturation(step, offered_rates=[10, 50, 300], n_per_rate=40)
    knee = saturation_knee(res)
    assert 60 <= knee <= 150, res        # the sink's capacity, ~100/s


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
