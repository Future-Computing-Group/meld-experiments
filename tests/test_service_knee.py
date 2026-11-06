"""Service throughput knee over the brain's /patch (scripts/run_service_knee.py).

A donor replica (same key, same M_nli) pre-signs deltas; per offered rate the target is reset and
pre-filled to a fixed store size, then one envelope per /patch is offered at that rate through
eval/sink_throughput.run_saturation. Against an in-process service whose admission takes a known
service time, the knee sits at the measured 1/service (the existing band test), and every row
carries the target's models, epoch ids and the offered rate.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _target(service_s, times):
    from meld.sync.node import BrainNode
    from meld.sync.service import BrainService

    def slow(kind, obj):
        if kind == "D":
            t = time.perf_counter()
            time.sleep(service_s)
            times.append(time.perf_counter() - t)
    return BrainService(BrainNode("T", encoder="hashing", nli="lexical", admit=slow))


def test_knee_at_one_over_service_time_and_rows_are_stamped():
    import run_service_knee as K
    from meld.eval.sink_throughput import saturation_knee
    envs = K.sign_envelopes([f"claim number {i} about topic {i % 7}" for i in range(140)],
                            encoder="hashing", nli="lexical", key=_target(0, []).node.key)
    times = []
    svc = _target(0.01, times)
    post = lambda path, body=b"": svc.handle("POST", path, body)        # noqa: E731
    get = lambda path: json.loads(svc.handle("GET", path)[1])           # noqa: E731
    rows = K.knee(post, get, envs[:20], envs[20:], rates=[20, 500], n_per_rate=60)
    cap = len(times) / sum(times)
    res = {r["offered_rate"]: r for r in rows}
    assert 0.7 * cap <= saturation_knee(res) <= 1.1 * cap, (cap, res)
    assert all(r["n"] == 60 and r["prefill"] == 20 and r["live_before"] == 20 for r in rows)
    assert all({"encoder", "m_nli", "epoch_ids", "git_sha", "host_cpu"} <= set(r) for r in rows)


def test_a_refused_envelope_fails_the_run_rather_than_counting_as_a_merge():
    import pytest
    import run_service_knee as K
    envs = K.sign_envelopes(["a", "b", "c"], encoder="hashing", nli="lexical", key=b"another-key")
    svc = _target(0, [])
    with pytest.raises(RuntimeError, match="403"):
        K.knee(lambda p, b=b"": svc.handle("POST", p, b), lambda p: json.loads(svc.handle("GET", p)[1]),
               envs[:1], envs[1:], rates=[50], n_per_rate=2)
