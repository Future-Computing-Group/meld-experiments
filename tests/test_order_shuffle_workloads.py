"""S3: shape tests for the two order-shuffle workload builders. Hashing
encoder for speed -- the real minilm/cross-encoder run is
scripts/run_order_shuffle.py, not this test."""
from __future__ import annotations

from smp.eval.order_shuffle_workloads import (build_gold_derived_workload,
                                               build_running_example_workload)
from smp.proto import MemoryUnit


def test_running_example_workload_shape():
    workload, seed_units = build_running_example_workload()
    assert len(seed_units) == 1
    assert isinstance(seed_units[0], MemoryUnit)
    assert len(workload) == 4
    units = [w for w in workload if isinstance(w, MemoryUnit)]
    events = [w for w in workload if not isinstance(w, MemoryUnit)]
    assert len(units) == 3         # messages 1-3 (claim deltas)
    assert len(events) == 1        # message 4 (status overrule)
    ck, link_id, effect = events[0]
    assert isinstance(ck, bytes) and ck
    assert effect == "OVERRULED"
    # message 1 and message 2 carry the SAME text, different cells -- the
    # cross-boundary pair the D2 scoped key distinguishes.
    assert units[0].claims[0].text == units[1].claims[0].text
    assert units[0].cell != units[1].cell


def test_gold_derived_workload_shape():
    workload = build_gold_derived_workload(encoder="hashing", n_sources=3)
    assert 12 <= len(workload) <= 20
    assert all(isinstance(u, MemoryUnit) for u in workload)
    texts = [u.claims[0].text for u in workload]
    assert len(texts) == len(set(texts))   # every seed contributes distinct content
