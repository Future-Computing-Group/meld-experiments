"""M1: global-brain sink saturation/throughput harness.

The throughput-limiting cost at the global merge sink is the per-delta merge compute
(one embedding + one NLI call + classify + status-CRDT update), not the transport. This
harness drives a single-worker sink with an offered load at controlled rates and reports
achieved throughput and publish-to-merge latency, so the analytic sink bound (manuscript
Section 9.2) becomes an evidenced saturation knee.

Below the sink's capacity (1/service-time) achieved throughput tracks the offered rate;
above it, throughput saturates near 1/service while queueing latency climbs. The harness is
transport- and model-agnostic: ``merge_step`` is the per-delta callable (a FakeMergeStep of
known latency for tests, the real embed+NLI+classify+status path for the measurement).
"""
from __future__ import annotations

import math
import queue
import threading
import time
from typing import Callable, Dict, List, Optional


class FakeMergeStep:
    """A merge step of fixed service time, for verifying the harness without the real
    (embed+NLI) cost. Capacity of a single-worker sink is then ~1/service_s merges/s."""

    def __init__(self, service_s: float):
        self.service_s = service_s

    def __call__(self, payload) -> None:  # noqa: ANN001
        time.sleep(self.service_s)


def _pct(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = max(0, min(len(vals) - 1, int(math.ceil(p / 100.0 * len(vals)) - 1)))
    return round(vals[idx], 2)


def run_saturation(merge_step: Callable, offered_rates: List[float], n_per_rate: int,
                   claim_source: Optional[Callable[[int], object]] = None) -> Dict:
    """For each offered rate (claims/s), inject ``n_per_rate`` claims spaced 1/rate apart
    into a queue drained by a single sink worker running ``merge_step``; report achieved
    throughput (merges/s over the inject-to-last-merge window), publish-to-merge p50/p95
    (ms), and the merged count. ``claim_source(i)`` produces the i-th payload (default: i)."""
    claim_source = claim_source or (lambda i: i)
    out: Dict[float, Dict] = {}
    for r in offered_rates:
        q: "queue.Queue" = queue.Queue()
        latencies: List[float] = []
        merged = [0]
        last_done = [0.0]

        def sink() -> None:
            while True:
                item = q.get()
                if item is None:
                    q.task_done()
                    break
                inject_t, payload = item
                merge_step(payload)
                now = time.perf_counter()
                latencies.append((now - inject_t) * 1000.0)
                merged[0] += 1
                last_done[0] = now
                q.task_done()

        worker = threading.Thread(target=sink, daemon=True)
        worker.start()
        interval = (1.0 / r) if r > 0 else 0.0
        start = time.perf_counter()
        for i in range(n_per_rate):
            target = start + i * interval
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            q.put((time.perf_counter(), claim_source(i)))
        q.put(None)
        worker.join()
        window = max(last_done[0] - start, 1e-9)
        out[r] = {"merged_per_s": round(merged[0] / window, 2),
                  "p50_ms": _pct(latencies, 50), "p95_ms": _pct(latencies, 95),
                  "n": merged[0], "offered_rate": r}
    return out


def saturation_knee(res: Dict) -> float:
    """The sink's capacity: the maximum achieved throughput across offered rates."""
    return max((v["merged_per_s"] for v in res.values()), default=0.0)


__all__ = ["FakeMergeStep", "run_saturation", "saturation_knee"]
