"""S3: candidate-cost instrumentation, WITHOUT changing merge_on_receive's
behaviour.

``merge_on_receive`` looks up ``_best_candidate`` as a module GLOBAL at call
time (both are defined at module scope in
``smp.protocol.merge_on_receive``), so monkeypatching that module attribute
for the duration of a context intercepts every call the real function makes
-- same arguments in, same return value out, just timed and sized -- with no
edit to merge_on_receive.py. Records, per call, the candidate-set size (the
number of local claims scanned) and the scan's wall-clock latency.
"""
from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class CandidateCostRecorder:
    candidate_set_sizes: List[int] = field(default_factory=list)
    scan_latencies_ms: List[float] = field(default_factory=list)

    def record(self, size: int, latency_s: float) -> None:
        self.candidate_set_sizes.append(size)
        self.scan_latencies_ms.append(latency_s * 1000.0)

    def summary(self) -> Dict[str, float]:
        if not self.candidate_set_sizes:
            return {"n_calls": 0}
        return {
            "n_calls": len(self.candidate_set_sizes),
            "candidate_set_size_median": statistics.median(self.candidate_set_sizes),
            "candidate_set_size_p95": _percentile(self.candidate_set_sizes, 0.95),
            "scan_latency_ms_median": statistics.median(self.scan_latencies_ms),
            "scan_latency_ms_p95": _percentile(self.scan_latencies_ms, 0.95),
        }


def _percentile(xs: Iterable[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def pooled_summary(recorders: Iterable[CandidateCostRecorder]) -> Dict[str, float]:
    """Pool call-level samples across many recorders (e.g. one per ordering)
    into a single distribution, so the reported median/p95 is over every
    ``_best_candidate`` call in the run, not a median of per-ordering medians."""
    pooled = CandidateCostRecorder()
    for r in recorders:
        pooled.candidate_set_sizes.extend(r.candidate_set_sizes)
        pooled.scan_latencies_ms.extend(r.scan_latencies_ms)  # already ms
    return pooled.summary()


@contextmanager
def instrument_best_candidate(recorder: CandidateCostRecorder):
    """Scoped monkeypatch of ``_best_candidate``; always restored on exit."""
    import smp.protocol.merge_on_receive as _mor

    original = _mor._best_candidate

    def wrapped(ic, local_claims, es, sigma_lo):
        t0 = time.perf_counter()
        result = original(ic, local_claims, es, sigma_lo)
        t1 = time.perf_counter()
        recorder.record(len(local_claims), t1 - t0)
        return result

    _mor._best_candidate = wrapped
    try:
        yield recorder
    finally:
        _mor._best_candidate = original


def novel_rate(patches, *, exclude_gate: str = "status-link") -> float:
    """INSERT fraction over a Patch stream, derived from the decisions
    (excludes synthetic status-link deliveries, which never run the
    classifier and so are not part of the admission procedure's novel rate)."""
    real = [p for p in patches if exclude_gate not in p.gates_that_fired]
    if not real:
        return float("nan")
    return sum(1 for p in real if p.decision == "INSERT") / len(real)


__all__ = ["CandidateCostRecorder", "instrument_best_candidate", "novel_rate",
           "pooled_summary"]
