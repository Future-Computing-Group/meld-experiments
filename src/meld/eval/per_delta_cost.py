"""M2-edge: per-delta merge-cost decomposition (embedding vs NLI).

The per-delta cost on real edge-class hardware, decomposed into its components, is what
substantiates "edge-deployable" on the costliest step. This times the embedding step and
the NLI step separately over N claim
pairs after a warmup, reporting mean +/- std (ms) each and the per-delta total. The HPC-GPU
half is the same harness on an HPC GPU node.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable, Dict


def _stats(xs):
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def measure_per_delta_cost(embed_fn: Callable[[], object], nli_fn: Callable[[], object],
                           n: int, warmup: int = 3) -> Dict:
    """Time ``embed_fn`` and ``nli_fn`` (one merge's embedding and NLI step) over ``n``
    iterations after ``warmup`` warm calls; report mean/std ms per component and the total."""
    for _ in range(warmup):
        embed_fn(); nli_fn()
    embeds, nlis, totals = [], [], []
    for _ in range(n):
        t0 = time.perf_counter()
        embed_fn()
        t1 = time.perf_counter()
        nli_fn()
        t2 = time.perf_counter()
        embeds.append((t1 - t0) * 1000.0)
        nlis.append((t2 - t1) * 1000.0)
        totals.append((t2 - t0) * 1000.0)
    em, es = _stats(embeds)
    nm, ns = _stats(nlis)
    tm, ts = _stats(totals)
    return {"embed_ms_mean": round(em, 2), "embed_ms_std": round(es, 2),
            "nli_ms_mean": round(nm, 2), "nli_ms_std": round(ns, 2),
            "total_ms_mean": round(tm, 2), "total_ms_std": round(ts, 2), "n": n}


__all__ = ["measure_per_delta_cost"]
