"""Seed-as-unit bootstrap statistics for the rung-4 multi-host campaign.

The experiment unit is the SEED: each seed contributes one summary number per metric
(its within-seed p50 latency, recall, agreement, ...). Confidence intervals are a
bootstrap ACROSS those seed-level values -- never across the pooled within-seed claims,
which would inflate n (pseudoreplication). Pair with
mem_delta_metrics.tost_equivalent / holm_correct for non-inferiority + FWER control.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def bootstrap_ci(seed_values: List[float], *, n_boot: int = 10000,
                 alpha: float = 0.05, agg: str = "mean",
                 seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the ``agg`` (mean|median) of seed-level values.

    Returns (point, lo, hi). One value -> a degenerate (point==lo==hi) interval (honest:
    no across-seed spread is estimable from a single unit). Empty input raises (no silent
    NaN). The resample is over the seed-level units, so this is seed-as-unit by construction.
    """
    if not seed_values:
        raise ValueError("bootstrap_ci needs >=1 seed-level value (no silent NaN)")
    vals = np.asarray(seed_values, dtype=float)
    aggfn = {"mean": np.mean, "median": np.median}.get(agg)
    if aggfn is None:
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
    point = float(aggfn(vals))
    if vals.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = aggfn(vals[idx], axis=1)
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi


__all__ = ["bootstrap_ci"]
