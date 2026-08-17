"""Seed-as-unit statistics for the rung-4 campaign (STATS-C1).

Each seed yields ONE summary number per metric (e.g. that seed's p50 latency, or its
recall); the CI is a bootstrap ACROSS those seed-level values -- never across the pooled
within-seed claims (that is the pseudoreplication the NPS review flagged). bootstrap_ci
is the primitive; TOST/Holm (mem_delta_metrics) consume its seed-level inputs.
"""
from __future__ import annotations

import statistics


def test_bootstrap_ci_brackets_mean_and_is_ordered():
    from meld.eval.campaign_stats import bootstrap_ci
    vals = [0.80, 0.82, 0.79, 0.85, 0.81, 0.83, 0.78, 0.84]
    point, lo, hi = bootstrap_ci(vals, n_boot=3000, alpha=0.05, agg="mean", seed=1)
    assert lo <= point <= hi
    assert abs(point - statistics.mean(vals)) < 0.02      # point estimate ~ the mean
    assert lo < hi                                        # non-degenerate CI


def test_bootstrap_ci_deterministic_under_seed():
    from meld.eval.campaign_stats import bootstrap_ci
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    a = bootstrap_ci(vals, n_boot=1000, seed=42)
    b = bootstrap_ci(vals, n_boot=1000, seed=42)
    assert a == b


def test_single_seed_is_degenerate_ci():
    from meld.eval.campaign_stats import bootstrap_ci
    point, lo, hi = bootstrap_ci([0.9], n_boot=100, seed=1)
    assert point == lo == hi == 0.9                       # one unit -> no spread (honest)


def test_median_aggregator_robust_to_outlier():
    from meld.eval.campaign_stats import bootstrap_ci
    vals = [1, 2, 3, 4, 100]
    point, _lo, _hi = bootstrap_ci(vals, agg="median", n_boot=3000, seed=1)
    assert 1 <= point <= 4                                # median ~3, not dragged by 100


def test_empty_raises_not_silent():
    from meld.eval.campaign_stats import bootstrap_ci
    import pytest
    with pytest.raises(ValueError):
        bootstrap_ci([], n_boot=100, seed=1)              # no silent NaN
