#!/usr/bin/env python3
"""Tests for the NPS-aligned routing baselines + the 4-way ablation invariants.

Run:  PYTHONPATH=src:cross-tier python3 -m pytest cross-tier/test_routing.py -q
"""
import os, random, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "cross-tier"))
from meld.sync.router import (broadcast_route, random_route, round_robin_route,  # noqa: E402
                             routing_metrics)

BRAINS = {"a": ["x"], "b": ["y"], "c": ["z"], "d": ["w"]}


def test_broadcast_reaches_all():
    assert broadcast_route(BRAINS) == set(BRAINS)


def test_random_delivers_exactly_k_and_is_seeded():
    rng = random.Random(7)
    d = random_route(BRAINS, k=2, rng=rng)
    assert len(d) == 2 and d <= set(BRAINS)
    # deterministic under a fixed seed
    assert random_route(BRAINS, k=2, rng=random.Random(7)) == random_route(BRAINS, k=2, rng=random.Random(7))


def test_round_robin_rotates_deterministically():
    assert round_robin_route(BRAINS, k=2, offset=0) == {"a", "b"}
    assert round_robin_route(BRAINS, k=2, offset=3) == {"d", "a"}  # wraps
    assert round_robin_route(BRAINS, k=1, offset=2) == {"c"}


def test_metrics_broadcast_single_need_wastes_n_minus_1_over_n():
    # 1 needed of 4 brains, broadcast -> wasted = 3/4
    delivered = [set(BRAINS)]
    needed = [{"a"}]
    m = routing_metrics(delivered=delivered, needed=needed)
    assert m["recall"] == 1.0
    assert abs(m["wasted_delivery_rate"] - 0.75) < 1e-9


def test_ablation_semantic_dominates():
    """On a separable workload, semantic = full recall at zero waste; random/round-robin
    spend the same budget at lower recall + higher waste."""
    from routing_ablation import run_ablation
    res = run_ablation()
    assert res["broadcast"]["recall"] == 1.0
    assert res["semantic"]["recall"] == 1.0
    assert res["semantic"]["wasted_delivery_rate"] == 0.0
    assert res["random"]["recall"] < res["semantic"]["recall"]
    assert res["round_robin"]["recall"] < res["semantic"]["recall"]
    assert res["semantic"]["wasted_delivery_rate"] < res["broadcast"]["wasted_delivery_rate"]
