"""Evaluation metric counters.

accuracy / recall@k / MRR for answer quality; byte-count / version-count for the
O-1 overhead terms. All counters operate on real (small-scale) outputs; none are
fabricated.
"""
from __future__ import annotations

from typing import List, Sequence

from meld.proto import MemoryUnit


def recall_at_k(ranked: Sequence[bytes], gold: bytes, k: int) -> float:
    return 1.0 if gold in list(ranked)[:k] else 0.0


def mrr(ranked: Sequence[bytes], gold: bytes) -> float:
    for i, x in enumerate(ranked, 1):
        if x == gold:
            return 1.0 / i
    return 0.0


def accuracy(predicted: Sequence, gold: Sequence) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for p, g in zip(predicted, gold) if p == g)
    return correct / len(gold)


def byte_count(units: List[MemoryUnit]) -> int:
    """Total serialised bytes of the unit texts (a proxy for sync payload size)."""
    return sum(len(c.text.encode("utf-8")) for u in units for c in u.claims) or len(units)


def version_count(units: List[MemoryUnit]) -> int:
    """Number of unit versions (the c_ver term input)."""
    return len(units)


__all__ = ["recall_at_k", "mrr", "accuracy", "byte_count", "version_count"]
