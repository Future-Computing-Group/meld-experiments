"""Merge-threshold calibration on a labeled claim-pair set (manuscript Section 4 /
M-2). The thresholds (theta_merge, sigma_lo) are CALIBRATED from data, never
eyeballed and never inverted: this module derives them by maximizing decision
accuracy on a held-out set of labeled claim pairs, then breaks ties by maximizing
the margin (distance from the chosen thresholds to the nearest observed
similarity), so the bands sit in the middle of the gaps between classes.

The fixture below is a small, deliberately CLEAR starter set spanning the three
similarity-driven outcomes (CONFLICT is a separate chi overlay, not a
sigma-threshold question, so the pairs are all non-contradicting):

    MERGE       -- genuine paraphrases (same claim, different surface form)
    RELATE      -- related but distinct claims (overlap, not identity)
    NO_RELATION -- unrelated claims (no actionable relation)

The full M-2 experiment should expand this set; the routine and the property
tests (tests/test_threshold_calibration.py) are the reproducible provenance for
whatever values ship in thresholds.yaml.
"""
from __future__ import annotations

from typing import List, Tuple

from meld.config import Thresholds
from meld.merge.classifier import classify
from meld.merge.signals import EmbeddingSimilarity

# (gold_decision, claim_a, claim_b) -- all non-contradicting (chi != contradicts),
# same frame (ctx_ok=True), distinct claim keys (kappa=0); the gold is the
# sigma-band outcome the classifier should produce.
CALIBRATION_PAIRS: List[Tuple[str, str, str]] = [
    ("MERGE", "Paris is the capital of France",
     "Paris is the capital city of France"),
    ("MERGE", "Mount Everest is the tallest mountain on Earth",
     "The tallest mountain on Earth is Mount Everest"),
    ("MERGE", "The novel was written by George Orwell",
     "George Orwell wrote the novel"),
    ("RELATE", "Paris is the capital of France",
     "France is a country in western Europe"),
    ("RELATE", "Mount Everest is the tallest mountain",
     "Mount Everest is located in the Himalayas"),
    ("RELATE", "George Orwell wrote the book 1984",
     "The book 1984 is a dystopian novel"),
    ("NO_RELATION", "Paris is the capital of France",
     "The cat sat quietly on the mat"),
    ("NO_RELATION", "Mount Everest is the tallest mountain",
     "I really enjoy playing tennis outdoors"),
    ("NO_RELATION", "George Orwell wrote the book 1984",
     "Photosynthesis occurs in green plants"),
]


def calibrate_thresholds(pairs, *, encoder: str, nli: str,
                         r_min: float, tau: float, step: float = 0.01) -> Thresholds:
    """Grid-search (theta_merge, sigma_lo) with sigma_lo < theta_merge, maximizing
    classification accuracy on ``pairs``, ties broken by maximum margin (the
    min distance from any observed sigma to the nearest of the two thresholds)."""
    es = EmbeddingSimilarity(encoder=encoder)
    samples = [(es.sigma(a, b), gold) for gold, a, b in pairs]
    n_steps = int(round(1.0 / step))
    grid = [round(i * step, 4) for i in range(1, n_steps)]
    best_key = None
    best = None
    for tm in grid:
        for sl in grid:
            if not (sl < tm):
                continue
            cfg = Thresholds(theta_merge=tm, sigma_lo=sl, r_min=r_min, tau=tau)
            acc = sum(
                classify(cfg=cfg, kappa=0, sigma=s, chi="neutral", ctx_ok=True) == g
                for s, g in samples
            )
            margin = min(min(abs(s - tm), abs(s - sl)) for s, _ in samples)
            key = (acc, margin)
            if best_key is None or key > best_key:
                best_key, best = key, cfg
    if best is None:  # pragma: no cover - grid always non-empty
        raise RuntimeError("calibration grid produced no candidate")
    return best


__all__ = ["CALIBRATION_PAIRS", "calibrate_thresholds"]
