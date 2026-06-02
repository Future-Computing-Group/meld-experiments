"""Semantic Patch routing vs broadcast (§7-r ablation).

A Patch is delivered to a brain iff the brain is relevant to the Patch's claim
(semantic routing), versus broadcasting to every brain. Relevance is the same
sigma signal the merge decision procedure uses (smp.merge.signals), so a brain the
router selects is one whose content the claim would plausibly MERGE/RELATE into --
not REJECT. The routing win over broadcast is equal recall of the brains that need
a Patch at a far lower wasted-delivery rate. Built on the SMP sigma signal
(smp.merge.signals).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from smp.merge.signals import EmbeddingSimilarity


def score_brains(patch_claim: str, brains: Dict[str, List[str]], *,
                 encoder: str) -> Dict[str, float]:
    """Per-brain relevance score = the brain's most-similar held claim to the Patch
    (sigma over ``encoder``). This is the scoring seam the NR router never exposed:
    selection is score-RANKED, not a positional slice of an opaque match list."""
    es = EmbeddingSimilarity(encoder=encoder)
    return {bid: (max(es.sigma(patch_claim, t) for t in texts) if texts else 0.0)
            for bid, texts in brains.items()}


def semantic_route(patch_claim: str, brains: Dict[str, List[str]], *,
                   encoder: str, sigma_lo: float,
                   top_k: Optional[int] = None) -> Set[str]:
    """Deliver the Patch only to brains scoring at/above ``sigma_lo`` (the relatedness
    floor) -- i.e. brains that would find it relevant. If ``top_k`` is set, keep only
    the ``top_k`` highest-scoring of those (score-ranked, never a positional slice).
    A Patch relevant to no brain is delivered to none (no force-route fallback)."""
    scores = score_brains(patch_claim, brains, encoder=encoder)
    eligible = [(bid, s) for bid, s in scores.items() if s >= sigma_lo]
    eligible.sort(key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        eligible = eligible[:top_k]
    return {bid for bid, _ in eligible}


def broadcast_route(brains: Dict[str, List[str]]) -> Set[str]:
    """The baseline: deliver to every brain regardless of relevance."""
    return set(brains)


def routing_metrics(*, delivered: List[Set[str]],
                    needed: List[Set[str]]) -> Dict[str, float]:
    """Aggregate routing quality over a list of per-Patch (delivered, needed) sets.

    precision = correct deliveries / total deliveries; recall = needed brains that
    were delivered / total needed; wasted_delivery_rate = deliveries to brains that
    did not need the Patch / total deliveries (the cost semantic routing reduces)."""
    tp = sum(len(d & n) for d, n in zip(delivered, needed))
    d_total = sum(len(d) for d in delivered)
    n_total = sum(len(n) for n in needed)
    return {
        "precision": tp / d_total if d_total else 0.0,
        "recall": tp / n_total if n_total else 0.0,
        "wasted_delivery_rate": (d_total - tp) / d_total if d_total else 0.0,
        "n_delivered": d_total, "n_needed": n_total, "n_correct": tp,
    }


__all__ = ["score_brains", "semantic_route", "broadcast_route", "routing_metrics"]
