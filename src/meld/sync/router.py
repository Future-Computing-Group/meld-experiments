"""Semantic Patch routing vs broadcast (§7-r ablation).

A Patch is delivered to a brain iff the brain is relevant to the Patch's claim
(semantic routing), versus broadcasting to every brain. Relevance is the same
sigma signal the merge decision procedure uses (meld.merge.signals), so a brain the
router selects is one whose content the claim would plausibly MERGE/RELATE into --
not REJECT. The routing win over broadcast is equal recall of the brains that need
a Patch at a far lower wasted-delivery rate. Built on the MELD sigma signal
(meld.merge.signals).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from meld.merge.signals import EmbeddingSimilarity


def score_brains(patch_claim: str, brains: Dict[str, List[str]], *,
                 encoder: str, es: "EmbeddingSimilarity" = None) -> Dict[str, float]:
    """Per-brain relevance score = the brain's most-similar held claim to the Patch
    (sigma over ``encoder``). Selection is score-RANKED. Pass a shared ``es`` to reuse its
    embedding cache across many claims in one publish (else peer content is re-encoded per
    claim -> O(claims x peer-content) encodings, which times out at scale)."""
    es = es or EmbeddingSimilarity(encoder=encoder)
    return {bid: (max(es.sigma(patch_claim, t) for t in texts) if texts else 0.0)
            for bid, texts in brains.items()}


def select_peers(scores: Dict[str, float], *, sigma_lo: float,
                 top_k: Optional[int] = None, margin: Optional[float] = None,
                 gap: bool = False) -> Set[str]:
    """Selection over per-brain scores. Eligible = score at/above ``sigma_lo`` (the floor).
    Budget-free cuts on the eligible list, sorted by score: ``margin`` keeps scores at/above
    max - margin; ``gap`` cuts at the largest drop between consecutive scores, the floor
    counting as the bottom score (so a list far above the floor is kept whole). ``top_k``
    (the budget) is applied last."""
    eligible = [(bid, s) for bid, s in scores.items() if s >= sigma_lo]
    eligible.sort(key=lambda kv: kv[1], reverse=True)
    if margin is not None and eligible:
        eligible = [(b, s) for b, s in eligible if s >= eligible[0][1] - margin]
    if gap and eligible:
        vals = [s for _, s in eligible] + [sigma_lo]
        drops = [vals[i] - vals[i + 1] for i in range(len(eligible))]
        eligible = eligible[:drops.index(max(drops)) + 1]
    if top_k is not None:
        eligible = eligible[:top_k]
    return {bid for bid, _ in eligible}


def semantic_route(patch_claim: str, brains: Dict[str, List[str]], *,
                   encoder: str, sigma_lo: float,
                   top_k: Optional[int] = None,
                   es: "EmbeddingSimilarity" = None,
                   margin: Optional[float] = None, gap: bool = False) -> Set[str]:
    """Deliver the Patch only to brains scoring at/above ``sigma_lo`` (the relatedness
    floor) -- i.e. brains that would find it relevant. If ``top_k`` is set, keep only
    the ``top_k`` highest-scoring of those (score-ranked, never a positional slice).
    ``margin`` / ``gap`` are the budget-free cuts of select_peers.
    A Patch relevant to no brain is delivered to none (no force-route fallback)."""
    scores = score_brains(patch_claim, brains, encoder=encoder, es=es)
    return select_peers(scores, sigma_lo=sigma_lo, top_k=top_k, margin=margin, gap=gap)


def broadcast_route(brains: Dict[str, List[str]]) -> Set[str]:
    """The baseline: deliver to every brain regardless of relevance."""
    return set(brains)


def random_route(brains: Dict[str, List[str]], *, k: int, rng) -> Set[str]:
    """Baseline (NPS ``random`` placement): deliver to a random k-subset of brains,
    blind to relevance. ``k`` is the delivery budget (typically the needed-count, so
    the comparison to semantic routing is at a matched budget)."""
    bids = list(brains)
    k = max(0, min(k, len(bids)))
    return set(rng.sample(bids, k))


def round_robin_route(brains: Dict[str, List[str]], *, k: int, offset: int) -> Set[str]:
    """Baseline (NPS ``round-robin`` placement): deliver to k brains in rotation from
    ``offset``, blind to relevance. Deterministic given the offset sequence."""
    bids = sorted(brains)
    if not bids:
        return set()
    k = max(0, min(k, len(bids)))
    return {bids[(offset + i) % len(bids)] for i in range(k)}


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


__all__ = ["score_brains", "select_peers", "semantic_route", "broadcast_route", "random_route",
           "round_robin_route", "routing_metrics"]
