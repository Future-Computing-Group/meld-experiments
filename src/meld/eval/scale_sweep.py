"""S-3: node-scaling of sync overhead --- how delivery cost behaves as the cluster grows.

The manuscript argues structurally that the semantic-routing advantage "widens with the number
of brains" and that "sink load grows with the rate of novel claims, not the number of brains",
but sweeps only N<=4. This harness sweeps N over a wide range and measures, per published
claim, how many brains each policy delivers to: broadcast delivers to all N (O(N)); semantic
routing delivers only to the brains that need the claim (O(needed), flat in N) at maintained
routing recall. So the per-claim sync overhead of MELD's transport is constant in cluster
size while broadcast's grows linearly --- the scaling face of the Section~7.6 routing result.

Domains are constructed to be cleanly separable (each carries a unique index keyword) so the
sweep isolates the \emph{delivery-count} scaling of the routing policy, not the encoder's
discrimination at scale (which the M-2 classification and Section~7.6 ablation address); recall
is reported so routing correctness is transparent at every N.
"""
from __future__ import annotations

from typing import Dict, List, Set

from meld.sync.router import semantic_route, broadcast_route, routing_metrics


def make_brains(n: int) -> Dict[str, List[str]]:
    """``n`` brains, each owning one distinct, separable interest domain."""
    return {f"b{i}": [f"domain number {i}: subject area {i} and its terminology"]
            for i in range(n)}


def make_workload(n: int, per_domain: int = 4):
    """``per_domain`` claims for each of the ``n`` domains; each claim is needed by exactly
    the brain owning its domain (ground-truth target by construction)."""
    items = []
    for i in range(n):
        for _ in range(per_domain):
            items.append((f"a claim concerning domain number {i}, subject area {i}", f"b{i}"))
    return items


def summarize_scale(n_brains: int, sem_delivered: List[Set[str]],
                    bc_delivered: List[Set[str]], needed: List[Set[str]]) -> Dict:
    """Pure: per-claim delivery overhead for semantic vs broadcast, plus routing recall."""
    sm = routing_metrics(delivered=sem_delivered, needed=needed)
    bm = routing_metrics(delivered=bc_delivered, needed=needed)
    nc = len(needed)
    return {
        "n_brains": n_brains,
        "n_claims": nc,
        "recall_semantic": round(sm["recall"], 3),
        "deliv_per_claim_semantic": round(sm["n_delivered"] / nc, 3) if nc else 0.0,
        "deliv_per_claim_broadcast": round(bm["n_delivered"] / nc, 3) if nc else 0.0,
        "waste_semantic": round(sm["wasted_delivery_rate"], 3),
        "waste_broadcast": round(bm["wasted_delivery_rate"], 3),
        "broadcast_over_semantic": round(bm["n_delivered"] / max(sm["n_delivered"], 1), 2),
    }


def scale_row(n: int, *, per_domain: int = 4, encoder: str = "minilm",
              sigma_lo: float, es=None) -> Dict:
    """One N: route a workload by semantic (top-1) and by broadcast over ``n`` brains."""
    brains = make_brains(n)
    wl = make_workload(n, per_domain=per_domain)
    needed = [{nb} for _, nb in wl]
    sem = [semantic_route(text, brains, encoder=encoder, sigma_lo=sigma_lo, top_k=1, es=es)
           for text, _ in wl]
    bc = [broadcast_route(brains) for _ in wl]
    return summarize_scale(n, sem, bc, needed)


__all__ = ["make_brains", "make_workload", "summarize_scale", "scale_row"]
