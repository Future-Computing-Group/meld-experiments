"""Pure decision core for the multi-host partition-heal convergence run.

This is the reusable, scrubbed logic the over-the-wire orchestrator replays: it carries no
host aliases, IPs, HTTP, or tc -- only the seed-driven delivery order, the per-seed metric
computation, and the seed-as-unit aggregation. The orchestrator wires these to real brains
(HTTP) and a real, healable partition (tc); the local-docker smoke wires them to the compose
stack with a delivery-based partition seam.

Model (status-CRDT, a 2|2 split holding one claim): while partitioned, side_a applies only
overrule link "La" (effect_a) and side_b only "Lb" (effect_b), so the two sides DIVERGE.
After heal, BOTH links are delivered to EVERY brain in a per-brain SEEDED order; the
status-CRDT join of {La->effect_a, Lb->effect_b} is order-independent and identical for every
brain, so agreement reconverges to 1.0. The last-writer-wins baseline is an ANALYTICAL
counterfactual read off the SAME delivery order: each brain keeps the effect of the LAST link
delivered to it, so the two sides stay split iff the effects differ. No separate LWW backend
is run.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Tuple

from smp.eval.campaign_stats import bootstrap_ci

LINK_A = "La"
LINK_B = "Lb"


def agreement(statuses: List[str]) -> float:
    """Fraction of brains agreeing on the modal status (1.0 == unanimous)."""
    return max(Counter(statuses).values()) / len(statuses) if statuses else 1.0


def seeded_delivery_orders(brains: List[str], links: List[Tuple[str, str]], *,
                           seed: int) -> Dict[str, List[Tuple[str, str]]]:
    """Per-brain (link_id, effect) delivery order for the resync, permuted by a seeded RNG.

    Same seed -> identical orders (the order-independence demonstration is reproducible).
    Each brain gets its own permutation so no single global order can be credited for the
    CRDT reconvergence.
    """
    rng = random.Random(seed)
    orders: Dict[str, List[Tuple[str, str]]] = {}
    for b in brains:
        order = list(links)
        rng.shuffle(order)
        orders[b] = order
    return orders


def per_seed_result(*, during_statuses: Dict[str, str],
                    delivery_orders: Dict[str, List[Tuple[str, str]]],
                    effect_a: str, effect_b: str) -> Dict[str, float]:
    """The three metrics for one seed from the measured during-statuses, the seeded delivery
    order, and the two effects.

    crdt_healed is the agreement over the status-CRDT join: every brain receives BOTH links,
    so its join is identical regardless of delivery order -> unanimous (1.0). lww_healed is
    the agreement over each brain's last-writer-wins survivor, i.e. the effect of the LAST
    (link_id, effect) in that brain's actual delivery order -- the analytical LWW
    counterfactual, read off the SAME order the CRDT path used, not a separate backend.
    """
    agreement_during = agreement(list(during_statuses.values()))

    join_token = "|".join(sorted([effect_a, effect_b]))     # order-independent join, all brains
    crdt_healed = agreement([join_token for _ in delivery_orders])

    lww_survivors = [order[-1][1] for order in delivery_orders.values()]
    lww_healed = agreement(lww_survivors)

    return {"agreement_during": agreement_during,
            "crdt_healed": crdt_healed,
            "lww_healed": lww_healed}


def aggregate_results(per_seed: List[Dict[str, float]]) -> Dict:
    """Mean + seed-as-unit percentile bootstrap CI per metric, plus reconvergence fractions.

    crdt_reconverges / lww_reconverges = fraction of seeds whose healed agreement == 1.0
    (the headline: the CRDT reconverges for every seed; LWW does not when the effects differ).
    """
    out: Dict = {"n_seeds": len(per_seed)}
    for metric in ("agreement_during", "crdt_healed", "lww_healed"):
        vals = [float(r[metric]) for r in per_seed]
        pt, lo, hi = bootstrap_ci(vals, n_boot=10000, alpha=0.05, agg="mean", seed=0)
        out[metric] = {"mean": pt, "lo": lo, "hi": hi}
    n = len(per_seed) or 1
    out["crdt_reconverges"] = sum(1 for r in per_seed if float(r["crdt_healed"]) == 1.0) / n
    out["lww_reconverges"] = sum(1 for r in per_seed if float(r["lww_healed"]) == 1.0) / n
    return out


__all__ = ["agreement", "seeded_delivery_orders", "per_seed_result", "aggregate_results",
           "LINK_A", "LINK_B"]
