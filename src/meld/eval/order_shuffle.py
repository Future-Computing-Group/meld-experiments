"""S3: the order-shuffle graph-structure convergence experiment.

Delivers ONE workload of deltas to a fresh sink Brain in K shuffled orders
(seeded, reproducible) and compares the resulting graph-state fingerprints
(graph_fingerprint.py) across orders. Drives the REAL merge path --
``meld.protocol.merge_on_receive.merge_on_receive`` into a ``Brain``, the same
substrate ``meld.eval.topology.run_funnel_merge`` uses for one delivery order
-- not the status-only replay in ``crdt_convergence.py``.

A workload item is either a claim delta (a ``MemoryUnit``, admitted through
``merge_on_receive``) or a bare status event -- ``(claim_key, link_id,
effect)`` -- delivered directly, the way an overrule/supersede link can reach
a receiver independent of the claim it targets. The status-CRDT's
order-independence is already established (crdt_convergence.py); interleaving
a status event into the SAME shuffle as the claim deltas here is a live check
that ``status_map`` stays invariant even while the claim-driven components can
diverge.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple, Union

from meld.brain.store import Brain
from meld.config import Thresholds
from meld.eval.candidate_cost import (CandidateCostRecorder, instrument_best_candidate,
                                     novel_rate, pooled_summary)
from meld.eval.graph_fingerprint import diff_fingerprints, fingerprint
from meld.merge.signals import EmbeddingSimilarity
from meld.protocol.merge_on_receive import merge_on_receive
from meld.protocol.messages import Patch
from meld.proto import Endpoint, MemoryUnit

StatusEvent = Tuple[bytes, str, str]     # (claim_key, link_id, effect)
WorkloadItem = Union[MemoryUnit, StatusEvent]


def _local_live_claims(brain: Brain) -> List[MemoryUnit]:
    # the §5.3/§5.5 candidacy filter: only status-live local units are
    # admission targets (matches meld.agent.memory.SMPMemory.write_unit).
    return [u for u in brain.extract_delta()
            if isinstance(u, MemoryUnit) and brain._unit_is_live(u)]


def _deliver(brain: Brain, item: WorkloadItem, *, cfg: Thresholds,
            es: EmbeddingSimilarity, nli: str) -> List[Patch]:
    if isinstance(item, MemoryUnit):
        patches = merge_on_receive(item, _local_live_claims(brain), cfg=cfg, es=es, nli=nli)
        for p in patches:
            brain.apply(p)
        return patches
    claim_key, link_id, effect = item
    patch = Patch(decision="STATUS", target=Endpoint(claim_key=claim_key),
                 emitted=[("STATUS", claim_key, (link_id, effect))],
                 gates_that_fired=["status-link"], version=1, signature=b"s")
    brain.apply(patch)
    return [patch]


def _seed_brain(seed_units: Sequence[MemoryUnit], *, encoder: str, nli: str,
                cfg: Thresholds, cache_dir: str | None) -> Tuple[Brain, EmbeddingSimilarity]:
    brain = Brain(encoder=encoder, cache_dir=cache_dir)
    es = EmbeddingSimilarity(encoder=encoder, cache_dir=cache_dir)
    for su in seed_units:
        _deliver(brain, su, cfg=cfg, es=es, nli=nli)
    return brain, es


def run_order_shuffle(workload: Sequence[WorkloadItem], *, k_orderings: int, seed: int,
                      encoder: str, nli: str, cfg: Thresholds,
                      probe_queries: Sequence[str] = (),
                      seed_units: Sequence[MemoryUnit] = (),
                      k_retrieval: int = 3, cache_dir: str | None = None) -> Dict[str, Any]:
    """Deliver ``workload`` to a fresh sink brain in ``k_orderings`` random
    orders (one shared ``random.Random(seed)`` advanced across draws, so the
    run is reproducible end to end). ``seed_units`` are applied identically
    and un-shuffled before the workload in every ordering (a receiver's fixed
    starting state, e.g. the running example's "B holds c1 active").
    Returns a report: distinct-fingerprint counts per component and jointly,
    the concrete diff for every divergent component, retrieval agreement over
    ``probe_queries`` (the same status-aware graph-walk read the eval uses),
    and the pooled candidate-cost distribution."""
    rng = random.Random(seed)
    n = len(workload)
    orderings: List[Dict[str, Any]] = []
    fps = []
    recorders: List[CandidateCostRecorder] = []

    for i in range(k_orderings):
        order = list(range(n))
        rng.shuffle(order)
        brain, es = _seed_brain(seed_units, encoder=encoder, nli=nli, cfg=cfg,
                                cache_dir=cache_dir)
        recorder = CandidateCostRecorder()
        all_patches: List[Patch] = []
        with instrument_best_candidate(recorder):
            for idx in order:
                all_patches.extend(_deliver(brain, workload[idx], cfg=cfg, es=es, nli=nli))
        fp = fingerprint(brain)
        fps.append(fp)
        recorders.append(recorder)
        retrieval = {q: [uid.hex() for uid, _s in
                         brain.graph_walk_read(q, encoder=encoder, k=k_retrieval)]
                    for q in probe_queries}
        real_patches = [p for p in all_patches if "status-link" not in p.gates_that_fired]
        orderings.append({
            "ordering_index": i, "order": order,
            "joint_digest": fp.joint_digest,
            "component_digests": dict(fp.component_digests),
            "decision_counts": dict(Counter(p.decision for p in real_patches)),
            "n_patches": len(all_patches),
            "novel_rate": novel_rate(all_patches),
            "retrieval": retrieval,
        })

    return _summarize(orderings, fps, recorders, k_orderings=k_orderings, seed=seed,
                      n_workload_items=n)


def _first_divergent_pair(orderings: List[Dict[str, Any]], fps: List, component: str
                          ) -> Tuple[int, int] | None:
    seen: Dict[str, int] = {}
    for i, o in enumerate(orderings):
        d = o["component_digests"][component]
        if d not in seen:
            seen[d] = i
        if len(seen) == 2:
            idxs = list(seen.values())
            return idxs[0], idxs[1]
    return None


def _summarize(orderings: List[Dict[str, Any]], fps: List, recorders: List[CandidateCostRecorder],
               *, k_orderings: int, seed: int, n_workload_items: int) -> Dict[str, Any]:
    components = ("live_claims", "status_map", "links", "representatives")
    distinct = {c: len({o["component_digests"][c] for o in orderings}) for c in components}
    distinct["joint"] = len({o["joint_digest"] for o in orderings})

    diffs: Dict[str, Any] = {}
    for c in components:
        if distinct[c] > 1:
            pair = _first_divergent_pair(orderings, fps, c)
            if pair is not None:
                i, j = pair
                diffs[c] = {"orderings_compared": [i, j],
                           "diff": diff_fingerprints(fps[i], fps[j]).get(c)}

    retrieval_agreement: Dict[str, Any] = {}
    if orderings and orderings[0]["retrieval"]:
        for q in orderings[0]["retrieval"]:
            results = [tuple(o["retrieval"][q]) for o in orderings]
            counts = Counter(results)
            retrieval_agreement[q] = {
                "agree": len(counts) == 1,
                "n_distinct_results": len(counts),
                "distinct_results": [list(r) for r in counts],
            }

    novel_rates = [o["novel_rate"] for o in orderings if o["novel_rate"] == o["novel_rate"]]  # drop NaN
    decision_count_variants = sorted({tuple(sorted(o["decision_counts"].items())) for o in orderings})

    return {
        "k_orderings": k_orderings, "seed": seed, "n_workload_items": n_workload_items,
        "distinct_fingerprints": distinct,
        "diffs": diffs,
        "retrieval_agreement": retrieval_agreement,
        "candidate_cost": pooled_summary(recorders),
        "novel_rate": {
            "median": _median(novel_rates), "min": min(novel_rates) if novel_rates else float("nan"),
            "max": max(novel_rates) if novel_rates else float("nan"),
        },
        "n_decision_count_variants": len(decision_count_variants),
        "decision_count_variants": [dict(v) for v in decision_count_variants],
        "reference_fingerprint": fps[0].readable if fps else {},
        "orderings": orderings,
    }


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


__all__ = ["run_order_shuffle", "WorkloadItem", "StatusEvent"]
