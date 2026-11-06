"""R2: overlap-fraction x k recall sweep + exact-dedup baseline.

recall@k is computed on the status-live store UPSTREAM of the answering LLM, so this sweep
needs only the encoder + the store -- no LLM. It answers two reviewer questions: (a) does
the recall/storage win hold across overlap fraction and k, or is it an operating-point
artefact of overlap 0.5 / k 5; and (b) is the win attributable to the SEMANTIC
merge or merely to store-size reduction -- the exact-dedup arm runs the same
distributed merge without an epoch (no similarity merge; exact copies are one node),
isolating the store-size component.

Orchestration reuses the tested build_condition + recall_at_k_and_mrr; aggregate_recall is
a pure per-cell mean.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from meld.agent.mem_conditions import build_condition
from meld.agent.recall_metrics import READ_TAU, recall_at_k_and_mrr
from meld.protocol.objects import EpochRecord

# condition label -> (build_condition name, without the epoch?)
_COND_BUILD = {
    "centralized": ("centralized", False),
    "distributed_merge": ("distributed_merge", False),
    "naive_union": ("naive_union", False),
    "exact_dedup": ("distributed_merge", True),   # no epoch: no similarity merge
}


def recall_rows(tasks, overlaps: List[float], ks: List[int], conditions: List[str], *,
                encoder: str, nli: str, epoch: Optional[EpochRecord], progress=None,
                cache_dir: str = None, policies: Optional[Dict] = None,
                n_brains: int = 3, dup_degree: int = 2,
                tau: Optional[float] = READ_TAU) -> List[Dict]:
    """One row per (overlap, condition, task, k): recall@k + the store's live-unit
    footprint (built once per (overlap, condition, task), read at each k).

    ``progress`` (optional): a callback invoked once per (cell, task) with counters
    {cell, n_cells, task, n_tasks, overlap, condition, elapsed_s}, so a long sweep emits
    per-batch progress and a wedge surfaces in seconds, not minutes.

    ``policies`` (optional): condition label -> baseline policy (meld.baselines); such a
    condition is the distributed merge with the policy in place of the global brain's
    admission. ``tau``: the read's near-duplicate threshold (recall_metrics.read_top)."""
    import time as _time
    rows: List[Dict] = []
    n_cells = len(overlaps) * len(conditions)
    cell = 0
    t0 = _time.perf_counter()
    for ov in overlaps:
        for cond in conditions:
            cell += 1
            policy = (policies or {}).get(cond)
            build_name, no_epoch = ("distributed_merge", False) if policy else _COND_BUILD[cond]
            for ti, task in enumerate(tasks, start=1):
                b = build_condition(task, build_name, encoder=encoder, nli=nli,
                                    epoch=None if no_epoch else epoch, overlap_fraction=ov,
                                    cache_dir=cache_dir, policy=policy,
                                    n_brains=n_brains, dup_degree=dup_degree)
                live = b.live_units if b.brain is not None else 0
                for k in ks:
                    rec = (recall_at_k_and_mrr(b.brain, task, k=k, source=b.split and b.split.source,
                                               tau=tau)["recall_at_k"]
                           if b.brain is not None else 0.0)
                    rows.append({"overlap": ov, "condition": cond, "k": k, "task_id": task.task_id,
                                 "recall": rec, "live_units": live})
                if progress is not None:
                    progress({"cell": cell, "n_cells": n_cells, "task": ti,
                              "n_tasks": len(tasks), "overlap": ov, "condition": cond,
                              "elapsed_s": round(_time.perf_counter() - t0, 1)})
    return rows


def aggregate_recall(rows: List[Dict]) -> Dict:
    """Mean recall@k and mean live-unit footprint per (overlap, condition, k) cell."""
    acc = defaultdict(lambda: {"recall": [], "live_units": []})
    for r in rows:
        key = (r["overlap"], r["condition"], r["k"])
        acc[key]["recall"].append(r["recall"])
        acc[key]["live_units"].append(r["live_units"])
    out: Dict = {}
    for key, v in acc.items():
        n = len(v["recall"])
        out[key] = {"mean_recall": round(sum(v["recall"]) / n, 4),
                    "mean_live_units": round(sum(v["live_units"]) / n, 2), "n": n}
    return out


__all__ = ["recall_rows", "aggregate_recall", "_COND_BUILD"]
