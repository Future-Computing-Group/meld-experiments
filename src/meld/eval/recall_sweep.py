"""R2: overlap-fraction x k recall sweep + exact-dedup baseline.

recall@k is computed on the status-live store UPSTREAM of the answering LLM, so this sweep
needs only the encoder + the store -- no LLM. It answers two reviewer questions: (a) does
the recall/storage win hold across overlap fraction and k, or is it an operating-point
artefact of overlap 0.5 / k 5; and (b) is the win attributable to the SEMANTIC
merge or merely to store-size reduction -- the exact-dedup arm runs the same
distributed merge with the semantic gate disabled (theta->1, sigma_lo->1, so only the
exact-key kappa path collapses duplicates), isolating the store-size component.

Orchestration reuses the tested build_condition + recall_at_k_and_mrr; aggregate_recall is
a pure per-cell mean.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from meld.agent.mem_conditions import build_condition
from meld.agent.recall_metrics import recall_at_k_and_mrr
from meld.config import Thresholds

# condition label -> (build_condition name, use-dedup-cfg?)
_COND_BUILD = {
    "centralized": ("centralized", False),
    "distributed_merge": ("distributed_merge", False),
    "naive_union": ("naive_union", False),
    "exact_dedup": ("distributed_merge", True),   # semantic gate disabled
}


def dedup_cfg(cfg: Thresholds) -> Thresholds:
    """A cfg with the semantic merge gate effectively disabled: theta_merge and sigma_lo
    pushed to ~1.0 so only the exact-key (kappa=1) path collapses duplicates. Paraphrased
    duplicates (sigma < 1) then never semantic-merge -- exact-dedup, not semantic dedup."""
    return Thresholds(theta_merge=0.999, sigma_lo=0.998,
                      r_min=cfg.r_min, tau=getattr(cfg, "tau", 0.0))


def recall_rows(tasks, overlaps: List[float], ks: List[int], conditions: List[str], *,
                encoder: str, nli: str, cfg: Thresholds, progress=None,
                cache_dir: str = None) -> List[Dict]:
    """One row per (overlap, condition, task, k): recall@k + the store's live-unit
    footprint (built once per (overlap, condition, task), read at each k).

    ``progress`` (optional): a callback invoked once per (cell, task) with counters
    {cell, n_cells, task, n_tasks, overlap, condition, elapsed_s}, so a long sweep emits
    per-batch progress and a wedge surfaces in seconds, not minutes."""
    import time as _time
    ddcfg = dedup_cfg(cfg)
    rows: List[Dict] = []
    n_cells = len(overlaps) * len(conditions)
    cell = 0
    t0 = _time.perf_counter()
    for ov in overlaps:
        for cond in conditions:
            cell += 1
            build_name, use_dedup = _COND_BUILD[cond]
            use_cfg = ddcfg if use_dedup else cfg
            for ti, task in enumerate(tasks, start=1):
                b = build_condition(task, build_name, encoder=encoder, nli=nli,
                                    cfg=use_cfg, overlap_fraction=ov, cache_dir=cache_dir)
                # ConditionBuild.brain is an SMPMemory; recall_at_k_and_mrr consumes the
                # inner Brain (graph_walk_read/get_unit live there).
                inner = b.brain.brain if b.brain is not None else None
                live = b.live_units if b.brain is not None else 0
                for k in ks:
                    rec = (recall_at_k_and_mrr(inner, task, encoder=encoder, k=k)["recall_at_k"]
                           if inner is not None else 0.0)
                    rows.append({"overlap": ov, "condition": cond, "k": k,
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


__all__ = ["recall_rows", "aggregate_recall", "dedup_cfg", "_COND_BUILD"]
