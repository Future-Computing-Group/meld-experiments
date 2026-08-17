#!/usr/bin/env python3
"""R2-LME: second-substrate recall sweep on the REAL LongMemEval corpus (arXiv 2410.10813).

Corroborates the M-1 headline (distributed merge is recall-non-inferior to centralized and
beats naive union) on a long-horizon chat-memory benchmark distinct from HotpotQA. recall@k
of the gold has_answer turns is computed on the status-live store, UPSTREAM of any LLM, under
the centralized / distributed-merge / naive-union (+ exact-dedup) conditions, reusing the
tested recall_sweep machinery via the LongMemEval->HotpotTask adapter. Local, no LLM, no cost.

Run (smoke):  PYTHONPATH=src .venv/bin/python scripts/run_r2_lme.py --smoke
Run (full):   PYTHONPATH=src .venv/bin/python scripts/run_r2_lme.py --n 40
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from meld.config import load_thresholds                            # noqa: E402
from meld.eval.longmemeval_real import load_longmemeval_tasks      # noqa: E402
from meld.eval.recall_sweep import recall_rows, aggregate_recall   # noqa: E402

MINILM = os.path.join(REPO, "src", "meld", "config", "thresholds-minilm.yaml")
CONDS = ["centralized", "distributed_merge", "naive_union", "exact_dedup"]
DEFAULT_DATA = os.path.join(REPO, "data", "longmemeval", "longmemeval_s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--nli", default="lexical")  # recall is sigma-driven; see run_r2_overlap_k
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "r2-lme.json"))
    a = ap.parse_args()
    cfg = load_thresholds(MINILM)

    n = 3 if a.smoke else a.n
    overlaps, ks = ([0.0, 0.5], [5]) if a.smoke else ([0.0, 0.5], [5, 10, 20])
    tasks = load_longmemeval_tasks(a.data, n=n, seed=0)
    print(f"[run] LongMemEval: {len(tasks)} tasks x overlaps {overlaps} x k {ks} x {len(CONDS)} conds, nli={a.nli}", flush=True)

    status_path = os.path.join(REPO, "outputs", "r2-lme-status.log")
    os.makedirs(os.path.dirname(status_path), exist_ok=True)

    def _progress(info):
        line = (f"[r2-lme] cell {info['cell']}/{info['n_cells']} "
                f"(ov={info['overlap']} {info['condition']}) task {info['task']}/{info['n_tasks']}"
                f" | elapsed {info['elapsed_s']}s")
        print(line, file=sys.stderr, flush=True)
        with open(status_path, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")

    emb_cache = os.path.join(REPO, "outputs", "emb-cache-lme")
    rows = recall_rows(tasks, overlaps=overlaps, ks=ks, conditions=CONDS,
                       encoder="minilm", nli=a.nli, cfg=cfg, progress=_progress,
                       cache_dir=emb_cache)
    agg = aggregate_recall(rows)
    cells = {f"{ov}|{c}|{k}": v for (ov, c, k), v in agg.items()}
    out = {"substrate": "LongMemEval (arXiv 2410.10813)", "data": os.path.basename(a.data),
           "n_tasks": len(tasks), "overlaps": overlaps, "ks": ks, "conditions": CONDS,
           "encoder": "all-MiniLM-L6-v2", "nli": a.nli, "cells": cells}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    k0 = ks[0]
    print(f"\n=== LongMemEval recall@{k0} / live-units by overlap (central / merge / union / dedup) ===")
    for ov in overlaps:
        def cell(c):
            v = agg[(ov, c, k0)]
            return f"{v['mean_recall']:.3f}/{v['mean_live_units']:.1f}"
        print(f"  overlap {ov}: {cell('centralized')}  {cell('distributed_merge')}  "
              f"{cell('naive_union')}  {cell('exact_dedup')}")
    print(f"(mean_recall@{k0}/mean_live_units)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
