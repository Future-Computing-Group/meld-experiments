#!/usr/bin/env python3
"""R2-LME: second-substrate recall sweep on the REAL LongMemEval corpus (arXiv 2410.10813).

Corroborates the M-1 headline (distributed merge is recall-non-inferior to centralized and
beats naive union) on a long-horizon chat-memory benchmark distinct from HotpotQA. recall@k
of the gold has_answer turns is computed on the status-live store, UPSTREAM of any LLM, under
the centralized / distributed-merge / naive-union (+ exact-dedup) conditions, reusing the
tested recall_sweep machinery via the LongMemEval->HotpotTask adapter. Local, no LLM, no cost.

The encoder and NLI are required arguments (deployed: --encoder minilm --nli mnli-cross-encoder);
every cell carries topology.stamps.

Run (smoke):  PYTHONPATH=src .venv/bin/python scripts/run_r2_lme.py --smoke --encoder E --nli N
Run (full):   PYTHONPATH=src .venv/bin/python scripts/run_r2_lme.py --n 40 --encoder minilm \
                  --nli mnli-cross-encoder --epoch EPOCH.json
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from meld.eval.runner_args import (add_baseline, add_key_file, add_read_tau, read_stamp, baseline_policy,  # noqa: E402
                                   baseline_stamp, runner_key)
from meld.config import read_epoch                            # noqa: E402
from meld.eval.longmemeval_real import load_longmemeval_tasks      # noqa: E402
from meld.eval.recall_sweep import recall_rows, aggregate_recall   # noqa: E402
from meld.eval.topology import stamps                          # noqa: E402
from meld.eval.substrates import eval_pool                     # noqa: E402

CONDS = ["centralized", "distributed_merge", "naive_union", "exact_dedup"]
DEFAULT_DATA = os.path.join(REPO, "data", "longmemeval", "longmemeval_s")


def main(argv=None, llm=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "r2-lme.json"))
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "outputs", "emb-cache-lme"))
    ap.add_argument("--epoch", default=None, help="signed epoch file (config.read_epoch); "
                    "without one no similarity merge happens")
    add_key_file(ap)
    add_baseline(ap)
    add_read_tau(ap)
    a = ap.parse_args(argv)
    policy = baseline_policy(a, llm)
    conds = CONDS + [a.baseline] * bool(policy)
    epoch = read_epoch(a.epoch, runner_key(a.key_file)) if a.epoch else None

    n = 3 if a.smoke else a.n
    overlaps, ks = ([0.0, 0.5], [5]) if a.smoke else ([0.0, 0.5], [5, 10, 20])
    tasks = eval_pool(load_longmemeval_tasks(a.data, seed=0))[:n]   # never a calibration-pool task
    print(f"[run] LongMemEval: {len(tasks)} tasks x overlaps {overlaps} x k {ks} x {len(conds)} conds, nli={a.nli}", flush=True)

    status_path = a.out + ".status"
    os.makedirs(os.path.dirname(status_path), exist_ok=True)

    def _progress(info):
        line = (f"[r2-lme] cell {info['cell']}/{info['n_cells']} "
                f"(ov={info['overlap']} {info['condition']}) task {info['task']}/{info['n_tasks']}"
                f" | elapsed {info['elapsed_s']}s")
        print(line, file=sys.stderr, flush=True)
        with open(status_path, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")

    rows = recall_rows(tasks, overlaps=overlaps, ks=ks, conditions=conds,
                       encoder=a.encoder, nli=a.nli, epoch=epoch, progress=_progress,
                       cache_dir=a.cache_dir, policies={a.baseline: policy} if policy else None,
                       tau=a.read_tau)
    agg = aggregate_recall(rows)
    st = {**stamps(a.encoder, a.nli, epoch), **read_stamp(a)}
    cells = {f"{ov}|{c}|{k}": {**v, **st, **(baseline_stamp(a) if c == a.baseline else {})}
             for (ov, c, k), v in agg.items()}
    out = {"substrate": "LongMemEval (arXiv 2410.10813)", "data": os.path.basename(a.data),
           "n_tasks": len(tasks), "overlaps": overlaps, "ks": ks, "conditions": conds, "cells": cells,
           **({"policy_counters": policy.counters} if policy else {})}
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
