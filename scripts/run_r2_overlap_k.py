#!/usr/bin/env python3
"""R2: overlap-fraction x k recall sweep + exact-dedup baseline, on real HotpotQA.

Answers (a) does the recall/storage win hold across overlap and k (is the headline
an overlap-0.5/k-5 artefact?) and (b) is it semantic or store-size (the exact_dedup
arm runs the same merge with the semantic gate off). recall@k is upstream of the LLM, so no
LLM is needed. The encoder and NLI are required arguments (deployed: --encoder minilm --nli
mnli-cross-encoder); every cell carries topology.stamps.

Run (smoke):  PYTHONPATH=src .venv/bin/python scripts/run_r2_overlap_k.py --smoke --encoder E --nli N
Run (full):   PYTHONPATH=src .venv/bin/python scripts/run_r2_overlap_k.py --n 200 --ks 1 3 5 10 20 \
                  --encoder minilm --nli mnli-cross-encoder --epoch EPOCH.json
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from meld.eval.runner_args import (add_baseline, add_key_file, add_read_tau, read_stamp, baseline_policy,  # noqa: E402
                                   baseline_stamp, runner_key)
from meld.config import read_epoch                              # noqa: E402
from meld.agent.hotpot_loader import load_hotpot_fixture, load_hotpot_real  # noqa: E402
from meld.eval.recall_sweep import recall_rows, aggregate_recall    # noqa: E402
from meld.eval.topology import stamps                            # noqa: E402
from meld.eval.substrates import eval_pool                       # noqa: E402

CONDS = ["centralized", "distributed_merge", "naive_union", "exact_dedup"]


def main(argv=None, llm=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "r2-overlap-k.json"))
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "outputs", "emb-cache"))
    ap.add_argument("--epoch", default=None, help="signed epoch file (config.read_epoch); "
                    "without one no similarity merge happens")
    add_key_file(ap)
    add_baseline(ap)
    add_read_tau(ap)
    a = ap.parse_args(argv)
    policy = baseline_policy(a, llm)
    conds = CONDS + [a.baseline] * bool(policy)
    epoch = read_epoch(a.epoch, runner_key(a.key_file)) if a.epoch else None

    if a.smoke:
        tasks = load_hotpot_fixture(n=4)
        overlaps, ks, nli = [0.0, 0.5], [5], a.nli
    else:
        # the calibration pool (meld.eval.substrates) is never evaluated
        tasks = eval_pool(load_hotpot_real(split="validation", config="distractor"))[:a.n]
        overlaps, ks, nli = [0.0, 0.25, 0.5, 0.75], a.ks, a.nli

    print(f"[run] {len(tasks)} tasks x overlaps {overlaps} x k {ks} x {len(conds)} conds, nli={nli}", flush=True)
    # emit per-(cell,task) progress to a status file (readable mid-run via `cat`) and
    # flushed stderr, so a wedge surfaces in seconds rather than minutes of blind waiting.
    status_path = a.out + ".status"
    os.makedirs(os.path.dirname(status_path), exist_ok=True)

    def _progress(info):
        line = (f"[r2] cell {info['cell']}/{info['n_cells']} "
                f"(ov={info['overlap']} {info['condition']}) task {info['task']}/{info['n_tasks']}"
                f" | elapsed {info['elapsed_s']}s")
        print(line, file=sys.stderr, flush=True)
        with open(status_path, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # shared on-disk embedding cache: each sentence embeds once across all cells/conditions
    rows = recall_rows(tasks, overlaps=overlaps, ks=ks, conditions=conds,
                       encoder=a.encoder, nli=nli, epoch=epoch, progress=_progress,
                       cache_dir=a.cache_dir, policies={a.baseline: policy} if policy else None,
                       tau=a.read_tau)
    agg = aggregate_recall(rows)
    # serialize (tuple keys -> "overlap|cond|k"); every cell stamped
    st = {**stamps(a.encoder, nli, epoch), **read_stamp(a)}
    cells = {f"{ov}|{c}|{k}": {**v, **st, **(baseline_stamp(a) if c == a.baseline else {})}
             for (ov, c, k), v in agg.items()}
    out = {"n_tasks": len(tasks), "overlaps": overlaps, "ks": ks, "conditions": conds, "cells": cells,
           **({"policy_counters": policy.counters} if policy else {})}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    # per-task rows (PH-4) beside the aggregate, so paired intervals and d_z can be formed
    with open(os.path.splitext(a.out)[0] + ".rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**r, **st, **(baseline_stamp(a) if r["condition"] == a.baseline else {})}) + "\n")

    # headline contrasts: merge vs union recall + live_units, across overlap, at k=5
    k0 = ks[0]
    print(f"\n=== recall@{k0} and live-units by overlap (merge vs union vs dedup vs central) ===")
    print(f"{'overlap':>8} {'central':>16} {'merge':>16} {'union':>16} {'dedup':>16}")
    for ov in overlaps:
        def cell(c):
            v = agg[(ov, c, k0)]
            return f"{v['mean_recall']:.3f}/{v['mean_live_units']:.1f}"
        print(f"{ov:>8} {cell('centralized'):>16} {cell('distributed_merge'):>16} "
              f"{cell('naive_union'):>16} {cell('exact_dedup'):>16}")
    print(f"(cells are mean_recall@{k0}/mean_live_units)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
