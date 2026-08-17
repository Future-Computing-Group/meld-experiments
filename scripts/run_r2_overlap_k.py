#!/usr/bin/env python3
"""R2: overlap-fraction x k recall sweep + exact-dedup baseline, on real HotpotQA.

Answers (a) does the recall/storage win hold across overlap and k (is the headline
an overlap-0.5/k-5 artefact?) and (b) is it semantic or store-size (the exact_dedup
arm runs the same merge with the semantic gate off). recall@k is upstream of the LLM, so no
LLM is needed. encoder=all-MiniLM-L6-v2, NLI=cross-encoder/nli-deberta-v3-small (the M-1 setup).

Run (smoke):  PYTHONPATH=src .venv/bin/python scripts/run_r2_overlap_k.py --smoke
Run (full):   PYTHONPATH=src .venv/bin/python scripts/run_r2_overlap_k.py --n 30
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from meld.config import load_thresholds                              # noqa: E402
from meld.agent.hotpot_loader import load_hotpot_fixture, load_hotpot_real  # noqa: E402
from meld.eval.recall_sweep import recall_rows, aggregate_recall    # noqa: E402

MINILM = os.path.join(REPO, "src", "meld", "config", "thresholds-minilm.yaml")
CONDS = ["centralized", "distributed_merge", "naive_union", "exact_dedup"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    # recall@k is driven by the sigma-based dedup (encoder), NOT the NLI (chi only routes
    # CONFLICT edges, which do not remove units), so the recall sweep uses the fast lexical
    # NLI by default; --nli mnli-cross-encoder spot-checks NLI-invariance of recall.
    ap.add_argument("--nli", default="lexical")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "r2-overlap-k.json"))
    a = ap.parse_args()
    cfg = load_thresholds(MINILM)

    if a.smoke:
        tasks = load_hotpot_fixture(n=4)
        overlaps, ks, nli = [0.0, 0.5], [5], a.nli
    else:
        tasks = load_hotpot_real(split="validation", config="distractor", n=a.n)
        overlaps, ks, nli = [0.0, 0.25, 0.5, 0.75], [5, 10, 20], a.nli

    print(f"[run] {len(tasks)} tasks x overlaps {overlaps} x k {ks} x {len(CONDS)} conds, nli={nli}", flush=True)
    # emit per-(cell,task) progress to a status file (readable mid-run via `cat`) and
    # flushed stderr, so a wedge surfaces in seconds rather than minutes of blind waiting.
    status_path = os.path.join(REPO, "outputs", "r2-status.log")
    os.makedirs(os.path.dirname(status_path), exist_ok=True)

    def _progress(info):
        line = (f"[r2] cell {info['cell']}/{info['n_cells']} "
                f"(ov={info['overlap']} {info['condition']}) task {info['task']}/{info['n_tasks']}"
                f" | elapsed {info['elapsed_s']}s")
        print(line, file=sys.stderr, flush=True)
        with open(status_path, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # shared on-disk embedding cache: each sentence embeds once across all cells/conditions
    # (fixes the per-build re-embedding slowness).
    emb_cache = os.path.join(REPO, "outputs", "emb-cache")
    rows = recall_rows(tasks, overlaps=overlaps, ks=ks, conditions=CONDS,
                       encoder="minilm", nli=nli, cfg=cfg, progress=_progress,
                       cache_dir=emb_cache)
    agg = aggregate_recall(rows)
    # serialize (tuple keys -> "overlap|cond|k")
    cells = {f"{ov}|{c}|{k}": v for (ov, c, k), v in agg.items()}
    out = {"n_tasks": len(tasks), "overlaps": overlaps, "ks": ks, "conditions": CONDS,
           "encoder": "all-MiniLM-L6-v2", "nli": nli, "cells": cells}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

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
