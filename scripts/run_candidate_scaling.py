#!/usr/bin/env python3
"""S8: run the claim-count scaling experiment for candidate selection and
persist the report as JSON.

Dense rung (minilm + cross-encoder NLI), matching the M-2/S3 headline
pipeline. Store sizes N_local in {16, 64, 256, 1024, 4096}, K=200 incoming
claims per size (seeded paraphrase/novel mix), against a FIXED store
snapshot per size (see candidate_scaling.py).

Run:  PYTHONPATH=src .venv/bin/python scripts/run_candidate_scaling.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from smp.agent.run_hotpot_gate import _calibrated_cfg          # noqa: E402
from smp.eval.candidate_scaling import run_candidate_scaling   # noqa: E402

OUT_DIR = os.path.join(ROOT, "outputs", "candidate-scaling")
CACHE_DIR = os.path.join(ROOT, "outputs", "emb-cache")  # shared minilm embedding cache (S3)

ENCODER = "minilm"
NLI = "mnli-cross-encoder"
DEFAULT_SIZES = [16, 64, 256, 1024, 4096]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                    help="comma-separated store sizes N_local")
    ap.add_argument("--k", type=int, default=200, help="incoming claims per store size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny sizes/K for a fast plumbing check")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "candidate-scaling.json"))
    args = ap.parse_args()

    sizes = [8, 32] if args.smoke else [int(s) for s in args.sizes.split(",")]
    k = 10 if args.smoke else args.k
    cfg = _calibrated_cfg(ENCODER)  # minilm-calibrated (theta_merge=0.90, sigma_lo=0.12)

    print(f"[run] store_sizes={sizes} k={k} encoder={ENCODER} nli={NLI} ...", flush=True)
    t0 = time.perf_counter()
    report = run_candidate_scaling(sizes, k_per_size=k, seed=args.seed, encoder=ENCODER,
                                   nli=NLI, cfg=cfg, cache_dir=CACHE_DIR)
    report["_runtime_s"] = time.perf_counter() - t0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(f"== candidate-scaling == runtime={report['_runtime_s']:.1f}s -> {args.out}")
    for row in report["per_size"]:
        cc = row["candidate_cost"]
        print(f"  N_local={row['n_local']:>5}  n_calls={cc['n_calls']:>4}  "
              f"scan_ms median={cc['scan_latency_ms_median']:.4f} p95={cc['scan_latency_ms_p95']:.4f}  "
              f"other_ms(matched) median={row['other_ms_matched_median']:.2f}  "
              f"decisions={row['decision_counts']}")
    fit = report["fit"]
    print(f"  fit: {fit['ms_per_1k_claims']:.4f} ms / 1k claims "
          f"(intercept {fit['intercept_ms']:.4f} ms, r2={fit['r2']:.4f})")
    print(f"  NLI reference (matched-delta, this run) = {fit['nli_reference_ms']:.2f} ms; "
          f"crossover ~= {fit['crossover_n_claims']:.0f} claims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
