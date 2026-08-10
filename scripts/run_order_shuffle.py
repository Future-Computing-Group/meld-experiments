"""S3: run the order-shuffle graph-structure convergence experiment on both
workloads (running example + gold-derived) and persist the reports as JSON.

Dense rung (minilm + cross-encoder NLI), matching the M-2 headline pipeline,
so the contradiction/paraphrase signals in the running example and gold set
actually fire as the manuscript describes them.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_order_shuffle.py [--k K] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import time

from smp.agent.run_hotpot_gate import _calibrated_cfg
from smp.eval.order_shuffle import run_order_shuffle
from smp.eval.order_shuffle_workloads import (GOLD_PROBE_QUERIES,
                                              RUNNING_EXAMPLE_PROBE_QUERIES,
                                              build_gold_derived_workload,
                                              build_running_example_workload)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "outputs", "order-shuffle")
CACHE_DIR = os.path.join(ROOT, "outputs", "emb-cache")

ENCODER = "minilm"
NLI = "mnli-cross-encoder"


def _warmup_caches(items, *, encoder: str, nli: str, cache_dir: str) -> None:
    """Populate the on-disk embedding cache and load the NLI model BEFORE
    timing starts. ``EmbeddingSimilarity`` embeds-then-caches per (encoder,
    text); once cached, every later ordering's ``_best_candidate`` calls hit
    the cache and never touch the model again -- but the UNwarmed first call
    would eat one-time model-load latency, which would otherwise sit in the
    candidate-cost distribution as a single wild outlier rather than
    reflecting steady-state per-call cost."""
    from smp.merge.signals import EmbeddingSimilarity, chi
    from smp.proto import MemoryUnit

    es = EmbeddingSimilarity(encoder=encoder, cache_dir=cache_dir)
    texts = sorted({c.text for it in items if isinstance(it, MemoryUnit) for c in it.claims})
    for t in texts:
        es.sigma(t, t)
    if nli != "lexical" and len(texts) >= 2:
        chi(texts[0], texts[1], model=nli)


def _run(name: str, workload, *, k: int, seed: int, probe_queries, seed_units=(), cfg=None):
    cfg = cfg or _calibrated_cfg(ENCODER)
    t0 = time.perf_counter()
    report = run_order_shuffle(workload, k_orderings=k, seed=seed, encoder=ENCODER, nli=NLI,
                               cfg=cfg, probe_queries=probe_queries, seed_units=seed_units,
                               cache_dir=CACHE_DIR)
    dt = time.perf_counter() - t0
    report["_runtime_s"] = dt
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"== {name} == k={k} n_items={len(workload)} runtime={dt:.1f}s -> {path}")
    print(f"  distinct fingerprints: {report['distinct_fingerprints']}")
    print(f"  diffs present for: {sorted(report['diffs'].keys())}")
    print(f"  retrieval_agreement: "
          f"{ {q: v['agree'] for q, v in report['retrieval_agreement'].items()} }")
    print(f"  candidate_cost: {report['candidate_cost']}")
    print(f"  novel_rate: {report['novel_rate']}")
    print(f"  n_decision_count_variants: {report['n_decision_count_variants']}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100, help="orderings per workload (default 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny K for a fast plumbing check")
    args = ap.parse_args()
    k = 3 if args.smoke else args.k

    cfg = _calibrated_cfg(ENCODER)  # minilm-calibrated (theta_merge=0.90, sigma_lo=0.12)

    re_workload, re_seed_units = build_running_example_workload()
    _warmup_caches(list(re_workload) + list(re_seed_units), encoder=ENCODER, nli=NLI,
                  cache_dir=CACHE_DIR)
    _run("running-example", re_workload, k=k, seed=args.seed,
        probe_queries=RUNNING_EXAMPLE_PROBE_QUERIES, seed_units=re_seed_units, cfg=cfg)

    gold_workload = build_gold_derived_workload(encoder=ENCODER, n_sources=3, cache_dir=CACHE_DIR)
    _warmup_caches(gold_workload, encoder=ENCODER, nli=NLI, cache_dir=CACHE_DIR)
    _run("gold-derived", gold_workload, k=k, seed=args.seed,
        probe_queries=GOLD_PROBE_QUERIES, cfg=cfg)


if __name__ == "__main__":
    main()
