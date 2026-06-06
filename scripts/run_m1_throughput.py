#!/usr/bin/env python3
"""M1: measure the global-brain sink saturation curve with the REAL merge step.

The per-delta merge cost is one embedding (all-MiniLM-L6-v2) + one NLI call
(cross-encoder/nli-deberta-v3-small) + the four-way classify; the status-CRDT update is
O(1) and negligible. We first warm up and time a single merge step to size the offered-rate
grid around the sink's capacity (1/service), then sweep below->above capacity and report
achieved throughput + publish-to-merge p50/p95 per offered rate, and the saturation knee.

Run (bounded smoke):  PYTHONPATH=src .venv/bin/python scripts/run_m1_throughput.py --smoke
Run (full):           PYTHONPATH=src .venv/bin/python scripts/run_m1_throughput.py
"""
import argparse, itertools, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from smp.config import load_thresholds                       # noqa: E402
from smp.merge.classifier import classify                    # noqa: E402
from smp.merge.signals import EmbeddingSimilarity, chi       # noqa: E402
from smp.merge.m2_gold import build_gold_set                 # noqa: E402
from smp.eval.sink_throughput import run_saturation, saturation_knee  # noqa: E402

MINILM = os.path.join(REPO, "src", "smp", "config", "thresholds-minilm.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="bounded smoke (small n, 4 rates)")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "m1-throughput.json"))
    a = ap.parse_args()

    cfg = load_thresholds(MINILM)
    es = EmbeddingSimilarity(encoder="minilm")
    # A corpus of distinct claim pairs to merge (cycle through the gold bank's pairs).
    gold = build_gold_set()
    pairs = [(r["a"], r["b"]) for r in gold]
    cyc = itertools.cycle(pairs)

    def merge_step(payload):
        a, b = payload
        s = es.sigma(a, b)
        c = chi(a, b, model="mnli-cross-encoder")
        classify(cfg=cfg, kappa=0, sigma=s, chi=c, ctx_ok=True)

    def claim_source(i):
        return next(cyc)

    # Warm up to steady state (model load + cross-encoder warm), then probe the sink's
    # CAPACITY directly with a saturating burst (offered >> capacity, so achieved == capacity).
    # A direct burst probe is robust where per-step cold timing under-estimates steady state.
    print("[warmup] loading encoder + NLI and warming to steady state ...", flush=True)
    for _ in range(12):
        merge_step(next(cyc))
    probe = run_saturation(merge_step, offered_rates=[10000.0], n_per_rate=40,
                           claim_source=claim_source)
    cap = probe[10000.0]["merged_per_s"]
    print(f"[probe] saturating-burst capacity ~{cap:.1f} merges/s", flush=True)

    rates = [round(cap * m, 2) for m in (0.5, 1.0, 1.5, 2.0, 3.0)]
    n_per_rate = 15 if a.smoke else 60
    print(f"[run] offered rates {rates} (claims/s), n_per_rate={n_per_rate}", flush=True)

    res = run_saturation(merge_step, offered_rates=rates, n_per_rate=n_per_rate,
                         claim_source=claim_source)
    knee = saturation_knee(res)
    out = {"service_ms_steady": round(1000.0 / cap, 2) if cap else 0.0,
           "capacity_probe": round(cap, 2),
           "saturation_knee_merges_per_s": round(knee, 2),
           "n_per_rate": n_per_rate, "encoder": "all-MiniLM-L6-v2",
           "nli": "cross-encoder/nli-deberta-v3-small", "curve": res}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    # sanity: below-capacity tracks offered; above-capacity saturates below offered
    lo, hi = rates[0], rates[-1]
    ok = (res[hi]["merged_per_s"] < hi) and (res[hi]["p95_ms"] >= res[lo]["p95_ms"])
    print(f"=== saturation observed: {ok} (knee {knee:.1f}/s) ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
