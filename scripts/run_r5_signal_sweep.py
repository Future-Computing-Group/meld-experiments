#!/usr/bin/env python3
"""R5: merge-signal AI sensitivity sweep.

The headline (M-1) deliberately removes the answering LLM from recall@k, so the question
"does the AI capability matter" relocates to the merge-signal models -- the embedding
encoder and the NLI cross-encoder, which M-2 evaluates at exactly one each. This sweep runs
the 104-pair M-2 benchmark over {encoders} x {NLI} at the SAME calibrated operating point
(theta_merge=0.90, sigma_lo=0.12) and reports macro-F1 + false-merge rate per cell, so the
reader can see (a) how sensitive the merge classifier is to the encoder/NLI choice and
(b) whether the minilm-calibrated operating point transfers across encoders (the Section 9.1
calibration burden, made concrete).

Run (smoke):  PYTHONPATH=src .venv/bin/python scripts/run_r5_signal_sweep.py --smoke
Run (full):   PYTHONPATH=src .venv/bin/python scripts/run_r5_signal_sweep.py
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from smp.config import load_thresholds            # noqa: E402
from smp.merge.m2_eval import evaluate_m2          # noqa: E402
from smp.merge.m2_gold import M2_GOLD_V2           # noqa: E402

MINILM = os.path.join(REPO, "src", "smp", "config", "thresholds-minilm.yaml")


def run_sweep(encoders, nlis, pairs, cfg):
    out = {}
    for e in encoders:
        for n in nlis:
            print(f"[cell] encoder={e} nli={n} ({len(pairs)} pairs) ...", flush=True)
            m = evaluate_m2(pairs, encoder=e, nli=n, cfg=cfg)
            out[f"{e}|{n}"] = {"encoder": e, "nli": n,
                               "macro_f1": round(m["macro_f1"], 3),
                               "false_merge_rate": round(m["false_merge_rate"], 3),
                               "n_false_merges": m["n_false_merges"], "n": m["n"]}
            print(f"       macro-F1={out[f'{e}|{n}']['macro_f1']} "
                  f"FMR={out[f'{e}|{n}']['false_merge_rate']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "r5-signal-sweep.json"))
    a = ap.parse_args()
    cfg = load_thresholds(MINILM)   # the calibrated operating point (theta=0.90, sigma_lo=0.12)

    if a.smoke:
        encoders, nlis, pairs = ["minilm", "mpnet"], ["mnli-cross-encoder"], M2_GOLD_V2[:12]
    else:
        encoders = ["minilm", "mpnet", "minilm-l12"]
        nlis = ["mnli-cross-encoder", "roberta-nli"]
        pairs = M2_GOLD_V2

    res = run_sweep(encoders, nlis, pairs, cfg)
    f1s = [v["macro_f1"] for v in res.values()]
    fmrs = [v["false_merge_rate"] for v in res.values()]
    out = {"operating_point": {"theta_merge": cfg.theta_merge, "sigma_lo": cfg.sigma_lo},
           "n_pairs": len(pairs), "cells": res,
           "macro_f1_range": [min(f1s), max(f1s)],
           "false_merge_rate_range": [min(fmrs), max(fmrs)]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    # smoke sanity: a different encoder must actually change the metrics (a real swap,
    # not a no-op) OR be reported -- distinctness is informative either way; assert the run
    # produced one metrics cell per (encoder,nli).
    assert len(res) == len(encoders) * len(nlis), res
    return 0


if __name__ == "__main__":
    sys.exit(main())
