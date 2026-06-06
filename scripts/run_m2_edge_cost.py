#!/usr/bin/env python3
"""M2-edge: measure the per-delta merge cost decomposition on CPU (edge-class).

Times the embedding step (all-MiniLM-L6-v2) and the NLI step
(cross-encoder/nli-deberta-v3-small) separately over N claim pairs, reporting mean +/- std
ms each and the per-delta total, so the manuscript's "edge-deployable" claim is substantiated
on its costliest step. Run on a quiet CPU (no concurrent compute) for clean numbers.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_m2_edge_cost.py --n 200
"""
import argparse, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from smp.merge.signals import EmbeddingSimilarity, chi    # noqa: E402
from smp.merge.m2_gold import M2_GOLD_V2                   # noqa: E402
from smp.eval.per_delta_cost import measure_per_delta_cost  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "m2-edge-cost.json"))
    a = ap.parse_args()

    es = EmbeddingSimilarity(encoder="minilm")
    pairs = [(g, x, y) for g, x, y in M2_GOLD_V2]
    cyc = itertools.cycle(pairs)
    # advance the cycle in lockstep so embed and nli time the SAME pair each iteration
    state = {"cur": pairs[0]}

    def embed_fn():
        state["cur"] = next(cyc)
        a_, b_ = state["cur"][1], state["cur"][2]
        es.sigma(a_, b_)

    def nli_fn():
        a_, b_ = state["cur"][1], state["cur"][2]
        chi(a_, b_, model="mnli-cross-encoder")

    print(f"[run] timing embed + NLI over n={a.n} pairs (CPU) ...", flush=True)
    res = measure_per_delta_cost(embed_fn, nli_fn, n=a.n, warmup=8)
    res["encoder"] = "all-MiniLM-L6-v2"
    res["nli"] = "cross-encoder/nli-deberta-v3-small"
    res["device"] = "cpu"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    print(f"=== per-delta: embed {res['embed_ms_mean']}+-{res['embed_ms_std']} ms, "
          f"NLI {res['nli_ms_mean']}+-{res['nli_ms_std']} ms, "
          f"total {res['total_ms_mean']} ms (NLI is {res['nli_ms_mean']/max(res['total_ms_mean'],1e-9)*100:.0f}% of cost) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
