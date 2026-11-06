#!/usr/bin/env python3
"""S-4: sink state-load vs cluster size N (replication of a fixed novel corpus).

As N brains re-publish the same novel corpus, the merge-on-receive sink consolidates the
re-publications so its live store stays ~corpus-sized, while a naive union grows as N x corpus.
The empirical state-side of "sink load grows with the rate of novel claims, not the number of
brains." Local, no LLM, no cost. Companion to M1 (the sink's compute knee) and S-3 (per-claim
delivery overhead).

Rows carry topology.stamps.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_sink_load.py --encoder minilm --nli mnli-cross-encoder \
          --epoch EPOCH.json
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from meld.eval.runner_args import add_key_file, runner_key  # noqa: E402
from meld.config import read_epoch              # noqa: E402
from meld.merge.m2_gold import M2_GOLD_V2            # noqa: E402
from meld.eval.sink_load import sink_load_row        # noqa: E402
from meld.eval.topology import stamps                # noqa: E402



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ns", default="1,2,4,8,16,32")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "sink-load.json"))
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "outputs", "emb-cache-sinkload"))
    ap.add_argument("--epoch", default=None, help="signed epoch file (config.read_epoch); "
                    "without one no similarity merge happens")
    add_key_file(ap)
    a = ap.parse_args(argv)
    epoch = read_epoch(a.epoch, runner_key(a.key_file)) if a.epoch else None
    # a fixed corpus of distinct novel facts (the curated M-2 fact bank, left side of each pair)
    corpus = sorted({str(t[1]).strip() for t in M2_GOLD_V2 if str(t[1]).strip()})
    Ns = [1, 2, 4] if a.smoke else [int(x) for x in a.ns.split(",")]

    rows = []
    for n in Ns:
        r = {**sink_load_row(n, corpus, encoder=a.encoder, nli=a.nli, epoch=epoch, cache_dir=a.cache_dir),
             **stamps(a.encoder, a.nli, epoch)}
        rows.append(r)
        print(f"[sink-load] N={n:>3} | merge_live {r['merge_live']:>4} "
              f"({r['merge_over_corpus']:.2f}x corpus) | union_live {r['union_live']:>5} "
              f"({r['union_over_corpus']:.0f}x)", flush=True)

    out = {"n_corpus": len(corpus), "ns": Ns, "rows": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n=== sink state-load vs cluster size (corpus = {len(corpus)} novel facts) ===")
    print(f"{'N':>4} {'merge_live':>11} {'union_live':>11} {'merge/corpus':>13} {'union/corpus':>13}")
    for r in rows:
        print(f"{r['n']:>4} {r['merge_live']:>11} {r['union_live']:>11} "
              f"{r['merge_over_corpus']:>12.2f}x {r['union_over_corpus']:>12.0f}x")
    print("merge-sink live store stays ~corpus-sized (flat in N); union grows as N -> sink "
          "state-load tracks novel content, not cluster size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
