"""PH-7 / PH-12: MELD on the baselines' labelled pair protocol (meld.eval.pair_protocol), one fresh
two-claim store per pair, under the deployed epoch of the pair's domain (--epoch DOMAIN=PATH,
read with config.read_epoch under the group key from --key-file); a domain without an epoch is
admitted without one (no similarity merge). Writes <out-dir>/rows.jsonl (one stamped row per pair,
with the signals the predicates read) and <out-dir>/meta.json (each epoch's id, scope and bin
table, the inputs' sha256). scripts/analyze_meld_pairs.py derives every arm from these rows.

Run:  PYTHONPATH=src python scripts/run_meld_pairs.py --pairs runs/wp6b/pairs/shard-*.jsonl \
          --epoch paws=EP/a0.02/paws-seed-a0.02.group.epoch.json --epoch sick=... --epoch vitaminc=... \
          --epoch lme=EP/substrate-r4-a0.05-lme/lme-seed-a0.05.group.epoch.json \
          --key-file KEY --encoder minilm --nli mnli-cross-encoder --out-dir OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from meld.config import read_epoch                                   # noqa: E402
from meld.eval.pair_protocol import meld_rows                        # noqa: E402
from meld.eval.runner_args import add_key_file, runner_key           # noqa: E402
from meld.protocol.objects import canonical_bytes, object_id         # noqa: E402


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--epoch", action="append", default=[], metavar="DOMAIN=PATH")
    add_key_file(ap)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--n", type=int, default=0, help="first N pairs (0 = all)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    key = runner_key(a.key_file)
    paths = dict(e.split("=", 1) for e in a.epoch)
    epochs = {d: read_epoch(p, key) for d, p in paths.items()}
    pairs = [json.loads(x) for f in a.pairs for x in open(f, encoding="utf-8") if x.strip()]
    pairs = [p for p in pairs if p.get("label")]
    pairs = pairs[:a.n] if a.n else pairs
    t0 = time.time()
    rows = meld_rows(pairs, epochs=epochs, encoder=a.encoder, nli=a.nli, key=key)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "rows.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {"config": {k: v for k, v in vars(a).items()}, "n": len(rows), "seconds": round(time.time() - t0, 1),
            "pairs_sha256": {p: sha256(p) for p in a.pairs},
            "epochs": {d: {"path": paths[d], "sha256": sha256(paths[d]),
                           "epoch_id": object_id(canonical_bytes(e)).hex(), "scope": e.scope, "alpha": e.alpha,
                           "bins": list(e.bins), "table": [list(t) for t in e.table]} for d, e in epochs.items()}}
    with open(os.path.join(a.out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"{len(rows)} pairs in {meta['seconds']} s -> {a.out_dir}")
    return meta


if __name__ == "__main__":
    main()
