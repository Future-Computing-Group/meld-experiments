"""E2-T2 typed-link value: recall@k and the stale-fact surfacing proxy under four read modes on one
store per world (eval/e2_links.py: node_only, typed, rewired, untyped), with paired contrasts of
typed against each other mode (bootstrap CI over worlds, dz).

Worlds are built as the recall workloads' calibration substrates build them (eval/substrates.py),
so a substrate epoch applies: ``hotpot`` one HotpotQA evaluation-pool task per store
(split_across_brains, --n-brains, --dup, --overlap); ``lme`` one LongMemEval-cleaned
evaluation-pool question per store (session_split, user turns are claims), question types
--types. Each store is the MELD star sink (topology.build, condition meld). recall@k: HotpotQA
gold supporting sentences (recall_at_k_and_mrr), LongMemEval answer turns (lme_recall); the stale
proxy on knowledge-update items. Rewiring is one seeded draw per world.

Without --epoch no similarity merge happens (stated in the output).

Run:  PYTHONPATH=src python scripts/run_e2t2_links.py --substrate hotpot --n 200 --ks 1 3 5 10 \
          --encoder minilm --nli mnli-cross-encoder --epoch EPOCH.json --out outputs/e2t2/hotpot.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter

from meld.agent.mem_conditions import split_across_brains
from meld.agent.recall_metrics import recall_at_k_and_mrr
from meld.config import read_epoch
from meld.eval import e2_links as L
from meld.eval import topology
from meld.eval.lme_supersession import lme_recall, session_split
from meld.eval.substrates import LME_TYPES, eval_pool
from meld.eval.runner_args import add_read_tau, read_stamp
from meld.sync.wire_delta import group_key

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def worlds(a):
    """(world id, plan, recall(reader, k) -> float or None, LongMemEval item or None)."""
    if a.substrate == "lme":
        with open(a.data, encoding="utf-8") as fh:
            items = eval_pool([x for x in json.load(fh) if x["question_type"] in a.types],
                              key=lambda x: x["question_id"])
        random.Random(a.seed).shuffle(items)
        for it in items[:a.n]:
            plan = session_split(it, n_brains=a.n_brains, dup_degree=a.dup, seed=a.seed)
            yield it["question_id"], plan, (lambda r, k, it=it, p=plan: lme_recall(r, it, k, p.source, tau=a.read_tau)), it
    else:
        from meld.agent.hotpot_loader import load_hotpot_fixture, load_hotpot_real
        tasks = load_hotpot_fixture() if a.data == "fixture" else eval_pool(load_hotpot_real())
        for t in tasks[:a.n]:
            plan = split_across_brains(t, n_brains=a.n_brains, overlap_fraction=a.overlap,
                                       dup_degree=a.dup)
            yield t.task_id, plan, (lambda r, k, t=t, p=plan: recall_at_k_and_mrr(
                r, t, k=k, source=p.source, tau=a.read_tau)["recall_at_k"]), None


def summarize(rows, ks, n_boot):
    out = {}
    for k in map(str, ks):
        rec = [{m: r["recall"][k][m] for m in L.MODES} for r in rows]
        st = [r["stale"][k] for r in rows if r["stale"]]
        out[k] = {"recall": {m: statistics.mean(x[m] for x in rec if x[m] is not None) for m in L.MODES},
                  "contrasts": {f"{a}-{b}": L.paired(rec, a, b, n_boot=n_boot) for a, b in L.CONTRASTS}}
        if st:
            out[k]["stale"] = {"n": len(st), **{
                f: {m: statistics.mean(x[m][f] for x in st) for m in L.MODES}
                for f in ("unflagged_stale", "stale_preferred", "flagged")},
                "contrasts": {f"{f}:{a}-{b}": L.paired([{m: x[m][f] for m in L.MODES} for x in st],
                                                       a, b, n_boot=n_boot)
                              for f in ("unflagged_stale", "stale_preferred") for a, b in L.CONTRASTS}}
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", choices=("hotpot", "lme"), required=True)
    ap.add_argument("--data", help="lme: the cleaned LongMemEval file; hotpot: 'fixture' for the fixture")
    ap.add_argument("--types", nargs="+", default=list(LME_TYPES))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--n-brains", type=int, default=3)
    ap.add_argument("--dup", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "outputs", "emb-cache"))
    ap.add_argument("--out", required=True)
    add_read_tau(ap)
    a = ap.parse_args(argv)
    if a.substrate == "lme":
        a.data = a.data or os.path.join(REPO, "data", "bench", "lme-cleaned", "longmemeval_s_cleaned.json")
    epoch = read_epoch(a.epoch, group_key(os.environ)) if a.epoch else None
    rows = []
    for w, (wid, plan, recall, item) in enumerate(worlds(a)):
        b = topology.build(plan, "meld", "star", encoder=a.encoder, nli=a.nli, epoch=epoch,
                           cache_dir=a.cache_dir)
        mem = b["memory"]
        readers = {m: L.ModeReader(mem, m, seed=a.seed * 1_000_003 + w) for m in L.MODES}
        row = {"world": wid, "live": b["live"], "decisions": b["decisions"],
               "links": dict(Counter(t for t, *_ in readers["typed"].links)),
               "recall": {str(k): {m: recall(r, k) for m, r in readers.items()} for k in a.ks},
               "stale": {str(k): {m: L.stale_proxy(r, item, k, plan.source, r.links, a.read_tau)
                                  for m, r in readers.items()} for k in a.ks} if item else None}
        if row["stale"] and row["stale"][str(a.ks[0])]["typed"] is None:
            row["stale"] = None
        rows.append(row)
        print(f"[e2t2] {w + 1}/{a.n} {wid}: live {b['live']}, links {row['links']}, "
              f"{round(b['wall_s'], 1)}s", file=sys.stderr, flush=True)
    rep = {"substrate": a.substrate, "n_worlds": len(rows), "ks": a.ks, "modes": list(L.MODES),
           "epoch": a.epoch or "none: no similarity merge",
           "config": {k: v for k, v in vars(a).items() if k not in ("out", "cache_dir")},
           **topology.stamps(a.encoder, a.nli, epoch), **read_stamp(a),
           "summary": summarize(rows, a.ks, a.n_boot), "rows": rows}
    for k, s in rep["summary"].items():
        print(f"recall@{k}: " + " ".join(f"{m} {v:.3f}" for m, v in s["recall"].items()))
        for c, v in s["contrasts"].items():
            print(f"  {c}: {v['delta']} [{v['lo']}, {v['hi']}] dz {v['dz']} n {v['n']}")
        if "stale" in s:
            print(f"  stale (n {s['stale']['n']}): unflagged {s['stale']['unflagged_stale']} "
                  f"preferred {s['stale']['stale_preferred']} flagged {s['stale']['flagged']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    return rep


if __name__ == "__main__":
    main()
