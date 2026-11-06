"""Order-shuffle harness (eval/order_shuffle.py) on the running example or the 26-seed gold
workload: K seeded orders folded one at a time on a fresh sink replica each (one provider, one
model instance, reused across orders); prints the distinct post-pass fingerprints (target 1), the
five divergence counters across orders before and after the pass (Sig reported, not a gate), the
orders with a pre-pass window, L, distinct top-k reads per probe query (running example: two
questions; gold: each seed's canonical sentence), and for the gold workload purity and typed-edge
miss rates. The report and each E2-T1 row carry topology.stamps.
With --rescope the epoch's table, selector and Cal root are carried onto the workload's scope and
policy (the running example has its own policy, scope "jur", for which no epoch is calibrated): the
record is rebuilt with that scope and M_pol in-process and the report names the source epoch id
(epoch_rescoped_from). An evaluation device; the rescoped record is not a calibrated epoch.
With --e2t1-epochs (gold workload, one epoch file per m, each its own cal_root) it also prints
E2-T1: the missed-link rate against m, one fold per epoch, with per-seed hits in --out.

Without --epoch every theta is +inf: no similarity merge (stated in the output). An epoch file
(config.save_epoch) must carry the workload's M_pol (scope "jur" for the running example,
"agent" for gold) and the run's M_nli.

Run:  PYTHONPATH=src python scripts/run_order_shuffle.py --encoder minilm --nli mnli-cross-encoder \
          --workload gold --epoch EPOCH.json [--k 200] [--out outputs/order-shuffle/gold.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

from meld.agent.memory import DEFAULT_POL
from meld.config import read_epoch
from meld.eval.order_shuffle import e2t1, gold_queries, gold_workload, run_order_shuffle, running_example
from meld.eval.topology import stamps
from meld.protocol.fold import nli_provider
from meld.protocol.objects import canonical_bytes, m_pol_of, object_id
from meld.sync.node import BrainNode
from meld.eval.runner_args import add_key_file, runner_key


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--workload", choices=("running", "gold"), required=True)
    ap.add_argument("--epoch")
    add_key_file(ap)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--e2t1-epochs", nargs="+", default=[])
    ap.add_argument("--rescope", action="store_true", help="carry --epoch onto the workload's scope and policy")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.rescope and not a.epoch:
        ap.error("--rescope needs --epoch")
    key = runner_key(a.key_file)
    epoch = read_epoch(a.epoch, key) if a.epoch else None
    provider = nli_provider(a.nli, own=True)        # orders never coexist: one model instance
    if a.workload == "running":
        pol, seed_objs, work = running_example(a.encoder)
        roles, scope, queries = None, "jur", ["How long must customer data be retained?",
                                              "When must customer data be deleted?"]
    else:
        pol, seed_objs, scope = DEFAULT_POL, [], "agent"
        work, roles = gold_workload(a.encoder)
        queries = gold_queries()
    source = epoch
    if a.rescope:
        epoch = dataclasses.replace(epoch, scope=scope, m_pol=m_pol_of(pol))
    rep = run_order_shuffle(seed_objs, work, k=a.k, seed=a.seed, scope=scope, roles=roles,
                            probe_queries=queries,
                            make_node=lambda i: BrainNode("sink", encoder=a.encoder, nli=a.nli, pol=pol,
                                                          epoch=epoch, provider=provider))
    rep["epoch"] = "none: no similarity merge" if epoch is None else a.epoch
    rep.update(stamps(a.encoder, a.nli, epoch, pol))
    if a.rescope:
        rep["epoch_rescoped_from"] = object_id(canonical_bytes(source)).hex()
    for f in ("epoch", "distinct_fingerprints", "post_pass", "pre_pass", "purity", "topk_distinct"):
        print(f"{f}: {rep[f]}")
    print(f"decision variants: {len(rep['decision_variants'])} (sequences in --out)")
    print(f"L max: {max(rep['L'])}; final misses max: {max(rep['final_misses'])}; not_last mean: "
          f"{sum(rep['not_last']) / len(rep['not_last']):.2f}; orders with a pre-pass window: "
          f"{rep['orders_with_window']}/{a.k}")
    if roles:
        print(f"edges before the pass: {rep['edges_pre']}\nedges after the pass: {rep['edges_post']}")
    if a.e2t1_epochs:
        if not roles:
            raise SystemExit("--e2t1-epochs needs --workload gold")
        eps = [read_epoch(f, key) for f in a.e2t1_epochs]
        rep["e2t1"] = e2t1(work, roles, eps, scope=scope,
                           make_node=lambda ep: BrainNode("sink", encoder=a.encoder, nli=a.nli,
                                                          pol=pol, epoch=ep, provider=provider))
        for ep in eps:
            rep["e2t1"][ep.m].update(stamps(a.encoder, a.nli, ep, pol))
        for m, row in rep["e2t1"].items():
            print(f"E2-T1 m={m}: miss {row['miss_rate']} group-level {row['group_miss_rate']} "
                  f"spurious links {row['spurious_links']} merges {row['spurious_merges']} L {row['L']}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return rep


if __name__ == "__main__":
    main()
