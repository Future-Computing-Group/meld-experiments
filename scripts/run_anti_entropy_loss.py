"""Lossy multi-replica convergence over the five replicated sets (eval/anti_entropy_eval.py).

The 26-seed gold workload with authored effects, N replicas (each its own model instance), each
receiving every object in its own seeded order with independent per-delivery loss. Per loss rate:
the post-pass authored divergence WITHOUT anti-entropy, and WITH it (ring DIGEST/PULL rounds, a
Sig exchange, a pass) the largest post-pass divergence counter (0 asserted), the pre-pass window,
rounds, anti-entropy bytes per message class, L and group purity; and the effects-last arm
(deltas in canonical order, then the effects: L = 1 expected); in both healed arms the top-k
reads of the gold probe queries must agree across replicas. Rows carry topology.stamps. Without
--epoch, no similarity merge (stated).

Run:  PYTHONPATH=src python scripts/run_anti_entropy_loss.py --encoder hashing --nli lexical \
          [--epoch EPOCH.json] [--out outputs/anti-entropy/loss.json]
"""
from __future__ import annotations

import argparse
import json
import os

from meld.agent.memory import DEFAULT_POL
from meld.config import read_epoch
from meld.eval.anti_entropy_eval import loss_sweep, with_effects
from meld.eval.order_shuffle import gold_queries, gold_workload
from meld.eval.topology import stamps
from meld.sync.experiment import node_factory
from meld.eval.runner_args import add_key_file, runner_key

LOSSES = [0.0, 0.003, 0.01, 0.05, 0.10]
N_REPLICAS = 12
SEEDS = [0, 1, 2, 3, 4]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--epoch")
    add_key_file(ap)
    ap.add_argument("--n", type=int, default=N_REPLICAS)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    epoch = read_epoch(a.epoch, runner_key(a.key_file)) if a.epoch else None
    deltas, roles = gold_workload(a.encoder)
    rep = loss_sweep(with_effects(deltas, roles, DEFAULT_POL), losses=LOSSES, n=a.n, seeds=SEEDS,
                     scope="agent", roles=roles, probe_queries=gold_queries(),
                     make_node=node_factory(encoder=a.encoder, nli=a.nli, epoch=epoch))
    rep["epoch"] = "none: no similarity merge" if epoch is None else a.epoch
    print(f"epoch: {rep['epoch']}; n={a.n}, seeds={SEEDS}")
    for r in rep["rows"]:
        r.update(stamps(a.encoder, a.nli, epoch, DEFAULT_POL))
        print(r)
        assert r["with_ae_post_pass_max"] == r["final_misses_max"] == r["effects_last_post_pass_max"] == 0, r["loss"]
        assert r["with_ae_topk_distinct_max"] == r["effects_last_topk_distinct_max"] == 1, r["loss"]
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return rep


if __name__ == "__main__":
    main()
