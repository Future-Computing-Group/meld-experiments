"""E2-T1 as a structural diagnostic: the missed-link rate of the 26-seed gold workload against
ground truth as a function of the selector's m, one fold per m (eval/order_shuffle.e2t1), with a
cluster bootstrap over seeds (eval/e2_links.seed_cluster_ci).

The gold workload is a constructed diagnostic with ground truth, not a calibration domain: it has
no calibration stream, so no epoch can carry a coverage guarantee for it. Every m therefore folds
under one fixed theta, the source epoch's (a substrate epoch of the same scope, M_pol and M_nli),
with only m changed (e2_links.structural_epochs: derived development cal_root per m). The rows
measure how the neighbourhood size changes which ground-truth edges the fold recovers at a fixed
operating point; they make no false-merge-rate claim. The derived epochs are written to --out-dir
under the group key, with the report.

Run:  PYTHONPATH=src python scripts/run_e2t1_structural.py --encoder minilm --nli mnli-cross-encoder \
          --source-epoch outputs/calibration-labelled/substrate-a0.05-hotpot/hotpot-seed-a0.05.group.epoch.json \
          --out-dir outputs/e2t1
"""
from __future__ import annotations

import argparse
import json
import os

from meld.agent.memory import DEFAULT_POL
from meld.config import read_epoch, save_epoch
from meld.eval.e2_links import seed_cluster_ci, structural_epochs
from meld.eval.order_shuffle import e2t1, gold_workload
from meld.eval.topology import stamps
from meld.protocol.fold import nli_provider
from meld.protocol.objects import canonical_bytes, object_id
from meld.sync.node import BrainNode
from meld.sync.wire_delta import group_key


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-epoch", required=True)
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--n-seeds", type=int, default=26)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    key = group_key(os.environ)
    src = read_epoch(a.source_epoch, key)
    eps = structural_epochs(src, a.ms)
    os.makedirs(a.out_dir, exist_ok=True)
    for ep in eps:
        save_epoch(ep, key, os.path.join(a.out_dir, f"structural-m{ep.m}.epoch.json"))
    deltas, roles = gold_workload(a.encoder, n_seeds=a.n_seeds)
    provider = nli_provider(a.nli, own=True)        # folds never coexist: one model instance
    by_m = e2t1(deltas, roles, eps, scope="agent",
                make_node=lambda ep: BrainNode("sink", encoder=a.encoder, nli=a.nli, pol=DEFAULT_POL,
                                               epoch=ep, provider=provider))
    for ep in eps:
        row = by_m[ep.m]
        row["ci"] = seed_cluster_ci(row["per_seed"], n_boot=a.n_boot)
        row.update(stamps(a.encoder, a.nli, ep, DEFAULT_POL))
        print(f"E2-T1 m={ep.m}: " + "; ".join(
            f"{t} miss {v['miss_rate']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for t, v in row["ci"].items())
            + f"; spurious links {row['spurious_links']} merges {row['spurious_merges']} L {row['L']}")
    rep = {"diagnostic": "structural: fixed theta, m varied, no coverage claim on the gold workload",
           "theta_source": a.source_epoch,
           "theta_source_epoch_id": object_id(canonical_bytes(src)).hex(),
           "n_seeds": a.n_seeds, "by_m": by_m}
    with open(os.path.join(a.out_dir, "e2t1.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    return rep


if __name__ == "__main__":
    main()
