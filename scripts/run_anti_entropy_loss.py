"""Anti-entropy loss-injection convergence experiment (driver).

Sweeps routing-loss rates and reports, per loss rate, the per-claim status
AGREEMENT FRACTION across brains WITHOUT the anti-entropy digest (loss is
permanent -> divergence grows) versus WITH periodic anti-entropy rounds (the
digest detects + reconciles the lost links -> agreement returns to 1.000), plus
the DIGEST COST (bytes exchanged + rounds to convergence). Includes the 0.3% loss
the reviewers cited (recall 0.997).

Pure simulation, deterministic in the seeds; sync-layer only (does NOT touch the
merge classifier / M-2 headline). Calls the unit-tested
``smp.eval.anti_entropy_eval``; the printed numbers are the asserted invariants.
Persists the full sweep to gitignored data/anti-entropy/.

Run:  PYTHONPATH=src python3 scripts/run_anti_entropy_loss.py
"""
from __future__ import annotations

import datetime
import json
import os

from smp.eval.anti_entropy_eval import loss_convergence_sweep

LOSSES = [0.0, 0.003, 0.01, 0.05, 0.10]      # incl. the reviewer-cited 0.3%
N_BRAINS = 12
SEEDS = [0, 1, 2, 3, 4]


def main(losses=LOSSES, n_brains=N_BRAINS, seeds=SEEDS) -> dict:
    rep = loss_convergence_sweep(losses=losses, n_brains=n_brains, seeds=seeds)
    rows = rep["rows"]

    print("== Anti-entropy digest: per-claim status convergence under routing loss "
          f"(n_brains={n_brains}, seeds={seeds}) ==")
    print(f"  {'loss':>6} | {'no-digest agree':>15} {'(max #maps)':>12} | "
          f"{'digest agree':>12} | {'bytes/round':>12} {'rounds':>7} {'repaired':>9}")
    print("  " + "-" * 86)
    for row in rows:
        loss = row["loss"]
        wo = row["without_digest_agreement"]
        nmaps = row["without_digest_max_distinct_maps"]
        wd = row["with_digest_agreement"]
        rounds = row["rounds_to_converge_mean"]
        bytes_total = row["digest_bytes_total_mean"]
        bytes_round = bytes_total / rounds if rounds else 0.0
        repaired = row["links_repaired_mean"]
        print(f"  {loss:>6.3f} | {wo:>15.3f} {nmaps:>12d} | "
              f"{wd:>12.3f} | {bytes_round:>12.0f} {rounds:>7.1f} {repaired:>9.1f}")

    # asserted invariants (the manuscript-grade claims)
    for row in rows:
        assert row["with_digest_agreement"] == 1.0, \
            f"digest must restore full agreement at loss={row['loss']}"
        if row["loss"] > 0.0:
            assert row["without_digest_agreement"] < 1.0, \
                f"loss={row['loss']} should degrade no-digest agreement"

    _persist(rep)
    return rep


def _persist(rep: dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(os.path.dirname(here), "data", "anti-entropy")
    os.makedirs(outdir, exist_ok=True)
    payload = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "sweep": rep}
    path = os.path.join(outdir, "loss_convergence.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  [persisted] {path}")


if __name__ == "__main__":
    main()
