"""E2-P2 calibration burden (evaluation only, meld.eval.portability): realised FM and recall on the
test stream as a function of the Cal records per bin. For each domain the full SEED epoch is built
from the cal split; its Cal draw is subsampled to at most m records per bin (--reps seeded draws per
m), each sub-epoch verified by recomputation and evaluated on the test stream; m = "all" is the full
draw. Every row carries n_min(alpha), the fewest Cal records that give a bin a finite threshold.
A domain whose cal or test stream holds more than --max-unresolved unlabelled tested pairs is refused.

Run:  PYTHONPATH=src python scripts/run_calibration_burden.py --key-file KEY --alpha 0.02 \\
          --annotations ... --labels ... --out-dir outputs/p2
"""
from __future__ import annotations

import argparse
import json
import os

from meld.eval import calibration as C
from meld.eval import portability as P


def main(argv=None, providers=P.provider) -> dict:
    ap = argparse.ArgumentParser()
    P.common_args(ap)
    ap.add_argument("--encoders", nargs="+", default=["minilm", "minilm-l12", "mpnet"])
    ap.add_argument("--nli", default="mnli-cross-encoder")
    ap.add_argument("--domains", nargs="+", choices=sorted(C.DOMAINS), default=sorted(C.DOMAINS))
    ap.add_argument("--m", nargs="+", type=int, default=[10, 20, 50, 100, 200, 300])
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args(argv)
    ctx = P.context(a)
    emit, rows = P.writer(a.out_dir)
    prov = providers(a.nli)
    for enc in a.encoders:
        for d in a.domains:
            row = {**ctx["stamp"], "kind": "burden", "encoder": enc, "nli": a.nli, "cal_domain": d, "test_domain": d}
            try:
                res = P.calibrate(ctx, d, encoder=enc, nli=a.nli, prov=prov)
            except ValueError as e:
                emit({**row, "status": "refused", "reason": str(e)})
                continue
            base = P.base_for(ctx, d, encoder=enc, nli=a.nli, prov=prov)
            full = P.score(ctx, base, res["epoch"], d)
            unres = (res["stats"]["unresolved"], full["unresolved"])
            if res["epoch"] is None or max(unres) > a.max_unresolved:
                emit({**row, "status": "refused", "cal_unresolved": unres[0], "test_unresolved": unres[1],
                      "reason": "no epoch" if res["epoch"] is None else f"unresolved above {a.max_unresolved}"})
                continue
            draws = [("all", 0, res["epoch"], res["cal"])]
            for m in a.m:
                for r in range(a.reps):
                    sub = P.subsample(res["cal"], m, seed=r)
                    draws.append((m, r, P.reepoch(res["epoch"], sub, res["sig"]), sub))
            for m, r, ep, sub in draws:
                ev = full if m == "all" else P.score(ctx, base, ep, d)
                per = {}
                for k in sub:
                    per[k.bin] = per.get(k.bin, 0) + 1
                emit({**row, "status": "ok", "m": m, "rep": r, "cal_per_bin": per, "theta": P.theta_table(ep),
                      "cal_unresolved": unres[0], **ev})
    report = {"stamp": ctx["stamp"], "rows": rows}
    with open(os.path.join(a.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    return report


if __name__ == "__main__":
    main()
