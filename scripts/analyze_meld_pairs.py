"""PH-7 / PH-12 tables from a run_meld_pairs.py output (stored signals only, no model):

  meld_literal  the admission decision in the two-claim store (bin 0 of the deployed epoch)
  meld_maxcal   the same pair at the threshold of the epoch's enabled bin with the most Cal
                records (the bin a deployed store of calibration size uses most); a domain whose
                epoch enables no bin, or has no epoch, keeps the literal decision
  meld_bin<k>   every enabled bin's threshold, per domain (in the JSON only)
  submitted     the v1.0.6 predicate at theta_merge 0.90, sigma_lo 0.12 (pair_protocol.submitted)

Each arm: overall and per stratum false merge, silent overwrite, missed contradiction, claim lost,
merge recall with Clopper-Pearson 95% intervals. The report is stamped with this checkout's git
sha and the rows file's sha256; the rows carry the run's stamps.

Run:  PYTHONPATH=src python scripts/analyze_meld_pairs.py --run OUT [--out OUT/analysis.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from meld.eval.pair_protocol import derive, score, submitted   # noqa: E402
from meld.eval.topology import _git_sha                          # noqa: E402
from meld.protocol.objects import INF16                          # noqa: E402


def enabled(table):
    """bin -> (n_g, theta_q) of the bins the table enables."""
    return {t[0]: (t[1], t[3]) for t in table if t[3] < INF16}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    path = os.path.join(a.run, "rows.jsonl")
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    meta = json.load(open(os.path.join(a.run, "meta.json"), encoding="utf-8"))
    en = {d: enabled(e["table"]) for d, e in meta["epochs"].items()}
    maxcal = {d: max(b.items(), key=lambda kv: (kv[1][0], -kv[0])) if b else None for d, b in en.items()}
    bins = sorted({k for b in en.values() for k in b})
    for r in rows:
        r["submitted"] = submitted(sigma=r["sigma"], chi=r["chi_ba"], identical=r["identical"])
        r["meld_literal"] = r["decision"]
        mc = maxcal.get(r["domain"])
        r["meld_maxcal"] = derive(r, mc[1][1]) if mc else r["decision"]
        for k in bins:
            b = en.get(r["domain"], {})
            r[f"meld_bin{k}"] = derive(r, b[k][1]) if k in b else None
    arms = {"meld_literal": score(rows, "meld_literal"), "meld_maxcal": score(rows, "meld_maxcal"),
            "submitted": score(rows, "submitted")}
    for k in bins:
        sub = [r for r in rows if r[f"meld_bin{k}"] is not None]
        arms[f"meld_bin{k}"] = score(sub, f"meld_bin{k}")
    calibrated = [r for r in rows if r["domain"] in meta["epochs"]]
    rep = {"git_sha": _git_sha(), "rows": path, "rows_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
           "n": len(rows), "run_stamps": sorted({(r["git_sha"], r["nli_id"], r["encoder"]) for r in rows}),
           "epoch_ids": {d: e["epoch_id"] for d, e in meta["epochs"].items()},
           "maxcal_bin": {d: (m[0], m[1][0], m[1][1]) if m else None for d, m in maxcal.items()},
           "arms": arms,
           "arms_calibrated_domains": {k: score(calibrated, k)["overall"]
                                       for k in ("meld_literal", "meld_maxcal", "submitted")}}
    print(f"n={rep['n']} maxcal bins (bin, n_g, theta_q): {rep['maxcal_bin']}")
    for arm in ("meld_literal", "meld_maxcal", "submitted"):
        o = arms[arm]["overall"]
        print(arm, {k: o[k]["rate"] for k in ("false_merge", "silent_overwrite", "missed_contradiction",
                                               "claim_lost", "merge_recall")})
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return rep


if __name__ == "__main__":
    main()
