"""E2-B4 recalibration under shift (evaluation only, meld.eval.portability).

Two arms, each reported before and after recalibration on the same held-out test items:
- domain shift A:B: the SEED epoch of domain A (cal split) applied to domain B's test stream;
- degraded signals D: the clean SEED epoch of domain D applied to D's test stream built under
  embedding noise (--noise, norm relative to the unit embedding) and NLI corruption (--corrupt,
  the seeded fraction of directions whose label slots are permuted), both at test time only.
Recalibration: the test stream's source items are split (seeded) into an audit window
(--audit-frac) and held-out items; the audit window's labelled non-mergeable tested pairs, one per
item, as the deployment observes them (same stream, same degraded signals), form a new SEED epoch
(seq 2, verified by recomputation), and both epochs are scored on the held-out items only.
A row whose audit or held-out pairs hold more than --max-unresolved unlabelled pairs is refused.

Run:  PYTHONPATH=src python scripts/run_recalibration_shift.py --key-file KEY --alpha 0.02 \\
          --shift paws:sick sick:paws --degraded paws sick vitaminc --annotations ... --out-dir outputs/b4
"""
from __future__ import annotations

import argparse
import json
import os
import random

from meld.eval import calibration as C
from meld.eval import portability as P


def main(argv=None, providers=P.provider) -> dict:
    ap = argparse.ArgumentParser()
    P.common_args(ap)
    ap.add_argument("--encoder", default="minilm")
    ap.add_argument("--nli", default="mnli-cross-encoder")
    ap.add_argument("--shift", nargs="*", default=[], metavar="A:B")
    ap.add_argument("--degraded", nargs="*", default=[], choices=sorted(C.DOMAINS))
    ap.add_argument("--noise", type=float, default=0.30)
    ap.add_argument("--corrupt", type=float, default=0.50)
    ap.add_argument("--audit-frac", type=float, default=0.5)
    a = ap.parse_args(argv)
    ctx = P.context(a)
    emit, rows = P.writer(a.out_dir)
    prov = providers(a.nli)
    eps = {}

    def epoch_of(d):
        if d not in eps:
            eps[d] = P.calibrate(ctx, d, encoder=a.encoder, nli=a.nli, prov=prov)
        return eps[d]

    arms = [("domain_shift", *s.split(":")) for s in a.shift] + [("degraded", d, d) for d in a.degraded]
    for arm, x, y in arms:
        row = {**ctx["stamp"], "kind": "shift", "arm": arm, "encoder": a.encoder, "nli": a.nli,
               "cal_domain": x, "test_domain": y, "audit_frac": a.audit_frac}
        res = epoch_of(x)
        tprov, kw = prov, {}
        if arm == "degraded":
            tprov = P.corrupt(prov, a.corrupt, seed=a.seed)
            kw = {"noise": a.noise, "noise_seed": a.seed}
            row.update(nli_corruption=a.corrupt)
        base = P.base_for(ctx, y, encoder=a.encoder, nli=a.nli, prov=tprov, **kw)
        row["noise"] = base["noise"]
        items = sorted(set(base["item"].values()))
        random.Random(a.seed).shuffle(items)
        k = round(a.audit_frac * len(items))
        audit, held = set(items[:k]), set(items[k:])
        rec = C.fold_calibrate([(P._clone(b["store"], y, kinds=("D",)), b["item"], b["pair"], {})
                                for b in base.get("chunks", [base])],
                               scope=y, pol=base["pol"], source=y, encoder=a.encoder, nli=a.nli,
                               provider=P.replay(base),
                               alpha=ctx["alpha"], B_0=a.B0, hw=ctx["hw"], seed=a.seed,
                               source_label=ctx["lookup"], annotations=ctx["ann"], max_unresolved=None)
        audit_cal = [c for c in rec["cal"] if c.item in audit]
        audit_unres = P.score(ctx, base, None, y, items=audit)["unresolved"]
        before_ep = res["epoch"] if x == y else res["epoch"] and P.transfer(res["epoch"], base["pol"], y)
        after_ep = P.reepoch(rec["epoch"], audit_cal, rec["sig"], seq=2) if audit_cal else None
        row.update(audit_items=len(audit), eval_items=len(held), audit_cal=len(audit_cal),
                   audit_unresolved=audit_unres, audit_item_ids=sorted(audit), eval_item_ids=sorted(held))
        for phase, ep in (("before", before_ep), ("after", after_ep)):
            ev = P.score(ctx, base, ep, y, items=held)
            r = {**row, "phase": phase, "theta": P.theta_table(ep), **ev}
            if ep is None:
                emit({**r, "status": "refused", "reason": "no epoch"})
            elif max(audit_unres, ev["unresolved"], res["stats"]["unresolved"]) > a.max_unresolved:
                emit({**r, "status": "refused", "reason": f"unresolved above {a.max_unresolved}"})
            else:
                emit({**r, "status": "ok"})
    report = {"stamp": ctx["stamp"], "rows": rows}
    with open(os.path.join(a.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    return report


if __name__ == "__main__":
    main()
