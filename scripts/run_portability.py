"""E2-P1 portability grid and transfer matrix (evaluation only, meld.eval.portability).

For every encoder x NLI cell (plus the in-domain ablation arm, NLI alias ``min-deberta-base-pawsx``,
whose PAWS epoch is calibrated on official PAWS validation pairs) and every domain: a SEED epoch from
the domain's cal split, evaluated on the domain's test stream (FM, recall, per stratum), and the
transfer matrix: the epoch of domain X applied to the test stream of domain Y. The selector reads
the encoder, so each encoder tests other pairs; their labels come from the label store, and a cell
whose cal or test stream holds more than --max-unresolved unlabelled tested pairs is refused (no
metrics) and listed in ``needs_labels`` of report.json. Rows: rows.jsonl, one per (cell, X, Y).

Run:  PYTHONPATH=src python scripts/run_portability.py --key-file KEY --alpha 0.02 \\
          --annotations outputs/annotate/combined-annotations-humanS3.jsonl \\
          --labels outputs/annotate/full/labels-humanS3.jsonl --out-dir outputs/p1
"""
from __future__ import annotations

import argparse
import json
import os
import time

from meld.eval import calibration as C
from meld.eval import portability as P
from meld.merge.signals import _SentenceTransformerEncoder

ENCODERS = ("minilm", "minilm-l12", "mpnet")
NLIS = ("mnli-cross-encoder", "deberta-base-nli", "roberta-large-mnli", P.ABLATION)


def main(argv=None, providers=P.provider) -> dict:
    ap = argparse.ArgumentParser()
    P.common_args(ap)
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--nlis", nargs="+", default=list(NLIS))
    ap.add_argument("--domains", nargs="+", choices=sorted(C.DOMAINS), default=sorted(C.DOMAINS))
    a = ap.parse_args(argv)
    ctx = P.context(a)
    emit, rows = P.writer(a.out_dir)
    needs = set()
    for enc in a.encoders:
        for nli in a.nlis:
            prov = providers(nli)
            cell = {**ctx["stamp"], "encoder": enc, "nli": nli,
                    "encoder_id": _SentenceTransformerEncoder._MODEL_IDS.get(enc, enc),
                    "nli_model": P.NLI_IDS.get(nli, f"min({P.NLI_IDS['deberta-base-nli']}, {P.PARA_ID})"
                                               if nli == P.ABLATION else nli)}
            cal, base, why = {}, {}, {}
            for d in a.domains:
                t0 = time.monotonic()
                try:
                    cal[d] = P.calibrate(ctx, d, encoder=enc, nli=nli, prov=prov)
                except ValueError as e:                   # cal rows still awaiting labels
                    why[d] = str(e)
                    continue
                t1 = time.monotonic()
                base[d] = P.base_for(ctx, d, encoder=enc, nli=nli, prov=prov)
                cal[d]["seconds"] = (t1 - t0, time.monotonic() - t1)
            for x in a.domains:
                for y in a.domains:
                    row = {**cell, "kind": "cell" if x == y else "transfer", "cal_domain": x, "test_domain": y}
                    if x in why or y in why:
                        emit({**row, "status": "refused", "reason": why.get(x) or why.get(y)})
                        needs.update(f"{enc}|{nli}|{d}" for d in (x, y) if d in why)
                        continue
                    res, b = cal[x], base[y]
                    ep = res["epoch"]
                    if ep is not None and x != y:
                        ep = P.transfer(ep, b["pol"], y)
                    ev = P.score(ctx, b, ep, y)
                    unres = {"cal": res["stats"]["unresolved"], "test": ev["unresolved"]}
                    row.update(cal_unresolved=unres["cal"], test_unresolved=unres["test"],
                               cal_labels=res["labels"], cal_stats=res["stats"],
                               cal_per_bin={g: n for g, n, *_ in ep.table} if ep else {},
                               theta=P.theta_table(res["epoch"]), disabled_bins=res["disabled_bins"],
                               test_nodes=b["nodes"], test_nli_calls=b["nli_calls"],
                               nli_calls_per_claim=b["nli_calls"] / max(1, b["nodes"]),
                               seconds_cal_fold=cal[x]["seconds"][0], seconds_test_fold=cal[y]["seconds"][1])
                    bad = [d for d, n in ((x, unres["cal"]), (y, ev["unresolved"])) if n > a.max_unresolved]
                    if res["epoch"] is None:
                        emit({**row, "status": "refused", "reason": "no non-mergeable Cal record, no epoch"})
                    elif bad:
                        needs.update(f"{enc}|{nli}|{d}" for d in bad)
                        emit({**row, "status": "refused", "reason": f"unresolved tested pairs above {a.max_unresolved}",
                              "test_by_class_counts": {c: {k: v for k, v in n.items() if k in ("unresolved", "excluded")}
                                                       for c, n in ev["by_class"].items()}})
                    else:
                        emit({**row, "status": "ok", **ev})
    report = {"stamp": ctx["stamp"], "rows": rows, "needs_labels": sorted(needs)}
    with open(os.path.join(a.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(json.dumps({"needs_labels": report["needs_labels"]}))
    return report


if __name__ == "__main__":
    main()
