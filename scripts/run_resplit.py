"""E2-B3 item re-split spread (meld.eval.resplit) for the nine cells PAWS, SICK, VitaminC x alpha 0.02/0.05/0.10,
evaluation only, on the local machine.

Per domain the cal stream is folded once without an epoch and the test split once per alpha under the
dev-signed SEED epoch (bodies identical to the group-signed epochs the Roihu run read; the epoch ids are
checked against the reports), with the Roihu run's labels (annotations, AMBIGUOUS as non-mergeable, source
labels; hashes checked against the reports). Refuses unless the seed-0 Cal draw rebuilds each epoch's
(n, theta) per bin and the realised per-bin false merges equal the Roihu reports
(--reports/<domain>-a<alpha>/strict/benchmark-test-report.json). Then --splits item re-splits per cell.

Writes to --out-dir: resplit.json (run block and one stamped row per cell and enabled bin),
resplit-summary.txt (the table), resplit-pairs-<domain>.json (the pools).
Run (repo root):  OMP_NUM_THREADS=2 .venv/bin/python -I scripts/run_resplit.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from meld.bench.manifest import manifest_hash  # noqa: E402
from meld.config import read_epoch  # noqa: E402
from meld.eval import benchmark_test as B  # noqa: E402
from meld.eval import calibration as C  # noqa: E402
from meld.eval import label_queue as Q  # noqa: E402
from meld.eval import resplit as R  # noqa: E402
from meld.eval.topology import stamps  # noqa: E402
from meld.protocol.fold import nli_provider  # noqa: E402
from meld.protocol.objects import INF16, canonical_bytes, object_id  # noqa: E402

ALPHAS = ("0.02", "0.05", "0.10")
EPOCH_DIR = {"paws": "a{a}", "sick": "a{a}", "vitaminc": "humanS3-a{a}"}   # as the Roihu run's epoch files
ENC, NLI = "minilm", "mnli-cross-encoder"
P = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", nargs="+", default=list(EPOCH_DIR), choices=list(EPOCH_DIR))
    ap.add_argument("--splits", type=int, default=4000)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--reports", default=P("outputs/wp6-results/wp6b/b3"))
    ap.add_argument("--out-dir", default=P("outputs/wp6-results/resplit"))
    a = ap.parse_args(argv)

    key = open(P("outputs/calibration-labelled/key"), "rb").read().strip()
    annf, labf = P("outputs/annotate/combined-annotations-final.jsonl"), P("outputs/annotate/labels-final.jsonl")
    manifest = json.load(open(P("data/bench/manifest.json")))
    lookup = Q.source_labels(P("data/bench"))
    ann = Q.with_ambiguous(Q.load_annotations(annf), Q.load_ambiguous(labf), exclude=False)
    lab = dict(source_label=lookup, annotations=ann)
    verdicts = B.load_verdicts(annf)
    base = nli_provider(NLI)
    memo = {}
    prov = lambda p, h: memo[p, h] if (p, h) in memo else memo.setdefault((p, h), base(p, h))  # noqa: E731
    hw = platform.machine()
    run = {"manifest_sha256": manifest_hash(manifest), "annotations_sha256": sha(annf), "labels_sha256": sha(labf),
           "ambiguous": "non-mergeable", "source_labels": True, "seed": 0, "hw": hw, "splits": a.splits,
           "split_seed": a.split_seed, "gates": "theta-independent gates (mask, pair_mask) replayed; group gates "
           "not replayed, so a re-split rate is an upper bound on that split's false-merge rate"}
    os.makedirs(a.out_dir, exist_ok=True)
    rows, lines = [], []
    for d in a.domain:
        t0 = time.monotonic()
        eps = {al: read_epoch(P("outputs/calibration-labelled", EPOCH_DIR[d].format(a=al), f"{d}-seed-a{al}.epoch.json"), key)
               for al in ALPHAS}
        rep = {al: json.load(open(os.path.join(a.reports, f"{d}-a{al}", "strict", "benchmark-test-report.json")))
               for al in ALPHAS}
        for al, r in rep.items():
            for k in ("manifest_sha256", "annotations_sha256", "labels_sha256", "ambiguous", "source_labels", "seed"):
                assert r["run"][k] == run[k], (d, al, k)
            assert r["domains"][0]["epoch_id"] == object_id(canonical_bytes(eps[al])).hex(), (d, al)
        cal_rows = C.cal_rows(manifest, d, lookup, ann)
        n_cal = len({t for r in cal_rows for t in (r["text_a"], r["text_b"])})
        cal = R.cal_pool(cal_rows, eps["0.02"], key=key, encoder=ENC, nli=NLI, provider=prov, **lab)
        test_rows = C.split_rows(manifest, d, "test")
        dump = {"cal": cal, "test": {}}
        for al in ALPHAS:
            ep = eps[al]
            assert R.seed_table(cal, ep.alpha) == {g: (n, t) for g, n, _, t, _ in ep.table}, (d, al, "rebuild")
            ev = B.evaluate(test_rows, ep, key=key, encoder=ENC, nli=NLI, provider=prov, hw=hw, seed=0,
                            verdicts=verdicts, max_unresolved=0, chunk=n_cal, **lab)
            test = R.test_pool(ev, ep, **lab)
            dump["test"][al] = test
            realised = R.merges(test)
            roihu = {int(g): (m["false_merge"]["k"], m["false_merge"]["n"])
                     for g, m in rep[al]["domains"][0]["summary"]["bin"].items() if m["false_merge"]["n"]}
            assert realised == roihu, (d, al, realised, roihu)
            theta = {g: t for g, _, _, t, _ in ep.table}
            ub = R.exceed(test, theta)
            rates = R.resplit(cal + test, ep.alpha, splits=a.splits, seed=a.split_seed)
            st = stamps(ENC, NLI, ep, ev["pol"])
            for g, t in sorted(theta.items()):
                if t >= INF16 or g not in realised:
                    continue
                k, n = realised[g]
                obs_ub = ub[g][0] / n
                row = {**st, "domain": d, "alpha": float(al), "bin": g, "theta_q": t, "cal_n": {x: y for x, y, *_ in ep.table}[g],
                       "fm_k": k, "fm_n": n, "observed": k / n, "observed_upper": obs_ub, "gate_slack": ub[g][0] - k,
                       **R.spread(rates[g], observed=obs_ub, alpha=float(al))}
                rows.append(row)
                lines.append(f"{d:<9}{al:>6}{g:>4}{row['cal_n']:>5}{k:>5}/{n:<5}{row['observed']:>8.4f}{row['gate_slack']:>4}"
                             f"{row['mean']:>8.4f}{row['sd']:>8.4f}{row['q025']:>8.4f}{row['q975']:>8.4f}"
                             f"{row['share_above_alpha']:>8.3f}{row['p_ge_observed']:>8.3f}")
        with open(os.path.join(a.out_dir, f"resplit-pairs-{d}.json"), "w") as fh:
            json.dump(dump, fh, default=list)
        print(f"{d}: {time.monotonic() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out_dir, "resplit.json"), "w") as fh:
        json.dump({"run": run, "rows": rows}, fh, indent=1)
    head = ("E2-B3 item re-split spread per cell and enabled bin (meld.eval.resplit, scripts/run_resplit.py)\n"
            f"git {rows[0]['git_sha']}  encoder {rows[0]['encoder']}  NLI {rows[0]['nli_id']}  hw {hw}  "
            f"splits {a.splits}  split seed {a.split_seed}; epoch ids per row in resplit.json\n"
            f"{run['gates']}; slack = upper-bound exceedances minus realised false merges at the epoch\n"
            "P = P(re-split rate >= observed); share = share of splits above alpha\n\n"
            f"{'domain':<9}{'alpha':>6}{'bin':>4}{'Cal':>5}{'FM k/n':>11}{'obs':>8}{'slk':>4}"
            f"{'mean':>8}{'sd':>8}{'q2.5':>8}{'q97.5':>8}{'share':>8}{'P':>8}\n")
    with open(os.path.join(a.out_dir, "resplit-summary.txt"), "w") as fh:
        fh.write(head + "\n".join(lines) + "\n")
    print(head + "\n".join(lines))


if __name__ == "__main__":
    main()
