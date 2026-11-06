"""PH-13: single-target miss and false merge by the selector's m with per-m epochs, evaluation only, local.

Per domain (PAWS, SICK, VitaminC) and m, the SEED epoch is rebuilt from the cal split with the released builder
(calibration.calibrate: the cal stream folded under the bootstrap selector (sigma_lo, sigma_c, m), one Cal record
per item, conformal.quantile_threshold per bin), signed with the local dev key (evaluation only; the group key is
not local), and the test split is folded under it in streams of the cal stream's length (benchmark_test.evaluate,
as E2-B3). Labels as E2-B3 / run_resplit.py: combined-annotations-final with AMBIGUOUS as non-mergeable, raw-source
labels, annotator verdicts. Unlabelled tested pairs are not invented: they are counted (cal and test) and left out
of every rate; in calibration they never become Cal records (fold_calibrate), which is the label-store rule.
Check: the m = 3 epoch's table equals the released epoch's.

Per (domain, alpha, m): false merge, merge recall and contradiction miss over tested pairs (E2-B3 metrics); the
share of arrivals whose tested pair differs from m = 3; and over the labelled manifest pairs of the test streams
(the later arrival's target is the earlier text): mergeable pairs by where the target was at the arrival's step
(the tested pair, inside K(b) but not tested, outside K(b)) and whether the two end in one group; CONFLICT-verdict
pairs by whether a CONTRADICTS link joins them. Single-target miss = a mergeable arrival whose target was outside
K(b) and that did not end in the target's group.

Run (repo root):  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python -I scripts/run_wp6e_per_m.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from meld.agent.memory import DEFAULT_POL  # noqa: E402
from meld.config import read_epoch, save_epoch  # noqa: E402
from meld.eval import benchmark_test as B  # noqa: E402
from meld.eval import calibration as C  # noqa: E402
from meld.eval import label_queue as Q  # noqa: E402
from meld.eval import wp6e as W  # noqa: E402
from meld.eval.topology import stamps  # noqa: E402
from meld.protocol.fold import nli_provider  # noqa: E402
from meld.protocol.objects import ALPHA_ONE  # noqa: E402

EPOCH_DIR = {"paws": "a{a}", "sick": "a{a}", "vitaminc": "humanS3-a{a}"}
ENC, NLI = "minilm", "mnli-cross-encoder"
PJ = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731


def rate(k, n):
    return {"k": k, "n": n, "rate": k / n if n else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", nargs="+", default=list(EPOCH_DIR), choices=list(EPOCH_DIR))
    ap.add_argument("--alpha", nargs="+", default=["0.02", "0.05", "0.10"])
    ap.add_argument("--ms", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--out-dir", default=PJ("outputs/wp6e/ph13"))
    a = ap.parse_args(argv)
    assert 3 in a.ms, "m = 3 is the released reference"
    key = open(PJ("outputs/calibration-labelled/key"), "rb").read().strip()
    annf, labf = PJ("outputs/annotate/combined-annotations-final.jsonl"), PJ("outputs/annotate/labels-final.jsonl")
    manifest = json.load(open(PJ("data/bench/manifest.json")))
    lookup = Q.source_labels(PJ("data/bench"))
    ann = Q.with_ambiguous(Q.load_annotations(annf), Q.load_ambiguous(labf), exclude=False)
    verdicts = B.load_verdicts(annf)
    base = nli_provider(NLI)
    memo = {}
    prov = lambda p, h: memo[p, h] if (p, h) in memo else memo.setdefault((p, h), base(p, h))  # noqa: E731
    hw = platform.machine()
    run = {"annotations_sha256": sha(annf), "labels_sha256": sha(labf), "ambiguous": "non-mergeable",
           "source_labels": True, "seed": 0, "hw": hw, "omp": os.environ.get("OMP_NUM_THREADS"),
           "key": "local dev key (outputs/calibration-labelled/key), evaluation only; epochs not group-signed",
           "unlabelled": "counted, never invented; left out of every rate; never a Cal record (fold_calibrate)"}
    os.makedirs(os.path.join(a.out_dir, "epochs"), exist_ok=True)
    rows, lines = [], []
    for d in a.domain:
        cal_rows, test_rows = C.cal_rows(manifest, d, lookup, ann), C.split_rows(manifest, d, "test")
        n_cal = len({t for r in cal_rows for t in (r["text_a"], r["text_b"])})
        for al in a.alpha:
            rel = read_epoch(PJ("outputs/calibration-labelled", EPOCH_DIR[d].format(a=al), f"{d}-seed-a{al}.epoch.json"), key)
            ref_tested = None
            for m in sorted(a.ms, key=lambda x: x != 3):            # m = 3 first: the reference tested pairs
                t0 = time.monotonic()
                sel = tuple(DEFAULT_POL["bootstrap_selector"][:2]) + (m,)
                cal = C.calibrate(cal_rows, scope=d, encoder=ENC, nli=NLI, provider=prov, key=key,
                                  alpha=round(float(al) * ALPHA_ONE), B_0=rel.bins[0], selector=sel, hw=rel.cal_hw,
                                  seed=0, source_label=lookup, annotations=ann, max_unresolved=None)
                ep = cal["epoch"]
                if m == 3:
                    assert ep.table == rel.table, (d, al, "m = 3 does not rebuild the released table")
                save_epoch(ep, key, os.path.join(a.out_dir, "epochs", f"{d}-a{al}-m{m}.epoch.json"))
                ev = B.evaluate(test_rows, ep, key=key, encoder=ENC, nli=NLI, provider=prov, hw=hw, seed=0,
                                source_label=lookup, annotations=ann, verdicts=verdicts, max_unresolved=None,
                                chunk=n_cal)
                tested = {(i, x.b): x.tested for i, c in enumerate(ev["chunks"]) for x in c["decisions"]}
                if m == 3:
                    ref_tested = tested
                diff = sum(tested[k] != ref_tested.get(k) for k in tested)
                tgt, con = Counter(), Counter()
                for c in ev["chunks"]:
                    text = {x.claim_key: x.text for _, x in c["store"].nodes().values()}
                    mg_pairs, cf_pairs = [], []
                    for fs, r in c["pair"].items():
                        x, y = tuple(fs)
                        _, mg, prov_ = Q.resolve(r, text[x], text[y], c["item"][x], c["item"][y], source=d,
                                                 source_label=lookup, annotations=ann)
                        v = B.verdict_of(r, text[x], text[y], c["item"][x], c["item"][y], source=d, mergeable=mg,
                                         prov=prov_, verdicts=verdicts)
                        if mg is True:
                            mg_pairs.append((x, y))
                        if v == "CONFLICT":
                            cf_pairs.append((x, y))
                    for o in W.target_outcomes(c["decisions"], mg_pairs):
                        tgt[o["where"], o["merged"]] += 1
                    for o, ln in zip(W.target_outcomes(c["decisions"], cf_pairs), W.linked(c["decisions"], cf_pairs)):
                        con[o["where"], ln] += 1
                n_mg, n_cf = sum(tgt.values()), sum(con.values())
                mk = ev["summary"]["marginal"]
                row = {**stamps(ENC, NLI, ep, ev["pol"]), "domain": d, "alpha": float(al), "m": m,
                       "released_epoch_table_equal": ep.table == rel.table,
                       "cal_unresolved": cal["stats"]["unresolved"], "cal_labels": cal["labels"],
                       "cal_records": cal["stats"]["cal"], "theta_q": {g: t for g, _, _, t, _ in ep.table},
                       "test_unresolved": ev["unresolved"], "test_labels": ev["labels"], "nodes": ev["nodes"],
                       "tested_pair_differs_from_m3": rate(diff, len(tested)),
                       "false_merge": mk["false_merge"], "merge_recall": mk["merge_recall"],
                       "contradiction_miss": mk["contradiction_miss"], "decisions": mk["decisions"],
                       "mergeable_pairs": n_mg,
                       "target": {f"{w}_{'merged' if g else 'not_merged'}": n for (w, g), n in sorted(tgt.items())},
                       "single_target_miss": rate(tgt["outside_K", False], n_mg),
                       "in_K_untested_miss": rate(tgt["in_K", False], n_mg),
                       "not_merged_with_target": rate(sum(n for (w, g), n in tgt.items() if not g), n_mg),
                       "conflict_pairs": n_cf,
                       "conflict": {f"{w}_{'linked' if g else 'unlinked'}": n for (w, g), n in sorted(con.items())},
                       "conflict_unlinked": rate(sum(n for (w, g), n in con.items() if not g), n_cf),
                       "conflict_unlinked_outside_K": rate(con["outside_K", False], n_cf),
                       "nli_calls": ev["nli_calls"]}
                rows.append(row)
                lines.append(f"{d:<9}{al:>6}{m:>4}{row['cal_unresolved']:>5}{row['test_unresolved']:>5}"
                             f"{diff:>6}{mk['false_merge']['rate']:>8.4f}{mk['merge_recall']['rate']:>8.4f}"
                             f"{row['single_target_miss']['rate']:>8.4f}{row['not_merged_with_target']['rate']:>8.4f}"
                             f"{row['conflict_unlinked']['rate'] if n_cf else float('nan'):>8.4f}"
                             f"{mk['contradiction_miss']['rate'] if mk['contradiction_miss']['n'] else float('nan'):>8.4f}")
                print(lines[-1], f"{time.monotonic() - t0:.0f}s memo {len(memo)}", flush=True)
                with open(os.path.join(a.out_dir, "per-m.json"), "w") as fh:
                    json.dump({"run": run, "rows": rows}, fh, indent=1, default=str)
    head = ("PH-13 single-target miss and false merge by m with per-m epochs (scripts/run_wp6e_per_m.py; meld.eval.wp6e)\n"
            f"git {rows[0]['git_sha']}  encoder {rows[0]['encoder']}  NLI {rows[0]['nli_id']}  hw {hw}  dev-key epochs "
            "(ids per row in per-m.json)\n"
            "unres = unlabelled tested pairs (cal / test, left out); diff = arrivals whose tested pair differs from m = 3; "
            "STmiss = mergeable labelled pair, target outside K, not merged; notM = mergeable pair not merged at all; "
            "CFunl = CONFLICT pair with no CONTRADICTS link; CMiss = E2-B3 contradiction miss on tested pairs\n\n"
            f"{'domain':<9}{'alpha':>6}{'m':>4}{'unrC':>5}{'unrT':>5}{'diff':>6}{'FM':>8}{'recall':>8}{'STmiss':>8}"
            f"{'notM':>8}{'CFunl':>8}{'CMiss':>8}\n")
    with open(os.path.join(a.out_dir, "per-m.txt"), "w") as fh:
        fh.write(head + "\n".join(lines) + "\n")
    print(head + "\n".join(lines))


if __name__ == "__main__":
    main()
