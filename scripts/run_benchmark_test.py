"""E2-B3: the benchmark's test split (PAWS, SICK, VitaminC) admitted under each domain's signed SEED
epoch, via meld.eval.benchmark_test. The test stream is built exactly as the calibration stream,
including its length: the test split is partitioned into independent streams of the calibration
stream's node count (seeded, last partial chunk kept), each folded as its own store, outcomes pooled;
labels come from the label store (manifest, then the raw-source lookup over --data unless
--no-source-labels, then --annotations; AMBIGUOUS items from --labels as in run_calibration), and
the run refuses while more than --max-unresolved test tested pairs lack a label. Reports per
domain, marginal over the pooled streams, per stratum, per calibration bin and per chunk: false merges on
non-mergeable tested pairs, merge recall, contradiction miss, RELATE precision, each with a
Clopper-Pearson 95% interval; and the bin coverage (tested decisions per bin against Cal records per
bin, with the share of each in bins the epoch disables); and the group purity and hidden-node share of
the folded streams (E2-B5, benchmark_test.group_purity).

Beside the given epoch it reports the training-conditional variant (Vovk 2012, Prop. 2b; formula
in meld.eval.benchmark_test) at --delta, built from the Cal set the calibration fold draws with
--seed (the report says whether that rebuild is byte-identical to the given epoch), and with
--draws N the realised test metrics of both over N calibration draws (draw seeds 0..N-1) with
per-bin quantiles of the false-merge rate. The variant is applied at test time only.

The epoch file is read with config.read_epoch under the group key from --key-file (raw bytes,
stripped) or SMP_GROUP_KEY; a MAC that fails refuses. Every output row carries topology.stamps
(git sha, M_nli, M_pol, epoch id and seq, model ids) plus the run's configuration. Writes
<out-dir>/benchmark-test-report.json.

Run:  PYTHONPATH=src python scripts/run_benchmark_test.py --domain paws \
          --epoch paws=outputs/calibration-labelled/a0.05/paws-seed-a0.05.group.epoch.json \
          --encoder minilm --nli mnli-cross-encoder --key-file KEY \
          --annotations outputs/annotate/combined-annotations-humanS3.jsonl \
          --labels outputs/annotate/full/labels-humanS3.jsonl --draws 50 --delta 0.05 --out-dir OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

from meld.agent.memory import DEFAULT_POL
from meld.bench.manifest import manifest_hash
from meld.config import read_epoch
from meld.eval import benchmark_test as B
from meld.eval import calibration as C
from meld.eval import label_queue as Q
from meld.eval.benchmark_provenance import REPO_ROOT
from meld.eval.topology import stamps
from meld.protocol.fold import nli_provider
from meld.protocol.objects import canonical_bytes, object_id
from meld.eval.runner_args import read_key_file as read_key

DOMAINS = ("paws", "sick", "vitaminc")


def flat(summary: dict) -> list:
    """(group, key, metrics) rows of a ``benchmark_test.summarize`` result."""
    return [("marginal", None, summary["marginal"])] + \
        [(g, k, m) for g in ("stratum", "bin", "chunk") for k, m in summary[g].items()]


def main(argv=None, provider=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "data", "bench", "manifest.json"))
    ap.add_argument("--domain", nargs="+", choices=DOMAINS, required=True)
    ap.add_argument("--epoch", action="append", required=True, metavar="DOMAIN=PATH",
                    help="the domain's signed epoch file (repeat per domain)")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--selector", nargs=3, type=int, metavar=("SIGMA_LO_Q", "SIGMA_C_Q", "M"),
                    default=DEFAULT_POL["bootstrap_selector"], help="must be the epoch's (checked)")
    ap.add_argument("--seed", type=int, default=0, help="stream shuffle and the given epoch's Cal draw")
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "bench"))
    ap.add_argument("--no-source-labels", action="store_true")
    ap.add_argument("--annotations", help="label store JSONL (meld.eval.label_queue); also the verdicts")
    ap.add_argument("--labels", help="labels.jsonl with the AMBIGUOUS items (default: beside --annotations)")
    ap.add_argument("--ambiguous", choices=("non-mergeable", "exclude"), default="non-mergeable")
    ap.add_argument("--max-unresolved", type=int, default=0, help="test tested pairs allowed without a label")
    ap.add_argument("--delta", type=float, default=0.05, help="training-conditional confidence")
    ap.add_argument("--draws", type=int, default=0, help="calibration draws for the spread (seeds 0..N-1)")
    ap.add_argument("--key-file")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    key = read_key(a.key_file) if a.key_file else (os.environb.get(b"SMP_GROUP_KEY") or b"").strip()
    if not key:
        raise SystemExit("no signing key: pass --key-file or set SMP_GROUP_KEY")
    paths = dict(e.split("=", 1) for e in a.epoch)
    if set(paths) != set(a.domain):
        ap.error(f"--epoch names {sorted(paths)}, --domain {sorted(a.domain)}")
    epochs = {d: read_epoch(paths[d], key) for d in a.domain}      # refuses a MAC that fails
    for d, ep in epochs.items():
        if ep.scope != d or (ep.sigma_lo_q, ep.sigma_c_q, ep.m) != tuple(a.selector):
            ap.error(f"{paths[d]}: scope {ep.scope!r} / selector differ from domain {d!r} / --selector")
    with open(a.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    lookup = None if a.no_source_labels else Q.source_labels(a.data)
    ann = Q.load_annotations(a.annotations) if a.annotations else None
    verdicts = B.load_verdicts(a.annotations) if a.annotations else None
    labels = a.labels or (a.annotations and os.path.join(os.path.dirname(a.annotations), "labels.jsonl"))
    if labels and (a.labels or os.path.exists(labels)):
        ann = Q.with_ambiguous(ann or {}, Q.load_ambiguous(labels), exclude=a.ambiguous == "exclude")
    else:
        labels = None
    cal = {d: C.cal_rows(manifest, d, lookup, ann) for d in a.domain}
    test = {d: C.split_rows(manifest, d, "test") for d in a.domain}
    provider = provider or nli_provider(a.nli)
    hw = platform.machine() or "unknown"
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest() if p else None  # noqa: E731
    run = {"manifest_sha256": manifest_hash(manifest), "annotations_sha256": sha(a.annotations),
           "labels_file": labels, "labels_sha256": sha(labels), "ambiguous": a.ambiguous,
           "source_labels": not a.no_source_labels, "seed": a.seed, "delta": a.delta, "hw": hw}
    report = {"run": run, "domains": [], "rows": []}
    for d in a.domain:
        t0 = time.monotonic()
        ep = epochs[d]
        kw = dict(key=key, encoder=a.encoder, nli=a.nli, provider=provider, hw=hw, seed=a.seed,
                  source_label=lookup, annotations=ann, verdicts=verdicts)
        n_cal = len({t for r in cal[d] for t in (r["text_a"], r["text_b"])})   # the cal stream's nodes
        ev = B.evaluate(test[d], ep, max_unresolved=a.max_unresolved, chunk=n_cal, **kw)
        purity = B.group_purity(ev, source=d, source_label=lookup, annotations=ann)   # before spread refolds
        draws = sorted(set(range(a.draws)) | {a.seed})
        sp = B.spread(cal[d], ev, epoch=ep, draws=draws, delta=a.delta, **kw)
        own = next(r for r in sp if r["draw"] == a.seed)
        if not own["is_given_epoch"]:
            print(f"WARNING {d}: the Cal draw with seed {a.seed} does not rebuild the given epoch; the "
                  "training-conditional variant is of the rebuilt draw", file=sys.stderr)
        alpha = ep.alpha / 1_000_000
        base = {**run, "domain": d, "alpha": alpha, "base_epoch_id": object_id(canonical_bytes(ep)).hex()}
        # the variant rows carry the base epoch's stamps; ``variant`` and ``table`` say what was applied
        blocks = [(None, "marginal", ep, ev["summary"], [list(t) for t in ep.table]),
                  (None, "training_conditional", ep, own["training_conditional"]["summary"],
                   own["training_conditional"]["table"])]
        blocks += [(r["draw"], v, ep, r[v]["summary"], r[v]["table"]) for r in sp if r["draw"] < a.draws
                   for v in ("marginal", "training_conditional")]
        for draw, variant, e, summ, table in blocks:
            st = {**stamps(a.encoder, a.nli, e, ev["pol"]), **base, "draw": draw, "variant": variant,
                  "table": table}
            if draw is not None:        # the draw's own (unsigned, in-process) epoch
                st["epoch_id"] = next(r["epoch_id"] for r in sp if r["draw"] == draw)
            report["rows"] += [{**st, "group": g, "key": k, **m} for g, k, m in flat(summ)]
        spread_summary = B.spread_summary([r for r in sp if r["draw"] < a.draws], alpha) if a.draws else None
        dom = {**stamps(a.encoder, a.nli, ep, ev["pol"]), **base, "epoch_file": paths[d],
               "epoch_table": [list(t) for t in ep.table], "rebuilt_is_given_epoch": own["is_given_epoch"],
               "tc_table": own["training_conditional"]["table"], "labels": ev["labels"],
               "unresolved": ev["unresolved"], "nodes": ev["nodes"], "chunk_size": n_cal,
               "chunks": [{k: c[k] for k in ("items", "texts", "max_bin")} for c in ev["chunks"]],
               "coverage": ev["coverage"], "tc_coverage": own["training_conditional"]["coverage"], "tested": len(ev["units"]),
               "untested": ev["untested"], "nli_calls": ev["nli_calls"], "summary": ev["summary"],
               "tc_summary": own["training_conditional"]["summary"], "draws": a.draws,
               "spread_summary": spread_summary, "group_purity": purity, "seconds": round(time.monotonic() - t0, 1)}
        report["domains"].append(dom)
        m, t = ev["summary"]["marginal"], own["training_conditional"]["summary"]["marginal"]
        print(json.dumps({"domain": d, "unresolved": ev["unresolved"], "labels": ev["labels"],
                          "fm": m["false_merge"], "recall": m["merge_recall"]["rate"], "chunks": len(ev["chunks"]),
                          "test_disabled_share": ev["coverage"]["test_disabled_share"],
                          "group_purity": purity, "tc_fm": t["false_merge"], "tc_recall": t["merge_recall"]["rate"],
                          "seconds": dom["seconds"]}))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "benchmark-test-report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    return report


if __name__ == "__main__":
    main()
