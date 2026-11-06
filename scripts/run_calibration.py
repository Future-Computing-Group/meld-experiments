"""SEED epochs from the benchmark's cal split (design r3 §4.1-4.3, §4.6, §7.4): one signed,
recomputation-verified epoch per domain scope under one encoder x NLI substrate (M_nli), via
meld.eval.calibration. Reads only the cal split; refuses a domain whose cal truth awaits
annotation (VitaminC today). Every tested pair is labelled by the label store (manifest, then the
raw-source lookup over --data unless --no-source-labels, then --annotations), and a cal row counts
as labelled when the store labels it; the adjudicator's AMBIGUOUS items (--labels, default the
labels.jsonl beside --annotations) are non-mergeable, or with --ambiguous exclude kept out of Cal
(``-ambx`` epoch file), counted per class in the report (provenance adjudicated_ambiguous); while any tested pair
is unresolved the epoch is refused, unless --allow-unresolved builds a development epoch
(``<domain>-seed-a<alpha>-dev.epoch.json``, ``development`` and the unresolved counts in the report,
a warning on stderr). The copy pairs of a recall workload are labelled by the annotators unless
--copies-validated (the annotators' labels of the copy sample, scripts/build_label_queue.py
--copy-sample) makes them source-labelled mergeable; it refuses unless the validated mergeable
rate is at least label_queue.COPY_MIN_RATE (Wilson lower bound in the report). Writes ``<domain>-seed-a<alpha>.epoch.json`` (config.save_epoch,
MAC under the group key from --key-file or SMP_GROUP_KEY; no built-in key) and
``calibration-report.json`` stamped with git sha, M_nli, M_pol, epoch id and seq, per-bin n_g and
theta, disabled bins, model ids and the manifest hash.

Run:  PYTHONPATH=src python scripts/run_calibration.py --domain paws sick --alpha 0.05 \
          --encoder minilm --nli mnli-cross-encoder --key-file KEY --out-dir outputs/calibration
The encoder and NLI are required (deployed: minilm, all-MiniLM-L6-v2, and mnli-cross-encoder,
nli-deberta-v3-small). Defaults: selector = DEFAULT_POL's bootstrap selector; B_0 = 256.

Recall workloads (--domain hotpot lme, meld.eval.substrates): the epoch is calibrated on the
cal-pool worlds built as the recall runners build them (--n-brains, --dup, --overlap, --per-store,
first --n-cal tasks), in the evaluation memory's scope and M_pol, so the recall runners accept it
with --epoch. No tested pair there has a manifest or raw-source label: the strict run refuses until
the annotators have labelled the queue of scripts/build_label_queue.py, and a development run
without annotations has no Cal (every bin disabled, no epoch file), reporting the populated bins
and the source items per bin (the most Cal records each bin can get).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

from meld.agent.memory import DEFAULT_POL
from meld.bench.manifest import manifest_hash
from meld.config import save_epoch
from meld.eval import calibration as C
from meld.eval import label_queue as Q
from meld.eval import substrates as S
from meld.eval.benchmark_provenance import REPO_ROOT
from meld.eval.runner_args import add_key_file, runner_key
from meld.merge.conformal import derive_B0, n_min
from meld.merge.signals import _NLI_MODEL_IDS, _SentenceTransformerEncoder
from meld.protocol.fold import nli_provider
from meld.protocol.objects import ALPHA_ONE, FIXED_ONE, INF16, canonical_bytes, object_id


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def main(argv=None, provider=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "data", "bench", "manifest.json"))
    ap.add_argument("--domain", nargs="+", choices=sorted(C.DOMAINS) + list(S.RECALL_DOMAINS), required=True)
    ap.add_argument("--alpha", choices=("0.02", "0.05", "0.10"), required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--selector", nargs=3, type=int, metavar=("SIGMA_LO_Q", "SIGMA_C_Q", "M"),
                    default=DEFAULT_POL["bootstrap_selector"])
    b0 = ap.add_mutually_exclusive_group()
    b0.add_argument("--B0", type=int, default=256)
    b0.add_argument("--derive-B0", nargs=3, metavar=("Q", "R", "T"),
                    help="B_0 = conformal.derive_B0(alpha, q, r, t)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", type=int, default=1)
    ap.add_argument("--created-at", type=int, default=0, help="epoch created_at (us); fixed for reproducibility")
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "bench"),
                    help="raw source files for the source-label lookup")
    ap.add_argument("--no-source-labels", action="store_true")
    ap.add_argument("--annotations", help="annotator labels (JSONL, format in meld.eval.label_queue)")
    ap.add_argument("--labels", help="the labels.jsonl of scripts/annotate.py, source of the AMBIGUOUS items "
                    "(default: labels.jsonl beside --annotations, if present)")
    ap.add_argument("--ambiguous", choices=("non-mergeable", "exclude"), default="non-mergeable",
                    help="AMBIGUOUS items: non-mergeable (default, conservative for false merges) or "
                    "excluded from Cal (sensitivity run)")
    ap.add_argument("--allow-unresolved", action="store_true",
                    help="development run: build the epoch although tested pairs lack a label")
    add_key_file(ap)
    ap.add_argument("--copies-validated", help="panel labels of the copy sample: copies become mergeable")
    ap.add_argument("--lme-data", default=os.path.join(REPO_ROOT, "data", "bench", "lme-cleaned",
                                                       "longmemeval_s_cleaned.json"))
    ap.add_argument("--n-cal", type=int, help="cal-pool tasks per recall domain (default all)")
    ap.add_argument("--world-seed", type=int, help="draw the --n-cal tasks by this seed (default: the first); match build_label_queue")
    ap.add_argument("--n-brains", type=int, default=3)
    ap.add_argument("--dup", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--per-store", type=int, default=1)
    ap.add_argument("--world-spec", help="JSON world spec (substrates.spec_worlds): replaces --n-cal ... --per-store; "
                    "match build_label_queue")
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, "outputs", "emb-cache"))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    if not (a.key_file or os.environ.get("SMP_GROUP_KEY")):
        raise SystemExit("no signing key: pass --key-file or set SMP_GROUP_KEY")
    key = runner_key(a.key_file)
    alpha = round(float(a.alpha) * ALPHA_ONE)
    B_0 = derive_B0(alpha, *a.derive_B0) if a.derive_B0 else a.B0
    with open(a.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    recall = [d for d in a.domain if d in S.RECALL_DOMAINS]
    if recall and list(a.selector) != DEFAULT_POL["bootstrap_selector"]:
        ap.error("the recall workloads fold under the evaluation memory's selector (DEFAULT_POL)")
    lookup = None if a.no_source_labels or len(recall) == len(a.domain) else Q.source_labels(a.data)
    ann = Q.load_annotations(a.annotations) if a.annotations else None
    try:
        copies = Q.copies_validated(a.copies_validated) if a.copies_validated else None
    except ValueError as e:
        ap.error(str(e))
    labels = a.labels or (a.annotations and os.path.join(os.path.dirname(a.annotations), "labels.jsonl"))
    if labels and (a.labels or os.path.exists(labels)):
        ann = Q.with_ambiguous(ann or {}, Q.load_ambiguous(labels), exclude=a.ambiguous == "exclude")
    else:
        labels = None
    rows = {d: C.cal_rows(manifest, d, lookup, ann) for d in a.domain if d not in recall}  # refuse before any model load
    spec, spec_stamp = S.load_spec(a.world_spec)
    worlds = {d: S.spec_worlds({d: spec[d]}, data=a.lme_data)[d] if spec else
              S.cal_worlds(d, data=a.lme_data, n=a.n_cal, n_brains=a.n_brains, dup_degree=a.dup,
                           overlap=a.overlap, per_store=a.per_store, seed=a.seed, pick=a.world_seed) for d in recall}
    provider = provider or nli_provider(a.nli)
    hw = platform.machine() or "unknown"
    stamp = {"git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
             "manifest_sha256": manifest_hash(manifest), "encoder": a.encoder,
             "encoder_id": _SentenceTransformerEncoder._MODEL_IDS.get(a.encoder, a.encoder),
             "nli": a.nli, "nli_model": _NLI_MODEL_IDS.get(a.nli, a.nli), "hw": hw,
             "alpha": float(a.alpha), "alpha_ppm": alpha, "n_min": n_min(alpha), "B_0": B_0,
             "B_0_rule": f"derived {a.derive_B0}" if a.derive_B0 else "given",
             "selector": list(a.selector), "seed": a.seed}
    report = {"epochs": []}
    for d in a.domain:
        t0 = time.monotonic()
        tol = None if a.allow_unresolved else 0
        if d in recall:
            res = C.calibrate_substrate(d, worlds[d], encoder=a.encoder, nli=a.nli, provider=provider,
                                        alpha=alpha, B_0=B_0, hw=hw, seed=a.seed, seq=a.seq,
                                        created_at=a.created_at, annotations=ann, max_unresolved=tol,
                                        cache_dir=a.cache_dir, copies_mergeable=copies is not None)
        else:
            res = C.calibrate(rows[d], scope=d, encoder=a.encoder, nli=a.nli, provider=provider,
                              key=key, alpha=alpha, B_0=B_0, selector=a.selector, hw=hw, seed=a.seed,
                              seq=a.seq, created_at=a.created_at, source_label=lookup, annotations=ann,
                              max_unresolved=tol)
        ep = res["epoch"]
        n_un = res["stats"]["unresolved"]
        if a.allow_unresolved:
            print(f"WARNING {d}: DEVELOPMENT EPOCH, {n_un} tested pairs UNRESOLVED {res['labels']}",
                  file=sys.stderr)
        name = (f"{d}-seed-a{a.alpha}{'-ambx' if a.ambiguous == 'exclude' else ''}"
                f"{'-dev' if a.allow_unresolved else ''}.epoch.json") if ep else None
        os.makedirs(a.out_dir, exist_ok=True)
        if ep is None:
            print(f"WARNING {d}: no non-mergeable tested pair is labelled, so no Cal and no epoch",
                  file=sys.stderr)
        else:
            save_epoch(ep, key, os.path.join(a.out_dir, name))
        epoch = {"m_nli": ep.m_nli.hex(), "m_pol": ep.m_pol.hex(),
                 "epoch_id": object_id(canonical_bytes(ep)).hex(), "seq": ep.seq, "policy": ep.policy,
                 "mondrian_c": ep.mondrian_c, "cal_root": ep.cal_root.hex(), "cal_hw": ep.cal_hw,
                 "verified": True} if ep else {"epoch_id": None, "verified": False}
        report["epochs"].append({
            **stamp, "domain": d, "strata": list(C.DOMAINS.get(d, ())), "epoch_file": name,
            "scope": next(iter(res["pol"]["scopes"])), "pol": res["pol"], **epoch,
            "substrate": {**{k: getattr(a, k) for k in ("n_cal", "world_seed", "n_brains", "dup", "overlap", "per_store")},
                          "world_spec": spec_stamp} if d in recall else None,
            "repeated_items": len(S.repeated_items(worlds[d])) if d in recall else None,
            "bins": [{"g": g, "n_g": n, "n_c_g": nc, "theta_q": th,
                      "theta": None if th == INF16 else th / FIXED_ONE}
                     for g, n, nc, th, _ in (ep.table if ep else ())],
            "populated_bins": res["populated_bins"], "disabled_bins": res["disabled_bins"],
            "items_by_bin": {str(g): n for g, n in res["items_by_bin"].items()},
            "stats": res["stats"], "labels": res["labels"], "unresolved": n_un,
            "development": a.allow_unresolved, "copies_validated": copies if d in recall else None, "source_labels": not a.no_source_labels,
            "annotations_sha256": hashlib.sha256(open(a.annotations, "rb").read()).hexdigest()
            if a.annotations else None, "contradiction_subtype_unlabelled": res.get("unlabelled_subtype"),
            "ambiguous": {"policy": a.ambiguous, "provenance": Q.AMBIGUOUS, "labels_file": labels,
                          "by_class": {c: n["ambiguous"] for c, n in res["labels"].items() if n.get("ambiguous")}},
            "seconds": round(time.monotonic() - t0, 1)})
        print(json.dumps({k: report["epochs"][-1][k] for k in
                          ("domain", "epoch_id", "bins", "populated_bins", "items_by_bin", "disabled_bins",
                           "stats", "labels", "seconds")}))
    with open(os.path.join(a.out_dir, "calibration-report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    return report


if __name__ == "__main__":
    main()
