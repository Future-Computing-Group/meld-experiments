"""Label queue for the tested pairs the fold produces on each split's stream (meld.eval.label_queue).

For every domain and split (cal, test) it builds the stream exactly as calibration does, runs the
fold's selector (encoder only, no NLI), classifies every tested pair (manifest, manifest_pending,
source_labelled, same_item, cross_item; raw-source lookup in PAWS train/test and SICK.txt), and
exports the pairs annotators must label in the annotation export format: label_queue_view_ab.jsonl,
label_queue_view_ba.jsonl and the private label_queue_key.jsonl (export.panel_export), deduplicated
against the existing panel queue. Also writes tested_pairs.jsonl (every tested pair with its class)
and label-queue-report.json (counts per domain, split and class; queue sizes; runtime). With
--annotations (the label store's JSONL, meld.eval.label_queue) annotated pairs leave the queue.

The recall workloads (--domain hotpot lme, meld.eval.substrates) have one split, the cal pool: their
worlds are built as the recall runners build them (--n-brains, --dup, --overlap, --per-store; first
--n-cal cal-pool tasks), every tested pair is classified (copy, same_item, cross_item; no pair has a
manifest or raw-source label) and every unlabelled one is queued. --copy-sample N also writes a
seeded sample of N copy pairs per recall domain as a panel queue (copy_validation_view_ab.jsonl,
copy_validation_view_ba.jsonl, copy_validation_key.jsonl), for the annotators to confirm the copies are
paraphrases. Copies leave the queue as source-labelled mergeable only with --copies-validated (the
annotators' labels of that sample), which refuses unless the validated mergeable rate is at least
label_queue.COPY_MIN_RATE and reports its Wilson lower bound. --world-seed draws the --n-cal tasks of
the cal pool by a seed (the report records the world ids); --no-copies keeps copies out of the queue;
--panel-view takes several files (earlier queues or labelled pairs) whose pairs stay out.
--world-spec (JSON: per recall domain a list of shapes, each cal task ids and the evaluation split,
substrates.spec_worlds) replaces the single-shape options, so one set holds worlds of several cells;
the report records the spec's sha256 and the source items placed in more than one world.

--chunked builds the test split as the evaluators fold it (meld.eval.benchmark_test.chunk_rows):
independent streams of at most the domain's calibration stream node count, the same seed; each
tested test pair records its stream under chunk, and the report the chunk size per domain.

Run:  PYTHONPATH=src python scripts/build_label_queue.py --domain paws sick vitaminc \
          --out-dir data/bench/label-queue
      PYTHONPATH=src python scripts/build_label_queue.py --domain hotpot lme --n-cal 40 \
          --out-dir data/bench/label-queue-substrates
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter

from meld.agent.memory import DEFAULT_POL
from meld.bench.export import panel_export, write_jsonl
from meld.bench.manifest import manifest_hash
from meld.eval import calibration as C
from meld.eval import label_queue as Q
from meld.eval import substrates as S
from meld.eval.benchmark_provenance import REPO_ROOT


def main(argv=None) -> dict:
    bench = os.path.join(REPO_ROOT, "data", "bench")
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(bench, "manifest.json"))
    ap.add_argument("--data", default=bench, help="raw source files for the source-label lookup")
    ap.add_argument("--no-source-labels", action="store_true")
    ap.add_argument("--annotations", help="annotator labels (JSONL, format in meld.eval.label_queue)")
    ap.add_argument("--panel-view", nargs="+", default=[os.path.join(bench, "panel_view_ab.jsonl")],
                    help="queues already annotated or queued (rows with text_a, text_b): their pairs stay out")
    ap.add_argument("--domain", nargs="+", choices=sorted(C.DOMAINS) + list(S.RECALL_DOMAINS), required=True)
    ap.add_argument("--split", nargs="+", choices=("cal", "test"), default=["cal", "test"])
    ap.add_argument("--encoder", default="minilm")
    ap.add_argument("--selector", nargs=3, type=int, metavar=("SIGMA_LO_Q", "SIGMA_C_Q", "M"),
                    default=DEFAULT_POL["bootstrap_selector"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunked", action="store_true",
                    help="test split in streams of the calibration stream's length (as the evaluators fold it)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lme-data", default=os.path.join(bench, "lme-cleaned", "longmemeval_s_cleaned.json"))
    ap.add_argument("--n-cal", type=int, help="cal-pool tasks per recall domain (default all)")
    ap.add_argument("--world-seed", type=int, help="draw the --n-cal tasks by this seed (default: the first)")
    ap.add_argument("--n-brains", type=int, default=3)
    ap.add_argument("--dup", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--per-store", type=int, default=1)
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, "outputs", "emb-cache"))
    ap.add_argument("--copy-sample", type=int, default=0, help="copy pairs per recall domain to validate")
    ap.add_argument("--copies-validated", help="panel labels of the copy sample: copies become mergeable")
    ap.add_argument("--world-spec", help="JSON world spec (substrates.spec_worlds): replaces --n-cal ... --per-store")
    ap.add_argument("--no-copies", action="store_true",
                    help="copies stay out of the label queue (the copy sample still draws from them)")
    a = ap.parse_args(argv)

    if set(a.domain) & set(S.RECALL_DOMAINS) and list(a.selector) != DEFAULT_POL["bootstrap_selector"]:
        ap.error("the recall workloads fold under the evaluation memory's selector (DEFAULT_POL)")
    with open(a.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    existing = []
    for p in a.panel_view:
        with open(p, encoding="utf-8") as fh:
            existing += [json.loads(x) for x in fh if x.strip()]
    lookup = None if a.no_source_labels or not set(a.domain) & set(C.DOMAINS) else Q.source_labels(a.data)
    ann = Q.load_annotations(a.annotations) if a.annotations else None
    try:
        copies = Q.copies_validated(a.copies_validated) if a.copies_validated else None
    except ValueError as e:
        ap.error(str(e))
    tested, queue, counts, seconds, sample, worlds, sizes, repeated = [], [], {}, {}, [], {}, {}, {}
    spec, spec_stamp = S.load_spec(a.world_spec)
    for d in a.domain:
        if d in S.RECALL_DOMAINS:
            t0 = time.monotonic()
            w = S.spec_worlds({d: spec[d]}, data=a.lme_data)[d] if spec else \
                S.cal_worlds(d, data=a.lme_data, n=a.n_cal, n_brains=a.n_brains, dup_degree=a.dup,
                             overlap=a.overlap, per_store=a.per_store, seed=a.seed, pick=a.world_seed)
            worlds[d] = [x[0] for x in w]
            repeated[d] = len(S.repeated_items(w))
            t = Q.classify_substrate(d, w, encoder=a.encoder, annotations=ann, cache_dir=a.cache_dir,
                                     copies_mergeable=copies is not None)
            seconds[f"{d}/cal"] = round(time.monotonic() - t0, 1)
            counts[f"{d}/cal"] = dict(Counter(r["class"] for r in t), worlds=len(w))
            tested += [dict(r, domain=d) for r in t]
            new = Q.queue_rows(t, source=d, existing=existing + queue)
            queue += new
            cp = [r for r in new if r["stratum"] == "tested_copy"]
            sample += random.Random(a.seed).sample(cp, min(a.copy_sample, len(cp)))
            if a.no_copies:
                queue = [r for r in queue if r["stratum"] != "tested_copy"]
            continue
        for sp in a.split:
            rows = C.split_rows(manifest, d, sp)
            if not rows:
                continue
            chunk = None
            if sp == "test" and a.chunked:
                chunk = sizes[d] = len({t for r in C.split_rows(manifest, d, "cal") for t in (r["text_a"], r["text_b"])})
            t0 = time.monotonic()
            t = Q.classify(rows, scope=d, encoder=a.encoder, selector=a.selector, source_label=lookup,
                           annotations=ann, seed=a.seed, chunk=chunk)
            seconds[f"{d}/{sp}"] = round(time.monotonic() - t0, 1)
            counts[f"{d}/{sp}"] = {c: n for c, n in Counter(r["class"] for r in t).items()}
            tested += [dict(r, domain=d) for r in t]
            queue += Q.queue_rows(t, source=d, existing=existing + queue)
    view, view_ba, key = panel_export(queue, seed=a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    out = lambda f: os.path.join(a.out_dir, f)  # noqa: E731
    write_jsonl(out("label_queue_view_ab.jsonl"), view)
    write_jsonl(out("label_queue_view_ba.jsonl"), view_ba)
    write_jsonl(out("label_queue_key.jsonl"), key)
    write_jsonl(out("tested_pairs.jsonl"), tested)
    if a.copy_sample:
        for f, rows in zip(("view_ab", "view_ba", "key"), panel_export(sample, seed=a.seed)):
            write_jsonl(out(f"copy_validation_{f}.jsonl"), rows)
    report = {"manifest_sha256": manifest_hash(manifest), "encoder": a.encoder, "selector": list(a.selector),
              "seed": a.seed, "substrate": {k: getattr(a, k) for k in ("n_cal", "n_brains", "dup", "overlap",
                                                                          "per_store")}, "world_seed": a.world_seed, "world_spec": spec_stamp, "repeated_items": repeated, "worlds": worlds, "chunk_size": sizes, "counts": counts, "seconds": seconds,
              "new_rows_by_stream": dict(Counter(f"{r['source']}/{r['split']}/{r['stratum']}" for r in queue)),
              "copies_validated": copies, "copy_sample": dict(Counter(r["source"] for r in sample)),
              "existing_panel_rows": len(existing), "new_rows": len(queue),
              "total_panel_rows": len(existing) + len(queue)}
    with open(out("label-queue-report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
