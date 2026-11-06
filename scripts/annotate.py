"""Annotation harness CLI (meld.annotate): batches, annotator runs, aggregation and the audit draw.

    PYTHONPATH=src python scripts/annotate.py batches --out outputs/annotate
    PYTHONPATH=src python scripts/annotate.py pilot --out outputs/annotate/pilot
    PYTHONPATH=src python scripts/annotate.py run --dir outputs/annotate --annotator A1 --backend subagent
    PYTHONPATH=src python scripts/annotate.py run --dir outputs/annotate --annotator A2 --backend http \
        --base-url http://HOST:8000/v1 --model Qwen/Qwen2.5-32B-Instruct
    PYTHONPATH=src python scripts/annotate.py aggregate --dir outputs/annotate --annotators A1 A2 A3 \
        --adjudicator A4 [--resolve mergeability] [--audit]

Layout of a run directory: batches/<id>.json (what annotators see), private.json (secret, item map,
input hashes: never shown to an annotator), exchange/<annotator>/ (subagent prompt and response files),
records/<annotator>/<id>.json (validated output with raw responses). ``aggregate`` writes
adjudication/ (the same layout, for the adjudicator), labels.jsonl (every decided pair),
annotations.jsonl (the label store format of meld.eval.label_queue), report.json and, with --audit
once nothing awaits adjudication, audit_view.jsonl and audit_private.json. Every step is idempotent:
``run`` skips batches that have a record, and the subagent backend reports pending prompts.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import secrets
import sys
from collections import Counter

from meld.annotate import aggregate as A
from meld.annotate import batch as B
from meld.bench.export import write_jsonl
from meld.eval.benchmark_provenance import REPO_ROOT

QUEUES = {"bench": (os.path.join("data", "bench"), "panel_"), "fold": (os.path.join("outputs", "label-queue"), "label_queue_")}


def _jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dump(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


def _inputs(root, queues=None):
    """queue -> {view_ab, view_ba, key} paths: QUEUES under ``root``, or ``queues`` (NAME=PATH_PREFIX each)."""
    pre = dict(q.split("=", 1) for q in queues) if queues else \
        {q: os.path.join(root, d, p) for q, (d, p) in QUEUES.items()}
    return {q: {f: f"{p}{f}.jsonl" for f in ("view_ab", "view_ba", "key")} for q, p in pre.items()}


def _load(paths):
    return {q: {f: _jsonl(p) for f, p in fs.items()} for q, fs in paths.items()}


def _sha(paths):
    return {q: {f: hashlib.sha256(open(p, "rb").read()).hexdigest() for f, p in fs.items()} for q, fs in paths.items()}


def _write_batches(out, batches, private):
    for b in batches:
        _dump(os.path.join(out, "batches", f"{b['batch_id']}.json"), b)
    _dump(os.path.join(out, "private.json"), private)


def cmd_batches(a):
    paths = _inputs(a.root, a.queue)
    data = _load(paths)
    secret = a.secret or secrets.token_hex(16)
    batches, items = B.make_batches({q: (d["view_ab"], d["view_ba"]) for q, d in data.items()}, secret=secret,
                                    seed=a.seed, size=a.size, n_repeats=a.repeats, min_gap=a.min_gap)
    _write_batches(a.out, batches, {"secret": secret, "seed": a.seed, "inputs": paths, "sha256": _sha(paths),
                                    "items": items})
    return {"batches": len(batches), "items": len(items), "repeats": sum(m["repeat"] for m in items.values())}


def cmd_pilot(a):
    paths = _inputs(a.root)
    data = _load(paths)
    secret = a.secret or secrets.token_hex(16)
    b, items = B.pilot_batch({q: (d["view_ab"], d["key"]) for q, d in data.items()},
                             n={"bench": a.n_bench, "fold": a.n_fold}, secret=secret, seed=a.seed)
    _write_batches(a.out, [b], {"secret": secret, "seed": a.seed, "inputs": paths, "sha256": _sha(paths),
                                "items": items})
    return {"batch": b["batch_id"], "items": len(items)}


def cmd_run(a):
    if a.backend == "subagent":
        be = B.SubagentBackend(os.path.join(a.dir, "exchange", a.annotator), model=a.model or "subagent")
    elif a.backend == "http":
        be = B.HttpBackend(a.base_url, a.model, temperature=a.temperature, seed=a.seed, max_tokens=a.max_tokens,
                           timeout=a.timeout, api_key=os.environ.get(a.api_key_env) if a.api_key_env else None)
    else:
        be = B.FakeBackend(lambda it: {"verdict": "MERGE" if it["text_a"].lower() == it["text_b"].lower()
                                       else "NO_RELATION", "subtype": None, "overlap_flag": False})
    n = Counter()
    for path in sorted(glob.glob(os.path.join(a.dir, "batches", "*.json"))):
        batch = _json(path)
        out = os.path.join(a.dir, "records", a.annotator, os.path.basename(path))
        if os.path.exists(out):
            n["done"] += 1
            continue
        rec = B.run_batch(batch, be, annotator=a.annotator)
        if rec is None:
            n["pending"] += 1
            continue
        _dump(out, rec)
        n["failed" if rec["labels"] is None else "done"] += 1
    return dict(n)


def _records(d, names):
    return [_json(p) for n in names for p in sorted(glob.glob(os.path.join(d, "records", n, "*.json")))]


def cmd_aggregate(a):
    private = _json(os.path.join(a.dir, "private.json"))
    data = _load(private["inputs"])
    if _sha(private["inputs"]) != private["sha256"]:
        raise SystemExit("queue files changed since the batches were cut")
    keys = {q: d["key"] for q, d in data.items()}
    views_ab = {q: d["view_ab"] for q, d in data.items()}
    votes, repeats = A.collect(_records(a.dir, a.annotators), private["items"])
    agg = A.aggregate(votes, a.annotators, resolve=a.resolve)
    adj_dir = os.path.join(a.dir, "adjudication")
    adj, adj_items = A.adjudication_batches(agg, votes, views_ab, secret=private["secret"], seed=private["seed"],
                                            size=a.size)
    keep = {f"{b['batch_id']}.json" for b in adj}
    for path in glob.glob(os.path.join(adj_dir, "batches", "*.json")):     # batches of an earlier aggregation
        if os.path.basename(path) not in keep:
            if os.path.exists(os.path.join(adj_dir, "records", a.adjudicator, os.path.basename(path))):
                raise SystemExit(f"{path} is answered but no longer drawn: move adjudication/ aside first")
            os.remove(path)
    _write_batches(adj_dir, adj, {"items": adj_items})
    final = A.finalize(agg, A.adjudicated(_records(adj_dir, [a.adjudicator]), adj_items))
    write_jsonl(os.path.join(a.dir, "labels.jsonl"), [dict(f, pair_id=p) for p, f in sorted(final.items())])
    write_jsonl(os.path.join(a.dir, "annotations.jsonl"), A.annotation_rows(final, keys, views_ab))
    rep = A.report(final, votes, repeats, keys, a.annotators)
    rep["pending_adjudication"] = len(agg) - len(final)
    if a.resolve != "verdict":
        rep["resolve"] = a.resolve
        rep["pending_by_reason"] = dict(sorted(Counter(agg[p]["reason"] for p in agg if p not in final).items()))
    _dump(os.path.join(a.dir, "report.json"), rep)
    if a.audit:
        if rep["pending_adjudication"]:
            raise SystemExit("adjudication pending: draw the audit after it")
        view, priv = A.audit_draw(votes, A.rows_of(keys, views_ab), keys, allocation=json.loads(a.audit_alloc),
                                  seed=private["seed"], secret=secrets.token_hex(16), n_repeats=a.audit_repeats,
                                  min_gap=a.audit_min_gap)
        write_jsonl(os.path.join(a.dir, "audit_view.jsonl"), view)
        _dump(os.path.join(a.dir, "audit_private.json"), priv)
    return {"pairs": rep["n_pairs"], "final": rep["n_final"], "paths": rep["paths"],
            "pending_adjudication": rep["pending_adjudication"], "adjudication_batches": len(adj)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("batches", "pilot"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--root", default=REPO_ROOT, help="directory holding data/bench and outputs/label-queue")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--secret", help="item-id key (default: random, stored in private.json)")
    sub.choices["batches"].add_argument("--size", type=int, default=20)
    sub.choices["batches"].add_argument("--repeats", type=int, default=60, help="hidden repeats per orientation")
    sub.choices["batches"].add_argument("--min-gap", type=int, default=50)
    sub.choices["batches"].add_argument("--queue", action="append", metavar="NAME=PATH_PREFIX",
                                        help="queue files PATH_PREFIX{view_ab,view_ba,key}.jsonl in place of "
                                             "the default queues (repeatable)")
    sub.choices["pilot"].add_argument("--n-bench", type=int, default=20)
    sub.choices["pilot"].add_argument("--n-fold", type=int, default=20)
    p = sub.add_parser("run")
    p.add_argument("--dir", required=True)
    p.add_argument("--annotator", required=True)
    p.add_argument("--backend", choices=("subagent", "http", "fake"), required=True)
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=600, help="seconds per HTTP request")
    p.add_argument("--api-key-env", help="environment variable holding the endpoint's API key")
    p = sub.add_parser("aggregate")
    p.add_argument("--dir", required=True)
    p.add_argument("--annotators", nargs="+", required=True)
    p.add_argument("--adjudicator", required=True)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--resolve", choices=("verdict", "mergeability"), default="verdict",
                   help="decide the full verdict, or the merge flag only")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--audit-alloc", default='{"S4": 80, "S2": 40, "S7": 80}')
    p.add_argument("--audit-repeats", type=int, default=60)
    p.add_argument("--audit-min-gap", type=int, default=50)
    a = ap.parse_args(argv)
    if a.cmd == "run" and a.backend == "http" and not (a.base_url and a.model):
        ap.error("--backend http needs --base-url and --model")
    res = {"batches": cmd_batches, "pilot": cmd_pilot, "run": cmd_run, "aggregate": cmd_aggregate}[a.cmd](a)
    json.dump(res, sys.stdout, indent=1)
    print()
    return res


if __name__ == "__main__":
    main()
