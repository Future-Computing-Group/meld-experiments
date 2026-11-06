"""Build the claim-pair benchmark from the downloaded sources.

    PYTHONPATH=src python -m meld.bench.build [--data data/bench] [--seed 0]

Expects under --data: paws/{train,test}.parquet (HF google-research-datasets/paws,
labeled_final), vitaminc/{dev,test}.jsonl (vitaminc_flagging.zip),
lme-cleaned/longmemeval_s_cleaned.json, scifact/claims_{train,dev}.jsonl,
sick/SICK.txt (from SICK.zip, Zenodo 2787612).
Writes manifest.json (every pair with labels), the annotation queue in both side
orders (panel_view_ab.jsonl, panel_view_ba.jsonl) with its private key
(panel_key.jsonl: swapped for ab, swapped_ba for ba), and records the benchmark
identity in data/benchmark_provenance.json (G0). With --audit-panel (panel output
rows {pair_id, group, labels}) it also writes the human audit view with
unannounced repeats (audit_view.jsonl) and its private mapping (audit_private.json,
holding the repeat-id secret, the key and the draw with inclusion probabilities).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from collections import Counter

from meld.bench import loaders as L
from meld.bench.export import audit_export, panel_export, panel_queue, write_jsonl
from meld.bench.manifest import build_manifest, record_identity
from meld.eval.benchmark_provenance import PROVENANCE_PATH, REPO_ROOT

INPUTS = ("paws/train.parquet", "paws/test.parquet", "vitaminc/dev.jsonl", "vitaminc/test.jsonl",
          "lme-cleaned/longmemeval_s_cleaned.json", "scifact/claims_train.jsonl", "scifact/claims_dev.jsonl", "sick/SICK.txt")


def _sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "bench"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-panel", help="JSONL of panel output rows {pair_id, group, labels}")
    ap.add_argument("--audit-alloc", default='{"S4": 80, "S2": 40, "S7": 80}', help="JSON audit size per group")
    a = ap.parse_args(argv)
    d = lambda rel: os.path.join(a.data, rel)  # noqa: E731
    vit, lme, dup = Counter(), Counter(), Counter()
    pairs = (L.load_paws(d("paws/train.parquet"), origin_split="dev")      # PAWS train is the dev pool
             + L.load_paws(d("paws/test.parquet"), origin_split="test")
             + L.load_vitaminc(d("vitaminc/dev.jsonl"), origin_split="dev", stats=vit)
             + L.load_vitaminc(d("vitaminc/test.jsonl"), origin_split="test", stats=vit)
             + L.load_lme_ku(d("lme-cleaned/longmemeval_s_cleaned.json"), stats=lme)
             + L.load_scifact_conflicts([d("scifact/claims_train.jsonl"), d("scifact/claims_dev.jsonl")])
             + L.load_sick(d("sick/SICK.txt"))
             + L.load_gold())
    pairs = L.dedupe_pairs(pairs, stats=dup)
    m = build_manifest(pairs, seed=a.seed)
    m["filter_stats"] = {"vitaminc": dict(sorted(vit.items())), "longmemeval": dict(sorted(lme.items())),
                         "dedupe": dict(sorted(dup.items()))}
    mpath = d("manifest.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=1)
    view, view_ba, key = panel_export(panel_queue(m["pairs"], seed=a.seed), seed=a.seed)
    write_jsonl(d("panel_view_ab.jsonl"), view)
    write_jsonl(d("panel_view_ba.jsonl"), view_ba)
    write_jsonl(d("panel_key.jsonl"), key)
    if a.audit_panel:
        with open(a.audit_panel, encoding="utf-8") as fh:
            panel = [json.loads(line) for line in fh if line.strip()]
        av, private = audit_export(m["pairs"], panel, json.loads(a.audit_alloc), seed=a.seed,
                                   secret=secrets.token_hex(16))
        write_jsonl(d("audit_view.jsonl"), av)
        with open(d("audit_private.json"), "w", encoding="utf-8") as fh:
            json.dump(private, fh, ensure_ascii=False, indent=1)
    entry = record_identity(m, PROVENANCE_PATH, manifest_path=os.path.relpath(mpath, REPO_ROOT),
                            files={rel: _sha256(d(rel)) for rel in INPUTS})
    print(json.dumps({"manifest_sha256": entry["manifest_sha256"], "counts": m["counts"],
                      "filter_stats": m["filter_stats"], "panel_queue": len(view)}, indent=1))
    return entry


if __name__ == "__main__":
    main()
