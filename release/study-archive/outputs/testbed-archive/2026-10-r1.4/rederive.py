#!/usr/bin/env python3
"""Independent re-derivation of the testbed r1.4 headlines from the raw archived files.

Stdlib only; imports nothing from the meld package or the drivers, so an aggregation bug there
cannot propagate here. Usage: python3 rederive.py ARCHIVE_DIR  (prints JSON to stdout)
"""
import json, random, statistics, sys
from collections import defaultdict
from pathlib import Path


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def boot_ci(xs, n=10000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed)
    ms = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return [round(statistics.fmean(xs), 4), round(ms[int(0.025 * n)], 4), round(ms[int(0.975 * n) - 1], 4)]


def eval_c(p):
    r = rows(p)
    shas = {x["stamp"]["git_sha"] for x in r}
    return {
        "n_seeds": len(r), "seed_range": [min(x["seed"] for x in r), max(x["seed"] for x in r)], "git_sha": sorted(shas),
        "epoch_ids": sorted({e for x in r for e in x["stamp"]["epoch_ids"]}),
        "agreement_during": boot_ci([x["agreement_during"] for x in r]),
        "crdt_healed": boot_ci([x["crdt_healed"] for x in r]),
        "lww_healed": boot_ci([x["lww_healed"] for x in r]),
        "crdt_reconverged_seeds": sum(x["crdt_healed"] == 1.0 for x in r),
        "lww_reconverged_seeds": sum(x["lww_healed"] == 1.0 for x in r),
        "roots_equal_after": sum(x["roots_equal_after"] for x in r),
        "one_fingerprint": sum(x["fingerprints_distinct"] == 1 for x in r),
        "L_counts": {str(k): sum(x["L"] == k for x in r) for k in sorted({x["L"] for x in r})},
        "resync_rounds": {str(k): sum(x["resync_rounds"] == k for x in r) for k in sorted({x["resync_rounds"] for x in r})},
        "partition_verified": sum(x["partition_verified"] for x in r),
        "cross_side_failed_during": sum(x["cross_side_failed_during"] for x in r),
        "ae_bytes_mean": {k: statistics.fmean(x["ae_bytes"][k] for x in r) for k in ("DIGEST", "PULL", "PATCH")},
    }


def grid(paths):
    by = defaultdict(list)
    shas = set()
    for p in paths:
        for x in rows(p):
            c, res = x["cell"], x["result"]
            if "error" in res or "p50_ms" not in res:
                by[("FAILED",)].append(c)
                continue
            rtt, n, routing, seed = c.split("_")
            by[(routing, int(rtt[3:]), int(n[1:]))].append(res)
            shas.add(res["stamp"]["git_sha"])
    out = {"git_sha": sorted(shas), "cells": {}}
    for k, rs in sorted(by.items(), key=lambda kv: str(kv[0])):
        if k[0] == "FAILED":
            out["failed_cells"] = rs
            continue
        dl = sum(r["delivered_total"] for r in rs)
        rn, rd = sum(r["recall_num"] for r in rs), sum(r["recall_den"] for r in rs)
        cls = defaultdict(lambda: [0, 0])
        for r in rs:
            for c, v in r["bytes"].items():
                cls[c][0] += v["count"]; cls[c][1] += v["bytes"]
        pooled_p50 = [b["p50_ms"] for r in rs for b in r["latency_per_brain"].values() if b.get("p50_ms") is not None]
        out["cells"]["%s rtt%d N%d" % k] = {
            "n_cells": len(rs),
            "p50_ms_brainA": boot_ci([r["p50_ms"] for r in rs]),
            "p95_ms_brainA": boot_ci([r["p95_ms"] for r in rs]),
            "p50_ms_all_brains_mean": round(statistics.fmean(pooled_p50), 3) if pooled_p50 else None,
            "recall": [rn, rd, round(rn / rd, 5) if rd else None],
            "delivered_total": dl, "deliveries_per_cell": round(dl / len(rs), 2),
            "waste": round((dl - rn) / dl, 5) if dl else None,
            "bytes_per_cell": {c: {"count": round(v[0] / len(rs), 1), "bytes": round(v[1] / len(rs), 1)} for c, v in sorted(cls.items())},
            "epoch_held_all": all(all(r["epoch_held"].values()) for r in rs),
            "publish_wall_s_mean": round(statistics.fmean(r["publish_wall_s"] for r in rs), 2),
            "rtt_meas_ms_mean": round(statistics.fmean(r["rtt_meas_ms"] or 0 for r in rs), 2),
        }
    return out


def block(p):
    """Interest-level arm (one checkpoint per arm): per-cell recall, deliveries per published
    delta (published = needed / (replication - 1): each delta is needed by its other homes) and
    wasted rate, each with a seed-as-unit bootstrap CI, and brain A p50."""
    r = [x["result"] for x in rows(p) if "p50_ms" in x["result"]]
    rep = {x["stamp"]["replication"] for x in r}
    assert len(rep) == 1, rep
    rep = rep.pop()
    return {"n_cells": len(r), "replication": rep, "git_sha": sorted({x["stamp"]["git_sha"] for x in r}),
            "cells": sorted({x["stamp"]["routing"] for x in r}),
            "recall": boot_ci([x["recall_num"] / x["recall_den"] for x in r]),
            "recall_pooled": [sum(x["recall_num"] for x in r), sum(x["recall_den"] for x in r)],
            "deliveries_per_delta": boot_ci([x["delivered_total"] / (x["recall_den"] / (rep - 1)) for x in r]),
            "wasted_rate": boot_ci([x["wasted_rate"] for x in r]),
            "p50_ms_brainA": boot_ci([x["p50_ms"] for x in r]),
            "epoch_held_all": all(all(x["epoch_held"].values()) for x in r)}


def main(d):
    d = Path(d)
    out = {}
    if (d / "eval_c_r14.jsonl").exists():
        out["M-a eval-c"] = eval_c(d / "eval_c_r14.jsonl")
    g = sorted(d.glob("campaign_r14_*.jsonl"))
    for p in sorted(d.glob("campaign_r14_block*.jsonl")):
        out["block " + p.stem] = block(p)
    g = [p for p in g if "block" not in p.name]
    full = [p for p in g if "pilot" not in p.name]
    if full:
        out["M-b/M-c grid"] = grid(full)
    pil = [p for p in g if "pilot" in p.name]
    if pil:
        out["pilot"] = grid(pil)
    for p in sorted(d.glob("knee*.json")):
        out["M-f " + p.name] = json.loads(p.read_text())
    for p in sorted(d.glob("m3a-wan-2*.json")):
        k = json.loads(p.read_text())
        t = k.get("trials", [])
        hm = sorted(x["heal_ms"] for x in t if x["converged"])
        out["M-e " + p.name] = {"trials": len(t), "reconverged": sum(x["converged"] for x in t),
                                "diverged_first": sum(x["diverged_first"] for x in t),
                                "heal_ms_p50": statistics.median(hm) if hm else None,
                                "heal_ms_max": hm[-1] if hm else None, "transport": k.get("transport")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
