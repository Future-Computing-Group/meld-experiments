"""wp6h: budget-free semantic routing rules, calibrated on HotpotQA train, measured on the multi-host testbed workload.

Rules (router.select_peers, one parameter each): R1 floor sigma >= tau; R2 margin sigma >= max - delta
(floor 0.12 as on the testbed); R3 largest-gap cut with the floor tau as the bottom score.

TEST = load_hotpot_claims(200) (validation/distractor, the multi-host testbed workload). CAL = HotpotQA train
supporting facts, disjoint from TEST by question id and title (asserted). Workload cache in
outputs/wp6h-routing-cal/cache (never the home-dir cache). Models as on the testbed brains: minilm,
mnli-cross-encoder, the HotpotQA r4 alpha 0.05 epoch (test-key copy of the released record).

Modes (each writes outputs/wp6h-routing-cal/<mode>.json):
  anchor  run_sync, N 4, replication 2, top_k 1, sigma_lo 0.12, 15 seeds (testbed recall 0.9965)
  agree   pre-ingest decisions vs run_sync decisions on a few CAL cells, per rule
  cal     pre-ingest sweep per rule over CAL cells; picks each rule's parameter and the winner
  test    run_sync on TEST for every arm (calibrated rules, top_k = needed, top_k unset, broadcast)

Run: OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python scripts/run_wp6h_routing_cal.py MODE [--ns 3,4,8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from meld.config import load_epoch
from meld.protocol.objects import EpochRecord, from_bytes, signed
from meld.eval.campaign_stats import bootstrap_ci
from meld.merge.signals import EmbeddingSimilarity
from meld.protocol.fold import nli_provider
from meld.sync.experiment import run_sync
from meld.sync.node import BrainNode
from meld.sync.router import select_peers
from meld.sync.wire_delta import group_key
from meld.sync.workload import build_workload, load_hotpot_claims

OUT = Path("outputs/wp6h-routing-cal")
CACHE = OUT / "cache"
EPOCH = "outputs/release-staging/epochs/substrate-r4-a0.05-hotpot/hotpot-seed-a0.05.group.epoch.json"
ENC, NLI, FLOOR = "minilm", "mnli-cross-encoder", 0.12
SEEDS = list(range(1, 16))
REPS = [2, 3]
GRID = [round(x, 3) for x in np.arange(0.0, 1.0001, 0.005)]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------- data

def _stream(split, n, skip_titles=frozenset()):
    """(qid, text, title) supporting facts in stream order, as workload._stream_hotpot builds them;
    a question touching a title in ``skip_titles`` is skipped whole."""
    from datasets import load_dataset
    out = []
    for item in load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True):
        ctx, sf = item["context"], item["supporting_facts"]
        cmap = {t: s for t, s in zip(ctx["title"], ctx["sentences"])}
        pairs = list(zip(sf["title"], sf["sent_id"]))
        if skip_titles & {t for t, _ in pairs}:
            continue
        for title, sid in pairs:
            sents = cmap.get(title, [])
            if 0 <= sid < len(sents):
                out.append((item["id"], f"{title}: {sents[sid]}".strip(), title))
                if len(out) >= n:
                    return out
    return out


def splits():
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "splits.json"
    if p.exists():
        d = json.loads(p.read_text())
    else:
        test_rows = _stream("validation", 200)
        test_titles = frozenset(t for _, _, t in test_rows)
        cal_rows = _stream("train", 200, skip_titles=test_titles)
        d = {"test": test_rows, "cal": cal_rows}
        p.write_text(json.dumps(d))
    test = load_hotpot_claims(200, cache_dir=str(CACHE))
    assert [tuple(x) for x in test] == [(t, ti) for _, t, ti in d["test"]], "TEST != load_hotpot_claims(200)"
    tq, cq = {q for q, _, _ in d["test"]}, {q for q, _, _ in d["cal"]}
    tt, ct = {t for _, _, t in d["test"]}, {t for _, _, t in d["cal"]}
    assert not (tq & cq), "CAL and TEST share a question"
    assert not (tt & ct), "CAL and TEST share a title"
    assert not ({x for _, x, _ in d["test"]} & {x for _, x, _ in d["cal"]}), "CAL and TEST share a claim"
    cal = [(t, ti) for _, t, ti in d["cal"]]
    meta = {"test_sha256": hashlib.sha256((CACHE / "hotpot_distractor_validation_n200.json").read_bytes()).hexdigest(),
            "test_questions": len(tq), "test_titles": len(tt), "cal_questions": len(cq), "cal_titles": len(ct),
            "cal_split": "train", "disjoint_by": ["question id", "title", "claim text"]}
    return test, cal, meta


# ---------------------------------------------------------------- scoring

def score(deliveries, needed):
    """Per-cell counts over every published delta: (src, text, delivered peers)."""
    num = den = tot = 0
    for src, text, got in deliveries:
        need = set(needed[text]) - {src}
        num, den, tot = num + len(set(got) & need), den + len(need), tot + len(got)
    return {"recall_num": num, "recall_den": den, "delivered": tot, "n_deltas": len(deliveries),
            "recall": num / den if den else 1.0, "waste": (tot - num) / tot if tot else 0.0,
            "per_delta": tot / len(deliveries) if deliveries else 0.0}


def arm_kw(arm, needed_k):
    """routing kwargs of an arm: (routing_mode, sigma_lo, top_k, margin, gap)."""
    kind, p = arm
    return {"R1": ("semantic", p, None, None, False), "R2": ("semantic", FLOOR, None, p, False),
            "R3": ("semantic", p, None, None, True), "topk": ("semantic", FLOOR, needed_k, None, False),
            "unset": ("semantic", FLOOR, None, None, False),
            "broadcast": ("broadcast", 0.0, None, None, False)}[kind]


_PROVIDERS = []


def _provider(i):
    """Replica i's own NLI provider (own model instance, as node_factory), reused across cells."""
    while len(_PROVIDERS) <= i:
        _PROVIDERS.append(nli_provider(NLI, own=True))
    return _PROVIDERS[i]


def run_cell_sync(claims, n, rep, seed, arm, epoch, es):
    ids = [f"b{i}" for i in range(n)]
    wl = build_workload(claims, ids, replication=rep, seed=seed)
    nodes = {b: BrainNode(b, encoder=ENC, nli=NLI, epoch=epoch, provider=_provider(i),
                          cache_dir=str(CACHE / "emb")) for i, b in enumerate(ids)}
    assert all(nd.objects.ids("Ep", "agent") for nd in nodes.values()), "epoch not held"
    for b in ids:
        for text in wl.assignments[b]:
            nodes[b].ingest(text)
    peers = {b: [p for p in ids if p != b] for b in ids}
    mode, lo, k, margin, gap = arm_kw(arm, rep - 1)
    res = run_sync(nodes, peers, routing_mode=mode, encoder=ENC, sigma_lo=lo, top_k=k,
                   margin=margin, gap=gap, es=es, clock=time.monotonic)
    out = score(res.deliveries, wl.needed)
    out["decisions"] = {f"{s}|{t}": sorted(g) for s, t, g in res.deliveries}
    return out


def pre_ingest_scores(claims, es):
    texts = sorted({t for t, _ in claims})
    v = np.stack([es._embed(t) for t in texts])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return {t: i for i, t in enumerate(texts)}, v @ v.T


def pre_ingest_cell(claims, n, rep, seed, idx, sim, loo=False):
    """Per published delta (src, text) the peer scores against PRE-INGEST peer content; ``loo``
    drops the delta's own text from every peer's content (no peer holds an exact copy)."""
    ids = [f"b{i}" for i in range(n)]
    wl = build_workload(claims, ids, replication=rep, seed=seed)
    cols = {b: [idx[t] for t in wl.assignments[b]] for b in ids}
    rows = []
    for b in ids:
        for text in dict.fromkeys(wl.assignments[b]):     # one delta per distinct text, as ingest
            r = sim[idx[text]]
            own = idx[text] if loo else -1
            rows.append((b, text, {p: float(max((r[j] for j in cols[p] if j != own), default=0.0))
                                   for p in ids if p != b}))
    return rows, wl.needed


def apply_rule(rows, arm, needed_k):
    mode, lo, k, margin, gap = arm_kw(arm, needed_k)
    if mode == "broadcast":
        return [(s, t, set(sc)) for s, t, sc in rows]
    return [(s, t, select_peers(sc, sigma_lo=lo, top_k=k, margin=margin, gap=gap)) for s, t, sc in rows]


def ci(vals):
    p, lo, hi = bootstrap_ci(vals, n_boot=10000, seed=0)
    return [round(p, 5), round(lo, 5), round(hi, 5)]


def summarise(cells):
    """cells: list of per-seed score dicts -> point + seed-bootstrap CI per metric, pooled counts."""
    out = {m: ci([c[m] for c in cells]) for m in ("recall", "waste", "per_delta")}
    out.update({k: sum(c[k] for c in cells) for k in ("recall_num", "recall_den", "delivered", "n_deltas")})
    out["seeds"] = len(cells)
    return out


def epoch_and_es():
    """The released r4 record (as on the testbed: same record bytes, re-signed under the run key, here the
    in-process node key on insertion)."""
    received, _ = load_epoch(EPOCH)
    epoch = from_bytes(EpochRecord, received)
    assert signed(epoch, group_key({}))[0] == received, "record bytes changed"
    return epoch, EmbeddingSimilarity(encoder=ENC)


def stamp():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout)
    return {"git_sha": sha, "src_dirty": dirty, "encoder": ENC, "nli": NLI, "epoch_file": EPOCH,
            "epoch_id": "59b847dab1ffa1000f665c6425e6d66a (released group-file record, re-signed in-process)",
            "floor": FLOOR, "seeds": SEEDS, "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def save(name, d):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(d, indent=1, sort_keys=True))
    log(f"wrote {OUT / name}.json")


# ---------------------------------------------------------------- modes

def mode_anchor(a):
    test, _, meta = splits()
    epoch, es = epoch_and_es()
    cells, t0 = [], time.time()
    for s in SEEDS[: a.seeds]:
        c = run_cell_sync(test, 4, 2, s, ("topk", None), epoch, es)
        c.pop("decisions")
        cells.append(c)
        log(f"anchor seed {s}: recall {c['recall']:.4f} deliveries {c['delivered']} ({time.time() - t0:.0f}s)")
    save("anchor", {"stamp": stamp(), "data": meta, "cell": "N 4, replication 2, top_k 1, sigma_lo 0.12",
                    "fivegtn": {"recall": 0.9965, "recall_num": 5979, "recall_den": 6000, "delivered_per_cell": 400},
                    "summary": summarise(cells), "cells": cells, "wall_s": round(time.time() - t0)})


def _calibrate(cal_rows):
    """Per rule: minimise mean waste s.t. mean recall >= 0.99; ties -> higher recall, then the
    median of the tied parameter values."""
    chosen, curves = {}, {}
    for rule in ("R1", "R2", "R3"):
        curve = []
        for p in GRID:
            sc = [score(apply_rule(rows, (rule, p), rep - 1), need) for (rep, rows, need) in cal_rows]
            curve.append({"param": p, "recall": float(np.mean([c["recall"] for c in sc])),
                          "waste": float(np.mean([c["waste"] for c in sc])),
                          "per_delta": float(np.mean([c["per_delta"] for c in sc]))})
        curves[rule] = curve
        ok = [c for c in curve if c["recall"] >= 0.99]
        if not ok:
            chosen[rule] = None
            continue
        best_w = min(c["waste"] for c in ok)
        tied = [c for c in ok if c["waste"] == best_w]
        best_r = max(c["recall"] for c in tied)
        tied = [c for c in tied if c["recall"] == best_r]
        chosen[rule] = dict(tied[len(tied) // 2], tied_range=[tied[0]["param"], tied[-1]["param"]])
    # rules tied on (waste, recall) -> the widest tied parameter interval (robustness to the parameter)
    ranked = sorted((r for r in chosen if chosen[r]), key=lambda r: (
        chosen[r]["waste"], -chosen[r]["recall"], -(chosen[r]["tied_range"][1] - chosen[r]["tied_range"][0])))
    return chosen, curves, (ranked[0] if ranked else None)


def mode_cal(a):
    _, cal, meta = splits()
    _, es = epoch_and_es()
    idx, sim = pre_ingest_scores(cal, es)
    cal_rows = [(rep,) + pre_ingest_cell(cal, n, rep, s, idx, sim) for n in a.ns for rep in REPS for s in SEEDS]
    chosen, curves, winner = _calibrate(cal_rows)
    save("cal", {"stamp": stamp(), "data": meta, "ns": a.ns, "reps": REPS, "grid": [GRID[0], GRID[-1], 0.005],
                 "criterion": "min mean waste s.t. mean recall >= 0.99 over CAL cells; ties: higher recall, then median tied param; winner across rules: lowest waste, higher recall, then widest tied interval",
                 "scoring": "pre-ingest peer content (see agree.json)", "chosen": chosen, "winner": winner,
                 "curves": curves})
    log(f"chosen {json.dumps(chosen)} winner {winner}")


def mode_loo(a):
    """Diagnostic, pre-ingest only: the delta's exact text removed from peer content, so selection
    has to find the topic home from the topic's OTHER claims. Calibrated on CAL as in mode_cal,
    then applied to TEST (pre-ingest)."""
    test, cal, meta = splits()
    _, es = epoch_and_es()
    ci_, cs = pre_ingest_scores(cal, es)
    cal_rows = [(rep,) + pre_ingest_cell(cal, n, rep, s, ci_, cs, loo=True) for n in a.ns for rep in REPS for s in SEEDS]
    chosen, curves, winner = _calibrate(cal_rows)
    ti, ts = pre_ingest_scores(test, es)
    arms = [(r, chosen[r]["param"]) for r in ("R1", "R2", "R3") if chosen[r]]
    arms += [("topk", None), ("unset", None), ("broadcast", None)]
    table = {}
    for n in a.ns:
        for rep in REPS:
            cells = [pre_ingest_cell(test, n, rep, s, ti, ts, loo=True) for s in SEEDS]
            for arm in arms:
                table[f"N{n}-r{rep}-{arm[0]}"] = dict(summarise([score(apply_rule(rows, arm, rep - 1), need)
                                                                 for rows, need in cells]), param=arm[1])
    save("loo", {"stamp": stamp(), "data": meta, "ns": a.ns, "reps": REPS, "scoring": "pre-ingest, exact copy removed",
                 "chosen_on_cal": chosen, "winner_on_cal": winner, "cal_curves": curves, "test_table": table})
    log(f"loo chosen {json.dumps(chosen)} winner {winner}")


def mode_agree(a):
    _, cal, meta = splits()
    epoch, es = epoch_and_es()
    chosen = json.loads((OUT / "cal.json").read_text())["chosen"]
    idx, sim = pre_ingest_scores(cal, es)
    res, t0 = [], time.time()
    for n, rep, s in [(3, 2, 1), (4, 3, 2), (8, 2, 3)]:
        rows, need = pre_ingest_cell(cal, n, rep, s, idx, sim)
        for rule in ("R1", "R2", "R3"):
            if not chosen[rule]:
                continue
            arm = (rule, chosen[rule]["param"])
            pre = {f"{s_}|{t}": sorted(g) for s_, t, g in apply_rule(rows, arm, rep - 1)}
            live = run_cell_sync(cal, n, rep, s, arm, epoch, es)["decisions"]
            same = sum(pre[k] == live.get(k) for k in pre)
            res.append({"n": n, "rep": rep, "seed": s, "rule": rule, "param": arm[1], "deltas": len(pre),
                        "live_deltas": len(live), "agree": same / len(pre)})
            log(f"agree N{n} r{rep} s{s} {rule}: {same}/{len(pre)} ({time.time() - t0:.0f}s)")
    save("agree", {"stamp": stamp(), "data": meta, "cells": res,
                   "agreement": sum(r["agree"] * r["deltas"] for r in res) / sum(r["deltas"] for r in res)})


def mode_test(a):
    test, _, meta = splits()
    epoch, es = epoch_and_es()
    cal = json.loads((OUT / "cal.json").read_text())
    arms = [(r, cal["chosen"][r]["param"]) for r in ("R1", "R2", "R3") if cal["chosen"][r]]
    arms += [("topk", None), ("unset", None), ("broadcast", None)]
    path = OUT / "test_cells.jsonl"
    done = set()
    if path.exists():
        done = {(r["arm"], r["n"], r["rep"], r["seed"]) for r in map(json.loads, path.open())}
    t0, k = time.time(), 0
    total = len(arms) * len(a.ns) * len(REPS) * len(SEEDS)
    with path.open("a") as fh:
        for n in a.ns:
            for rep in REPS:
                for arm in arms:
                    for s in SEEDS:
                        k += 1
                        key = (arm[0], n, rep, s)
                        if key in done:
                            continue
                        c = run_cell_sync(test, n, rep, s, arm, epoch, es)
                        c.pop("decisions")
                        fh.write(json.dumps(dict(c, arm=arm[0], param=arm[1], n=n, rep=rep, seed=s)) + "\n")
                        fh.flush()
                    log(f"test {k}/{total} N{n} r{rep} {arm[0]}: last recall {c['recall']:.4f} ({time.time() - t0:.0f}s)")
    rows = [json.loads(x) for x in path.open()]
    table = {}
    for n in a.ns:
        for rep in REPS:
            for arm in arms:
                cells = [r for r in rows if (r["arm"], r["n"], r["rep"]) == (arm[0], n, rep)]
                table[f"N{n}-r{rep}-{arm[0]}"] = dict(summarise(cells), arm=arm[0], param=arm[1], n=n, rep=rep)
            table[f"N{n}-r{rep}-analytic"] = {"oracle": {"recall": 1.0, "waste": 0.0, "per_delta": rep - 1},
                                              "broadcast_waste": (n - rep) / (n - 1),
                                              "random_matched_budget_recall": (rep - 1) / (n - 1)}
    save("test", {"stamp": stamp(), "data": meta, "ns": a.ns, "reps": REPS, "arms": arms,
                  "winner_on_cal": cal["winner"], "table": table, "wall_s_this_invocation": round(time.time() - t0)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["anchor", "agree", "cal", "test", "loo"])
    ap.add_argument("--ns", default="3,4,8", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--seeds", type=int, default=15)
    a = ap.parse_args()
    assert os.environ.get("OMP_NUM_THREADS") == "2", "set OMP_NUM_THREADS=2"
    {"anchor": mode_anchor, "agree": mode_agree, "cal": mode_cal, "test": mode_test, "loo": mode_loo}[a.mode](a)
    sys.exit(0)
