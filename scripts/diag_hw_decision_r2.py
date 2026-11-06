#!/usr/bin/env python3
"""Cross-hardware decision diagnostic for the R2 overlap x k MELD cells (wp6c flag 7).

Rebuilds the distributed_merge condition of run_r2_overlap_k.py task by task on this machine and
records, for every step of the GLOBAL brain's admission, how close each decision-relevant comparison
came to its cut: the test s > theta (fixed16 and float), the contradiction predicate p_c > max(p_e, p_n)
on every direction read (fixed16 and float), the selection floor sigma_q >= sigma_lo and the m-th/(m+1)-th
cut, the sigma_c cut, the int8 rounding distance of each delta's embedding, and created_at ties in the
canonical order (broken by delta id, which hashes q). One JSON line per (overlap, task): live units,
recall@k, decision counts and the near-cut events. Read-only on the protocol: it wraps classifier.step
and the cross-encoder's probs to observe, and changes nothing they return.

  OMP_NUM_THREADS=2 PYTHONPATH=<src at 62005af> python scripts/diag_hw_decision_r2.py \
      --epoch E.json --key-file K --overlaps 0.5 0.75 --n 200 --cache-dir C --out OUT.jsonl
"""
import argparse, hashlib, json, os, sys

import numpy as np

from meld.config import read_epoch
from meld.eval.runner_args import read_key_file
from meld.agent.hotpot_loader import load_hotpot_real
from meld.agent.mem_conditions import build_condition
from meld.agent.recall_metrics import recall_at_k_and_mrr
from meld.eval.substrates import eval_pool
from meld.merge import classifier, signals
from meld.protocol.objects import FIXED_ONE, contradicts
from meld.proto import canon

FLOAT = {}          # (premise text, hypothesis text) -> float softmax (p_c, p_e, p_n)
EVENTS = []         # near-cut events of the current build
_orig_probs = signals._CrossEncoderNLI.probs
_orig_step = classifier.step


def _probs(self, a, b):
    p = _orig_probs(self, a, b)
    FLOAT[a, b] = p
    return p


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def _cos(x, y):
    x, y = x.astype(np.int64), y.astype(np.int64)
    nn = int(x @ x) * int(y @ y)
    return int(x @ y) / np.sqrt(nn) if nn else 0.0


def _step(state, b, ctx):
    prior = [state.nodes[k] for k in state.keys]
    dec = _orig_step(state, b, ctx)
    if ctx.sig.issuer != "agent" or dec.decision == "REJECT":      # global brain only
        return dec
    lo, sc = ctx.sigma_lo / FIXED_ONE, ctx.sigma_c / FIXED_ONE
    cos = sorted(((_cos(b.q, n.q), n) for n in prior), key=lambda t: -t[0])
    above = [c for c, _ in cos if c >= lo]
    ev = {"b": _h(b.text), "decision": dec.decision, "n_prior": dec.n_prior}
    near_floor = min((abs(c - lo) for c, _ in cos), default=None)
    ev["sel_floor_margin"] = near_floor
    if len(above) > ctx.m:
        ev["sel_mcut_gap"] = above[ctx.m - 1] - above[ctx.m]
    ev["sigma_c_margin"] = min((abs(_cos(b.q, state.nodes[r].q) - sc) for r in dec.K[1:]), default=None)
    if dec.tested is not None and dec.s_q is not None:
        a_e = state.nodes[dec.tested[0]]
        pf, pb = FLOAT.get((b.text, a_e.text)), FLOAT.get((a_e.text, b.text))
        ev.update(a_e=_h(a_e.text), s_q=dec.s_q, theta_q=dec.theta_q, test_margin_q=dec.s_q - dec.theta_q,
                  b_text=b.text, a_text=a_e.text)
        if pf and pb:
            ev["s_float"] = min(pf[1], pb[1])
            ev["test_margin_float_units"] = min(pf[1], pb[1]) * FIXED_ONE - dec.theta_q - 0.5
    cm = []
    for r in dec.K:
        n = state.nodes[r]
        for p, h in ((b, n), (n, b)):
            pay = ctx.sig._get(p, h)
            if pay is not None:
                f = FLOAT.get((p.text, h.text))
                cm.append({"q": pay[0] - max(pay[1], pay[2]),
                           "f": (f[0] - max(f[1], f[2])) * FIXED_ONE if f else None,
                           "contra": contradicts(pay)})
    if cm:
        ev["contra_margin_q"] = min(abs(c["q"]) for c in cm)
        ev["contra_margin_f"] = min((abs(c["f"]) for c in cm if c["f"] is not None), default=None)
    EVENTS.append(ev)
    return dec


def _int8_half_dist(v):
    v = np.asarray(v, dtype=np.float64)
    x = 127.0 * v / np.abs(v).max()
    return float(np.min(np.abs(np.abs(x - np.floor(x)) - 0.5)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--overlaps", type=float, nargs="+", default=[0.5, 0.75])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tasks", type=int, nargs="*", help="0-based task indices (default all of --n)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    signals._CrossEncoderNLI.probs = _probs
    classifier.step = _step
    epoch = read_epoch(a.epoch, read_key_file(a.key_file))
    tasks = eval_pool(load_hotpot_real(split="validation", config="distractor"))[:a.n]
    idx = a.tasks if a.tasks else range(len(tasks))
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            done = {(r["overlap"], r["task"]) for r in map(json.loads, fh)}
    for ov in a.overlaps:
        for i in idx:
            if (ov, i) in done:
                continue
            EVENTS.clear()
            b = build_condition(tasks[i], "distributed_merge", encoder="minilm", nli="mnli-cross-encoder",
                                epoch=epoch, overlap_fraction=ov, cache_dir=a.cache_dir)
            rec = {k: recall_at_k_and_mrr(b.brain, tasks[i], k=k, source=b.split.source)["recall_at_k"]
                   for k in a.ks}
            store = b.brain.store
            ds = [d for _, d in store.nodes().values()]
            ca = {}
            for d in ds:
                ca.setdefault(d.created_at, []).append(d)
            ties = sum(1 for v in ca.values() if len(v) > 1)
            half = min(_int8_half_dist(b.brain._es._embed(canon(d.text))) for d in ds) if ds else None
            last = {e["b"]: e for e in EVENTS}            # a refold re-runs steps: keep each node's last
            row = {"overlap": ov, "task": i, "live": b.live_units, "stored": b.stored_units,
                   "recall": rec, "n_merge": b.n_merge, "n_relate": b.n_relate, "n_conflict": b.n_conflict,
                   "created_at_ties": ties, "int8_min_half_dist": half, "events": list(last.values())}
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[diag] ov={ov} task={i} live={b.live_units} merges={b.n_merge}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
