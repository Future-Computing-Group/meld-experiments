"""WP6e evaluation helpers, evaluation only (the protocol is used as frozen).

PH-10, the E2-P2 card's burden criterion (planning/06-e2-evaluation.md): a draw is adequate when its
macro-F1 (merge / no merge over the tested pairs with a label) is within 0.02 of the test-set oracle's
and its false-merge rate within +0.01 of the oracle's; m* is the smallest m with at least 90 % adequate
draws. ``replay_counts`` decides a tested pair as MERGE iff s > theta of its bin and no
theta-independent gate fired (``blocked``: mask, pair_mask), so it ignores the group gates and is an
upper bound on the realised merges (as meld.eval.resplit). ``oracle`` is the per-bin threshold vector
with the largest replayed macro-F1, optionally under a false-merge cap, found exactly: macro-F1 is a
function of the pooled (TP, FP) alone, so a max-plus convolution over the bins gives the largest TP
for every FP count.

PH-13, ``target_outcomes`` and ``linked``: for labelled pairs (x, y) of one fold, the later arrival's
target is the earlier node; where it was at the arrival's step (the tested pair, inside K but not
tested, outside K) and whether the two ended in one consolidation group; ``linked`` says whether a
CONTRADICTS link joins the pair.

PH-18, ``summary_ms``: median and p95 (linear interpolation) of per-delta times.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def macro_f1(tp: int, fp: int, fn: int, tn: int) -> float:
    """Mean of the MERGE-class and the no-merge-class F1 (an empty class's F1 is 0)."""
    f = lambda t, p, n: 2 * t / (2 * t + p + n) if t else 0.0  # noqa: E731
    return (f(tp, fp, fn) + f(tn, fn, fp)) / 2


def replay_counts(pairs: Sequence[dict], theta: Dict[int, int]) -> Tuple[int, int, int, int]:
    """(TP, FP, FN, TN) when a pair merges iff not blocked and s > theta[bin] (no row: disabled)."""
    tp = fp = n_m = n_n = 0
    for p in pairs:
        hit = not p["blocked"] and p["s_q"] is not None and p["bin"] in theta and p["s_q"] > theta[p["bin"]]
        if p["mergeable"]:
            n_m, tp = n_m + 1, tp + hit
        else:
            n_n, fp = n_n + 1, fp + hit
    return tp, fp, n_m - tp, n_n - fp


def oracle(pairs: Sequence[dict], max_fm: Optional[float] = None):
    """(theta per bin, (TP, FP, FN, TN), macro-F1) of the replayed optimum; see the module doc."""
    n_m = sum(bool(p["mergeable"]) for p in pairs)
    n_n = len(pairs) - n_m
    best = np.full(n_n + 1, -1, np.int64)       # best[f]: largest TP with exactly f false merges
    best[0] = 0
    choice = []
    for g in sorted({p["bin"] for p in pairs}):
        ps = [p for p in pairs if p["bin"] == g]
        cuts = [-1] + sorted({p["s_q"] for p in ps if p["s_q"] is not None})
        opts = [replay_counts(ps, {g: t})[:2] for t in cuts]
        new = np.full_like(best, -1)
        arg = np.full(best.shape, -1, np.int64)
        for k, (tp, fp) in enumerate(opts):
            cand = np.full_like(best, -1)
            cand[fp:] = np.where(best[:len(best) - fp] >= 0, best[:len(best) - fp] + tp, -1)
            better = cand > new
            new[better], arg[better] = cand[better], k
        choice.append((g, cuts, opts, arg))
        best = new
    cap = n_n if max_fm is None else int(np.floor(max_fm * n_n + 1e-9))
    f1 = [macro_f1(int(best[f]), f, n_m - int(best[f]), n_n - f) if best[f] >= 0 else -1.0
          for f in range(cap + 1)]
    f = int(np.argmax(f1))
    theta = {}
    for g, cuts, opts, arg in reversed(choice):    # backtrack the chosen cut per bin
        k = int(arg[f])
        theta[g] = cuts[k]
        f -= opts[k][1]
    counts = replay_counts(pairs, theta)
    return theta, counts, macro_f1(*counts)


def card_mstar(draws: Dict[int, List[Tuple[float, Optional[float]]]], *, oracle_f1: float, oracle_fm: float,
               f1_tol: float = 0.02, fm_tol: float = 0.01, share: float = 0.9) -> dict:
    """Share of adequate (macro-F1, FM) draws per m and the smallest m with at least ``share`` (None if none)."""
    ok = lambda f1, fm: f1 >= oracle_f1 - f1_tol - 1e-12 and fm is not None and fm <= oracle_fm + fm_tol + 1e-12  # noqa: E731
    sh = {m: sum(ok(*d) for d in ds) / len(ds) for m, ds in sorted(draws.items())}
    return {"share": sh, "m_star": next((m for m, s in sh.items() if s >= share - 1e-12), None)}


def target_outcomes(decisions, pairs: Sequence[Tuple]) -> List[dict]:
    """Per labelled pair: {'merged': one group at the end, 'where': tested | in_K | outside_K}."""
    pos = {d.b: i for i, d in enumerate(decisions)}
    group = {}
    for d in decisions:
        group[d.b] = group[d.tested[0]] if d.decision == "MERGE" else d.b
    out = []
    for x, y in pairs:
        t, a = sorted((x, y), key=pos.get)
        d = decisions[pos[a]]
        where = "tested" if d.tested == (t, a) else "in_K" if t in d.K else "outside_K"
        out.append({"merged": group[t] == group[a], "where": where})
    return out


def linked(decisions, pairs: Sequence[Tuple]) -> List[bool]:
    """Whether a CONTRADICTS link joins each pair (either direction)."""
    con = {frozenset(l[1:3]) for d in decisions for l in d.links if l[0] == "CONTRADICTS"}
    return [frozenset(p) in con for p in pairs]


def summary_ms(xs: Sequence[float]) -> dict:
    a = np.asarray(xs, float)
    return {"n": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "mean": float(a.mean())}


__all__ = ["macro_f1", "replay_counts", "oracle", "card_mstar", "target_outcomes", "linked", "summary_ms"]
