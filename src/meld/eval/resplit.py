"""E2-B3 item re-split spread of the per-bin false-merge rate, evaluation only.

The marginal bound of a SEED epoch is an average over calibration sets. ``benchmark_test.spread``
redraws only which tested pair of a fixed cal item is the Cal record; this module also redraws which
items are calibration items. The cal stream is folded once without an epoch (as calibration folds
it) and the test split once under the epoch (``benchmark_test.evaluate``); every non-mergeable
tested pair of either fold is kept as (side, item, bin, s_q, blocked). A split pools the cal and test
items, takes as many cal items as the cal fold had (items with a non-mergeable tested pair), draws
one pair per cal item as its Cal record (as ``calibration.fold_calibrate``), rebuilds each bin's
threshold with ``conformal.quantile_threshold`` and counts, per bin, the test-side pairs with
s > theta. Selection is theta-independent (merge/classifier.py), so tested pairs and scores are the
same under every threshold; each pair stays in the stream it was folded in.

Gates. A pair with s > theta is a non-merge when a gate fires. The theta-independent gates ("mask"
on a*, "pair_mask" on the tested pair; NLI of fixed pairs only) are replayed exactly: such a pair is
``blocked`` and never counts. The group gates ("group_contradiction", "group_cannot_link",
"cal_veto") read the groups the earlier merges built, which change with theta, and are not
replayed, so a re-split count is an upper bound on the false merges of that split (the bound
covers it: a false merge needs s > theta). ``merges`` gives the realised count under the given
epoch, ``exceed`` the bound at the same thresholds; their difference is the slack.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from meld.eval import benchmark_test as B
from meld.eval import calibration as C
from meld.merge.conformal import quantile_threshold
from meld.proto import CANON_VERSION
from meld.protocol.fold import readjudicate
from meld.protocol.objects import INF16, m_nli

FIXED_GATES = ("mask", "pair_mask")


def _pairs(decisions, us, side: str) -> List[dict]:
    ds = [d for d in decisions if d.tested is not None]
    assert len(ds) == len(us)
    return [{"side": side, "item": (side, u["chunk"], u["item"]), "bin": u["bin"], "s_q": d.s_q,
             "blocked": any(g in FIXED_GATES for g in d.gates), "decision": d.decision}
            for d, u in zip(ds, us) if u["mergeable"] is False]


def _units(decisions, store, item, pair, *, epoch, chunk=0, **lab):
    us = B.units(decisions, store, item, pair, source=epoch.scope, B_0=epoch.bins[0], chunk=chunk, **lab)
    for u, d in zip(us, [d for d in decisions if d.tested is not None]):
        u["item"] = item[d.tested[1]]       # the arriving node's item, as fold_calibrate keys it
    return us


def cal_pool(cal_rows: Sequence[dict], epoch, *, key: bytes, encoder: str, nli: str, provider, seed: int = 0,
             source_label=None, annotations=None) -> List[dict]:
    """The non-mergeable tested pairs of the cal stream folded without an epoch (bootstrap selector)."""
    sel = (epoch.sigma_lo_q, epoch.sigma_c_q, epoch.m)
    store, _, item, pair = C.stream(cal_rows, scope=epoch.scope, encoder=encoder, selector=sel, key=key, seed=seed)
    res = readjudicate(store, epoch.scope, m_nli=m_nli(encoder, nli, "int8", CANON_VERSION), provider=provider,
                       issuer=C.AUTHOR, hw=epoch.cal_hw)
    if res.epoch is not None:
        raise ValueError("the cal stream must fold without an epoch")
    us = _units(res.decisions, store, item, pair, epoch=epoch, source_label=source_label, annotations=annotations)
    return _pairs(res.decisions, us, "cal")


def test_pool(ev: dict, epoch, *, source_label=None, annotations=None) -> List[dict]:
    """The non-mergeable tested pairs of ``benchmark_test.evaluate``'s chunks, with their decisions."""
    out = []
    for i, c in enumerate(ev["chunks"]):
        us = _units(c["decisions"], c["store"], c["item"], c["pair"], epoch=epoch, chunk=i,
                    source_label=source_label, annotations=annotations)
        out += _pairs(c["decisions"], us, "test")
    return out


def _by_item(pairs, side=None) -> Dict[tuple, List[dict]]:
    by = defaultdict(list)
    for p in pairs:
        if side is None or p["side"] == side:
            by[p["item"]].append(p)
    return by


def seed_table(pairs: Sequence[dict], alpha: int, seed: int = 0) -> Dict[int, tuple]:
    """Per bin, (Cal records, threshold) of the Cal set ``fold_calibrate`` draws with ``seed`` from the
    cal side (items in sorted order, one pair each, pairs in fold order): the epoch table's (n, theta)."""
    rng, by = random.Random(seed), _by_item(pairs, "cal")
    cal = [rng.choice(by[it]) for it in sorted(by)]
    return {g: (sum(p["bin"] == g for p in cal), quantile_threshold([p["s_q"] for p in cal if p["bin"] == g], alpha))
            for g in sorted({p["bin"] for p in cal})}


def exceed(pairs: Sequence[dict], theta: Dict[int, int]) -> Dict[int, tuple]:
    """Per bin of the test side: (pairs with s > theta and no theta-independent gate, pairs)."""
    out = defaultdict(lambda: [0, 0])
    for p in pairs:
        if p["side"] == "test":
            c = out[p["bin"]]
            c[0] += not p["blocked"] and p["s_q"] is not None and p["s_q"] > theta.get(p["bin"], INF16)
            c[1] += 1
    return {g: tuple(c) for g, c in sorted(out.items())}


def merges(pairs: Sequence[dict]) -> Dict[int, tuple]:
    """Per bin of the test side: (realised MERGE decisions, pairs)."""
    out = defaultdict(lambda: [0, 0])
    for p in pairs:
        if p["side"] == "test":
            out[p["bin"]][0] += p["decision"] == "MERGE"
            out[p["bin"]][1] += 1
    return {g: tuple(c) for g, c in sorted(out.items())}


def resplit(pairs: Sequence[dict], alpha: int, *, splits: int = 4000, seed: int = 0) -> Dict[int, np.ndarray]:
    """Per bin, the test-side exceedance rate (``exceed``) of each of ``splits`` item re-splits (module
    doc); NaN for a split with no test pair in the bin."""
    by = _by_item(pairs)
    keys = sorted(by)
    n_cal = sum(k[0] == "cal" for k in keys)
    bins = sorted({p["bin"] for p in pairs})
    # flat arrays: pair -> item index, bin, score, blocked
    it = np.array([i for i, k in enumerate(keys) for _ in by[k]])
    gb = np.array([p["bin"] for k in keys for p in by[k]])
    s = np.array([-1 if p["s_q"] is None else p["s_q"] for k in keys for p in by[k]])
    ok = ~np.array([p["blocked"] or p["s_q"] is None for k in keys for p in by[k]])
    size = np.array([len(by[k]) for k in keys])
    start = np.concatenate([[0], np.cumsum(size)[:-1]])
    rng = np.random.default_rng(seed)
    out = {g: np.empty(splits) for g in bins}
    for r in range(splits):
        cal = rng.permutation(len(keys))[:n_cal]
        pick = start[cal] + (rng.random(n_cal) * size[cal]).astype(int)
        theta = np.full(max(bins) + 1, INF16)
        for g in np.unique(gb[pick]):
            theta[g] = quantile_threshold(s[pick][gb[pick] == g].tolist(), alpha)
        test = np.ones(len(keys), bool)
        test[cal] = False
        t = test[it]
        hit = t & ok & (s > theta[gb])
        for g in bins:
            n = np.sum(t & (gb == g))
            out[g][r] = np.sum(hit & (gb == g)) / n if n else np.nan
    return out


def spread(rates: np.ndarray, *, observed: float, alpha: float) -> dict:
    """Mean, SD, 2.5/97.5 percentiles, share above ``alpha`` and P(rate >= ``observed``) over the splits."""
    r = rates[~np.isnan(rates)]
    return {"splits": int(r.size), "mean": float(r.mean()), "sd": float(r.std()),
            "q025": float(np.quantile(r, 0.025)), "q975": float(np.quantile(r, 0.975)),
            "share_above_alpha": float(np.mean(r > alpha)), "p_ge_observed": float(np.mean(r >= observed - 1e-12))}


__all__ = ["FIXED_GATES", "cal_pool", "test_pool", "seed_table", "exceed", "merges", "resplit",
           "spread"]
