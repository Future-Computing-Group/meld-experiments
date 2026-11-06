"""Link evaluation: typed-link value (E2-T2) and the missed-link diagnostic's epochs and CIs (E2-T1).

Read modes (E2-T2). Every mode is the store's own ``graph_walk_read`` with the link list it
propagates over substituted for the duration of one read; the store is never changed:

  * ``node_only`` -- no links: the graph term is zero for every node (equal to beta_graph = 0);
  * ``typed``     -- the links as the protocol resolves them (``Store.resolved_links``);
  * ``rewired``   -- the same links with their destinations permuted: every node keeps its out-
                     and in-degree, every edge keeps its type, source and weight, no self-loop
                     where a swap can avoid one (controls for "any edge adds mass");
  * ``untyped``   -- the same edges with the type erased: one type ``LINK`` and one weight, the
                     mean weight of the store's links (the total propagated mass is kept).

The stale-fact surfacing proxy reads a LongMemEval knowledge-update item (``ku_truth``: the old
and the new value turn) under a mode: ``unflagged_stale`` is 1 when the top-k holds the old value
without the new one and without a CONTRADICTS link between them in the mode's link list;
``stale_preferred`` is 1 when the old value is in the top-k and the new one is absent or ranked
below it; ``flagged`` is 1 when such a link exists. Only ``typed`` and ``rewired`` lists carry
CONTRADICTS edges, so the flag exists only where link types are read.

E2-T1. ``structural_epochs`` derives one epoch per neighbourhood size m from a source epoch: theta
table, selector thresholds and every other field unchanged, m replaced, and cal_root a derived
development root (no calibration population of the gold workload exists, so the rows carry no
coverage claim). ``seed_cluster_ci`` is the cluster bootstrap over seeds of e2t1's per-seed hits.
"""
from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from meld.agent.recall_metrics import READ_TAU, recalled_sources
from meld.eval.campaign_stats import bootstrap_ci
from meld.eval.lme_supersession import ku_truth
from meld.eval.order_shuffle import EDGES
from meld.proto import h16

MODES = ("node_only", "typed", "rewired", "untyped")
CONTRASTS = (("typed", "node_only"), ("typed", "rewired"), ("typed", "untyped"))


def rewire(links: Sequence, rng: random.Random) -> List:
    """Degree-preserving rewiring (module doc): the destination column permuted, self-loops
    swapped away where a partner exists."""
    dst = [d for _, _, d, _ in links]
    rng.shuffle(dst)
    n = len(dst)
    for i in range(n):
        if dst[i] == links[i][1]:
            # ponytail: O(n^2) worst case; per-store link lists are a few hundred edges
            for j in rng.sample(range(n), n):
                if dst[j] != links[i][1] and dst[i] != links[j][1]:
                    dst[i], dst[j] = dst[j], dst[i]
                    break
    return [(t, s, d, w) for (t, s, _, w), d in zip(links, dst)]


def mode_links(store, mode: str, seed: int = 0) -> List:
    """The link list a read in ``mode`` propagates over."""
    links = list(store.resolved_links())
    if mode == "typed":
        return links
    if mode == "node_only":
        return []
    if mode == "rewired":
        return rewire(links, random.Random(seed))
    if mode == "untyped":
        w = sum(x[3] for x in links) / len(links) if links else 0.0
        return [("LINK", s, d, w) for _, s, d, _ in links]
    raise ValueError(f"unknown read mode {mode!r}; choose from {MODES}")


@contextmanager
def _links_as(store, links):
    store.resolved_links = lambda: links       # instance attribute shadows the method for one read
    try:
        yield
    finally:
        del store.resolved_links


class ModeReader:
    """A memory read in one mode: ``recall(query, k)`` as ``SMPMemory.recall``; ``links`` is the
    mode's link list (fixed at construction, so a rewiring is one draw per store)."""

    def __init__(self, mem, mode: str, seed: int = 0):
        self.mem, self.mode = mem, mode
        self.links = mode_links(mem.store, mode, seed)

    def recall(self, query: str, k: int = 3) -> List[str]:
        with _links_as(self.mem.store, self.links):
            return self.mem.recall(query, k=k)


def stale_proxy(reader, item: dict, k: int, source: Dict[str, str], links,
                tau: Optional[float] = READ_TAU) -> Optional[Dict]:
    """Module doc; None for an item without knowledge-update ground truth. ``source`` maps a placed
    text (a paraphrase copy) to its original turn; the top k is recall_metrics.read_top at ``tau``."""
    truth = ku_truth(item)
    if truth is None:
        return None
    old, new = truth
    ranked = recalled_sources(reader, item["question"], k, source, tau)
    o = ranked.index(old) if old in ranked else None
    n = ranked.index(new) if new in ranked else None
    store = reader.mem.store
    rep = store.derived.rep

    def keys(text):
        return {rep.get(d.claim_key, d.claim_key) for _, d in store.nodes().values()
                if source.get(d.text, d.text) == text}
    ko, kn = keys(old), keys(new)
    flag = any(t == "CONTRADICTS" and ((s in ko and d in kn) or (s in kn and d in ko))
               for t, s, d, _ in links)
    return {"unflagged_stale": float(o is not None and n is None and not flag),
            "stale_preferred": float(o is not None and (n is None or o < n)),
            "flagged": float(flag)}


def paired(rows: Sequence[Dict], a: str, b: str, *, n_boot: int = 10000, seed: int = 0) -> Dict:
    """mean(a - b) over the rows where both are present, its percentile bootstrap CI over rows
    (paired: one resample of rows serves both), and dz = mean / sd of the differences."""
    d = [r[a] - r[b] for r in rows if r.get(a) is not None and r.get(b) is not None]
    if not d:
        return {"n": 0, "delta": None, "lo": None, "hi": None, "dz": None}
    point, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return {"n": len(d), "delta": point, "lo": lo, "hi": hi, "dz": point / sd if sd > 0 else None}


def structural_epochs(src, ms: Sequence[int]) -> List:
    """One epoch per m (module doc)."""
    return [replace(src, m=m, mac="",
                    cal_root=h16(b"structural-diagnostic\x1f" + src.cal_root + m.to_bytes(2, "big")))
            for m in ms]


def seed_cluster_ci(per_seed: Dict, *, n_boot: int = 10000, alpha: float = 0.05,
                    seed: int = 0) -> Dict:
    """type -> miss rate with its cluster-bootstrap CI: seeds resampled with replacement, the
    rate recomputed as one minus pooled hits over pooled edges."""
    rng = np.random.default_rng(seed)
    out = {}
    for typ, pairs in EDGES.items():
        h = np.array([v[typ] for v in per_seed.values()], dtype=float)
        boot = 1 - h[rng.integers(0, h.size, (n_boot, h.size))].sum(1) / (len(pairs) * h.size)
        out[typ] = {"miss_rate": float(1 - h.sum() / (len(pairs) * h.size)),
                    "lo": float(np.percentile(boot, 100 * alpha / 2)),
                    "hi": float(np.percentile(boot, 100 * (1 - alpha / 2))), "n_seeds": int(h.size)}
    return out


__all__ = ["MODES", "CONTRASTS", "rewire", "mode_links", "ModeReader", "stale_proxy", "paired",
           "structural_epochs", "seed_cluster_ci"]
