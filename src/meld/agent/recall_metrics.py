"""recall@k and MRR of the gold supporting facts in a brain's graph-walk read.

This is the retrieval-quality observable for the merge evaluation: a memory-merge
protocol's job is store hygiene (dedup + consistency), whose direct effect is on
WHICH units reach the recalled top-k -- not on downstream answer token-F1, which a
fixed-k cap and the LLM's own competence both confound. recall@k (did the gold
supporting sentence make the top-k?) and MRR (how high?) measure the protocol's
effect at the layer it acts on, and are the systems-relevant primary metric the
manuscript Section 7.2 names alongside the answer score.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from typing import Dict, List, Optional

import numpy as np

from meld.agent.hotpot_loader import HotpotTask

# Near-duplicate suppression threshold of the read (wp6c-nd), fixed in advance at 0.95 and not
# tuned. None is the reported wp6c read (the store's top k as is); any tau > 1 equals it too.
READ_TAU: Optional[float] = 0.95
# What the near-duplicate test compares (r1.13): the replicated int8 embeddings, exactly. r1.12
# compared a float cosine under the reader's encoder; every row carries this stamp to tell them apart.
READ_SIM = "int8-q"


def gold_supporting_sentences(task: HotpotTask) -> List[str]:
    """The gold supporting sentences: resolve each (title, sent_id) supporting
    fact against the item's context paragraphs."""
    by_title = {title: list(sents) for title, sents in task.context}
    out: List[str] = []
    for entry in task.supporting_facts:
        title, sid = entry[0], entry[1]
        sents = by_title.get(title)
        if sents and isinstance(sid, int) and 0 <= sid < len(sents):
            s = (sents[sid] or "").strip()
            if s:
                out.append(s)
    return out


def _matches(gold: str, text: str) -> bool:
    g = gold.strip().lower()
    return bool(g) and g in text.strip().lower()


def near_dup(qa: bytes, qb: bytes, tau: float) -> bool:
    """Two replicated int8 embeddings (``Delta.q``) are near-duplicates at ``tau``: dot(a, b) >= 0
    and dot^2 * den^2 >= num^2 * |a|^2 |b|^2, with tau = num/den read exactly from its decimal
    form (0.95 -> 19/20, so tau^2 = 361/400). Integer arithmetic only (int8 widened to int64, a
    384-dim dot stays far below 2^63; the comparison is in Python ints). A zero q has no
    direction and is never a near-duplicate."""
    a = np.frombuffer(qa, np.int8).astype(np.int64)
    b = np.frombuffer(qb, np.int8).astype(np.int64)
    dot, na, nb = int(a @ b), int(a @ a), int(b @ b)
    t = Fraction(repr(tau))
    return dot >= 0 and na > 0 and nb > 0 and dot * dot * t.denominator ** 2 >= t.numerator ** 2 * na * nb


def _live_q(store) -> Dict[str, bytes]:
    """Listed text -> the q of its live node with the smallest key (bytes order). A listed entry is
    a live unit's text, every live node of a unit carries that text, so this is a function of the
    replicated derived view alone (live set + held deltas): two replicas with the same view assign
    every listed text the same q."""
    nodes, out = store.nodes(), {}
    for key in sorted(store.live_nodes()):
        d = nodes[key][1]
        out.setdefault(d.text, d.q)
    return out


def read_top(mem, query: str, k: int, tau: Optional[float] = READ_TAU) -> List[str]:
    """The top-k read every recall metric uses. ``tau`` None (or > 1, which no pair can reach):
    the store's own top k (``mem.recall(query, k)``). Otherwise near-duplicate suppression: walk
    the store's full fused ranking (``mem.recall`` with k = all units, a prefix-stable sort, so no
    depth can cut a survivor short), skip a unit whose replicated int8 embedding (``_live_q``) is a
    ``near_dup`` of an already listed unit's, and stop at k. The decision reads replicated data only
    (no encoder, no float), so replicas holding the same derived view list the same entries
    whenever their scores agree. Identical for every memory that reads through ``recall`` over a
    ``Store`` (MELD, the ablations, the baselines, the E2-T2 read modes); a reader without one is
    refused, never read with a float fallback."""
    if tau is None or tau > 1:
        return mem.recall(query, k=k)
    store = getattr(mem, "store", None) or getattr(getattr(mem, "mem", None), "store", None)
    if store is None:                                   # an E2-T2 ModeReader wraps the memory
        raise TypeError(f"{type(mem).__name__} exposes no Store, hence no replicated q to read")
    qs = _live_q(store)
    out: List[str] = []
    for t in mem.recall(query, k=sys.maxsize):
        if not any(near_dup(qs[t], qs[o], tau) for o in out):
            out.append(t)
            if len(out) == k:
                break
    return out


def recalled_sources(mem, query: str, k: int, source: Optional[Dict[str, str]] = None,
                     tau: Optional[float] = READ_TAU) -> List[str]:
    """The top-k read (``read_top``) mapped to source texts, in rank order, each source once. The
    read returns live units only (a merged group is read as its representative), so a copy, or a
    group whose representative is a copy, is scored as the text it was copied from (``source``:
    placed text -> original; a text absent from it is its own source)."""
    source = source or {}
    return list(dict.fromkeys(source.get(t, t) for t in read_top(mem, query, k, tau)))


def recall_at_k_and_mrr(mem, task: HotpotTask, *, k: int = 5,
                        source: Optional[Dict[str, str]] = None,
                        tau: Optional[float] = READ_TAU) -> Dict[str, float]:
    """recall@k = fraction of gold supporting sentences whose unit is in the
    status-live top-k read; MRR = reciprocal rank of the first gold-supporting
    source in that read (0 if none). Units are mapped to their source texts first
    (``recalled_sources``). Matching is gold-sentence-substring-of-unit-text (robust to a
    title prefix added when context is fed through the brain)."""
    gold = gold_supporting_sentences(task)
    if not gold:
        return {"recall_at_k": 0.0, "mrr": 0.0, "n_gold": 0}
    ranked = recalled_sources(mem, task.question, k, source, tau)
    matched = sum(1 for g in gold if any(_matches(g, t) for t in ranked))
    rr = 0.0
    for i, t in enumerate(ranked, start=1):
        if any(_matches(g, t) for g in gold):
            rr = 1.0 / i
            break
    return {"recall_at_k": matched / len(gold), "mrr": rr, "n_gold": len(gold)}


__all__ = ["READ_SIM", "READ_TAU", "near_dup", "gold_supporting_sentences", "read_top", "recalled_sources", "recall_at_k_and_mrr"]
