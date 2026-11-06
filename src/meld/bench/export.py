"""Annotation exports and the human audit sample.

The annotation view is what a blinded annotator (model or human) sees: an opaque
pair id and two texts, in a shuffled stream with a random side order per pair.
Scope/time badges are stripped from the view for every stratum (only one
stratum has them, so showing them would reveal it); they stay in the key file
and are given to MELD. Labels, strata, sources, badges and the side order live
only in the key file. ``reversed_view`` gives the opposite side order of every
pair (each pair is labelled in both orders), ``with_repeats`` adds unannounced
repeat items for the human audit, and ``panel_queue`` selects what the
annotation panel labels.

The audit drawer selects the human audit sample within each group (stratum
group) by systematic probability-proportional-to-size sampling: fixed size per
group, exactly known first-order inclusion probabilities, with panel
disagreements given a larger size measure. Every item's probability is logged,
so the audit supports inverse-probability (design-based) estimation.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
from collections import Counter
from typing import Dict, List, Sequence

_KEY_FIELDS = ("label", "subtype", "label_provenance", "mergeable", "stratum", "source", "source_item_id",
               "split", "origin_split", "badges_a", "badges_b")


def annotation_export(rows: Sequence[dict], *, seed: int):
    """rows: manifest pair dicts. Returns (view, key), both in the same shuffled order."""
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r["pair_id"])
    rng.shuffle(rows)
    view, key = [], []
    for r in rows:
        sw = rng.random() < 0.5
        a, b = ("b", "a") if sw else ("a", "b")
        view.append({"pair_id": r["pair_id"], "text_a": r["text_" + a], "text_b": r["text_" + b],
                     "badges_a": {}, "badges_b": {}})
        key.append({"pair_id": r["pair_id"], "swapped": sw, **{f: r.get(f) for f in _KEY_FIELDS}})
    return view, key


def unswap(view_row: dict, key_row: dict):
    """(text_a, text_b) in the manifest's original side order (badges are in the key)."""
    v = view_row
    return (v["text_b"], v["text_a"]) if key_row["swapped"] else (v["text_a"], v["text_b"])


def reversed_view(view: Sequence[dict], *, seed: int) -> List[dict]:
    """Every pair in the opposite side order, in a fresh shuffled stream."""
    out = [dict(r, text_a=r["text_b"], text_b=r["text_a"], badges_a=r["badges_b"], badges_b=r["badges_a"])
           for r in view]
    random.Random(seed).shuffle(out)
    return out


def panel_export(rows: Sequence[dict], *, seed: int):
    """(view_ab, view_ba, key): the queue in both side orders from one export; key.swapped
    describes view_ab, key.swapped_ba (= not swapped) describes view_ba."""
    ab, key = annotation_export(rows, seed=seed)
    key = [dict(k, swapped_ba=not k["swapped"]) for k in key]
    return ab, reversed_view(ab, seed=seed + 1), key


def with_repeats(view: Sequence[dict], *, n: int = 60, min_gap: int = 50, seed: int, secret: str):
    """Insert n unannounced repeats, each at least min_gap positions after its original and
    under a fresh opaque id keyed by ``secret`` (keep the secret with the private mapping only,
    so repeat ids cannot be derived from visible ids). Returns (view with repeats,
    [{pair_id, repeat_id}])."""
    if len(view) - min_gap + 1 < n:
        raise ValueError(f"a view of {len(view)} items cannot hold {n} repeats {min_gap} apart")
    rng = random.Random(seed)
    after = {}
    rec = []
    for i in sorted(rng.sample(range(len(view) - min_gap + 1), n)):
        rid = hmac.new(secret.encode(), view[i]["pair_id"].encode(), hashlib.sha256).hexdigest()[:16]
        after.setdefault(rng.randint(i + min_gap - 1, len(view) - 1), []).append(dict(view[i], pair_id=rid))
        rec.append({"pair_id": view[i]["pair_id"], "repeat_id": rid})
    out = []
    for j, r in enumerate(view):
        out += [r] + after.get(j, [])
    return out, rec


def audit_export(rows: Sequence[dict], panel: Sequence[dict], allocation: Dict[str, int], *, seed: int,
                 secret: str, n_repeats: int = 60, min_gap: int = 50):
    """The human audit: draw the sample from panel output, export it blinded (one side order,
    no panel labels), add unannounced repeats. Returns (view, private mapping with the
    secret, repeats, key and the draw with its inclusion probabilities)."""
    draw = draw_audit(panel, allocation, seed=seed)
    drawn = {r["pair_id"] for r in draw["sample"]}
    view, key = annotation_export([r for r in rows if r["pair_id"] in drawn], seed=seed)
    view, repeats = with_repeats(view, n=n_repeats, min_gap=min_gap, seed=seed, secret=secret)
    return view, {"secret": secret, "repeats": repeats, "key": key, "draw": draw}


def panel_queue(rows: Sequence[dict], *, seed: int, validation_frac: float = 0.1) -> List[dict]:
    """Manifest rows the annotation panel labels: S3 and S4 cal + test (S3 as a label gate,
    S4 for verdict and subtype), all of S5, S6 and S7 (S5 rule-verified pairs included as a
    check), and a validation_frac sample of S1, S2 and S8 per split (checks annotators against
    source labels; for S8 it also measures SICK label noise). Each (stratum, split) sample has its
    own generator, so resizing one group never moves another group's sample."""
    rows = sorted(rows, key=lambda r: r["pair_id"])
    out = [r for r in rows if (r["stratum"] in ("S3_vitc_nonfactual", "S4_vitc_factual") and r["split"] != "dev")
           or r["stratum"] in ("S5_lme_ku", "S6_scifact_conflict", "S7_organic")]
    for st in ("S1_paws_para", "S2_paws_nonpara", "S8_sick_para", "S8_sick_oneway", "S8_sick_contra"):
        for sp in ("dev", "cal", "test"):
            g = [r for r in rows if r["stratum"] == st and r["split"] == sp]
            out += random.Random(f"{seed}:{st}:{sp}").sample(g, round(validation_frac * len(g)))
    return sorted(out, key=lambda r: r["pair_id"])


def write_jsonl(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


# ------------------------------------------------------------------ audit sample

def inclusion_probabilities(w: Sequence[float], n: int) -> List[float]:
    """pi_i proportional to w_i with sum min(n, N), items whose share would exceed 1 fixed at 1
    (so every item gets pi = 1 when n >= N)."""
    if any(x <= 0 for x in w):
        raise ValueError("size-measure weights must be positive")
    pi, free = [0.0] * len(w), set(range(len(w)))
    while free:
        rem, tot = n - (len(w) - len(free)), sum(w[i] for i in free)
        over = [i for i in free if rem * w[i] / tot >= 1]
        if not over:
            for i in free:
                pi[i] = rem * w[i] / tot
            break
        for i in over:
            pi[i] = 1.0
            free.discard(i)
    return pi


def _systematic(pi: Sequence[float], rng: random.Random) -> List[int]:
    order = list(range(len(pi)))
    rng.shuffle(order)
    u, cum, out = rng.random(), 0.0, []
    for i in order:
        lo, cum = cum, cum + pi[i]
        if math.floor(cum - u) > math.floor(lo - u):
            out.append(i)
    return out


def draw_audit(panel: Sequence[dict], allocation: Dict[str, int], *, seed: int,
               disagreement_weight: float = 4.0) -> dict:
    """panel rows: {pair_id, group, labels: [panel labels]}; allocation: audit size per group.
    Returns {population: [{pair_id, group, pi}], sample: [{pair_id, group, pi, weight}],
    unallocated: {group: N} for panel groups absent from the allocation (not audited),
    short: {group: [N, n]} for groups with fewer items than allocated (all taken)}."""
    if disagreement_weight <= 0:
        raise ValueError("disagreement_weight must be positive")
    ids = [r["pair_id"] for r in panel]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pair_id in panel rows")
    rng = random.Random(seed)
    population, sample = [], []
    sizes = Counter(r["group"] for r in panel)
    for g in sorted(allocation):
        rows = sorted((r for r in panel if r["group"] == g), key=lambda r: r["pair_id"])
        w = [disagreement_weight if len(set(r["labels"])) > 1 else 1.0 for r in rows]
        pi = inclusion_probabilities(w, allocation[g])
        population += [{"pair_id": r["pair_id"], "group": g, "pi": p} for r, p in zip(rows, pi)]
        sample += [{"pair_id": rows[i]["pair_id"], "group": g, "pi": pi[i], "weight": 1.0 / pi[i]}
                   for i in sorted(_systematic(pi, rng))]
    return {"population": population, "sample": sample,
            "unallocated": {g: n for g, n in sorted(sizes.items()) if g not in allocation},
            "short": {g: [sizes[g], n] for g, n in sorted(allocation.items()) if sizes[g] < n}}


def ht_mean(sample: Sequence[dict], values: Dict[str, float], population_size: int) -> float:
    """Horvitz-Thompson estimate of the population mean of ``values`` from the audit sample."""
    return sum(r["weight"] * values[r["pair_id"]] for r in sample) / population_size
