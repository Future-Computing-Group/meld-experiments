"""From annotator records to final labels, agreement and accuracy statistics, and the audit draw.

``collect`` maps validated batch records back to pairs through the private item map: votes[pair_id]
[annotator][orientation], with the hidden repeat items kept apart for the consistency check.
``aggregate`` decides each pair in this order: an annotator's vote counts only when its verdict is
the same in both side orders; two or more annotators without such a vote send the pair to
adjudication (reason ``orientation``); all annotators agreeing is ``unanimous``; a strict majority is
``majority``; otherwise adjudication (``disagreement``). The subtype is the most frequent subtype among
the winning annotators' votes in both orders, a tie going to adjudication (``subtype``).
With ``resolve="mergeability"`` only the merge flag is decided: a vote is MERGE or NON_MERGE and counts
when that flag is the same in both side orders; the same routing then gives the paths
``mergeability_unanimous`` and ``mergeability_majority`` with verdict MERGE or NON_MERGE and no subtype.
``adjudication_batches`` shows the adjudicator the texts and the anonymous votes (no annotator names,
no source label, no stratum); ``finalize`` takes the adjudicator's verdict (path ``adjudicated``;
AMBIGUOUS is kept and never written as a label). ``annotation_rows`` writes the label store format of
``meld.eval.label_queue`` with the texts back in the manifest's side order (``export.unswap``).

``report`` gives, per stratum, each annotator's and the aggregate's accuracy against the
dataset-provided labels where they exist (mergeability, and the four-way verdict where the source
fixes it), the error correlation between annotators on those items, Krippendorff's alpha over the
annotators (verdict, and verdict with subtype), pairwise Cohen's kappa, orientation consistency, the
decision paths, and repeat consistency per annotator. ``audit_draw`` hands the votes to
``export.audit_export`` so the human audit over-samples disagreements with known inclusion
probabilities.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence

from meld.annotate.batch import SUBTYPES_OF, _cut, item_id
from meld.bench.export import audit_export, unswap

AUDIT_GROUP = {"S2_paws_nonpara": "S2", "S4_vitc_factual": "S4", "S7_organic": "S7"}


def _vote(e: dict) -> dict:
    return {k: e[k] for k in ("verdict", "subtype", "overlap_flag", "confidence", "rationale")}


def collect(records: Sequence[dict], items: Dict[str, dict]):
    """(votes, repeats), each pair_id -> annotator -> orientation -> vote."""
    votes, repeats = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))
    for rec in records:
        for iid, e in (rec["labels"] or {}).items():
            m = items[iid]
            (repeats if m["repeat"] else votes)[m["pair_id"]][rec["annotator"]][m["orientation"]] = _vote(e)
    plain = lambda d: {p: {a: dict(o) for a, o in by.items()} for p, by in d.items()}  # noqa: E731
    return plain(votes), plain(repeats)


def aggregate(votes: Dict[str, dict], annotators: Sequence[str], resolve: str = "verdict") -> Dict[str, dict]:
    if resolve not in ("verdict", "mergeability"):
        raise ValueError(f"resolve: {resolve!r}")
    merge_only = resolve == "mergeability"
    flag = (lambda v: "MERGE" if v == "MERGE" else "NON_MERGE") if merge_only else (lambda v: v)  # noqa: E731
    out = {}
    for pid, by in votes.items():
        cons = {}
        for a in annotators:
            o = by.get(a, {})
            if "ab" in o and "ba" in o and flag(o["ab"]["verdict"]) == flag(o["ba"]["verdict"]):
                cons[a] = flag(o["ab"]["verdict"])
        if len(annotators) - len(cons) >= 2:
            out[pid] = {"path": "adjudicate", "reason": "orientation"}
            continue
        top, k = Counter(cons.values()).most_common(1)[0] if cons else (None, 0)
        if 2 * k <= len(annotators):
            out[pid] = {"path": "adjudicate", "reason": "disagreement"}
            continue
        win = [v for a in annotators if cons.get(a) == top for v in by[a].values()]
        subtype = None
        if not merge_only and SUBTYPES_OF[top] != (None,):
            c = Counter(v["subtype"] for v in win).most_common()
            if len(c) > 1 and c[0][1] == c[1][1]:
                out[pid] = {"path": "adjudicate", "reason": "subtype"}
                continue
            subtype = c[0][0]
        path = "unanimous" if k == len(annotators) else "majority"
        out[pid] = {"path": f"mergeability_{path}" if merge_only else path, "verdict": top,
                    "subtype": subtype, "overlap_flag": 2 * sum(v["overlap_flag"] for v in win) > len(win)}
    return out


def adjudication_batches(agg: Dict[str, dict], votes: Dict[str, dict], views_ab: Dict[str, Sequence[dict]], *,
                         secret: str, seed: int = 0, size: int = 20):
    """(batches, items: item_id -> {pair_id}) for the pairs ``aggregate`` sent to adjudication."""
    text = {r["pair_id"]: (r["text_a"], r["text_b"]) for v in views_ab.values() for r in v}
    pids = sorted(p for p, a in agg.items() if a["path"] == "adjudicate")
    random.Random(f"{seed}:adjudication").shuffle(pids)
    stream, items = [], {}
    for pid in pids:
        vs = [{"verdict": v["verdict"], "subtype": v["subtype"], "overlap_flag": v["overlap_flag"],
               "rationale": v["rationale"], "order": "as shown" if o == "ab" else "reversed"}
              for a in sorted(votes[pid]) for o, v in sorted(votes[pid][a].items())]
        random.Random(f"{seed}:{pid}").shuffle(vs)
        iid = item_id(secret, "adjudication", pid)
        items[iid] = {"pair_id": pid}
        stream.append({"item_id": iid, "text_a": text[pid][0], "text_b": text[pid][1], "votes": vs})
    return _cut(stream, "j", 0, size, seed), items


def adjudicated(records: Sequence[dict], items: Dict[str, dict]) -> Dict[str, dict]:
    return {items[iid]["pair_id"]: _vote(e) for rec in records for iid, e in (rec["labels"] or {}).items()}


def finalize(agg: Dict[str, dict], adj: Dict[str, dict]) -> Dict[str, dict]:
    """Final labels; pairs awaiting an adjudicator's answer are left out."""
    out = {}
    for pid, a in agg.items():
        if a["path"] != "adjudicate":
            out[pid] = dict(a)
        elif pid in adj:
            e = adj[pid]
            out[pid] = {"path": "adjudicated", "reason": a["reason"], "verdict": e["verdict"],
                        "subtype": e["subtype"], "overlap_flag": e["overlap_flag"]}
    return out


def rows_of(keys: Dict[str, Sequence[dict]], views_ab: Dict[str, Sequence[dict]]) -> List[dict]:
    """Manifest-shaped rows (key fields, texts in the manifest's side order) of both queues."""
    view = {r["pair_id"]: r for v in views_ab.values() for r in v}
    out = []
    for q in sorted(keys):
        for k in keys[q]:
            ta, tb = unswap(view[k["pair_id"]], k)
            out.append(dict(k, text_a=ta, text_b=tb, queue=q))
    return out


def annotation_rows(final: Dict[str, dict], keys: Dict[str, Sequence[dict]],
                    views_ab: Dict[str, Sequence[dict]]) -> List[dict]:
    """Final labels in the label store format (``label_queue.load_annotations``)."""
    row = {r["pair_id"]: r for r in rows_of(keys, views_ab)}
    out = []
    for pid in sorted(final):
        f = final[pid]
        if f["verdict"] == "AMBIGUOUS":
            continue
        out.append({"pair_id": pid, "text_a": row[pid]["text_a"], "text_b": row[pid]["text_b"],
                    "mergeable": f["verdict"] == "MERGE", "label_provenance": f["path"], "verdict": f["verdict"],
                    "subtype": f["subtype"], "overlap_flag": f["overlap_flag"],
                    "contradiction": None if f["verdict"] == "NON_MERGE" else f["verdict"] == "CONFLICT"})
    return out


# ------------------------------------------------------------------ statistics

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p, d = k / n, 1 + z * z / n
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p + z * z / (2 * n)) / d - h, (p + z * z / (2 * n)) / d + h


def cohen_kappa(x: Sequence, y: Sequence) -> Optional[float]:
    n = len(x)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[c] * cy[c] for c in cx) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def krippendorff_alpha(units: Sequence[Sequence]) -> Optional[float]:
    """Nominal alpha; units are the values of one item across coders, None for missing."""
    o = Counter()
    for vals in units:
        c = Counter(v for v in vals if v is not None)
        m = sum(c.values())
        if m < 2:
            continue
        for a in c:
            for b in c:
                o[a, b] += c[a] * (c[b] - (a == b)) / (m - 1)
    nc = Counter()
    for (a, _), v in o.items():
        nc[a] += v
    n = sum(nc.values())
    de = sum(nc[a] * nc[b] for a in nc for b in nc if a != b) / (n - 1) if n > 1 else 0
    if de == 0:
        return 1.0 if n > 1 else None
    return 1 - sum(v for (a, b), v in o.items() if a != b) / de


def _acc(pred: Sequence, gold: Sequence) -> dict:
    k, n = sum(p == g for p, g in zip(pred, gold)), len(gold)
    return {"n": n, "correct": k, "accuracy": k / n if n else None, "wilson": list(wilson(k, n)),
            "kappa": cohen_kappa(pred, gold)}


def report(final: Dict[str, dict], votes: Dict[str, dict], repeats: Dict[str, dict],
           keys: Dict[str, Sequence[dict]], annotators: Sequence[str]) -> dict:
    key = {k["pair_id"]: k for ks in keys.values() for k in ks}
    by_st = defaultdict(list)
    for pid in votes:
        by_st[key[pid]["stratum"]].append(pid)
    accuracy, verdict_accuracy, agreement = {}, {}, {}
    for st, pids in sorted(by_st.items()):
        gold = [p for p in pids if key[p].get("mergeable") is not None]
        if gold:
            acc = {a: _acc([v["verdict"] == "MERGE" for p in gold for v in votes[p].get(a, {}).values()],
                           [key[p]["mergeable"] for p in gold for _ in votes[p].get(a, {})]) for a in annotators}
            fin = [p for p in gold if p in final and final[p]["verdict"] != "AMBIGUOUS"]
            acc["aggregate"] = _acc([final[p]["verdict"] == "MERGE" for p in fin], [key[p]["mergeable"] for p in fin])
            err = {a: {p: (votes[p][a]["ab"]["verdict"] == "MERGE") != key[p]["mergeable"]
                       for p in gold if "ab" in votes[p].get(a, {})} for a in annotators}
            acc["error_correlation"] = {}
            for a, b in combinations(annotators, 2):
                both = [p for p in err[a] if p in err[b]]
                if both:
                    ea, eb = (sum(err[x][p] for p in both) / len(both) for x in (a, b))
                    acc["error_correlation"][f"{a}|{b}"] = {
                        "n": len(both), "both_wrong": sum(err[a][p] and err[b][p] for p in both) / len(both),
                        "independent": ea * eb}
            accuracy[st] = acc
        vg = [p for p in pids if key[p].get("label") and key[p].get("label_provenance") == "source"]
        if vg:
            verdict_accuracy[st] = {a: _acc([v["verdict"] for p in vg for v in votes[p].get(a, {}).values()],
                                            [key[p]["label"] for p in vg for _ in votes[p].get(a, {})])
                                    for a in annotators}
        ab = lambda p, a, f: f(votes[p][a]["ab"]) if "ab" in votes[p].get(a, {}) else None  # noqa: E731
        verdict = lambda v: v["verdict"]  # noqa: E731
        label = lambda v: f"{v['verdict']}/{v['subtype']}"  # noqa: E731
        kappa = {}
        for a, b in combinations(annotators, 2):
            both = [p for p in pids if ab(p, a, verdict) and ab(p, b, verdict)]
            kappa[f"{a}|{b}"] = cohen_kappa([ab(p, a, verdict) for p in both], [ab(p, b, verdict) for p in both])
        paths = Counter(final[p]["path"] if p in final else "pending" for p in pids)
        agreement[st] = {
            "n": len(pids),
            "alpha_verdict": krippendorff_alpha([[ab(p, a, verdict) for a in annotators] for p in pids]),
            "alpha_label": krippendorff_alpha([[ab(p, a, label) for a in annotators] for p in pids]),
            "kappa": kappa,
            "orientation_consistency": {a: _consistency(votes, pids, a) for a in annotators},
            "paths": dict(sorted(paths.items())),
            "ambiguous": sum(p in final and final[p]["verdict"] == "AMBIGUOUS" for p in pids)}
    rep = {}
    for a in annotators:
        hits = [repeats[p][a][o]["verdict"] == votes[p][a][o]["verdict"]
                for p in repeats for o in repeats[p].get(a, {}) if o in votes.get(p, {}).get(a, {})]
        rep[a] = {"n": len(hits), "agree": sum(hits) / len(hits) if hits else None}
    return {"n_pairs": len(votes), "n_final": len(final),
            "paths": dict(sorted(Counter(f["path"] for f in final.values()).items())),
            "accuracy": accuracy, "verdict_accuracy": verdict_accuracy, "agreement": agreement,
            "repeat_consistency": rep}


def _consistency(votes, pids, a) -> Optional[float]:
    both = [votes[p][a] for p in pids if {"ab", "ba"} <= set(votes[p].get(a, {}))]
    return sum(o["ab"]["verdict"] == o["ba"]["verdict"] for o in both) / len(both) if both else None


# ------------------------------------------------------------------ audit

def audit_draw(votes: Dict[str, dict], rows: Sequence[dict], keys: Dict[str, Sequence[dict]], *,
               allocation: Dict[str, int], seed: int, secret: str, n_repeats: int = 60, min_gap: int = 50):
    """The human audit through ``export.audit_export``: one row per pair with its group (S2, S4, S7,
    else the stratum) and every annotator vote, so disagreements get the larger size measure."""
    st = {k["pair_id"]: k["stratum"] for ks in keys.values() for k in ks}
    panel = [{"pair_id": p, "group": AUDIT_GROUP.get(st[p], st[p]),
              "labels": [f"{v['verdict']}/{v['subtype']}" for a in sorted(votes[p]) for v in votes[p][a].values()]}
             for p in sorted(votes)]
    return audit_export(rows, panel, allocation, seed=seed, secret=secret, n_repeats=n_repeats, min_gap=min_gap)
