"""Labels follow the fold: the tested pairs the fold's selector produces on a split's stream, each
classified for labelling, and the annotation queue rows for those without a label.

The stream is the one ``calibration.stream`` builds (cal or test split alike; the test split as
the evaluators fold it, in streams of the calibration stream's length, with ``chunk``). The selector
(``fold.tested_pairs``, the fold's ``select_tested_pair``) reads the processed prefix, sigma_q and
node-level DISTINGUISHES only, never theta, Sig or the epoch table, so the tested pairs of a split
depend on the encoder and selector alone and no NLI model is loaded here. Classes:

  manifest           a manifest pair with a final mergeability
  manifest_pending   a manifest pair whose mergeability awaits annotation (None or source_filtered)
  source_labelled    not a manifest pair, but the raw source labels the two texts as a pair
  same_item          two texts of one source item that nobody pairs: annotators label it
  cross_item         two texts of different items: annotators label it (no label by construction)
  copy               (recall workloads) two placements of one source text, a ``light_paraphrase``
                     copy and its original or two copies: annotators label it, unless a validated
                     panel sample of copies (``copies_validated``) makes copies source-labelled
                     mergeable (``copies_mergeable``, provenance ``copy_validated``)

``classify_substrate`` classifies the cal-pool worlds of a recall workload (``substrates``) the
same way; there no pair has a manifest or raw-source label. ``queue_rows`` turns the annotator
classes into manifest-shaped rows for ``export.panel_export``.

Label store. ``resolve`` gives a tested pair its mergeability from, in this order: (1) the manifest
(a manifest pair with a final mergeability), (2) the raw-source lookup ``source_label``, (3) the
annotator labels ``annotations``; otherwise None (unresolved). A pair of two items is never labelled
by construction. ``load_annotations`` reads the annotator labels, a JSONL file of final labels, one
object per line:

  {"pair_id": "<manifest pair_id or queue pair_id>", "text_a": "...", "text_b": "...",
   "mergeable": true | false, "label_provenance": "<who decided, e.g. panel or adjudicated>"}

Each line needs ``pair_id`` or both texts (either or both; texts match as an unordered, case- and
whitespace-folded pair), a boolean ``mergeable`` and a non-empty ``label_provenance``. Two lines that
give one pair opposite labels are refused.

Copies. ``copies_validated`` reads the annotators' labels of a seeded sample of copy pairs (the
labels.jsonl of scripts/annotate.py, verdict per pair, or label store rows): a copy is mergeable when
its verdict is MERGE (AMBIGUOUS counts against), or, without a verdict, when ``mergeable`` is true.
It refuses unless the mergeable rate is at least ``COPY_MIN_RATE`` and reports the Wilson lower
bound; only then may a runner pass ``copies_mergeable``.

AMBIGUOUS. The adjudicator's AMBIGUOUS items (``verdict`` AMBIGUOUS in the labels.jsonl of scripts/annotate.py, which
``annotation_rows`` leaves out of the label store) enter through ``with_ambiguous`` with provenance
``adjudicated_ambiguous``: non-mergeable by default (for false-merge control, merging an item the
adjudicator could not decide counts as an error), or excluded (mergeability None with a provenance:
decided, never a Cal record, not unresolved) for a sensitivity run.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from meld.bench.schema import pair_id_of
from meld.eval import calibration as C
from meld.protocol.fold import tested_pairs

CLASSES = ("manifest", "manifest_pending", "source_labelled", "same_item", "cross_item", "copy")
NEEDS = ("same_item", "cross_item", "copy")
_KEY = b"label-queue"           # the stream's MAC key; tested pairs do not depend on it


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def _key(x: str, y: str) -> frozenset:
    return frozenset((_norm(x), _norm(y)))


def queue_id(source: str, item_a: str, item_b: str, text_a: str, text_b: str) -> str:
    """The pair_id ``queue_rows`` gives the tested pair (unordered in texts and items)."""
    return pair_id_of(source, "+".join(sorted({item_a, item_b})), *sorted((text_a, text_b)))


def load_annotations(path: str) -> Dict[object, Tuple[bool, str]]:
    """Annotator labels (format in the module doc): pair_id or text key -> (mergeable, provenance)."""
    out: Dict[object, Tuple[bool, str]] = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            keys = ([r["pair_id"]] if r.get("pair_id") else []) + \
                ([_key(r["text_a"], r["text_b"])] if r.get("text_a") and r.get("text_b") else [])
            if not keys or not isinstance(r.get("mergeable"), bool) or not r.get("label_provenance"):
                raise ValueError(f"{path}:{n}: needs pair_id or text_a and text_b, a boolean mergeable "
                                 "and a label_provenance")
            lab = (r["mergeable"], str(r["label_provenance"]))
            for k in keys:
                if out.setdefault(k, lab)[0] != lab[0]:
                    raise ValueError(f"{path}:{n}: opposite labels for one pair")
    return out


AMBIGUOUS = "adjudicated_ambiguous"


def load_ambiguous(path: str) -> set:
    """The pair_ids the adjudicator marked AMBIGUOUS in a labels.jsonl (scripts/annotate.py)."""
    with open(path, encoding="utf-8") as fh:
        return {r["pair_id"] for r in map(json.loads, filter(str.strip, fh)) if r["verdict"] == "AMBIGUOUS"}


def with_ambiguous(annotations: dict, ids: Iterable[str], exclude: bool = False) -> dict:
    """``annotations`` plus the AMBIGUOUS ``ids``: non-mergeable, or excluded (None) when ``exclude``."""
    both = set(ids) & set(annotations)
    if both:
        raise ValueError(f"{len(both)} pairs both labelled and AMBIGUOUS, e.g. {sorted(both)[:3]}")
    return {**annotations, **{i: (None if exclude else False, AMBIGUOUS) for i in ids}}


def resolve(r: Optional[dict], text_a: str, text_b: str, item_a: str, item_b: str, *, source: str,
            source_label: Optional[Callable[[str, str], Optional[bool]]] = None,
            annotations: Optional[dict] = None, copy: bool = False,
            copies_mergeable: bool = False) -> Tuple[str, Optional[bool], Optional[str]]:
    """(class, mergeable, provenance) of a tested pair; ``r`` its manifest row or None, ``source`` the
    domain (the queue's source), ``copy`` whether the two texts are placements of one source text,
    ``copies_mergeable`` whether copies are source-labelled mergeable (``copies_validated`` passed).
    Manifest beats source beats annotator; None when nobody labels it."""
    if r is not None and not (r["mergeable"] is None or r["label_provenance"] == "source_filtered"):
        return "manifest", r["mergeable"], "manifest"
    if r is None and copy and copies_mergeable:
        return "source_labelled", True, "copy_validated"
    cls = ("manifest_pending" if r is not None else "copy" if copy else
           "same_item" if item_a == item_b else "cross_item")
    m = source_label(text_a, text_b) if source_label else None
    if m is not None:
        return ("source_labelled" if r is None else cls), m, "source"
    ann = annotations or {}
    for k in (r and r["pair_id"], queue_id(source, item_a, item_b, text_a, text_b), _key(text_a, text_b)):
        if k in ann:
            return cls, ann[k][0], ann[k][1]
    return cls, None, None


def source_labels(data: str) -> Callable[[str, str], Optional[bool]]:
    """The raw-source lookup (text_a, text_b) -> mergeable or None, unordered and case/whitespace
    folded, from the raw PAWS train/test
    files (label 1) and SICK.txt (both-way entailment); VitaminC labels need the filters, so none."""
    import csv

    import pandas as pd

    from meld.bench import loaders as L

    lab = {}
    for f in ("paws/train.parquet", "paws/test.parquet"):
        for r in pd.read_parquet(os.path.join(data, f)).itertuples(index=False):
            lab[_key(L.detok_paws(r.sentence1), L.detok_paws(r.sentence2))] = r.label == 1
    with open(os.path.join(data, "sick", "SICK.txt"), encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            lab[_key(r["sentence_A"], r["sentence_B"])] = \
                (r["entailment_AB"], r["entailment_BA"]) == ("A_entails_B", "B_entails_A")
    return lambda x, y: lab.get(_key(x, y))


def classify(rows: Sequence[dict], *, scope: str, encoder: str, selector: Sequence[int],
             source_label: Optional[Callable[[str, str], Optional[bool]]] = None,
             annotations: Optional[dict] = None, seed: int = 0, chunk: Optional[int] = None) -> List[dict]:
    """Every tested pair (a_e, b) of the stream over ``rows`` (one split), in fold order, with its
    class, mergeability and its provenance by ``resolve`` (None when annotators must label it).
    With ``chunk`` the rows are folded as the evaluators fold the test split
    (``benchmark_test.chunk_rows``: independent streams of at most ``chunk`` texts, same seed), and
    each tested pair records its stream's index under ``chunk``."""
    if chunk:
        from meld.eval.benchmark_test import chunk_rows
        return [dict(r, chunk=i) for i, part in enumerate(chunk_rows(rows, chunk, seed))
                for r in classify(part, scope=scope, encoder=encoder, selector=selector,
                                  source_label=source_label, annotations=annotations, seed=seed)]
    store, _, item, pair = C.stream(rows, scope=scope, encoder=encoder, selector=selector, key=_KEY, seed=seed)
    text = {d.claim_key: d.text for _, d in store.nodes().values()}
    origin = {r["source_item_id"]: r.get("origin_split") for r in rows}
    split = rows[0]["split"]
    out = []
    for a, b, n in tested_pairs(store, scope, sigma_lo=selector[0], m=selector[2]):
        ta, tb = text[a], text[b]
        r = pair.get(frozenset((a, b)))
        cls, mergeable, prov = resolve(r, ta, tb, item[a], item[b], source=scope, source_label=source_label,
                                       annotations=annotations)
        out.append({"split": split, "class": cls, "mergeable": mergeable, "provenance": prov, "n_prior": n,
                    "pair_id": r["pair_id"] if r else None, "text_a": ta, "text_b": tb,
                    "item_a": item[a], "item_b": item[b], "origin_split": origin[item[a]]})
    return out


def classify_substrate(domain: str, worlds, *, encoder: str, annotations: Optional[dict] = None,
                       cache_dir: Optional[str] = None, copies_mergeable: bool = False) -> List[dict]:
    """Every tested pair of every world (``substrates``: (id, plan, item, source)), in fold order
    per world, classified by ``resolve`` (no manifest, no raw-source label; copies by source text)
    under the evaluation memory's selector."""
    from meld.agent.memory import DEFAULT_POL, DEFAULT_SCOPE
    from meld.eval import substrates as S
    sel = DEFAULT_POL["bootstrap_selector"]
    out = []
    for wid, plan, item, src in worlds:
        store = S.world_store(plan, encoder=encoder, cache_dir=cache_dir)
        k_item, k_src = S.keyed(store, item, src)
        text = {d.claim_key: d.text for _, d in store.nodes().values()}
        for a, b, n in tested_pairs(store, DEFAULT_SCOPE, sigma_lo=sel[0], m=sel[2]):
            cls, mergeable, prov = resolve(None, text[a], text[b], k_item[a], k_item[b], source=domain,
                                           annotations=annotations, copy=k_src[a] == k_src[b],
                                           copies_mergeable=copies_mergeable)
            out.append({"split": "cal", "class": cls, "mergeable": mergeable, "provenance": prov,
                        "n_prior": n, "pair_id": None, "text_a": text[a], "text_b": text[b],
                        "item_a": k_item[a], "item_b": k_item[b], "origin_split": None, "world": wid})
    return out


COPY_MIN_RATE = 0.98


def copies_validated(path: str, min_rate: float = COPY_MIN_RATE) -> dict:
    """The annotators' validation of a copy sample (module doc): n, mergeable, rate and the Wilson 95%
    lower bound; refuses (ValueError) when the sample is empty or the rate is below ``min_rate``."""
    from meld.annotate.aggregate import wilson
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    k = sum(r["verdict"] == "MERGE" if "verdict" in r else r.get("mergeable") is True for r in rows)
    n = len(rows)
    if not n:
        raise ValueError(f"{path}: no validated copy pairs")
    st = {"path": path, "n": n, "mergeable": k, "rate": k / n, "wilson_lower": wilson(k, n)[0], "min_rate": min_rate}
    if k / n < min_rate:
        raise ValueError(f"copies not validated: mergeable rate {k / n:.3f} ({k}/{n}, Wilson lower bound "
                         f"{st['wilson_lower']:.3f}) below {min_rate}")
    return st


def queue_rows(tested: Iterable[dict], *, source: str, existing: Iterable[dict] = ()) -> List[dict]:
    """Manifest-shaped rows (label to annotate) for the tested pairs annotators must label, once per
    unordered text pair (case and whitespace folded), never for a pair already labelled and never for
    a pair ``existing`` (rows with text_a, text_b: the annotation queue) already holds."""
    seen = {_key(r["text_a"], r["text_b"]) for r in existing}
    out = []
    for t in tested:
        k = _key(t["text_a"], t["text_b"])
        if t["class"] not in NEEDS or t["mergeable"] is not None or k in seen:
            continue
        seen.add(k)
        a, b = sorted((t["text_a"], t["text_b"]))
        it = "+".join(sorted({t["item_a"], t["item_b"]}))
        out.append({"pair_id": queue_id(source, t["item_a"], t["item_b"], a, b), "text_a": a, "text_b": b, "source": source,
                    "source_item_id": it, "stratum": "tested_" + t["class"], "split": t["split"],
                    "origin_split": t["origin_split"], "label": None, "subtype": None,
                    "label_provenance": "to_annotate", "mergeable": None, "badges_a": {}, "badges_b": {}})
    return sorted(out, key=lambda r: r["pair_id"])


__all__ = ["CLASSES", "NEEDS", "AMBIGUOUS", "COPY_MIN_RATE", "copies_validated", "queue_id", "load_annotations", "load_ambiguous", "with_ambiguous", "resolve", "source_labels", "classify",
           "classify_substrate",
           "queue_rows"]
