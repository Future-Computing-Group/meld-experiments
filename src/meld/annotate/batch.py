"""Annotation batches, the prompt, the strict output check and the annotator backends.

Batches. ``make_batches`` takes both annotation queues (the benchmark's annotation queue and the
fold-derived label queue, each as its two side-order views from ``export.panel_export``), merges each
orientation into one seeded stream (so nothing in an item's position reveals its queue or stratum),
adds unannounced repeat items with ``export.with_repeats`` (consistency checks), and cuts the
stream into batches. An annotator sees only ``{batch_id, verdict_order, items: [{item_id, text_a,
text_b}]}``: item ids are keyed hashes of (orientation, pair id), so they differ between the two
orders of one pair and cannot be traced to a pair without the secret. The private item map (item id
to pair id, queue, orientation, repeat) stays with the caller. The verdict menu order is drawn per
batch (option-order bias), and verdicts are words, never letters.

Backends. One call, ``complete(batch, prompt, attempt) -> raw text or None``; None means the answer
is pending. ``SubagentBackend`` only writes prompt files and reads response files (an orchestrator
runs the agents); ``HttpBackend`` posts to an OpenAI-compatible endpoint (for example vLLM serving an
open-weight model) with the response schema for guided decoding; ``FakeBackend`` answers by a rule,
for tests. ``run_batch`` validates the output with ``validate`` and retries once, with the errors, on
malformed output; every raw response is kept in the record.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import re
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from meld.bench.export import with_repeats

RUBRIC_PATH = os.path.join(os.path.dirname(__file__), "rubric.md")
VERDICTS = ("MERGE", "RELATE", "CONFLICT", "NO_RELATION")
SUBTYPES_OF = {"MERGE": (None,), "RELATE": ("ELABORATION", "SCOPE"),
               "CONFLICT": ("SUPERSESSION", "CORRECTION", "SCOPE"), "NO_RELATION": (None,),
               "AMBIGUOUS": (None,)}
FIELDS = ("item_id", "verdict", "subtype", "overlap_flag", "confidence", "rationale")
ORIENTATIONS = ("ab", "ba")

RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["labels"],
    "properties": {"labels": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": list(FIELDS),
        "properties": {"item_id": {"type": "string"},
                       "verdict": {"enum": list(SUBTYPES_OF)},
                       "subtype": {"enum": [None, "SUPERSESSION", "CORRECTION", "ELABORATION", "SCOPE"]},
                       "overlap_flag": {"type": "boolean"},
                       "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                       "rationale": {"type": "string"}}}}}}


def batch_schema(batch: dict) -> dict:
    """RESPONSE_SCHEMA pinned to ``batch``: one entry per item, in batch order, each with its own item id
    (open-weight models under guided decoding otherwise close the array early or drop, repeat or miscopy
    ids; the id is bookkeeping, the label fields stay free)."""
    labels = RESPONSE_SCHEMA["properties"]["labels"]
    entry = labels["items"]
    pos = [dict(entry, properties=dict(entry["properties"], item_id={"enum": [it["item_id"]]})) for it in batch["items"]]
    return dict(RESPONSE_SCHEMA, properties={"labels": {"type": "array", "prefixItems": pos, "items": False,
                                                        "minItems": len(pos), "maxItems": len(pos)}})


def item_id(secret: str, tag: str, pair_id: str) -> str:
    return hmac.new(secret.encode(), f"{tag}:{pair_id}".encode(), hashlib.sha256).hexdigest()[:16]


def _menu(seed: int, batch_id: str) -> List[str]:
    v = list(VERDICTS)
    random.Random(f"{seed}:{batch_id}").shuffle(v)
    return v


def _cut(stream: Sequence[dict], prefix: str, start: int, size: int, seed: int) -> List[dict]:
    out = []
    for k in range(0, len(stream), size):
        bid = f"{prefix}{start + k // size + 1:04d}"
        out.append({"batch_id": bid, "verdict_order": _menu(seed, bid), "items": list(stream[k:k + size])})
    return out


def make_batches(views: Dict[str, Tuple[Sequence[dict], Sequence[dict]]], *, secret: str, seed: int = 0,
                 size: int = 20, n_repeats: int = 60, min_gap: int = 50):
    """views: queue name -> (view_ab, view_ba). Returns (batches, items: item_id -> {pair_id, queue,
    orientation, repeat}). Deterministic given the views, seed and secret."""
    queue_of = {}
    for q, (ab, _) in sorted(views.items()):
        for r in ab:
            if queue_of.setdefault(r["pair_id"], q) != q:
                raise ValueError(f"pair {r['pair_id']} in two queues")
    batches, items = [], {}
    for k, o in enumerate(ORIENTATIONS):
        rows = sorted((r for q in sorted(views) for r in views[q][k]), key=lambda r: r["pair_id"])
        random.Random(f"{seed}:{o}").shuffle(rows)
        rep_of = {}
        if n_repeats:
            rows, rec = with_repeats(rows, n=n_repeats, min_gap=min_gap, seed=seed + k, secret=secret)
            rep_of = {r["repeat_id"]: r["pair_id"] for r in rec}
        stream = []
        for r in rows:
            iid = item_id(secret, o, r["pair_id"])
            pid = rep_of.get(r["pair_id"], r["pair_id"])
            items[iid] = {"pair_id": pid, "queue": queue_of[pid], "orientation": o, "repeat": r["pair_id"] in rep_of}
            stream.append({"item_id": iid, "text_a": r["text_a"], "text_b": r["text_b"]})
        batches += _cut(stream, "b", len(batches), size, seed)
    return batches, items


def pilot_batch(queues: Dict[str, Tuple[Sequence[dict], Sequence[dict]]], *, n: Dict[str, int], secret: str,
                seed: int = 0, splits: Dict[str, str] = None):
    """One batch of n[q] items per queue, side order ab, no repeats: from the benchmark queue its dev
    items, from the fold queue its cal items (it has no dev split). queues: q -> (view_ab, key)."""
    splits = splits or {"bench": "dev", "fold": "cal"}
    stream, items = [], {}
    for q in sorted(n):
        view, key = queues[q]
        ok = {k["pair_id"] for k in key if k["split"] == splits[q]}
        pool = sorted((r for r in view if r["pair_id"] in ok), key=lambda r: r["pair_id"])
        for r in random.Random(f"{seed}:pilot:{q}").sample(pool, n[q]):
            iid = item_id(secret, "pilot", r["pair_id"])
            items[iid] = {"pair_id": r["pair_id"], "queue": q, "orientation": "ab", "repeat": False}
            stream.append({"item_id": iid, "text_a": r["text_a"], "text_b": r["text_b"]})
    random.Random(f"{seed}:pilot").shuffle(stream)
    return _cut(stream, "p", 0, len(stream), seed)[0], items


# ------------------------------------------------------------------ prompt and output check

def build_prompt(batch: dict) -> str:
    """Rubric (the adjudication section only for items carrying votes), the batch's verdict menu,
    the items as JSON."""
    rubric = re.sub(r"<!--.*?-->\n", "", open(RUBRIC_PATH, encoding="utf-8").read())
    adjudicate = any("votes" in it for it in batch["items"])
    if not adjudicate:
        rubric = rubric.split("\n## Adjudication")[0] + "\n"
    menu = list(batch["verdict_order"]) + (["AMBIGUOUS"] if adjudicate else [])
    items = "\n".join(json.dumps(it, ensure_ascii=False) for it in batch["items"])
    return (f"{rubric}\n## This batch\n\nVerdicts for this batch, in no particular order: {', '.join(menu)}.\n\n"
            f"Items ({len(batch['items'])}), one JSON object per line:\n\n{items}\n")


def validate(raw: str, batch: dict):
    """(labels: item_id -> entry, []) or (None, [errors]) for a raw response to ``batch``."""
    adjudicate = any("votes" in it for it in batch["items"])
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw)
    try:
        obj = json.loads(text)
    except ValueError as e:
        return None, [f"not JSON: {e}"]
    if not isinstance(obj, dict) or set(obj) != {"labels"} or not isinstance(obj["labels"], list):
        return None, ['the answer must be one object {"labels": [...]}']
    want, err, out = {it["item_id"] for it in batch["items"]}, [], {}
    for e in obj["labels"]:
        if not isinstance(e, dict) or set(e) != set(FIELDS):
            err.append(f"entry {e!r}: needs exactly the fields {', '.join(FIELDS)}")
            continue
        i, v = e["item_id"], e["verdict"]
        if i not in want or i in out:
            err.append(f"item_id {i!r}: unknown or repeated")
        if v not in SUBTYPES_OF or (v == "AMBIGUOUS" and not adjudicate):
            err.append(f"item {i}: verdict {v!r} not allowed")
        elif e["subtype"] not in SUBTYPES_OF[v]:
            err.append(f"item {i}: subtype {e['subtype']!r} not allowed with {v}")
        if not isinstance(e["overlap_flag"], bool) or (e["overlap_flag"] and v == "MERGE"):
            err.append(f"item {i}: overlap_flag must be a boolean, never true with MERGE")
        if type(e["confidence"]) is not int or not 1 <= e["confidence"] <= 5:
            err.append(f"item {i}: confidence must be an integer 1-5")
        if not isinstance(e["rationale"], str) or not e["rationale"].strip():
            err.append(f"item {i}: rationale must be a non-empty sentence")
        out.setdefault(i, e)
    if want - set(out):
        err.append(f"missing items: {', '.join(sorted(want - set(out)))}")
    return (None, err) if err else (out, [])


def retry_note(errors: Sequence[str]) -> str:
    return ("\n## Your previous answer was not valid\n\n" + "\n".join(f"- {e}" for e in errors[:20])
            + "\n\nAnswer again with the complete JSON object for every item.\n")


def run_batch(batch: dict, backend, *, annotator: str) -> Optional[dict]:
    """The validated record of ``batch`` from ``backend``, one retry on malformed output; None while
    the backend's answer is pending."""
    prompt = build_prompt(batch)
    raws, errors, labels = [], [], None
    for attempt in (0, 1):
        raw = backend.complete(batch, prompt if attempt == 0 else prompt + retry_note(errors), attempt)
        if raw is None:
            return None
        raws.append(raw)
        labels, errors = validate(raw, batch)
        if labels is not None:
            break
    return {"batch_id": batch["batch_id"], "annotator": annotator, "backend": backend.describe(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "raw": raws, "labels": labels,
            "errors": errors}


# ------------------------------------------------------------------ backends

class FakeBackend:
    """Deterministic: rule(item) -> {verdict, subtype, overlap_flag} for every item."""

    def __init__(self, rule: Callable[[dict], dict]):
        self.rule = rule

    def complete(self, batch, prompt, attempt):
        return json.dumps({"labels": [dict(self.rule(it), item_id=it["item_id"], confidence=3, rationale="rule")
                                      for it in batch["items"]]})

    def describe(self):
        return {"backend": "fake"}


class SubagentBackend:
    """Writes ``<dir>/<batch_id>.prompt<k>.txt`` and reads ``<dir>/<batch_id>.response<k>.json``,
    written by an agent the orchestrator runs; never calls a model itself."""

    def __init__(self, directory: str, model: str = "subagent"):
        self.dir, self.model = directory, model

    def complete(self, batch, prompt, attempt):
        base = os.path.join(self.dir, f"{batch['batch_id']}.")
        if os.path.exists(base + f"response{attempt}.json"):
            with open(base + f"response{attempt}.json", encoding="utf-8") as fh:
                return fh.read()
        os.makedirs(self.dir, exist_ok=True)
        with open(base + f"prompt{attempt}.txt", "w", encoding="utf-8") as fh:
            fh.write(prompt)
        return None

    def describe(self):
        return {"backend": "subagent", "model": self.model}


class HttpBackend:
    """OpenAI-compatible chat completions (for example vLLM) with the batch's response schema
    (``batch_schema``) for guided decoding."""

    def __init__(self, base_url: str, model: str, *, temperature: float = 0.0, seed: int = 0,
                 max_tokens: int = 4096, api_key: Optional[str] = None, timeout: float = 600):
        self.url, self.model, self.api_key, self.timeout = base_url.rstrip("/") + "/chat/completions", model, \
            api_key, timeout
        self.params = {"temperature": temperature, "seed": seed, "max_tokens": max_tokens}

    def complete(self, batch, prompt, attempt):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], **self.params,
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "labels", "schema": batch_schema(batch)}}}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]

    def describe(self):
        return {"backend": "http", "url": self.url, "model": self.model, "schema": "per-batch", **self.params}
