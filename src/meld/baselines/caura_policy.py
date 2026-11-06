"""Caura's write path (caura-ai/caura backend-v3.34.5, commit fd1da78b) as a receive-path policy
for ``memory.BaselineMemory``: the strong-mode write gate, then the semantic contradiction
detector, with the prompts, parsers and constants vendored in ``_caura_prompts``.

Upstream, per write (``write_mode="strong"``, tenant defaults, one owner agent):

1. Exact-hash gate (``CheckExactDuplicate``): a live row with the same content (same tenant,
   fleet and agent) refuses the write with a 409, no call.
2. Semantic gate (``CheckSemanticDuplicate``, ``semantic_dedup_enabled`` default true): skipped
   when the content is identifier-bearing (``_content_is_identifier_bearing``); otherwise the
   nearest live row by cosine of the local embedder. Similarity >= 0.97 refuses the write (409,
   no call); 0.85 <= similarity < 0.97 asks the dedup judge (``DEDUP_PROMPT``, contents cut at
   500 characters) and refuses only on ``is_duplicate`` at confidence >= 0.60
   (``_judge_dedup``); below 0.85 the write lands.
3. Contradiction detection, semantic path (``_detect``, after the row is written): the live rows
   at cosine >= 0.45, nearest first, at most 20. One candidate: one ``CONTRADICTION_PROMPT``
   call; several: one ``BATCH_CONTRADICTION_PROMPT`` call (``[i] content[:500]`` lines, sparse
   reply expanded by ``_expand_sparse_batch``). Each judgment goes through
   ``_judge_contradiction`` (gate 1: different subject; gate 2: a non-conflict shape). Every
   contradicted row is the older one here and is set to ``conflicted``; the new row supersedes
   the first of them. ``conflicted`` rows leave default search (kept only for an exact lexical
   query match) and are demoted by half in ranking.

Every LLM call is ``complete_json``: one user message, ``response_format`` json_object,
temperature 0; the reply must be a JSON object (``json.loads``, no fence stripping). Any failure
(empty, unparsable, not an object) takes upstream's fallback: the dedup judge becomes the
whitespace/case heuristic at confidence 0.50 (never a refusal), the contradiction judge
abstains (no contradiction). Each fallback is counted in ``parse_fail``.

Not modelled (no effect on a two-claim store with prose claims, or off by default): the RDF
triple path and Path C (both need resolved subject entities: identifier-shaped subjects only
at write time, LLM entity extraction later), enrichment and the atomic-fact fan-out, the
subject preflight (inert while subject ids are unset), opt-in governance pre-gates, the fast
write mode (advisory near-duplicate only, merge off by default), provider timeouts and retries
(a retry at temperature 0 returns the same reply). Retrieval: the memory hands the policy its
top ``k`` live claims by MELD's encoder (``k`` = Caura's candidate window, 20); within them the
policy ranks by its own embedder, as Caura's storage does over the whole store.

``CauraCrossAgentPolicy`` ("caura-xagent") is the same path for an arrival written by a
different agent of the fleet: both dedup gates are owner-scoped upstream (CAURA-721), so only
the contradiction detector runs.

Mapping (``Verdict.event``): NEW (empty store) and NO_CANDIDATE (nothing at cosine >= 0.45) and
CLEAR (judged, nothing contradicted) -> INSERT; EXACT, DUP_AUTO, DUP_JUDGE (the write refused,
the arrival absorbed by the existing row) -> MERGE; CONFLICTED -> CONFLICT with the contradicted
rows retired (a REVOKE: out of recall, as ``conflicted`` is out of default search) and the
arrival stored. No outcome drops the arrival unabsorbed (no REJECT) and none keeps a
contradicted claim live beside a surfaced link (no RELATE).
"""
from __future__ import annotations

import json
from typing import Callable, List, Union

import numpy as np

from meld.baselines._caura_prompts import (BATCH_CONTRADICTION_PROMPT, CONTRADICTION_CANDIDATE_WINDOW,
                                           CONTRADICTION_PROMPT, CONTRADICTION_SIMILARITY_THRESHOLD,
                                           DEDUP_JUDGE_CONFIDENCE_THRESHOLD, DEDUP_PROMPT,
                                           SEMANTIC_DEDUP_AUTO_THRESHOLD, SEMANTIC_DEDUP_JUDGE_THRESHOLD,
                                           _CONF_FALLBACK, _content_is_identifier_bearing,
                                           _expand_sparse_batch, _fake_dedup_check, _judge_contradiction,
                                           _judge_dedup)
from meld.baselines.memory import Verdict

DEFAULT_EMBEDDER = "BAAI/bge-large-en-v1.5"     # common/embedding/_registry.py, EMBEDDING_PROVIDER=local
JSON_OBJECT = {"type": "json_object"}


class _Unusable(Exception):
    """A reply upstream's complete_json would raise on."""


def _complete_json(llm, prompt: str) -> dict:
    raw = llm.chat_json([{"role": "user", "content": prompt}], JSON_OBJECT)
    if not raw:
        raise _Unusable("empty content")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise _Unusable(str(e))
    if not isinstance(parsed, dict):
        raise _Unusable(type(parsed).__name__)
    return parsed


def _st_embedder(name: str) -> Callable[[List[str]], np.ndarray]:
    if name == "hashing":
        from meld.merge.signals import _HashingEncoder
        enc = _HashingEncoder()
        return lambda texts: np.array([enc.encode_one(t) for t in texts])
    model = []

    def embed(texts):
        if not model:
            import os
            from sentence_transformers import SentenceTransformer
            model.append(SentenceTransformer(name, device=os.environ.get("SMP_ST_DEVICE", "cpu")))
        return np.asarray(model[0].encode(list(texts), normalize_embeddings=True, batch_size=32,
                                          show_progress_bar=False), dtype=np.float64)
    return embed


class CauraPolicy:
    name = "caura"
    owner_scoped_gates = True

    def __init__(self, llm, *, k: int = CONTRADICTION_CANDIDATE_WINDOW,
                 embedder: Union[str, Callable, None] = None):
        self.llm, self.k = llm, k
        embedder = DEFAULT_EMBEDDER if embedder is None else embedder
        self.embedder_name = embedder if isinstance(embedder, str) else getattr(embedder, "__name__", "custom")
        self._embed = _st_embedder(embedder) if isinstance(embedder, str) else embedder
        self._cache: dict = {}
        self.counters = {"calls": 0, "parse_fail": 0, "dedup_calls": 0, "contradiction_calls": 0,
                         "batch_calls": 0, "embedder": self.embedder_name}

    def _vectors(self, texts: List[str]) -> np.ndarray:
        miss = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if miss:
            for t, v in zip(miss, self._embed(miss)):
                v = np.asarray(v, dtype=np.float64)
                n = np.linalg.norm(v)
                self._cache[t] = v / n if n else v
        return np.array([self._cache[t] for t in texts])

    def _call(self, prompt: str, key: str):
        self.counters["calls"] += 1
        self.counters[key] += 1
        try:
            return _complete_json(self.llm, prompt)
        except _Unusable:
            self.counters["parse_fail"] += 1
            return None

    def _is_duplicate(self, new: str, old: str) -> bool:
        raw = self._call(DEDUP_PROMPT.format(new_content=new[:500], old_content=old[:500]), "dedup_calls")
        is_dup, conf = (_fake_dedup_check(new, old), _CONF_FALLBACK) if raw is None else _judge_dedup(raw)
        return is_dup and conf >= DEDUP_JUDGE_CONFIDENCE_THRESHOLD

    def _contradicted(self, new: str, cands: List[str]) -> List[bool]:
        if len(cands) == 1:
            raw = self._call(CONTRADICTION_PROMPT.format(new_content=new[:500], old_content=cands[0][:500]),
                             "contradiction_calls")
            return [raw is not None and _judge_contradiction(raw)[0]]
        block = "\n".join(f"[{i}] {c[:500]}" for i, c in enumerate(cands))
        raw = self._call(BATCH_CONTRADICTION_PROMPT.format(new_content=new[:500], candidates_block=block),
                         "batch_calls")
        raws = [{} for _ in cands] if raw is None else _expand_sparse_batch(raw, len(cands))
        return [_judge_contradiction(r)[0] for r in raws]

    def __call__(self, new: str, old: List[str]) -> Verdict:
        if not old:
            return Verdict("INSERT", "NEW", add=(new,))
        if self.owner_scoped_gates and new in old:
            return Verdict("MERGE", "EXACT", target=old.index(new))
        v = self._vectors([new, *old])
        sims = v[1:] @ v[0]
        order = sorted(range(len(old)), key=lambda i: (-sims[i], i))
        events = []
        if self.owner_scoped_gates and not _content_is_identifier_bearing(new):
            j = order[0]
            if sims[j] >= SEMANTIC_DEDUP_AUTO_THRESHOLD:
                return Verdict("MERGE", "DUP_AUTO", target=j, events=(("dedup", j, float(sims[j]), "auto"),))
            if sims[j] >= SEMANTIC_DEDUP_JUDGE_THRESHOLD:
                dup = self._is_duplicate(new, old[j])
                events.append(("dedup", j, float(sims[j]), "duplicate" if dup else "accepted"))
                if dup:
                    return Verdict("MERGE", "DUP_JUDGE", target=j, events=tuple(events))
        cands = [i for i in order if sims[i] >= CONTRADICTION_SIMILARITY_THRESHOLD][:CONTRADICTION_CANDIDATE_WINDOW]
        if not cands:
            return Verdict("INSERT", "NO_CANDIDATE", add=(new,), events=tuple(events))
        verdicts = self._contradicted(new, [old[i] for i in cands])
        events += [("contradiction", i, float(sims[i]), bool(x)) for i, x in zip(cands, verdicts)]
        hit = tuple(i for i, x in zip(cands, verdicts) if x)
        if hit:
            return Verdict("CONFLICT", "CONFLICTED", hit, (new,), hit[0], tuple(events))
        return Verdict("INSERT", "CLEAR", add=(new,), events=tuple(events))


class CauraCrossAgentPolicy(CauraPolicy):
    name = "caura-xagent"
    owner_scoped_gates = False


__all__ = ["CauraPolicy", "CauraCrossAgentPolicy", "DEFAULT_EMBEDDER"]
