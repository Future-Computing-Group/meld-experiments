"""Graphiti's edge-resolution step (graphiti-core 0.30.2, ``resolve_extracted_edge`` through
``OpenAIGenericClient``) as a receive-path policy for ``memory.BaselineMemory``.

Upstream: with no candidate edges the new edge is stored without an LLM call; an existing
edge with the same normalized fact text is reused without a call; otherwise one call with
``resolve_edge`` (EXISTING FACTS = same-endpoint edges, FACT INVALIDATION CANDIDATES = a
wider search), the language instruction appended to the system message, every message
cleaned, and ``response_format`` the ``EdgeDuplicate`` JSON schema. The first in-range
duplicate becomes the resolved edge (the new edge is not stored); in-range contradicted
edges are invalidated (``invalid_at`` set) when they are older than the resolved edge.

Here a claim store has no entity endpoints, so the top-k retrieved claims are the EXISTING
FACTS and the invalidation list is empty (indices are continuous, so nothing is lost).
Events: NEW (no candidates), EXACT (normalized text match), DUPLICATE, CONTRADICTED, NONE,
PARSE_FAIL (unparsable or schema-invalid reply: nothing stored, counted). Mapping: an
invalidated claim -> CONFLICT (resolved by invalidation, nothing surfaced); a duplicate ->
MERGE; NEW or NONE -> INSERT; PARSE_FAIL -> REJECT.
"""
from __future__ import annotations

import json
from typing import List

from meld.baselines._graphiti_prompts import (EDGE_DUPLICATE_SCHEMA, _clean_input,
                                              _normalize_string_exact, _strip_code_fences,
                                              get_extraction_language_instruction, resolve_edge)
from meld.baselines.memory import Verdict

RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "EdgeDuplicate", "schema": EDGE_DUPLICATE_SCHEMA}}


def _ints(v) -> bool:
    return isinstance(v, list) and all(type(i) is int for i in v)


class GraphitiPolicy:
    name = "graphiti"

    def __init__(self, llm, *, k: int = 5):
        self.llm, self.k = llm, k
        self.counters = {"calls": 0, "parse_fail": 0}

    def __call__(self, new: str, old: List[str]) -> Verdict:
        if not old:
            return Verdict("INSERT", "NEW", add=(new,))
        n = _normalize_string_exact(new)
        for i, t in enumerate(old):
            if _normalize_string_exact(t) == n:
                return Verdict("MERGE", "EXACT", target=i)
        msgs = resolve_edge({"existing_edges": [{"idx": i, "fact": t} for i, t in enumerate(old)],
                             "new_edge": new, "edge_invalidation_candidates": []})
        msgs[0] = msgs[0]._replace(content=msgs[0].content + get_extraction_language_instruction())
        self.counters["calls"] += 1
        raw = self.llm.chat_json([{"role": m.role, "content": _clean_input(None, m.content)}
                                  for m in msgs], RESPONSE_FORMAT)
        try:
            r = json.loads(_strip_code_fences(raw or ""))
        except json.JSONDecodeError:
            r = None
        if not (isinstance(r, dict) and _ints(r.get("duplicate_facts"))
                and _ints(r.get("contradicted_facts"))):
            self.counters["parse_fail"] += 1
            return Verdict("REJECT", "PARSE_FAIL")
        dups = [i for i in r["duplicate_facts"] if 0 <= i < len(old)]
        resolved = dups[0] if dups else None
        # ponytail: Graphiti orders by extracted valid_at (one more LLM call); here every
        # retrieved claim is older than the arrival and, when a duplicate is resolved, the
        # other contradicted claims are taken as older than it (its own index is skipped).
        inval = tuple(dict.fromkeys(i for i in r["contradicted_facts"]
                                    if 0 <= i < len(old) and i != resolved))
        add = () if dups else (new,)
        if inval:
            return Verdict("CONFLICT", "CONTRADICTED", inval, add, inval[0])
        if dups:
            return Verdict("MERGE", "DUPLICATE", target=resolved)
        return Verdict("INSERT", "NONE", add=add)


__all__ = ["GraphitiPolicy", "RESPONSE_FORMAT"]
