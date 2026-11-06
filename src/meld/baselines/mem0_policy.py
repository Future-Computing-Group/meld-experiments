"""Mem0's write path (mem0ai 1.0.11, ``Memory._add_to_vector_store`` with ``infer=True``) as
receive-path policies for ``memory.BaselineMemory``, in two arms.

Upstream: one LLM call extracts the new facts from the message (system prompt
``USER_MEMORY_EXTRACTION_PROMPT``, user prompt ``"Input:\\n" + parse_messages(messages)``,
``response_format={"type": "json_object"}``; the reply's ``"facts"`` list, normalized; an empty,
unparsable or fact-less reply gives no facts). No facts: nothing is stored and the update call
is not made. Otherwise each fact retrieves its ``limit=5`` nearest memories, the retrieved ids
are mapped to "0".."n-1", and one LLM call with ``get_update_memory_messages(old, facts)``
returns a list of events. ADD stores the given text; UPDATE(id, text) rewrites memory id in
place (also when the text is unchanged: it is still reported as an UPDATE event); DELETE(id)
removes it; NONE changes nothing. An entry with an empty text or an id outside the mapping is
skipped. An empty or unparsable reply stores nothing.

``Mem0Policy`` ("mem0-update") isolates the update step: the arriving claim is the single new
fact and the extraction call is not made. ``Mem0FullPolicy`` ("mem0-full") runs the upstream
pipeline: extraction on the claim text, then one update call on all extracted facts.
Retrieval is the memory's own (top ``k`` live claims by encoder similarity), in both arms.

Storage: UPDATE with a new text retires the old claim and stores the rewrite; UPDATE with the
unchanged text leaves the store unchanged; DELETE retires the old claim; the arriving claim is
stored only if some ADD or UPDATE carries its exact text, and any other ADD/UPDATE text is
stored as a new claim. ``Verdict.events`` is upstream's returned event list with the retrieved
ids as indices: (event, index or None for ADD, text). Mapping of the arrival (first match): any
DELETE -> CONFLICT (a contradiction resolved by deletion, nothing surfaced), any UPDATE ->
MERGE, any ADD -> INSERT, otherwise -> MERGE (absorbed, event NONE); an empty or unparsable
update reply -> REJECT (event PARSE_FAIL, counted); no extracted facts -> REJECT (event
NO_FACTS, counted): nothing stored, the arriving claim is lost.
"""
from __future__ import annotations

import json
from typing import List

from meld.baselines._mem0_prompts import (ensure_json_instruction, extract_json,
                                          get_fact_retrieval_messages, get_update_memory_messages,
                                          normalize_facts, parse_messages, remove_code_blocks)
from meld.baselines.memory import Verdict


def _parse(raw: str):
    """Upstream's parse: fenced or bare JSON, then the outermost {...}; None on failure."""
    if not raw or not raw.strip():
        return None
    try:
        return json.loads(remove_code_blocks(raw), strict=False)
    except json.JSONDecodeError:
        try:
            return json.loads(extract_json(raw), strict=False)
        except json.JSONDecodeError:
            return None


def extraction_messages(text: str) -> List[dict]:
    """The fact-extraction request upstream sends for ``Memory.add(text)`` (a user message)."""
    system, user = ensure_json_instruction(*get_fact_retrieval_messages(
        parse_messages([{"role": "user", "content": text}]), False))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_facts(raw: str) -> List[str]:
    """Upstream's fact parse: any failure gives no facts."""
    try:
        cleaned = remove_code_blocks(raw)
        if not cleaned.strip():
            return []
        try:
            facts = json.loads(cleaned, strict=False)["facts"]
        except json.JSONDecodeError:
            facts = json.loads(extract_json(raw), strict=False)["facts"]
        return normalize_facts(facts)
    except Exception:
        return []


class Mem0Policy:
    name = "mem0-update"
    extract_first = False

    def __init__(self, llm, *, k: int = 5):
        self.llm, self.k = llm, k
        self.counters = {"calls": 0, "parse_fail": 0, "extract_calls": 0, "no_facts": 0}

    def extract(self, text: str) -> List[str]:
        self.counters["calls"] += 1
        self.counters["extract_calls"] += 1
        return parse_facts(self.llm.chat_json(extraction_messages(text), {"type": "json_object"}))

    def __call__(self, new: str, old: List[str]) -> Verdict:
        facts = self.extract(new) if self.extract_first else [new]
        if not facts:
            self.counters["no_facts"] += 1
            return Verdict("REJECT", "NO_FACTS")
        return self.update(facts, old)

    def update(self, facts: List[str], old: List[str]) -> Verdict:
        prompt = get_update_memory_messages([{"id": str(i), "text": t} for i, t in enumerate(old)],
                                            facts)
        self.counters["calls"] += 1
        actions = _parse(self.llm.chat_json([{"role": "user", "content": prompt}],
                                            {"type": "json_object"}))
        if not isinstance(actions, dict):
            self.counters["parse_fail"] += 1
            return Verdict("REJECT", "PARSE_FAIL")
        ids = {str(i): i for i in range(len(old))}
        revoke, add, events, target = [], [], [], None
        for r in actions.get("memory", []) if isinstance(actions.get("memory"), list) else []:
            if not isinstance(r, dict) or not r.get("text") or not isinstance(r["text"], str):
                continue
            text, ev, rid = r["text"], r.get("event"), r.get("id")
            i = ids.get(rid) if isinstance(rid, str) else None
            if ev == "ADD":
                add.append(text)
                events.append((ev, None, text))
            elif ev in ("UPDATE", "DELETE") and i is not None:
                events.append((ev, i, text))
                target = i if target is None else target
                if ev == "UPDATE" and text == old[i]:
                    continue                    # rewritten in place with the same text
                if ev == "UPDATE":
                    add.append(text)
                revoke.append(i)
        kinds = {e for e, _, _ in events}
        decision, event = next(((d, e) for e, d in (("DELETE", "CONFLICT"), ("UPDATE", "MERGE"),
                                                    ("ADD", "INSERT")) if e in kinds),
                               ("MERGE", "NONE"))
        return Verdict(decision, event, tuple(dict.fromkeys(revoke)), tuple(add), target,
                       tuple(events))


class Mem0FullPolicy(Mem0Policy):
    name = "mem0-full"
    extract_first = True


__all__ = ["Mem0Policy", "Mem0FullPolicy", "extraction_messages", "parse_facts"]
