"""LongMemEval-shaped conversation loader.

Turns a conversation slice into well-formed Concept (MemoryUnit) / Claim /
Evidence deltas. For the LOCAL smoke (0 cost, offline) this emits a small,
deterministic, SYNTHETIC LongMemEval-SHAPED fixture: user-assistant facts with
KNOWN coreferent (paraphrase) and contradiction pairs and gold answers. It
validates the PIPELINE end-to-end, exactly as a cross-domain control slice does
-- it is NOT a LongMemEval benchmark result.

Wiring the REAL LongMemEval corpus (500 questions, accuracy metric; arXiv
2410.10813, verified at source and distinct from LongMemBench) is gated behind
the benchmark identity gate G0 and is a later, data-bearing step; this fixture
is the offline stand-in so the merge engine + topology can be exercised now.
"""
from __future__ import annotations

import random
from typing import Dict, List

from meld.agent.memory import _cid, claim_key
from meld.proto import Claim, MemoryUnit

# Templated fact families. Each family has a base fact, a paraphrase (coreferent,
# should MERGE/RELATE) and a contradiction (should CONFLICT). Cells/frames vary
# so the Context gate has something to bite on.
_FAMILIES = [
    {
        "topic": "capital",
        "base": "Paris is the capital of France",
        "paraphrase": "The capital of France is Paris",
        "contradiction": "Lyon is the capital of France",
        "question": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "topic": "pet",
        "base": "Alice owns a dog named Rex",
        "paraphrase": "Alice has a dog called Rex",
        "contradiction": "Alice owns a cat named Rex",
        "question": "What pet does Alice own?",
        "answer": "dog",
    },
    {
        "topic": "meeting",
        "base": "The standup meeting is on Monday",
        "paraphrase": "The daily standup is scheduled for Monday",
        "contradiction": "The standup meeting is on Tuesday",
        "question": "When is the standup meeting?",
        "answer": "Monday",
    },
    {
        "topic": "budget",
        "base": "The project budget increased this quarter",
        "paraphrase": "This quarter the project budget went up",
        "contradiction": "The project budget decreased this quarter",
        "question": "Did the project budget change?",
        "answer": "increase",
    },
]


def _unit(text: str) -> MemoryUnit:
    # production key helper (meld.agent.memory), so this fixture's keys cannot
    # drift from the live write path's (default cell, §4.1).
    return MemoryUnit(
        id=_cid(text),
        retention=1.0,
        status="ACTIVE",
        claims=[Claim(claim_key=claim_key(text), text=text, confidence=0.9,
                      claim_status="ACTIVE")],
    )


def load_conversation_slice(n_conversations: int, seed: int = 0) -> List[Dict]:
    """Return ``n_conversations`` conversation dicts with units + gold pairs +
    a QA pair. Deterministic given ``seed``."""
    rng = random.Random(seed)
    convs: List[Dict] = []
    for i in range(n_conversations):
        fam = _FAMILIES[i % len(_FAMILIES)]
        base_u = _unit(fam["base"])
        para_u = _unit(fam["paraphrase"])
        contra_u = _unit(fam["contradiction"])
        # the conversation contributes the base + (alternating) a paraphrase or
        # contradiction to create merge pressure.
        include_contra = (i // len(_FAMILIES)) % 2 == 1 or rng.random() < 0.5
        units = [base_u, contra_u] if include_contra else [base_u, para_u]
        gold_pairs = [
            {"kind": "coreferent",
             "a": base_u.claims[0].claim_key, "b": para_u.claims[0].claim_key},
            {"kind": "contradiction",
             "a": base_u.claims[0].claim_key, "b": contra_u.claims[0].claim_key},
        ]
        convs.append({
            "conversation_id": i,
            "topic": fam["topic"],
            "units": units,
            "gold_pairs": gold_pairs,
            "question": fam["question"],
            "answer": fam["answer"],
            "answer_unit_id": base_u.id,
        })
    return convs


def all_claim_pairs(convs: List[Dict]) -> List[Dict]:
    """Flatten the labelled claim pairs across conversations (for M-2)."""
    pairs: List[Dict] = []
    text_by_key = {}
    for c in convs:
        for u in c["units"]:
            for cl in u.claims:
                text_by_key[cl.claim_key] = cl.text
    # rebuild paraphrase/contradiction text from families so M-2 has both texts
    for c in convs:
        fam = next(f for f in _FAMILIES if f["topic"] == c["topic"])
        base, para, contra = fam["base"], fam["paraphrase"], fam["contradiction"]
        pairs.append({"text_a": base, "text_b": para, "label": "coreferent"})
        pairs.append({"text_a": base, "text_b": contra, "label": "contradiction"})
        # an unrelated pair (different family) as a negative
        other = _FAMILIES[(_FAMILIES.index(fam) + 1) % len(_FAMILIES)]
        pairs.append({"text_a": base, "text_b": other["base"], "label": "unrelated"})
    return pairs


__all__ = ["load_conversation_slice", "all_claim_pairs"]
