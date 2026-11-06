"""HotpotQA (distractor) loader + the MELD brain-feed.

The competence benchmark is PIVOTED from GAIA (token-gated on Hugging Face) to
**HotpotQA distractor** (open, non-gated, CC BY-SA 4.0; ``hotpotqa/hotpot_qa``).
Verified at source; provenance in ``data/benchmark_provenance.json``.

Unlike GAIA, the real corpus is OPEN, so ``load_hotpot_real`` actually loads the
distractor validation split via ``datasets.load_dataset`` (run on the CSC node or
any networked host). For the LOCAL offline path a tiny deterministic fixture in
the HF distractor SCHEMA exercises the agent <-> MELD-brain <-> scorer harness with
NO network.

LOAD-BEARING MELD DECISION (the MELD-on-message choice): an item's ``context``
paragraphs are fed THROUGH the MELD brain as merged observations
(``feed_context_through_brain``: each sentence -> PUBLISH -> admission, chunked
into recallable claims), NOT injected as a raw prompt blob. So the
distractor paragraphs accrue in the brain and a later ``recall`` retrieves the
relevant evidence; the gate measures recall + multi-hop reasoning over
brain-resident memory, which is precisely what MELD contributes (GAIA's competence
is tool-use-bound; HotpotQA's is recall/reason-bound -> a better MELD fit).

Schema note: the HF config exposes ``context`` as a dict of parallel
``title``/``sentences`` lists (and ``supporting_facts`` as ``title``/``sent_id``
lists), NOT the original JSON list-of-[title, [sentences]] tuples. The loader
normalizes BOTH forms to ``[[title, [sentences]], ...]``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, List

from meld.agent.memory import SMPMemory
from meld.merge.classifier import Decision


@dataclass
class HotpotTask:
    task_id: str
    question: str
    answer: str               # gold
    level: str                # "easy" | "medium" | "hard"
    qtype: str = ""           # "bridge" | "comparison"
    # context: list of [title, [sentences]] paragraphs (distractors + gold)
    context: List[List[Any]] = field(default_factory=list)
    # supporting_facts: list of [title, sent_id]
    supporting_facts: List[List[Any]] = field(default_factory=list)


def _tid(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------
# schema normalization (HF dict form AND original list form -> list form)
# --------------------------------------------------------------------------


def normalize_context(context: Any) -> List[List[Any]]:
    """Return ``[[title, [sentences]], ...]`` from either the HF dict form
    (``{"title": [...], "sentences": [[...], ...]}``) or the original JSON list
    form (already ``[[title, [sentences]], ...]``)."""
    if isinstance(context, dict):
        titles = context.get("title", [])
        sents = context.get("sentences", [])
        return [[t, list(s)] for t, s in zip(titles, sents)]
    # already the original list form
    return [[p[0], list(p[1])] for p in (context or [])]


def normalize_supporting_facts(sf: Any) -> List[List[Any]]:
    """Return ``[[title, sent_id], ...]`` from either the HF dict form
    (``{"title": [...], "sent_id": [...]}``) or the original list form."""
    if isinstance(sf, dict):
        return [[t, i] for t, i in zip(sf.get("title", []),
                                       sf.get("sent_id", []))]
    return [[p[0], p[1]] for p in (sf or [])]


# --------------------------------------------------------------------------
# offline fixture (HF distractor schema, deterministic) — wiring only
# --------------------------------------------------------------------------

# Real-style multi-hop items in the HF distractor schema (context = dict of
# parallel title/sentences lists; a couple of DISTRACTOR paragraphs each). These
# are NOT HotpotQA items (the real corpus is loaded by load_hotpot_real); they
# validate the loader + brain-feed + scorer + gate wiring with no network.
_FIXTURE = [
    {
        "question": "What is the nationality of the director of the film "
                    "Inception?",
        "answer": "British",
        "level": "medium",
        "type": "bridge",
        "context": {
            "title": ["Inception", "Christopher Nolan", "Tenet"],
            "sentences": [
                ["Inception is a 2010 science fiction film.",
                 "It was directed by Christopher Nolan."],
                ["Christopher Nolan is a British film director.",
                 "He was born in London."],
                ["Tenet is a 2020 film also directed by Nolan."],
            ],
        },
        "supporting_facts": {"title": ["Inception", "Christopher Nolan"],
                             "sent_id": [1, 0]},
    },
    {
        "question": "Which city is the capital of the country whose flag has "
                    "blue, white and red vertical bands?",
        "answer": "Paris",
        "level": "hard",
        "type": "bridge",
        "context": {
            "title": ["Flag of France", "Paris", "Berlin"],
            "sentences": [
                ["The flag of France has three vertical bands of blue, white "
                 "and red."],
                ["Paris is the capital and largest city of France."],
                ["Berlin is the capital of Germany."],
            ],
        },
        "supporting_facts": {"title": ["Flag of France", "Paris"],
                             "sent_id": [0, 0]},
    },
    {
        "question": "Were the films Inception and Tenet directed by the same "
                    "person? Answer yes or no.",
        "answer": "yes",
        "level": "easy",
        "type": "comparison",
        "context": {
            "title": ["Inception", "Tenet", "Dunkirk"],
            "sentences": [
                ["Inception was directed by Christopher Nolan."],
                ["Tenet was directed by Christopher Nolan."],
                ["Dunkirk is a 2017 war film."],
            ],
        },
        "supporting_facts": {"title": ["Inception", "Tenet"], "sent_id": [0, 0]},
    },
    {
        "question": "In which city was the director of Inception born?",
        "answer": "London",
        "level": "medium",
        "type": "bridge",
        "context": {
            "title": ["Inception", "Christopher Nolan", "Interstellar"],
            "sentences": [
                ["Inception was directed by Christopher Nolan."],
                ["Christopher Nolan was born in London, England."],
                ["Interstellar is a 2014 film by Nolan."],
            ],
        },
        "supporting_facts": {"title": ["Inception", "Christopher Nolan"],
                             "sent_id": [0, 1]},
    },
    {
        "question": "What is the capital of the country where the river Seine "
                    "flows through its largest city?",
        "answer": "Paris",
        "level": "hard",
        "type": "bridge",
        "context": {
            "title": ["Seine", "Paris", "Thames"],
            "sentences": [
                ["The Seine flows through Paris, the largest city of France."],
                ["Paris is the capital of France."],
                ["The Thames flows through London."],
            ],
        },
        "supporting_facts": {"title": ["Seine", "Paris"], "sent_id": [0, 1]},
    },
    {
        "question": "Did Inception and Dunkirk share the same director? Answer "
                    "yes or no.",
        "answer": "yes",
        "level": "easy",
        "type": "comparison",
        "context": {
            "title": ["Inception", "Dunkirk", "Memento"],
            "sentences": [
                ["Inception was directed by Christopher Nolan."],
                ["Dunkirk was directed by Christopher Nolan."],
                ["Memento is a 2000 film by Nolan."],
            ],
        },
        "supporting_facts": {"title": ["Inception", "Dunkirk"], "sent_id": [0, 0]},
    },
]


def _task_from_raw(raw: dict, salt: str = "") -> HotpotTask:
    tid = raw.get("id") or _tid(salt + raw["question"])
    return HotpotTask(
        task_id=str(tid),
        question=raw["question"],
        answer=raw["answer"],
        level=raw.get("level", "medium"),
        qtype=raw.get("type", ""),
        context=normalize_context(raw.get("context")),
        supporting_facts=normalize_supporting_facts(raw.get("supporting_facts")),
    )


def load_hotpot_fixture(n: int = 4) -> List[HotpotTask]:
    """Return ``n`` deterministic HotpotQA-shaped fixture tasks (cycles)."""
    tasks: List[HotpotTask] = []
    for i in range(n):
        raw = _FIXTURE[i % len(_FIXTURE)]
        tasks.append(_task_from_raw(raw, salt=f"{i}:"))
    return tasks


# --------------------------------------------------------------------------
# the MELD brain feed (load-bearing MELD-on-message choice)
# --------------------------------------------------------------------------


def feed_context_through_brain(task: HotpotTask, memory: SMPMemory) -> List[Decision]:
    """Feed the item's ``context`` paragraphs THROUGH the MELD brain as admitted
    observations: each sentence (with its title) is published as one signed delta
    and admitted (PUBLISH -> receive gate -> admission). So the distractor
    paragraphs accrue in the brain and a later ``recall`` retrieves the relevant
    evidence (multi-hop reasoning over brain-resident memory).

    Returns the admission decisions (none for an exact copy of a held claim)."""
    patches: List[Decision] = []
    for title, sentences in task.context:
        # one observation per sentence keyed by its title, so each gold/distractor
        # sentence is an independently recallable claim (chunked claims path).
        for sent in sentences:
            text = f"{title}: {sent}".strip()
            patches.extend(memory.write(text))
    return patches


# --------------------------------------------------------------------------
# the REAL (open, non-gated) loader
# --------------------------------------------------------------------------


def load_hotpot_real(split: str = "validation", config: str = "distractor",
                     n: int | None = None, cache_dir: str | None = None,
                     levels: list | None = None) -> List[HotpotTask]:
    """Load the REAL HotpotQA distractor split.

    HotpotQA is OPEN / NON-GATED (CC BY-SA 4.0), so this needs no HF token. It is
    a CSC-time / networked-host path (``datasets`` is not in the offline ``.venv``).
    On the CSC compute node run it offline by prefetching the dataset into
    ``HF_HOME`` on a login node first and setting ``HF_HUB_OFFLINE=1``.

    ``levels`` optionally filters to e.g. ``["medium", "hard"]`` for the gate
    slice; ``n`` caps the count (level-stratified by the caller's slice builder).
    """
    from datasets import load_dataset  # CSC-time dep; absent in offline .venv

    ds = load_dataset("hotpotqa/hotpot_qa", config, split=split,
                      cache_dir=cache_dir, trust_remote_code=True)
    tasks: List[HotpotTask] = []
    for row in ds:
        if levels and row.get("level") not in levels:
            continue
        tasks.append(_task_from_raw(dict(row)))
        if n is not None and len(tasks) >= n:
            break
    return tasks


def stratified_slice(tasks: List[HotpotTask], per_level: dict) -> List[HotpotTask]:
    """Build a level-stratified slice: ``per_level`` maps level -> count. Keeps
    input order within each level (deterministic given a deterministic source)."""
    out: List[HotpotTask] = []
    remaining = dict(per_level)
    for t in tasks:
        if remaining.get(t.level, 0) > 0:
            out.append(t)
            remaining[t.level] -= 1
    return out


__all__ = ["HotpotTask", "load_hotpot_fixture", "load_hotpot_real",
           "feed_context_through_brain", "normalize_context",
           "normalize_supporting_facts", "stratified_slice"]
