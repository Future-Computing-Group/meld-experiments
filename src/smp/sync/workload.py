"""Campaign workload assignment + the eval-r needed-oracle.

Topic-labelled claims are homed on brains (topic-skewed, ``replication`` brains per
topic). Each brain ingests the claims of the topics it homes -> its topical content.
The needed-set of a claim is the set of brains homing its topic: an INDEPENDENT ground
truth (topic labels, not the router's sigma), so eval-r precision/recall measured against
it is not circular. Deterministic under ``seed`` (seeded Random; seed = the experiment
unit, STATS-C1). HotpotQA loading (datasets-gated) lives in load_hotpot_claims().
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple


@dataclass
class WorkloadCell:
    assignments: Dict[str, List[str]]        # brain_id -> claim texts to pre-ingest
    needed: Dict[str, FrozenSet]             # claim text -> brain_ids that NEED it (topic home)
    topic_home: Dict[str, Tuple[str, ...]]   # topic -> sorted home brain_ids


def build_workload(claims: List[Tuple[str, str]], brain_ids: List[str], *,
                   replication: int, seed: int) -> WorkloadCell:
    """Assign (text, topic) claims to ``brain_ids``; each topic homed on ``replication``
    brains; return per-brain assignments + the needed-oracle (topic-home sets)."""
    if not (1 <= replication <= len(brain_ids)):
        raise ValueError(f"replication must be in [1, {len(brain_ids)}], got {replication}")
    rng = random.Random(seed)
    topics = sorted({t for _, t in claims})
    topic_home: Dict[str, Tuple[str, ...]] = {}
    for t in topics:
        bs = list(brain_ids)
        rng.shuffle(bs)
        topic_home[t] = tuple(sorted(bs[:replication]))
    assignments: Dict[str, List[str]] = {b: [] for b in brain_ids}
    needed: Dict[str, FrozenSet] = {}
    for text, topic in claims:
        homes = topic_home[topic]
        for b in homes:
            assignments[b].append(text)
        needed[text] = frozenset(homes)
    return WorkloadCell(assignments=assignments, needed=needed, topic_home=topic_home)


def load_hotpot_claims(n: int, *, split: str = "validation",
                       config: str = "distractor"):  # pragma: no cover - datasets-gated (VM runtime)
    """Load ~n HotpotQA supporting-fact sentences as (claim_text, topic) pairs, topic =
    the source paragraph TITLE (so claims sharing a title share a topic -> a meaningful
    eval-r needed-set). Streams the open `hotpotqa/hotpot_qa` distractor split (no full
    download, no brain/protobuf import -> the orchestrator venv stays numpy+datasets).
    Handles both HF schema forms (dict of parallel lists, or list of tuples)."""
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", config, split=split, streaming=True)
    claims = []
    for item in ds:
        ctx, sf = item["context"], item["supporting_facts"]
        if isinstance(ctx, dict):
            cmap = {t: s for t, s in zip(ctx.get("title", []), ctx.get("sentences", []))}
        else:
            cmap = {p[0]: list(p[1]) for p in (ctx or [])}
        if isinstance(sf, dict):
            pairs = list(zip(sf.get("title", []), sf.get("sent_id", [])))
        else:
            pairs = [(p[0], p[1]) for p in (sf or [])]
        for title, sid in pairs:
            sents = cmap.get(title, [])
            if 0 <= sid < len(sents):
                claims.append((f"{title}: {sents[sid]}".strip(), title))
                if len(claims) >= n:
                    return claims
    return claims


__all__ = ["WorkloadCell", "build_workload", "load_hotpot_claims"]
