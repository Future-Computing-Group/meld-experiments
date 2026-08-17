"""Campaign workload assignment + the eval-r needed-oracle.

Topic-labelled claims are homed on brains (topic-skewed, ``replication`` brains per
topic). Each brain ingests the claims of the topics it homes -> its topical content.
The needed-set of a claim is the set of brains homing its topic: an INDEPENDENT ground
truth (topic labels, not the router's sigma), so eval-r precision/recall measured against
it is not circular. Deterministic under ``seed`` (seeded Random; seed = the experiment
unit, STATS-C1). HotpotQA loading (datasets-gated) lives in load_hotpot_claims().
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple


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


def load_hotpot_claims(n: int, *, split: str = "validation", config: str = "distractor",
                       cache_dir: Optional[str] = None,
                       _fetch: Optional[Callable] = None) -> List[Tuple[str, str]]:
    """Load ~n HotpotQA supporting-fact sentences as (claim_text, topic) pairs.

    The campaign calls this once PER CELL. To avoid re-streaming the full HotpotQA parquet
    over the network every cell (which wedged the grid when HF throttled the resolve
    endpoint), the result is cached to a local JSON keyed by (n, split, config): the first
    cell fetches + stores, every later cell reads from disk (zero network). Override the
    directory with ``cache_dir`` or ``$SMP_WORKLOAD_CACHE`` (default ~/.cache/meld-workload).
    ``_fetch`` is injectable for tests; production uses the streaming HF fetch below."""
    cdir = Path(cache_dir or os.environ.get("SMP_WORKLOAD_CACHE")
                or Path.home() / ".cache" / "meld-workload")
    cache_path = cdir / f"hotpot_{config}_{split}_n{n}.json"
    if cache_path.exists():
        return [tuple(x) for x in json.loads(cache_path.read_text())]
    fetch = _fetch or _stream_hotpot
    claims = [tuple(c) for c in fetch(n, split=split, config=config)]
    cdir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(claims))
    return claims


def _stream_hotpot(n: int, *, split: str,
                   config: str) -> List[Tuple[str, str]]:  # pragma: no cover - datasets-gated (VM runtime)
    """Stream the open `hotpotqa/hotpot_qa` split; topic = the source paragraph TITLE (so
    claims sharing a title share a topic -> a meaningful eval-r needed-set). No brain/protobuf
    import -> the orchestrator venv stays numpy+datasets. Handles both HF schema forms (dict
    of parallel lists, or list of tuples)."""
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
