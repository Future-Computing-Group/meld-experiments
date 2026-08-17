"""GAIA task loader: offline GAIA-shaped fixture (real corpus is gated).

The real GAIA corpus (``gaia-benchmark/GAIA`` on Hugging Face) is GATED: it
requires login, agreeing to share contact info, and agreeing NOT to reshare the
validation/test sets. It is wired ONLY on the gated CSC run, after the benchmark
identity gate (G0). For the LOCAL WIRING smoke this module emits a tiny,
deterministic, GAIA-SHAPED fixture so the agent <-> MELD-brain <-> scorer harness
runs end-to-end with NO network and NO GAIA data.

Each fixture task carries the in-context CLUE(S) the agent must read and remember
(through the MELD merge path) in order to answer -- so the fixture exercises the
memory integration, not raw model recall. The fixture is NOT a GAIA result; it
validates wiring only. ``load_gaia_real`` is the gated hook (raises locally).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List


@dataclass
class GaiaTask:
    task_id: str
    question: str
    gold: str
    level: str               # "L1" | "L2" | "L3"
    clues: List[str] = field(default_factory=list)
    file_path: str = ""      # local path to a real GAIA attachment (empty for text-only / fixture)


def _tid(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


# GAIA-shaped fixture tasks. Each needs the agent to remember clue(s) and answer
# in the GAIA short-answer format. Levels reflect the GAIA semantics (number of
# steps / tools): L1 single clue, L2 combine two clues (a step), L3 multi-step.
_FIXTURE = [
    {
        "question": "What is the capital of the country whose flag is blue, white, and red in vertical bands?",
        "gold": "Paris",
        "level": "L2",
        "clues": ["France's flag has blue, white, and red vertical bands",
                  "Paris is the capital of France"],
    },
    {
        "question": "What pet does Alice own?",
        "gold": "dog",
        "level": "L1",
        "clues": ["Alice owns a dog named Rex"],
    },
    {
        "question": "On which day is the standup meeting, and is the budget up or down? Answer as 'day, direction'.",
        "gold": "Monday, up",
        "level": "L2",
        "clues": ["The standup meeting is on Monday",
                  "The project budget increased this quarter"],
    },
    {
        "question": "How many sides does the shape on the warning sign have?",
        "gold": "3",
        "level": "L1",
        "clues": ["The warning sign shows a triangle"],
    },
    {
        "question": "What is the capital of Alice's country if Alice lives in France?",
        "gold": "Paris",
        "level": "L2",
        "clues": ["Alice lives in France", "Paris is the capital of France"],
    },
    {
        "question": "Name the pet's species and the meeting day, comma-separated.",
        "gold": "dog, Monday",
        "level": "L3",
        "clues": ["Alice owns a dog named Rex",
                  "The standup meeting is on Monday"],
    },
]


def load_gaia_fixture(n: int = 4) -> List[GaiaTask]:
    """Return ``n`` deterministic GAIA-shaped fixture tasks (cycles if n > len)."""
    tasks: List[GaiaTask] = []
    for i in range(n):
        spec = _FIXTURE[i % len(_FIXTURE)]
        tid = _tid(f"{i}:{spec['question']}")
        tasks.append(GaiaTask(task_id=tid, question=spec["question"],
                              gold=spec["gold"], level=spec["level"],
                              clues=list(spec["clues"])))
    return tasks


def _row_to_task(row: dict) -> GaiaTask:
    """Map one HF GAIA row -> GaiaTask. Pure (no dataset/token) so it is unit-testable.
    clues stay empty -- on GAIA the agent GATHERS evidence (via tools/files) rather than
    being handed clues; the file attachment, if any, is carried in file_path."""
    return GaiaTask(
        task_id=str(row["task_id"]),
        question=row["Question"],
        gold=str(row.get("Final answer", "")),
        level=f"L{row['Level']}",
        clues=[],
        file_path=(row.get("file_path") or ""),
    )


def load_gaia_real(n: int | None = None, *, level: str = "2",
                   split: str = "validation"):  # pragma: no cover - gated (token + cached data)
    """Load the REAL GAIA ``2023_level{level}`` ``split`` slice as GaiaTasks. Gated: needs an
    authenticated HF token (HUGGING_FACE_HUB_TOKEN/HF_TOKEN) + accepted dataset terms. The
    validation split carries gold answers (for the mem-delta); test has none."""
    import os

    from datasets import load_dataset
    tok = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError(
            "GAIA is gated: set HUGGING_FACE_HUB_TOKEN (and accept the dataset terms).")
    ds = load_dataset("gaia-benchmark/GAIA", f"2023_level{level}", split=split, token=tok)
    out: List[GaiaTask] = []
    for row in ds:
        out.append(_row_to_task(row))
        if n is not None and len(out) >= n:
            break
    return out


__all__ = ["GaiaTask", "load_gaia_fixture", "load_gaia_real", "_row_to_task"]
