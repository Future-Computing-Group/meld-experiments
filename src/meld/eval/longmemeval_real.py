"""Load the REAL LongMemEval corpus (arXiv 2410.10813) as HotpotTask retrieval tasks.

Each LongMemEval question ships a haystack of chat sessions (distractors + gold) with the
evidence turns flagged ``has_answer``. Mapping one item to a HotpotTask --- haystack session
-> context paragraph [session_id, [turn-contents]], each has_answer turn -> gold
supporting_fact [session_id, turn_index] --- lets the tested recall machinery (recall_sweep)
run unchanged on this second substrate: recall@k of the gold evidence turns in the
status-live store, upstream of any LLM, under the centralized / distributed-merge /
naive-union conditions. Distinct from the synthetic ``longmemeval_loader`` fixture.

The corpus file is gitignored (data-bearing; download with huggingface_hub from
``xiaowu0162/longmemeval``, file ``longmemeval_s`` or ``longmemeval_oracle``).
"""
from __future__ import annotations

import json
import random
from typing import List

from meld.agent.hotpot_loader import HotpotTask


def item_to_task(item: dict) -> HotpotTask:
    """Map one LongMemEval item to a HotpotTask (pure; no I/O)."""
    sids = item["haystack_session_ids"]
    sessions = item["haystack_sessions"]
    context: List = []
    supporting: List = []
    for sid, sess in zip(sids, sessions):
        context.append([sid, [t.get("content", "") for t in sess]])
        for idx, turn in enumerate(sess):
            if turn.get("has_answer"):
                supporting.append([sid, idx])
    return HotpotTask(
        task_id=item.get("question_id", ""),
        question=item["question"],
        answer=item.get("answer", ""),
        level="lme",
        qtype=item.get("question_type", ""),
        context=context,
        supporting_facts=supporting,
    )


def load_longmemeval_tasks(path: str, n: int = None, seed: int = 0,
                           require_gold: bool = True) -> List[HotpotTask]:
    """Load LongMemEval items from ``path`` and return HotpotTask retrieval tasks.

    ``require_gold`` skips items whose haystack carries no has_answer turn (no retrievable
    gold -> recall undefined). ``seed`` fixes the shuffle so a size-``n`` slice is reproducible.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    rng = random.Random(seed)
    rng.shuffle(data)
    tasks: List[HotpotTask] = []
    for item in data:
        task = item_to_task(item)
        if require_gold and not task.supporting_facts:
            continue
        tasks.append(task)
        if n is not None and len(tasks) >= n:
            break
    return tasks


__all__ = ["item_to_task", "load_longmemeval_tasks"]
