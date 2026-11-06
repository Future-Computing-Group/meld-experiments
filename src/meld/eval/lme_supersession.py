"""LongMemEval as a trace of independently evolving agent memories, LLM-free.

One world per LongMemEval item (worlds are never pooled: two simulated users' facts would
create false conflicts). ``session_split`` orders the item's haystack sessions by date and
gives each session an owner agent (round-robin over date order, or seeded random); with
duplication degree d the session is also held by the next d - 1 agents, copy j as paraphrase
variant j - 1 of every turn (``light_paraphrase``), so a copy is a tested pair at the sink, not
an exact copy that the exact-key path absorbs. Turns of the chosen roles are claims. Natural
supersession: a knowledge-update item states the old and the new value of one fact in two
sessions, which usually land on different agents.

``ku_truth`` is the ground truth of a knowledge-update item: (old turn, new turn) ordered by
session date, for items with exactly two answer turns in two sessions, else None.
``stale_fact_rate`` is 1.0 when the top-k read for the question surfaces the old turn (or a
copy of it) and not the new one, 0.0 otherwise, None without ground truth; the rate over
items is the mean. ``lme_recall`` is recall@k of the item's answer turns.
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Sequence, Tuple

from meld.agent.mem_conditions import SplitPlan, light_paraphrase
from meld.agent.recall_metrics import READ_TAU, recalled_sources


def _answer_turns(item: dict, roles: Sequence[str] = ("user",)):
    """(date, session index, turn index, text) of every answer turn of the roles."""
    return [(item["haystack_dates"][s], s, i, t["content"])
            for s, sess in enumerate(item["haystack_sessions"])
            for i, t in enumerate(sess) if t.get("has_answer") and t["role"] in roles]


def session_split(item: dict, *, n_brains: int, dup_degree: int = 1, order: str = "date",
                  assign: str = "round_robin", seed: int = 0,
                  roles: Sequence[str] = ("user",)) -> SplitPlan:
    """Per-agent claim lists, each in world-time order; ``plan.when`` is the world time of every
    placed text (session rank * 10_000 + turn index, copies share it) and ``plan.source`` maps
    every placed text to its original turn text."""
    if not 1 <= dup_degree <= n_brains:
        raise ValueError("dup_degree must be in [1, n_brains]")
    sessions = list(range(len(item["haystack_sessions"])))
    if order == "date":
        sessions.sort(key=lambda s: item["haystack_dates"][s])
    elif order != "given":
        raise ValueError(f"unknown order {order!r}")
    rng = random.Random(seed)
    if assign not in ("round_robin", "random"):
        raise ValueError(f"unknown assign {assign!r}")
    plan = SplitPlan(per_brain=[[] for _ in range(n_brains)])
    gold = {(s, i) for _, s, i, _ in _answer_turns(item, roles)}
    plan.n_gold = len(gold)
    for rank, s in enumerate(sessions):
        owner = rank % n_brains if assign == "round_robin" else rng.randrange(n_brains)
        for i, turn in enumerate(item["haystack_sessions"][s]):
            text = (turn.get("content") or "").strip()
            if turn["role"] not in roles or not text:
                continue
            for j in range(dup_degree):
                placed = text if j == 0 else light_paraphrase(text, variant=j - 1)
                plan.per_brain[(owner + j) % n_brains].append(placed)
                plan.source.setdefault(placed, text)
                plan.when.setdefault(placed, rank * 10_000 + i)
            if (s, i) in gold and dup_degree > 1:
                plan.n_duplicated += 1
                plan.n_paraphrased += 1
    return plan


def ku_truth(item: dict, roles: Sequence[str] = ("user",)) -> Optional[Tuple[str, str]]:
    """(old text, new text) of a knowledge-update item, by session date; None otherwise."""
    if item.get("question_type") != "knowledge-update":
        return None
    ans = sorted(_answer_turns(item, roles))
    if len(ans) != 2 or ans[0][1] == ans[1][1]:
        return None
    return ans[0][3], ans[1][3]


def _top(brain, item: dict, k: int, source: Optional[Dict[str, str]], tau: Optional[float] = READ_TAU):
    return set(recalled_sources(brain, item["question"], k, source, tau))


def stale_fact_rate(brain, item: dict, k: int, source: Optional[Dict[str, str]] = None,
                    roles: Sequence[str] = ("user",), tau: Optional[float] = READ_TAU) -> Optional[float]:
    truth = ku_truth(item, roles)
    if truth is None:
        return None
    top = _top(brain, item, k, source, tau)
    return float(truth[0] in top and truth[1] not in top)


def lme_recall(brain, item: dict, k: int, source: Optional[Dict[str, str]] = None,
               roles: Sequence[str] = ("user",), tau: Optional[float] = READ_TAU) -> Optional[float]:
    """Fraction of the item's answer turns (of the roles) whose text or a copy is in the top k;
    None when the item has none."""
    gold = {t for *_, t in _answer_turns(item, roles)}
    if not gold:
        return None
    return len(gold & _top(brain, item, k, source, tau)) / len(gold)


__all__ = ["session_split", "ku_truth", "stale_fact_rate", "lme_recall"]
