"""A-3 token-cost economics: recall-context token cost, cleaned vs uncleaned brain.

A CLEANED, linked brain (distributed_merge -- MELD collapses paraphrase duplicates into
canonical claims via the merge decision procedure) surfaces a smaller recalled context per
query than an UNCLEANED brain (naive_union -- every duplicate stays a live unit). Over
repeated queries the input-token cost (the recalled context fed to the agent's LLM) is
correspondingly lower. Token cost is approximated by a word count (no tiktoken locally; the
proxy factor cancels in the cleaned/uncleaned ratio). Reuses the mem_conditions builders +
recall -- no LLM, no testbed.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from meld.agent.mem_conditions import build_condition
from meld.protocol.objects import EpochRecord
from meld.agent.hotpot_loader import HotpotTask


def _recall_tokens(brain, query: str, k: int) -> int:
    """Word-count proxy for the input tokens of one query's recalled context."""
    return sum(len(text.split()) for text in brain.recall(query, k=k))


def token_economics(task: HotpotTask, queries: List[str], *, encoder: str, nli: str,
                    epoch: Optional[EpochRecord], k: int = 20) -> Dict[str, float]:
    """Sum the recalled-context token cost over ``queries`` for the cleaned
    (distributed_merge) and uncleaned (naive_union) brain built from the same task."""
    cleaned = build_condition(task, "distributed_merge", encoder=encoder, nli=nli, epoch=epoch)
    uncleaned = build_condition(task, "naive_union", encoder=encoder, nli=nli, epoch=epoch)
    ct = sum(_recall_tokens(cleaned.brain, q, k) for q in queries)
    ut = sum(_recall_tokens(uncleaned.brain, q, k) for q in queries)
    return {"cleaned_tokens": ct, "uncleaned_tokens": ut,
            "reduction_rate": (ut - ct) / ut if ut else 0.0,
            "cleaned_live_units": cleaned.live_units,
            "uncleaned_live_units": uncleaned.live_units}


__all__ = ["token_economics"]
