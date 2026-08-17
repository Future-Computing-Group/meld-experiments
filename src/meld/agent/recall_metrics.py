"""recall@k and MRR of the gold supporting facts in a brain's graph-walk read.

This is the retrieval-quality observable for the merge evaluation: a memory-merge
protocol's job is store hygiene (dedup + consistency), whose direct effect is on
WHICH units reach the recalled top-k -- not on downstream answer token-F1, which a
fixed-k cap and the LLM's own competence both confound. recall@k (did the gold
supporting sentence make the top-k?) and MRR (how high?) measure the protocol's
effect at the layer it acts on, and are the systems-relevant primary metric the
manuscript Section 7.2 names alongside the answer score.
"""
from __future__ import annotations

from typing import Dict, List

from meld.agent.hotpot_loader import HotpotTask
from meld.brain.store import Brain


def gold_supporting_sentences(task: HotpotTask) -> List[str]:
    """The gold supporting sentences: resolve each (title, sent_id) supporting
    fact against the item's context paragraphs."""
    by_title = {title: list(sents) for title, sents in task.context}
    out: List[str] = []
    for entry in task.supporting_facts:
        title, sid = entry[0], entry[1]
        sents = by_title.get(title)
        if sents and isinstance(sid, int) and 0 <= sid < len(sents):
            s = (sents[sid] or "").strip()
            if s:
                out.append(s)
    return out


def _unit_text(brain: Brain, uid: bytes) -> str:
    u = brain.get_unit(uid)
    return " ".join(c.text for c in u.claims) if u else ""


def _matches(gold: str, text: str) -> bool:
    g = gold.strip().lower()
    return bool(g) and g in text.strip().lower()


def recall_at_k_and_mrr(brain: Brain, task: HotpotTask, *, encoder: str,
                        k: int = 5) -> Dict[str, float]:
    """recall@k = fraction of gold supporting sentences whose unit is in the
    status-live top-k read; MRR = reciprocal rank of the first gold-supporting
    unit (0 if none). Matching is gold-sentence-substring-of-unit-text (robust to a
    title prefix added when context is fed through the brain)."""
    gold = gold_supporting_sentences(task)
    if not gold:
        return {"recall_at_k": 0.0, "mrr": 0.0, "n_gold": 0}
    ranked = [_unit_text(brain, uid)
              for uid, _ in brain.graph_walk_read(task.question, encoder=encoder, k=k)]
    matched = sum(1 for g in gold if any(_matches(g, t) for t in ranked))
    rr = 0.0
    for i, t in enumerate(ranked, start=1):
        if any(_matches(g, t) for g in gold):
            rr = 1.0 / i
            break
    return {"recall_at_k": matched / len(gold), "mrr": rr, "n_gold": len(gold)}


__all__ = ["gold_supporting_sentences", "recall_at_k_and_mrr"]
