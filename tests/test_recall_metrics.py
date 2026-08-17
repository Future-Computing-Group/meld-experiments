"""Fix #4 — recall@k / MRR of the gold supporting facts in the brain's graph-walk
read. This is the retrieval-quality observable for the merge evaluation (the
systems-relevant primary metric; QA token-F1 is secondary): does the recalled
top-k actually contain the gold supporting sentences the answer needs?
"""
from __future__ import annotations

from meld.agent.hotpot_loader import feed_context_through_brain, load_hotpot_fixture
from meld.agent.memory import SMPMemory
from meld.agent.run_hotpot_gate import _calibrated_cfg


def _mem():
    return SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg("hashing"))


def test_gold_supporting_sentences_extracted_from_task():
    from meld.agent.recall_metrics import gold_supporting_sentences
    task = load_hotpot_fixture(n=1)[0]
    gold = gold_supporting_sentences(task)
    assert len(gold) >= 1
    assert all(isinstance(s, str) and s.strip() for s in gold)


def test_recall_at_k_finds_gold_after_feeding_context():
    # feed the item's context through the brain; the gold supporting sentences
    # must be retrievable in the top-k (positive recall) and the first hit gives a
    # positive MRR.
    from meld.agent.recall_metrics import recall_at_k_and_mrr
    task = load_hotpot_fixture(n=1)[0]
    mem = _mem()
    feed_context_through_brain(task, mem)
    m = recall_at_k_and_mrr(mem.brain, task, encoder="hashing", k=10)
    assert m["n_gold"] >= 1
    assert m["recall_at_k"] > 0.0
    assert 0.0 < m["mrr"] <= 1.0


def test_recall_at_k_is_zero_for_an_empty_store():
    from meld.agent.recall_metrics import recall_at_k_and_mrr
    task = load_hotpot_fixture(n=1)[0]
    m = recall_at_k_and_mrr(_mem().brain, task, encoder="hashing", k=10)
    assert m["recall_at_k"] == 0.0
    assert m["mrr"] == 0.0
