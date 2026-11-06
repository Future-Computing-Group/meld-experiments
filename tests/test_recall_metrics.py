"""Fix #4 — recall@k / MRR of the gold supporting facts in the brain's graph-walk
read. This is the retrieval-quality observable for the merge evaluation (the
systems-relevant primary metric; QA token-F1 is secondary): does the recalled
top-k actually contain the gold supporting sentences the answer needs?
"""
from __future__ import annotations

from meld.agent.hotpot_loader import feed_context_through_brain, load_hotpot_fixture
from meld.agent.memory import SMPMemory


def _mem():
    return SMPMemory(encoder="hashing", nli="lexical")


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
    m = recall_at_k_and_mrr(mem, task, k=10)
    assert m["n_gold"] >= 1
    assert m["recall_at_k"] > 0.0
    assert 0.0 < m["mrr"] <= 1.0


def test_recall_at_k_is_zero_for_an_empty_store():
    from meld.agent.recall_metrics import recall_at_k_and_mrr
    task = load_hotpot_fixture(n=1)[0]
    m = recall_at_k_and_mrr(_mem(), task, k=10)
    assert m["recall_at_k"] == 0.0
    assert m["mrr"] == 0.0


def test_a_copy_kept_as_the_merged_representative_scores_its_source():
    """A paraphrased copy that a merge keeps as the group's representative is the text the read
    returns; with the plan's source map it is scored as the sentence it was copied from."""
    from meld.agent.mem_conditions import light_paraphrase
    from meld.agent.recall_metrics import gold_supporting_sentences, recall_at_k_and_mrr
    from test_agent_meld_memory import agent_epoch
    task = load_hotpot_fixture(n=1)[0]
    gold = gold_supporting_sentences(task)[0]
    title = next(t for t, sents in task.context if gold in sents)
    orig, copy = f"{title}: {gold}", f"{title}: {light_paraphrase(gold)}"
    mem = SMPMemory(encoder="hashing", nli="lexical", epoch=agent_epoch("hashing", "lexical"))
    mem.write(copy)
    mem.write(orig)
    assert mem.footprint() == {"stored": 2, "live": 1} and mem.recall(task.question, k=5) == [copy]
    assert recall_at_k_and_mrr(mem, task, k=5)["recall_at_k"] == 0.0
    m = recall_at_k_and_mrr(mem, task, k=5, source={copy: orig})
    assert m["recall_at_k"] == 1 / len(gold_supporting_sentences(task)) and m["mrr"] == 1.0


def test_copies_of_one_source_count_once():
    from types import SimpleNamespace
    from meld.agent.recall_metrics import gold_supporting_sentences, recall_at_k_and_mrr
    task = load_hotpot_fixture(n=1)[0]
    a, b = gold_supporting_sentences(task)[:2]
    brain = SimpleNamespace(recall=lambda q, k: ["x", "a1", "a2", "y"][:k])   # no Store: plain read
    m = recall_at_k_and_mrr(brain, task, k=5, source={"a1": a, "a2": a}, tau=None)
    assert len(gold_supporting_sentences(task)) == 2 and a != b
    assert m["recall_at_k"] == 0.5 and m["mrr"] == 0.5
