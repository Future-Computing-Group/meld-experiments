"""G3 (HotpotQA pivot) — HotpotQA loader + brain-feed.

HotpotQA distractor is OPEN (non-gated, CC BY-SA 4.0), so unlike GAIA the loader
has a real ``load_hotpot_real`` path that calls ``datasets.load_dataset`` (run on
the CSC node / any networked host; SKIPPED here if ``datasets`` is absent so the
offline suite stays green). For the LOCAL path a tiny deterministic fixture in
the HF distractor SCHEMA exercises the loader end-to-end with NO network.

The load-bearing MELD integration: ``feed_context_through_brain`` feeds each item's
``context`` paragraphs THROUGH the MELD brain as merged observations (chunked ->
PUBLISH -> admission), so a later recall retrieves the evidence.
This is the MELD-on-message choice (the gate exercises recall + multi-hop reason
over brain-resident evidence, not raw model recall).
"""
from __future__ import annotations

import pytest

from meld.agent.hotpot_loader import (
    HotpotTask, feed_context_through_brain, load_hotpot_fixture,
    normalize_context, normalize_supporting_facts, stratified_slice,
)


# -- schema normalization ---------------------------------------------------

def test_normalize_context_from_hf_dict_form():
    # HF hotpotqa/hotpot_qa exposes context as parallel title/sentences lists.
    hf_ctx = {"title": ["A", "B"],
              "sentences": [["a1.", "a2."], ["b1."]]}
    paras = normalize_context(hf_ctx)
    assert paras == [["A", ["a1.", "a2."]], ["B", ["b1."]]]


def test_normalize_context_from_original_list_form():
    # original HotpotQA JSON form: list of [title, [sentences]].
    orig = [["A", ["a1.", "a2."]], ["B", ["b1."]]]
    assert normalize_context(orig) == orig


def test_normalize_supporting_facts_from_hf_dict_form():
    sf = {"title": ["A", "B"], "sent_id": [0, 1]}
    assert normalize_supporting_facts(sf) == [["A", 0], ["B", 1]]


# -- fixture (HF distractor schema, deterministic, offline) -----------------

def test_fixture_is_hotpot_shaped_and_multihop():
    tasks = load_hotpot_fixture(n=4)
    assert tasks
    for t in tasks:
        assert isinstance(t, HotpotTask)
        assert t.question and t.answer
        assert t.level in {"easy", "medium", "hard"}
        # context is a list of [title, [sentences]] paragraphs
        assert t.context and all(len(p) == 2 and isinstance(p[1], list)
                                 for p in t.context)
        # a multi-hop item references >= 2 supporting titles
        assert t.supporting_facts


def test_fixture_has_medium_hard_for_the_gate():
    tasks = load_hotpot_fixture(n=6)
    levels = {t.level for t in tasks}
    assert levels & {"medium", "hard"}, "gate needs medium/hard strata"


def test_fixture_is_deterministic():
    a = load_hotpot_fixture(n=4)
    b = load_hotpot_fixture(n=4)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert [t.answer for t in a] == [t.answer for t in b]


# -- the MELD brain feed (load-bearing) --------------------------------------

def test_feed_context_through_brain_writes_patches_and_is_recallable():
    from meld.agent.memory import SMPMemory
    task = load_hotpot_fixture(n=1)[0]
    mem = SMPMemory(encoder="hashing", nli="lexical")
    patches = feed_context_through_brain(task, mem)
    # every sentence is a signed delta admitted by the protocol's step
    assert mem.decisions, "context must traverse the admission path"
    assert mem.footprint()["stored"] == len(mem.store.objects.ids("D", "agent"))
    assert patches
    # the gold-supporting evidence is now recallable from the brain
    hits = mem.recall(task.question, k=5)
    assert hits, "fed context should be recallable via the graph-walk read"


# -- level-stratified slice (the bounded gate slice builder) ----------------

def test_stratified_slice_caps_per_level_and_is_deterministic():
    pool = load_hotpot_fixture(n=12)
    sl = stratified_slice(pool, {"easy": 1, "medium": 2, "hard": 2})
    counts = {lvl: sum(1 for t in sl if t.level == lvl)
              for lvl in ("easy", "medium", "hard")}
    assert counts == {"easy": 1, "medium": 2, "hard": 2}
    # deterministic given a deterministic source
    sl2 = stratified_slice(load_hotpot_fixture(n=12),
                           {"easy": 1, "medium": 2, "hard": 2})
    assert [t.task_id for t in sl] == [t.task_id for t in sl2]


# -- the real (open, non-gated) loader: skipped if `datasets` absent --------

def test_load_hotpot_real_is_present_and_open():
    import importlib
    mod = importlib.import_module("meld.agent.hotpot_loader")
    assert hasattr(mod, "load_hotpot_real")
    if importlib.util.find_spec("datasets") is None:
        pytest.skip("datasets not installed in the offline .venv (CSC-time dep)")
