"""R2-LME: adapt the REAL LongMemEval corpus into HotpotTask retrieval tasks (TDD).

LongMemEval (arXiv 2410.10813) ships each question with a haystack of chat sessions
(distractors + the gold session(s)); the evidence turns are flagged ``has_answer``. To run
the *tested* recall machinery (recall_sweep) on this second substrate unchanged, we map one
LongMemEval item to a HotpotTask: each haystack session -> a context paragraph
[session_id, [turn-contents]], and each has_answer turn -> a gold supporting_fact
[session_id, turn_index]. This pins that mapping on an inline fixture (no data-file
dependency); the file loader + the recall run are the data-bearing step.

Run:  PYTHONPATH=src python3 -m pytest tests/test_longmemeval_real.py -q
"""
from meld.eval.longmemeval_real import item_to_task

_ITEM = {
    "question_id": "q1",
    "question_type": "single-session-user",
    "question": "What degree did I get?",
    "answer": "Business Administration",
    "haystack_session_ids": ["s1", "s2"],
    "haystack_sessions": [
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        [{"role": "user", "content": "I graduated in Business Administration",
          "has_answer": True},
         {"role": "assistant", "content": "great"}],
    ],
    "answer_session_ids": ["s2"],
}


def test_item_to_task_maps_sessions_to_context_and_has_answer_to_gold():
    t = item_to_task(_ITEM)
    assert t.task_id == "q1"
    assert t.question == "What degree did I get?"
    assert t.answer == "Business Administration"
    assert t.qtype == "single-session-user"
    # one context paragraph per haystack session, turn-contents as sentences
    assert [p[0] for p in t.context] == ["s1", "s2"]
    assert t.context[1][1] == ["I graduated in Business Administration", "great"]
    # the single has_answer turn becomes the gold supporting fact (session s2, turn 0)
    assert t.supporting_facts == [["s2", 0]]


def test_item_with_no_gold_turn_has_empty_supporting_facts():
    item = dict(_ITEM, haystack_sessions=[
        [{"role": "user", "content": "hi"}],
        [{"role": "user", "content": "no evidence here"}],
    ])
    t = item_to_task(item)
    assert t.supporting_facts == []
