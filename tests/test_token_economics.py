"""A-3 single-agent token-cost economics (motivating data point, not a contribution).

Repeated queries against a CLEANED, linked brain (distributed_merge: SMP dedups paraphrase
duplicates into canonical claims) recall a smaller context than against an UNCLEANED brain
(naive_union: every duplicate stays a live unit). Smaller recalled context = fewer input
tokens fed to the agent's LLM, summed over repeated queries. Token cost is approximated by a
word count (no tiktoken locally; the proxy factor cancels in the cleaned/uncleaned ratio).
Pure measurement over the existing mem_conditions builders + recall -- no LLM, no testbed.
"""
from __future__ import annotations


def _cfg():
    from smp.agent.run_hotpot_gate import _calibrated_cfg
    return _calibrated_cfg("hashing")


def _task():
    from smp.agent.hotpot_loader import load_hotpot_fixture
    return load_hotpot_fixture(1)[0]


def test_cleaned_brain_recall_costs_fewer_tokens_than_uncleaned():
    from smp.agent.token_economics import token_economics
    task = _task()
    queries = [task.question] * 3            # repeated queries (the economics axis)
    r = token_economics(task, queries, encoder="hashing", nli="lexical", cfg=_cfg(), k=20)
    assert r["uncleaned_tokens"] > 0
    assert r["cleaned_tokens"] < r["uncleaned_tokens"]   # dedup -> smaller recalled context
    assert r["reduction_rate"] > 0.0


def test_reduction_scales_with_repeated_queries():
    from smp.agent.token_economics import token_economics
    task = _task()
    one = token_economics(task, [task.question], encoder="hashing", nli="lexical", cfg=_cfg(), k=20)
    five = token_economics(task, [task.question] * 5, encoder="hashing", nli="lexical", cfg=_cfg(), k=20)
    # absolute token savings accumulate over repeated queries; the per-query rate is stable
    assert (five["uncleaned_tokens"] - five["cleaned_tokens"]) > \
           (one["uncleaned_tokens"] - one["cleaned_tokens"])
    assert abs(five["reduction_rate"] - one["reduction_rate"]) < 1e-9
