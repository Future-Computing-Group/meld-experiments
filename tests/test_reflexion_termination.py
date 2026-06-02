"""G3 real-LLM-contract regression: the reflexion baseline must ALWAYS terminate
with a parseable final answer within ``max_steps`` -- even when the model never
emits ``final_answer`` on its own.

Root cause (caught on the local real-model ext-smoke, qwen2.5:7b): on one task
the plan-execute-reflect loop ran 9 turns emitting only non-final actions
(recall / no-parseable-action) and returned ``answer=None`` (pred=''), a
contract failure. The MockLLM oracle hid it because the oracle always scripts a
final_answer turn. These tests drive the agent with a model that NEVER emits
final_answer and assert termination with a synthesized final answer.
"""
from __future__ import annotations

from smp.agent.memory import SMPMemory
from smp.agent.registry import make_agent
from smp.agent.tools import FinalAnswerTool, RecallTool
from smp.config import Thresholds, load_thresholds


def _calibrated_cfg() -> Thresholds:
    # delegate to the canonical encoder-matched calibrated thresholds (was a
    # duplicated inverted-0.5/0.55 copy that tripped the ordering invariant)
    from smp.agent.run_hotpot_gate import _calibrated_cfg as _canonical
    return _canonical("hashing")


def _mem() -> SMPMemory:
    return SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())


class _NeverFinalLLM:
    """A model that NEVER emits final_answer: it always recalls (mirrors the
    qwen failure where the loop spun on recall and never concluded). Used to
    prove the agent forces a final-answer synthesis instead of returning None."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        # When the agent's forced-synthesis turn asks for the final answer
        # directly, comply -- so we can also assert the synthesis path is used.
        if "FINAL ANSWER" in prompt and "synthes" in prompt.lower():
            return "Thought: concluding.\nFINAL ANSWER: dog"
        return "Thought: I should look it up.\nAction: recall[Alice's pet]"


class _NoParseLLM:
    """A model that emits NO parseable action and NO final answer on every turn
    (the worst case). The agent must still terminate with a (best-effort) final
    answer within max_steps rather than looping or returning None."""

    def complete(self, prompt: str) -> str:
        return "Thought: hmm, I am unsure how to proceed."


def test_reflexion_terminates_with_final_answer_when_model_never_finalizes():
    mem = _mem()
    mem.write("Alice owns a dog named Rex")
    llm = _NeverFinalLLM()
    agent = make_agent("reflexion", llm=llm, memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    res = agent.run("What pet does Alice own?")
    assert res.answer is not None, "reflexion returned None (contract failure)"
    assert res.answer.strip() != "", "reflexion returned an empty final answer"
    assert res.n_steps <= 6, f"ran {res.n_steps} > max_steps"


def test_reflexion_terminates_even_with_no_parseable_action_at_all():
    mem = _mem()
    mem.write("Alice owns a dog named Rex")
    llm = _NoParseLLM()
    agent = make_agent("reflexion", llm=llm, memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    res = agent.run("What pet does Alice own?")
    assert res.answer is not None, "reflexion returned None on all-unparseable"
    assert res.answer.strip() != "", "reflexion returned an empty final answer"
    assert res.n_steps <= 6, f"ran {res.n_steps} > max_steps"


class _FinalizesWithTemplateEchoLLM:
    """A model that 'finalizes' but echoes the prompt placeholder as its answer
    (caught on the HotpotQA ext-smoke, reflexion/mistral): the tightened payload
    is empty. The agent must NOT emit an empty answer -- it must fall back to the
    best recalled fact (never an empty contract)."""

    def complete(self, prompt: str) -> str:
        return "Thought: done.\nAction: final_answer[[YOUR FINAL ANSWER]]"


def test_reflexion_falls_back_when_final_payload_is_a_template_echo():
    mem = _mem()
    mem.write("Alice owns a dog named Rex")
    agent = make_agent("reflexion", llm=_FinalizesWithTemplateEchoLLM(),
                       memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    res = agent.run("What pet does Alice own?")
    assert res.answer is not None and res.answer.strip() != "", (
        f"empty answer on a template-echo finalize: {res.answer!r}")
    assert "your final answer" not in res.answer.lower()


def test_react_falls_back_when_final_payload_is_a_template_echo():
    mem = _mem()
    mem.write("Alice owns a dog named Rex")
    agent = make_agent("react", llm=_FinalizesWithTemplateEchoLLM(),
                       memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    res = agent.run("What pet does Alice own?")
    assert res.answer is not None and res.answer.strip() != "", (
        f"empty answer on a template-echo finalize: {res.answer!r}")
    assert "your final answer" not in res.answer.lower()


def test_reflexion_never_exceeds_max_steps_budget():
    """The total turn budget (plan + execute + reflect + execute + synthesis)
    must stay within max_steps so the gated run's cost is bounded."""
    mem = _mem()
    llm = _NeverFinalLLM()
    agent = make_agent("reflexion", llm=llm, memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=4)
    res = agent.run("What pet does Alice own?")
    assert res.n_steps <= 4
    assert res.answer is not None and res.answer.strip() != ""
