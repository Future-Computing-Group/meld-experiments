"""G3 — the HF smolagents ``ToolCallingAgent`` baseline (Apache-2.0), wired as a
REAL second public framework behind the SAME interface and the SAME SMP brain.

smolagents is an OPTIONAL dependency: the offline path never imports it, and the
adapter lazy-imports it so the protobuf-6.x venv pin stays unperturbed (verified:
installing smolagents pulls no protobuf/transformers/vllm). These tests SKIP if
smolagents is not installed, so the offline suite still passes on a minimal venv.

We drive the real ``ToolCallingAgent`` with an OFFLINE adapter ``Model`` backed by
our deterministic ``MockLLM`` (no network, no weights). The point is to prove the
off-the-shelf public framework is swappable through the registry and that ITS
tool calls write through the SAME SMP brain via the SAME tool objects.
"""
from __future__ import annotations

import pytest

smolagents = pytest.importorskip("smolagents",
                                 reason="smolagents optional; offline suite skips")

from smp.agent.llm import MockLLM
from smp.agent.memory import SMPMemory
from smp.agent.react_agent import AgentResult
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


def _scripted_toolcalls():
    """Scripted smolagents tool-call turns: remember a clue, then final_answer.
    The adapter Model turns these MockLLM turns into smolagents ChatMessage
    tool-calls, so the REAL ToolCallingAgent loop dispatches our SMP-brain tools.
    """
    return [
        "Thought: store the clue.\nAction: remember[Paris is the capital of France]",
        "Thought: answer.\nAction: final_answer[Paris]",
    ]


def test_smolagents_baseline_constructs_and_runs():
    mem = _mem()
    agent = make_agent("smolagents-toolcalling",
                       llm=MockLLM(_scripted_toolcalls()), memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    result = agent.run("What is the capital of France?")
    assert isinstance(result, AgentResult)
    assert result.answer is not None
    assert "paris" in result.answer.lower()


def test_smolagents_baseline_writes_through_the_smp_brain():
    """The real smolagents loop must dispatch the SMP-brain-backed tools, so a
    Patch must reach the brain audit log (the load-bearing integration property
    holds for the off-the-shelf framework too)."""
    mem = _mem()
    agent = make_agent("smolagents-toolcalling",
                       llm=MockLLM(_scripted_toolcalls()), memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    agent.run("What is the capital of France?")
    assert len(mem.brain.audit_log) >= 1, "smolagents wrote no Patch"
    assert any(p.decision in ("MERGE", "RELATE", "CONFLICT")
               for p in mem.brain.audit_log)


def test_smolagents_uses_the_real_framework_class():
    """Guard that this is the REAL HF framework, not a re-implementation: the
    adapter must hold a genuine smolagents.ToolCallingAgent instance."""
    from smolagents import ToolCallingAgent
    mem = _mem()
    agent = make_agent("smolagents-toolcalling",
                       llm=MockLLM(_scripted_toolcalls()), memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    assert isinstance(agent.inner, ToolCallingAgent)
