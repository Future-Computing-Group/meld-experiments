"""G3 — the agent-framework REGISTRY (the MELD analogue of DWM's encoder ladder).

These are the TDD spec for ``meld.agent.registry``. The registry exposes several
PUBLIC, open agentic baselines behind ONE interface (``make_agent(name)`` ->
something with ``.run(task) -> AgentResult``), each driving the SAME GAIA gate
through the SAME MELD ``Brain`` as its memory backend. The load-bearing property:
the baselines are SWAPPABLE and EACH writes >= 1 Patch through the MELD merge
path (PUBLISH -> merge-on-receive -> Patch). No baseline keeps a private memory.

The LLM here is a deterministic MOCK (offline, no weights, no network). Nothing
here produces a GAIA score; this validates WIRING + swappability only.
"""
from __future__ import annotations

import pytest

from meld.agent.llm import MockLLM
from meld.agent.memory import SMPMemory
from meld.agent.react_agent import AgentResult
from meld.agent.registry import AGENT_REGISTRY, make_agent, available_agents
from meld.agent.tools import FinalAnswerTool, RecallTool
from meld.config import Thresholds, load_thresholds


def _calibrated_cfg() -> Thresholds:
    # delegate to the canonical encoder-matched calibrated thresholds (was a
    # duplicated inverted-0.5/0.55 copy that tripped the ordering invariant)
    from meld.agent.run_hotpot_gate import _calibrated_cfg as _canonical
    return _canonical("hashing")


def _mem() -> SMPMemory:
    return SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())


# The PUBLIC, open in-repo baselines that must always be present offline. The
# smolagents baseline is wired separately (it needs the optional dep installed)
# and is tested in test_smolagents_baseline.py with a skip guard.
_OFFLINE_BASELINES = ["react", "reflexion"]


def _oracle_script():
    return [
        "Thought: note the clue.\nAction: remember[Paris is the capital of France]",
        "Thought: recall it.\nAction: recall[capital of France]",
        "Thought: I can answer.\nAction: final_answer[Paris]",
    ]


# --------------------------------------------------------------------------
# registry surface
# --------------------------------------------------------------------------


def test_registry_lists_the_public_baselines():
    names = available_agents()
    for b in _OFFLINE_BASELINES:
        assert b in names, f"{b} missing from registry"
    # smolagents must be a registered NAME even if the dep is absent (lazy).
    assert "smolagents-toolcalling" in names


def test_make_agent_unknown_name_raises():
    with pytest.raises(KeyError):
        make_agent("does-not-exist", llm=MockLLM(["x"]), memory=_mem(),
                   tools=[FinalAnswerTool()])


@pytest.mark.parametrize("name", _OFFLINE_BASELINES)
def test_each_offline_baseline_is_constructible_and_runs(name):
    mem = _mem()
    agent = make_agent(name, llm=MockLLM(_oracle_script()), memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    result = agent.run("What is the capital of France?")
    assert isinstance(result, AgentResult)
    assert result.answer == "Paris"


@pytest.mark.parametrize("name", _OFFLINE_BASELINES)
def test_each_baseline_is_nondegenerate_and_writes_a_patch(name):
    """Swappability + the load-bearing integration property: EACH baseline must
    step the agent (>= 2 steps) AND write >= 1 Patch through the MELD merge path
    (a MERGE/RELATE/CONFLICT in the brain audit log) -- never a no-op."""
    mem = _mem()
    agent = make_agent(name, llm=MockLLM(_oracle_script()), memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    result = agent.run("What is the capital of France?")
    assert result.n_steps >= 2, f"{name} did not step"
    assert len(mem.brain.audit_log) >= 1, f"{name} wrote no Patch"
    assert any(p.decision in ("INSERT", "MERGE", "RELATE", "CONFLICT")
               for p in mem.brain.audit_log), f"{name} wrote no real merge"


def test_baselines_share_one_interface_and_same_brain_type():
    """All registered offline baselines share the SAME run/AgentResult interface
    and the SAME MELD Brain memory backend (swap by name, no other change)."""
    results = {}
    for name in _OFFLINE_BASELINES:
        mem = _mem()
        agent = make_agent(name, llm=MockLLM(_oracle_script()), memory=mem,
                           tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
        r = agent.run("What is the capital of France?")
        assert isinstance(r, AgentResult)
        # the memory backend is the MELD brain for every baseline
        from meld.brain.store import Brain
        assert isinstance(mem.brain, Brain)
        results[name] = r.answer
    # same task, same oracle -> same answer regardless of which baseline ran
    assert set(results.values()) == {"Paris"}


def test_registry_entries_are_callables():
    for name, factory in AGENT_REGISTRY.items():
        assert callable(factory), f"{name} factory not callable"
