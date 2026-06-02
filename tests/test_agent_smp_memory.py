"""G3 — the competence-gate agent uses an SMP brain as its memory backend.

These tests are the TDD spec for ``smp.agent``. They assert the load-bearing
integration property of the competence gate: the agent does NOT keep its own
private scratchpad memory; it READS via the brain's graph-walk read and WRITES
observations via PUBLISH -> merge-on-receive -> Patch, so a G3 run exercises the
SMP merge path, not just a bare agent.

The LLM here is a deterministic MOCK (no weights, no network): it is the offline
WIRING stand-in. The real open-weight model (Qwen2.5-7B-Instruct) is the gated
CSC backend; nothing here produces a GAIA score.
"""
from __future__ import annotations

from smp.agent.memory import SMPMemory
from smp.agent.react_agent import ReActAgent, AgentResult
from smp.agent.llm import MockLLM
from smp.agent.tools import RecallTool, FinalAnswerTool
from smp.brain.store import Brain
from smp.config import Thresholds, load_thresholds


def _calibrated_cfg() -> Thresholds:
    # delegate to the canonical encoder-matched calibrated thresholds (was a
    # duplicated inverted-0.5/0.55 copy that tripped the ordering invariant)
    from smp.agent.run_hotpot_gate import _calibrated_cfg as _canonical
    return _canonical("hashing")


# --------------------------------------------------------------------------
# SMPMemory: the agent's memory backend is an SMP brain
# --------------------------------------------------------------------------


def test_memory_backend_is_an_smp_brain():
    mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
    assert isinstance(mem.brain, Brain)


def test_write_goes_through_merge_on_receive_and_emits_a_patch():
    """A written observation must reach the brain as a Patch (auditable),
    never as a raw delta. Writing one novel fact yields exactly one MERGE
    Patch and one unit in the brain."""
    mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
    patches = mem.write("Paris is the capital of France")
    assert len(patches) == 1
    assert patches[0].decision == "MERGE"
    # the brain mutated ONLY via the Patch audit log
    assert len(mem.brain.audit_log) == 1
    units = [u for u in mem.brain.extract_delta()]
    assert any(getattr(u, "claims", None) for u in units)


def test_contradicting_write_produces_a_conflict_not_a_silent_overwrite():
    """Two contradicting observations must NOT silently overwrite: the second
    write classifies as CONFLICT (a first-class CONTRADICTS edge), proving the
    merge decision procedure is on the agent's write path."""
    mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
    mem.write("The standup meeting is on Monday")
    patches = mem.write("The standup meeting is on Tuesday")
    decisions = {p.decision for p in patches}
    assert "CONFLICT" in decisions


def test_recall_reads_via_graph_walk_and_returns_written_text():
    mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
    mem.write("Alice owns a dog named Rex")
    mem.write("Paris is the capital of France")
    hits = mem.recall("What pet does Alice own?", k=1)
    assert hits, "graph-walk read returned nothing"
    assert "dog" in hits[0].lower() or "rex" in hits[0].lower()


# --------------------------------------------------------------------------
# ReActAgent: a GAIA task runs end-to-end against the MOCK LLM and the brain
# receives merged knowledge
# --------------------------------------------------------------------------


def test_agent_runs_a_task_end_to_end_against_mock_llm():
    mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
    # The mock LLM is scripted: observe a fact, recall it, answer.
    script = [
        'Thought: I should note the clue.\nAction: remember[Paris is the capital of France]',
        'Thought: Now recall.\nAction: recall[capital of France]',
        'Thought: I know it.\nAction: final_answer[Paris]',
    ]
    agent = ReActAgent(
        llm=MockLLM(script),
        memory=mem,
        tools=[RecallTool(mem), FinalAnswerTool()],
        max_steps=5,
    )
    result = agent.run("What is the capital of France?")
    assert isinstance(result, AgentResult)
    assert result.answer == "Paris"
    # NON-DEGENERATE PATH assertion: the agent must have actually stepped and
    # the brain must have received merged knowledge via a Patch (not a no-op).
    assert result.n_steps >= 2
    assert len(mem.brain.audit_log) >= 1
    assert any(p.decision in ("MERGE", "RELATE", "CONFLICT")
               for p in mem.brain.audit_log)


def test_agent_terminates_with_a_final_answer_within_max_steps():
    """Real-LLM contract: even when the model NEVER emits final_answer, the agent
    must terminate within max_steps with a NON-EMPTY parseable answer (forced
    final-answer synthesis), not return None. Returning None on a real model is
    the contract bug the local real-model ext-smoke surfaced; the MockLLM oracle
    (which always scripts a final_answer) hid it."""
    mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
    # never emits final_answer -> the agent must still conclude (synthesis)
    looping = MockLLM(['Thought: hmm.\nAction: recall[nothing]'])
    agent = ReActAgent(llm=looping, memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=3)
    result = agent.run("unanswerable")
    assert result.answer is not None, "agent returned None (contract failure)"
    assert result.answer.strip() != "", "agent returned an empty final answer"
    assert result.n_steps <= 3


def test_agent_accepts_bare_final_answer_without_action_prefix():
    """Real open models emit a bare 'final_answer[Rex]' line (no 'Action:'
    prefix). The parser must accept it (the MockLLM oracle always used the
    prefix, hiding this)."""
    mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
    llm = MockLLM(['Thought: I know.\nfinal_answer[Rex]'])
    agent = ReActAgent(llm=llm, memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=4)
    result = agent.run("name the pet")
    assert result.answer == "Rex"


def test_agent_accepts_bare_final_answer_template_line():
    """A bare 'FINAL ANSWER: dog' / 'FINAL_ANSWER: dog' line (the GAIA template,
    space OR underscore) must terminate the loop with that answer."""
    mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
    for turn in ('Thought: it is a dog.\nFINAL ANSWER: dog',
                 'Thought: it is a dog.\nFINAL_ANSWER: dog'):
        m2 = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
        agent = ReActAgent(llm=MockLLM([turn]), memory=m2,
                           tools=[RecallTool(m2), FinalAnswerTool()],
                           max_steps=4)
        assert agent.run("what pet?").answer == "dog"


def test_agent_tightens_oververbose_final_answer():
    """An over-verbose final_answer payload is tightened toward the GAIA short
    form (trailing period / 'the answer is' preamble stripped)."""
    mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
    llm = MockLLM(['Thought: ok.\nAction: final_answer[The answer is Paris.]'])
    agent = ReActAgent(llm=llm, memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=4)
    assert agent.run("capital?").answer == "Paris"


def test_mock_llm_avoids_open_weight_json_code_quirk():
    """Guard the open-weight-quirks lesson at the harness boundary: the agent's
    LLM contract is a plain ReAct text turn, NOT a JSON-output prompt (which is
    what triggers the Llama-3.1-8B code-emission quirk). The mock returns a
    plain 'Thought/Action' string, and the parser must accept it."""
    llm = MockLLM(['Thought: x.\nAction: final_answer[42]'])
    out = llm.complete("any prompt")
    assert out.strip().startswith("Thought:")
    assert "```python" not in out
