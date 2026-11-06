"""G3 — the competence-gate agent uses an MELD brain as its memory backend.

These tests are the TDD spec for ``meld.agent``. They assert the load-bearing
integration property of the competence gate: the agent does NOT keep its own
private scratchpad memory; it READS via the brain's graph-walk read and WRITES
observations as signed deltas admitted by the protocol's step, so a G3 run exercises
the MELD merge path, not just a bare agent.

The LLM here is a deterministic MOCK (no weights, no network): it is the offline
WIRING stand-in. The real open-weight model (Qwen2.5-7B-Instruct) is the gated
CSC backend; nothing here produces a GAIA score.
"""
from __future__ import annotations

from meld.agent.memory import DEFAULT_POL, SMPMemory
from meld.agent.react_agent import ReActAgent, AgentResult
from meld.agent.llm import MockLLM
from meld.agent.tools import RecallTool, FinalAnswerTool
from meld.brain.store import Store
from meld.protocol.objects import FIXED_ONE, INF16, EpochRecord, m_pol_of


def agent_epoch(mem_or_encoder="hashing", nli="lexical", *, theta=0.5, pol=None, sel=None,
                scope="agent"):
    """A test epoch for an SMPMemory (scope "agent", authority "calibration"): every bin's
    theta is ``theta`` (fixed point from a float); the selector defaults to the policy's
    bootstrap selector. Built in the test; a deployment receives the authority's record."""
    from meld.proto import CANON_VERSION
    from meld.protocol.objects import m_nli
    pol = pol or DEFAULT_POL
    sel = sel or pol["bootstrap_selector"]
    th = round(theta * FIXED_ONE)
    return EpochRecord(scope=scope, seq=1, m_nli=m_nli(mem_or_encoder, nli, "int8", CANON_VERSION),
                       m_pol=m_pol_of(pol), policy="SEED", alpha=20_000, alpha_c=20_000,
                       mondrian_c=False, cal_root=b"\x00" * 16, cal_hw="cpu",
                       bins=(256, "doubling"), table=((0, 100, 0, th, INF16),),
                       sigma_lo_q=sel[0], sigma_c_q=sel[1], m=sel[2], author="calibration",
                       created_at=0)


def _mem(**kw):
    return SMPMemory(encoder="hashing", nli="lexical", **kw)


# --------------------------------------------------------------------------
# SMPMemory: the agent's memory backend is an MELD replica
# --------------------------------------------------------------------------


def test_memory_backend_is_an_meld_replica():
    mem = _mem()
    assert isinstance(mem.store, Store)


def test_write_is_a_signed_delta_admitted_and_never_an_authored_effect():
    """A written observation enters D as a MAC-verified delta and is admitted; writing one
    novel fact yields exactly one INSERT decision and one node; nothing enters A."""
    mem = _mem()
    decisions = mem.write("Paris is the capital of France")
    assert [d.decision for d in decisions] == ["INSERT"] == [d.decision for d in mem.decisions]
    assert mem.footprint() == {"stored": 1, "live": 1}
    assert len(mem.store.objects.ids("D", "agent")) == 1 and mem.store.objects.ids("A", "agent") == ()
    assert mem.write("Paris is the capital of France") == []        # an exact copy is one node


def test_contradicting_write_produces_a_conflict_not_a_silent_overwrite():
    """Two contradicting observations must NOT silently overwrite: the second
    write is a CONFLICT (a CONTRADICTS link, both claims live), with or without an epoch."""
    for epoch in (None, agent_epoch()):
        mem = _mem(epoch=epoch)
        mem.write("The standup meeting is on Monday")
        decisions = {p.decision for p in mem.write("The standup meeting is not on Monday")}
        assert "CONFLICT" in decisions and mem.footprint()["live"] == 2


def test_paraphrase_merges_only_under_an_epoch():
    a, b = "the valve on line two is open", "the valve on line two is open now"
    bare, cal = _mem(), _mem(epoch=agent_epoch())
    for mem in (bare, cal):
        mem.write(a)
        mem.write(b)
    assert bare.decisions[-1].decision == "RELATE" and bare.footprint()["live"] == 2
    assert cal.decisions[-1].decision == "MERGE" and cal.footprint() == {"stored": 2, "live": 1}


def test_recall_reads_via_graph_walk_and_returns_written_text():
    mem = _mem()
    mem.write("Alice owns a dog named Rex")
    mem.write("Paris is the capital of France")
    hits = mem.recall("What pet does Alice own?", k=1)
    assert hits, "graph-walk read returned nothing"
    assert "dog" in hits[0].lower() or "rex" in hits[0].lower()


def test_receive_admits_a_peer_delta_once_and_drops_a_forged_one():
    from meld.agent.memory import signed
    peer, mem = _mem(origin="peer"), _mem()
    peer.write("Paris is the capital of France")
    (did,) = peer.store.objects.ids("D", "agent")
    received, mac, _ = peer.store.objects.raw(did)
    assert mem.receive(received, "0" * 64) == [] and mem.footprint()["stored"] == 0
    assert [d.decision for d in mem.receive(received, mac)] == ["INSERT"]
    assert mem.receive(received, mac) == []                           # redelivery decides nothing


# --------------------------------------------------------------------------
# ReActAgent: a GAIA task runs end-to-end against the MOCK LLM and the brain
# receives merged knowledge
# --------------------------------------------------------------------------


def test_agent_runs_a_task_end_to_end_against_mock_llm():
    mem = _mem()
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
    assert len(mem.decisions) >= 1
    assert any(p.decision in ("INSERT", "MERGE", "RELATE", "CONFLICT")
               for p in mem.decisions)


def test_agent_terminates_with_a_final_answer_within_max_steps():
    """Real-LLM contract: even when the model NEVER emits final_answer, the agent
    must terminate within max_steps with a NON-EMPTY parseable answer (forced
    final-answer synthesis), not return None. Returning None on a real model is
    the contract bug the local real-model ext-smoke surfaced; the MockLLM oracle
    (which always scripts a final_answer) hid it."""
    mem = _mem()
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
    mem = _mem()
    llm = MockLLM(['Thought: I know.\nfinal_answer[Rex]'])
    agent = ReActAgent(llm=llm, memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=4)
    result = agent.run("name the pet")
    assert result.answer == "Rex"


def test_agent_accepts_bare_final_answer_template_line():
    """A bare 'FINAL ANSWER: dog' / 'FINAL_ANSWER: dog' line (the GAIA template,
    space OR underscore) must terminate the loop with that answer."""
    mem = _mem()
    for turn in ('Thought: it is a dog.\nFINAL ANSWER: dog',
                 'Thought: it is a dog.\nFINAL_ANSWER: dog'):
        m2 = _mem()
        agent = ReActAgent(llm=MockLLM([turn]), memory=m2,
                           tools=[RecallTool(m2), FinalAnswerTool()],
                           max_steps=4)
        assert agent.run("what pet?").answer == "dog"


def test_agent_tightens_oververbose_final_answer():
    """An over-verbose final_answer payload is tightened toward the GAIA short
    form (trailing period / 'the answer is' preamble stripped)."""
    mem = _mem()
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
