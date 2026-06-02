"""G3 — end-to-end: a tool-using agent answers a GAIA-shaped task where the
ANSWER is only in an attached file / web page, and that evidence reaches the SMP
brain as a merged observation (the SMP-on-message decision), then is RECALLED to
answer. This is the L2-achievability path: file/web tools + memory, not raw recall
of a pre-seeded clue.

WIRING smoke (oracle MockLLM), NOT a GAIA result.
"""
from __future__ import annotations

from smp.agent.llm import MockLLM
from smp.agent.memory import SMPMemory
from smp.agent.registry import make_agent
from smp.agent.tools import (CalculatorTool, FileReaderTool, FinalAnswerTool,
                             OfflineWebSearchTool, RecallTool)
from smp.config import Thresholds, load_thresholds


def _mem():
    # canonical encoder-matched calibrated thresholds (was an inline inverted
    # 0.5/0.55 pair that tripped the Thresholds ordering invariant)
    from smp.agent.run_hotpot_gate import _calibrated_cfg
    return SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg("hashing"))


def test_agent_answers_from_a_file_via_brain_observation(tmp_path):
    """The answer is ONLY in the attached file: the agent reads it (content is
    merged into the brain), recalls it, and answers. Proves the file->brain->
    recall->answer path the GAIA L2 tool surface needs."""
    doc = tmp_path / "report.txt"
    doc.write_text("Quarterly figures: the launch venue is Helsinki.")
    mem = _mem()
    tools = [RecallTool(mem), FileReaderTool(memory=mem), FinalAnswerTool()]
    script = [
        f"Thought: read the file.\nAction: read_file[{doc}]",
        "Thought: recall the venue.\nAction: recall[launch venue]",
        "Thought: answer.\nAction: final_answer[Helsinki]",
    ]
    agent = make_agent("react", llm=MockLLM(script), memory=mem, tools=tools,
                       max_steps=6)
    result = agent.run("What is the launch venue in the attached report?")
    assert result.answer == "Helsinki"
    # the file content traversed the SMP merge path (not just a raw tool string)
    assert any(p.decision in ("MERGE", "RELATE", "CONFLICT")
               for p in mem.brain.audit_log)


def test_agent_answers_from_web_cache_via_brain_observation():
    cache = {"capital of france": "Paris is the capital of France."}
    mem = _mem()
    tools = [RecallTool(mem), OfflineWebSearchTool(memory=mem, cache=cache),
             FinalAnswerTool()]
    script = [
        "Thought: search.\nAction: web_search[capital of France]",
        "Thought: recall.\nAction: recall[capital of France]",
        "Thought: answer.\nAction: final_answer[Paris]",
    ]
    agent = make_agent("react", llm=MockLLM(script), memory=mem, tools=tools,
                       max_steps=6)
    result = agent.run("What is the capital of France?")
    assert result.answer == "Paris"
    assert len(mem.brain.audit_log) >= 1


def test_agent_uses_calculator_for_a_numeric_l2_step():
    mem = _mem()
    tools = [RecallTool(mem), CalculatorTool(), FinalAnswerTool()]
    script = [
        "Thought: compute.\nAction: calculator[12 * 12]",
        "Thought: answer.\nAction: final_answer[144]",
    ]
    agent = make_agent("react", llm=MockLLM(script), memory=mem, tools=tools,
                       max_steps=5)
    result = agent.run("What is 12 squared?")
    assert result.answer == "144"
