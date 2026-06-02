"""G3 real-LLM-contract regression: the smolagents adapter must dispatch the
model's tool calls to the CORRECT tool under smolagents' own tool-calling
protocol -- distinguishing a ``recall`` call from a ``final_answer`` call.

Root cause (caught on the local real-model ext-smoke, qwen2.5:7b): the real
``ToolCallingAgent`` feeds the model smolagents' OWN JSON tool-calling system
prompt, so qwen replies with a JSON tool call::

    Action:
    { "name": "recall", "arguments": {"arg": "Alice's pet"} }

The adapter's ``_AdapterModel`` only parsed the plain-ReAct ``Action: tool[arg]``
text form, so the JSON did not match and fell through to the "no parseable
action -> wrap raw text as final_answer" path, dumping the literal JSON blob as
the final answer (a tool-format confusion). These tests drive the adapter model
with the JSON form and assert recall vs final_answer dispatch distinctly.

smolagents is optional; the offline suite skips if it is not installed.
"""
from __future__ import annotations

import pytest

smolagents = pytest.importorskip("smolagents",
                                 reason="smolagents optional; offline suite skips")

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


class _JsonToolCallLLM:
    """Emits the smolagents-native JSON tool-call form qwen produces under the
    real ToolCallingAgent prompt: a recall first, then a final_answer."""

    def __init__(self):
        self._i = -1
        self.script = [
            'Action:\n{\n  "name": "recall",\n'
            '  "arguments": {"query": "Alice\'s pet"}\n}',
            'Action:\n{\n  "name": "final_answer",\n'
            '  "arguments": {"answer": "dog"}\n}',
        ]

    def complete(self, prompt: str) -> str:
        self._i += 1
        return self.script[min(self._i, len(self.script) - 1)]


def test_smolagents_dispatches_json_recall_then_final_answer():
    """A JSON ``recall`` call must hit the recall tool (NOT be mistaken for a
    final answer), and the subsequent JSON ``final_answer`` call must set the
    real final answer -- so the answer is 'dog', not the literal JSON blob."""
    mem = _mem()
    mem.write("Alice owns a dog named Rex")
    agent = make_agent("smolagents-toolcalling", llm=_JsonToolCallLLM(),
                       memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    res = agent.run("What pet does Alice own?")
    assert isinstance(res, AgentResult)
    assert res.answer is not None
    ans = res.answer.strip()
    # the final answer must NOT be a raw tool-call blob
    assert '"name"' not in ans and "recall" not in ans.lower(), (
        f"final answer is a mis-dispatched tool-call blob: {ans!r}")
    assert ans.lower() == "dog"


class _JsonRecallOnceLLM:
    """First turn: a JSON recall (must be dispatched to recall, observed); only
    then a final answer. Proves recall is NOT swallowed as the final answer."""

    def __init__(self):
        self._i = -1
        self.script = [
            'Action:\n{"name": "recall", "arguments": {"query": "pet"}}',
            'Action:\n{"name": "final_answer", "arguments": {"answer": "dog"}}',
        ]

    def complete(self, prompt: str) -> str:
        self._i += 1
        return self.script[min(self._i, len(self.script) - 1)]


class _MultiCallInOneTurnLLM:
    """A smaller model (e.g. Mistral-7B) may dump a WHOLE multi-step trajectory
    in ONE turn: several Action:{...}/Observation: blocks ending in final_answer.
    The adapter must dispatch the FIRST tool call (one action per turn), NOT
    greedily span to the last brace and leak the whole blob as a final answer."""

    def __init__(self):
        self._i = -1
        self.script = [
            # turn 1: a whole trajectory dumped at once (first call = recall)
            'Action:\n{\n  "name": "recall",\n'
            '  "arguments": {"query": "Alice\'s pet"}\n}\n'
            'Observation: "Alice owns a dog."\n\n'
            'Action:\n{\n  "name": "final_answer",\n  "arguments": "cat"\n}',
            # turn 2: a clean single final_answer
            'Action:\n{"name": "final_answer", "arguments": {"answer": "dog"}}',
        ]

    def complete(self, prompt: str) -> str:
        self._i += 1
        return self.script[min(self._i, len(self.script) - 1)]


def test_smolagents_dispatches_first_call_when_model_dumps_many_in_one_turn():
    """The FIRST tool call in a multi-call dump must be dispatched (here recall),
    NOT the whole blob leaked as a final answer (and NOT the WRONG later call --
    'cat' in turn 1 must not win over the real recall-then-final flow)."""
    mem = _mem()
    mem.write("Alice owns a dog named Rex")
    agent = make_agent("smolagents-toolcalling", llm=_MultiCallInOneTurnLLM(),
                       memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    res = agent.run("What pet does Alice own?")
    assert res.answer is not None
    ans = res.answer.strip()
    assert '"name"' not in ans and "observation" not in ans.lower(), (
        f"leaked a multi-call blob as the final answer: {ans!r}")
    # turn 1's first call (recall) is dispatched, not its trailing final_answer
    # 'cat'; turn 2's clean final_answer 'dog' wins.
    assert ans.lower() == "dog"
    assert res.n_steps >= 2


class _HallucinatedToolLLM:
    """A model (caught: Mistral-7B on HotpotQA) dumps a multi-call trajectory
    whose FIRST call names a tool that is NOT wired for this run (e.g.
    ``web_search`` when only recall+final_answer are provided). The adapter must
    SKIP the unknown-named call and dispatch the first call naming a KNOWN tool,
    NOT fall through and leak the whole blob as a final answer."""

    def __init__(self):
        self._i = -1
        self.script = [
            'Action:\n{\n  "name": "web_search",\n'
            '  "arguments": "Director of Inception"\n}\n'
            'Observation: Christopher Nolan\n\n'
            'Action:\n{\n  "name": "recall",\n'
            '  "arguments": {"query": "director nationality"}\n}',
            'Action:\n{"name": "final_answer", "arguments": {"answer": "British"}}',
        ]

    def complete(self, prompt: str) -> str:
        self._i += 1
        return self.script[min(self._i, len(self.script) - 1)]


def test_smolagents_skips_unknown_tool_and_dispatches_first_known_call():
    """The first KNOWN tool call (recall) is dispatched even though an unknown
    'web_search' call precedes it; the whole blob is NOT leaked as the answer."""
    mem = _mem()
    mem.write("Christopher Nolan is British")
    agent = make_agent("smolagents-toolcalling", llm=_HallucinatedToolLLM(),
                       memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    res = agent.run("What is the nationality of the director of Inception?")
    assert res.answer is not None
    ans = res.answer.strip()
    assert '"name"' not in ans and "web_search" not in ans.lower(), (
        f"leaked an unknown-tool blob as the final answer: {ans!r}")
    assert ans.lower() == "british"


def test_smolagents_recall_writes_no_final_and_advances_to_final():
    mem = _mem()
    mem.write("Alice owns a dog named Rex")
    agent = make_agent("smolagents-toolcalling", llm=_JsonRecallOnceLLM(),
                       memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()], max_steps=6)
    res = agent.run("What pet does Alice own?")
    assert res.answer is not None
    assert res.answer.strip().lower() == "dog"
    # at least 2 tool calls happened (recall + final_answer), i.e. recall was
    # not collapsed into a single mis-dispatched final_answer
    assert res.n_steps >= 2
