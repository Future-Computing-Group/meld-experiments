"""Agent-framework REGISTRY -- the SMP analogue of DWM's encoder ladder.

``make_agent(name, *, llm, memory, tools, max_steps)`` returns an agent that
exposes ONE interface (``run(task) -> AgentResult``) regardless of the underlying
framework, and drives the SAME GAIA gate through the SAME SMP ``Brain`` as its
memory backend. Swapping a baseline is a one-line config change (the name).

Registered PUBLIC, open baselines (HARD CONSTRAINT: public + open, NOT FUSILLI):

  "react"                in-repo minimal ReAct loop (Yao et al., arXiv 2210.03629).
  "reflexion"            in-repo plan-then-execute / Reflexion variant
                         (Shinn et al., arXiv 2303.11366; Wang et al., 2305.04091)
                         -- a genuinely distinct design, same SMP brain backend.
  "smolagents-toolcalling"
                         the REAL HF ``smolagents.ToolCallingAgent`` (Apache-2.0),
                         lazy-imported so the offline path never loads it and the
                         protobuf-6.x venv pin stays unperturbed. Verified:
                         installing smolagents pulls no protobuf/transformers/vllm.

All three run the SAME open models behind ``OpenWeightLLM`` (Qwen2.5-7B-Instruct
default; Mistral-7B-Instruct / Gemma-2-9B-it cross-family fallbacks) and the SAME
SMP-brain-backed tools, so the registry is the model x framework x tool matrix.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from smp.agent.memory import SMPMemory
from smp.agent.react_agent import AgentResult, ReActAgent
from smp.agent.reflexion_agent import ReflexionAgent


# --------------------------------------------------------------------------
# in-repo baselines
# --------------------------------------------------------------------------


def _make_react(*, llm, memory: SMPMemory, tools: List[object],
                max_steps: int = 8) -> ReActAgent:
    return ReActAgent(llm=llm, memory=memory, tools=tools, max_steps=max_steps)


def _make_reflexion(*, llm, memory: SMPMemory, tools: List[object],
                    max_steps: int = 8) -> ReflexionAgent:
    return ReflexionAgent(llm=llm, memory=memory, tools=tools,
                          max_steps=max_steps)


# --------------------------------------------------------------------------
# smolagents ToolCallingAgent adapter (lazy-import; optional dep)
# --------------------------------------------------------------------------


class SmolagentsAdapter:
    """Drives a REAL ``smolagents.ToolCallingAgent`` behind our interface.

    The adapter (a) wraps our ``complete(prompt)->str`` LLM as a smolagents
    ``Model`` whose ``generate`` parses a plain ReAct turn ("Action: tool[arg]")
    into a smolagents tool-call ChatMessage; (b) wraps our SMP-brain-backed tools
    (and the ``remember`` verb) as smolagents ``Tool`` objects so the framework's
    own loop dispatches them, writing through the SAME SMP brain; (c) runs
    ``ToolCallingAgent.run`` and returns our ``AgentResult``.

    smolagents is imported LAZILY here so the offline path never touches it.
    """

    def __init__(self, *, llm, memory: SMPMemory, tools: List[object],
                 max_steps: int = 8):
        try:
            from smolagents import ToolCallingAgent  # lazy, optional dep
        except Exception as e:  # pragma: no cover - exercised only without dep
            raise RuntimeError(
                "smolagents is not installed; the 'smolagents-toolcalling' "
                "baseline is an optional public-framework path. Install it "
                "(no protobuf/transformers/vllm pulled) or use 'react'.") from e

        self.memory = memory
        self.max_steps = max_steps
        self._last_answer: Optional[str] = None
        self._n_tool_calls = 0

        smol_tools = self._wrap_tools(tools)
        model = self._wrap_model(llm)
        self.inner = ToolCallingAgent(tools=smol_tools, model=model,
                                      max_steps=max_steps)

    # -- tool wrapping ----------------------------------------------------
    def _wrap_tools(self, tools: List[object]):
        from smolagents import Tool

        adapter = self
        wrapped = []

        # a SMP-brain "remember" tool (the ReAct loop handles this verb directly;
        # smolagents needs it as an explicit Tool so the framework can call it)
        class RememberTool(Tool):
            name = "remember"
            description = ("Store an observed fact into memory (it is merged via "
                           "the SMP merge decision procedure).")
            inputs = {"fact": {"type": "string", "description": "the fact"}}
            output_type = "string"

            def forward(self, fact: str) -> str:  # noqa: D401
                patches = adapter.memory.write(fact)
                decs = ",".join(p.decision for p in patches) or "none"
                return f"remembered ({decs})"

        wrapped.append(RememberTool())

        for t in tools:
            wrapped.append(adapter._wrap_one(Tool, t))
        return wrapped

    def _wrap_one(self, Tool, tool):
        # final_answer is special-cased by smolagents; capture it to set the
        # adapter's answer. Other tools delegate to the underlying __call__.
        is_final = getattr(tool, "name", "") == "final_answer"
        adapter = self

        class _Wrapped(Tool):
            name = getattr(tool, "name", "tool")
            description = getattr(tool, "description", "") or name
            inputs = {"arg": {"type": "string", "description": "the argument"}}
            output_type = "string"

            def forward(self, arg: str) -> str:  # noqa: D401
                if is_final:
                    adapter._set_answer(arg)
                return str(tool(arg))

        return _Wrapped()

    def _set_answer(self, ans: str) -> None:
        from smp.agent.gaia_scorer import tighten_gaia_answer
        self._last_answer = tighten_gaia_answer(ans)

    # -- model wrapping ---------------------------------------------------
    def _wrap_model(self, llm):
        from smolagents import Model
        from smolagents.models import (ChatMessage, ChatMessageToolCall,
                                        ChatMessageToolCallFunction, MessageRole)
        import json
        import re as _re

        action_re = _re.compile(r"Action:\s*([a-z_]+)\s*\[(.*?)\]\s*$",
                                _re.IGNORECASE | _re.DOTALL | _re.MULTILINE)
        # smolagents' OWN tool-calling system prompt makes the model reply with a
        # JSON tool call (optionally prefixed by "Action:"):
        #   {"name": "recall", "arguments": {"query": "..."}}
        # We must dispatch THAT to the named tool (NOT mistake it for a final
        # answer). A smaller model (e.g. Mistral-7B) may dump a WHOLE multi-step
        # trajectory (several Action:{...}/Observation: blocks) in one turn; the
        # tool-calling contract is ONE action per turn, so we extract the FIRST
        # balanced {...} object carrying a "name" key and dispatch only that --
        # the loop then proceeds step by step (a greedy {.*} would span to the
        # last brace, produce invalid JSON, and leak the whole blob).
        adapter = self

        def _first_tool_json(text, known=None):
            """Return the first brace-balanced JSON object that is a tool call,
            parsed; else None. String-aware (ignores braces inside quotes). If
            ``known`` (a set of tool names) is given, skip objects whose ``name``
            is NOT a known tool and keep scanning -- so a model that dumps a
            multi-call trajectory whose FIRST call names an UNWIRED tool (e.g.
            'web_search' when only recall+final_answer exist; caught on Mistral-7B
            over HotpotQA) still dispatches the first KNOWN call instead of
            falling through and leaking the whole blob as a final answer."""
            i, n = 0, len(text)
            first_named = None  # fallback: the first 'name'-bearing obj seen
            while i < n:
                if text[i] != "{":
                    i += 1
                    continue
                depth, j, in_str, esc = 0, i, False, False
                while j < n:
                    c = text[j]
                    if in_str:
                        if esc:
                            esc = False
                        elif c == "\\":
                            esc = True
                        elif c == '"':
                            in_str = False
                    elif c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            blob = text[i:j + 1]
                            try:
                                obj = json.loads(blob)
                            except Exception:  # noqa: BLE001
                                obj = None
                            if isinstance(obj, dict) and "name" in obj:
                                name = str(obj.get("name", "")).strip().lower()
                                if known is None or name in known:
                                    return obj
                                if first_named is None:
                                    first_named = obj
                            break  # this object isn't a tool call; scan onward
                    j += 1
                i += 1
            # no KNOWN-named call found; return the first name-bearing obj (if any)
            # so the caller can still decide (it will fall to final_answer if the
            # name is unknown), preserving prior single-call behavior.
            return first_named

        def _known_names(tools_to_call_from):
            names = {"remember", "final_answer"}
            for t in (tools_to_call_from or []):
                n = getattr(t, "name", None)
                if n:
                    names.add(n)
            return names

        class _AdapterModel(Model):
            def generate(self, messages, stop_sequences=None,
                         response_format=None, tools_to_call_from=None, **kw):
                # flatten the conversation to a plain prompt for our LLM
                prompt = _flatten(messages)
                turn = llm.complete(prompt)
                tool_name, arg = _parse_tool_call(turn, tools_to_call_from)
                adapter._n_tool_calls += 1
                # map the single-string arg onto the tool's declared input name
                arg_name = _arg_name(tools_to_call_from, tool_name)
                tc = ChatMessageToolCall(
                    id=f"call_{adapter._n_tool_calls}",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name=tool_name,
                        arguments=json.dumps({arg_name: arg})),
                )
                return ChatMessage(role=MessageRole.ASSISTANT, content=turn,
                                   tool_calls=[tc])

        def _parse_tool_call(turn, tools_to_call_from):
            """Parse the model's turn into (tool_name, single-string arg),
            understanding BOTH contract shapes the open model may emit:
              1. smolagents-native JSON: {"name": t, "arguments": {...}}
              2. plain-ReAct text:       Action: tool[arg]
            Only if NEITHER parses (and the text is not itself a recognized tool
            call) does it fall back to a final_answer with the raw text."""
            text = turn or ""
            known_tool_names = _known_names(tools_to_call_from)
            # (1) JSON tool call (the real smolagents protocol surface): dispatch
            # the FIRST balanced {...} carrying a KNOWN tool "name" (one action
            # per turn; skip a leading unwired-tool call rather than leak it).
            obj = _first_tool_json(text, known=known_tool_names)
            if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                name = obj["name"].strip().lower()
                if name in known_tool_names:
                    return name, _coerce_arg(obj.get("arguments"))
            # (2) plain-ReAct text form
            m = action_re.search(text)
            if m and m.group(1).lower() in known_tool_names:
                return m.group(1).lower(), m.group(2).strip()
            # (3) no recognized tool call -> final answer with the raw text
            return "final_answer", text.strip()

        def _coerce_arg(arguments):
            """Reduce a tool-call 'arguments' payload to the single string our
            tools accept. A dict -> its first/sole value; anything else -> str."""
            if isinstance(arguments, dict):
                if not arguments:
                    return ""
                # prefer a conventional key, else the first value
                for key in ("query", "answer", "arg", "fact", "expr"):
                    if key in arguments:
                        return str(arguments[key])
                return str(next(iter(arguments.values())))
            if arguments is None:
                return ""
            return str(arguments)

        def _arg_name(tools_to_call_from, tool_name):
            if tools_to_call_from:
                for t in tools_to_call_from:
                    if getattr(t, "name", None) == tool_name:
                        names = list(getattr(t, "inputs", {}) or {})
                        if names:
                            return names[0]
            # smolagents' built-in final_answer uses 'answer'
            return "answer" if tool_name == "final_answer" else "arg"

        def _flatten(messages) -> str:
            parts: List[str] = []
            for msg in messages or []:
                content = getattr(msg, "content", None)
                if content is None and isinstance(msg, dict):
                    content = msg.get("content")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content)
                if content:
                    parts.append(str(content))
            return "\n".join(parts)

        return _AdapterModel()

    # -- the common interface --------------------------------------------
    def run(self, task: str, clues: Optional[List[str]] = None) -> AgentResult:
        for clue in (clues or []):
            self.memory.write(clue)
        try:
            out = self.inner.run(task, max_steps=self.max_steps)
        except Exception:  # noqa: BLE001 - framework may raise at budget end
            out = None
        answer = self._last_answer
        if answer is None and out is not None:
            answer = str(out)
        # n_steps: prefer the framework's logged steps; fall back to tool calls
        n_steps = max(self._n_tool_calls, 1)
        try:
            n_steps = max(n_steps, len(self.inner.memory.steps))
        except Exception:  # noqa: BLE001
            pass
        return AgentResult(answer=answer, n_steps=n_steps, transcript=[],
                           final_raw=str(answer or ""))


def _make_smolagents(*, llm, memory: SMPMemory, tools: List[object],
                     max_steps: int = 8) -> SmolagentsAdapter:
    return SmolagentsAdapter(llm=llm, memory=memory, tools=tools,
                             max_steps=max_steps)


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

AGENT_REGISTRY: Dict[str, Callable[..., object]] = {
    "react": _make_react,
    "reflexion": _make_reflexion,
    "smolagents-toolcalling": _make_smolagents,
}

# license + lineage provenance for the wired baselines (for the plan / paper).
AGENT_PROVENANCE = {
    "react": {"kind": "in-repo", "license": "this repo (Apache-2.0-compatible)",
              "cite": "Yao et al., arXiv 2210.03629"},
    "reflexion": {"kind": "in-repo",
                  "license": "this repo (Apache-2.0-compatible)",
                  "cite": "Shinn et al., arXiv 2303.11366; Wang et al., 2305.04091"},
    "smolagents-toolcalling": {"kind": "off-the-shelf (HF smolagents)",
                               "license": "Apache-2.0",
                               "cite": "huggingface/smolagents"},
}


def available_agents() -> List[str]:
    return list(AGENT_REGISTRY)


def make_agent(name: str, *, llm, memory: SMPMemory, tools: List[object],
               max_steps: int = 8):
    if name not in AGENT_REGISTRY:
        raise KeyError(
            f"unknown agent {name!r}; available: {available_agents()}")
    return AGENT_REGISTRY[name](llm=llm, memory=memory, tools=tools,
                                max_steps=max_steps)


__all__ = ["AGENT_REGISTRY", "AGENT_PROVENANCE", "make_agent",
           "available_agents", "SmolagentsAdapter"]
