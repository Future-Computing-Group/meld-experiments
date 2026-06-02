"""A minimal ReAct tool-agent (public pattern: Yao et al., arXiv 2210.03629).

The loop, per step:
  1. build a plain-text ReAct prompt (task + GAIA system contract + recent
     scratchpad + remembered facts);
  2. ask the LLM for ONE turn ("Thought: ... Action: tool[arg]");
  3. parse the action and dispatch:
       remember[fact]     -> SMPMemory.write (PUBLISH -> merge-on-receive ->
                             Patch); the brain receives MERGED knowledge.
       recall[query]      -> SMPMemory.recall (graph-walk read).
       final_answer[ans]  -> stop, return the answer.
  4. append the observation to the scratchpad.

The LLM contract is a PLAIN ReAct text turn (NOT a JSON-output prompt), honoring
the open-weight-quirks lesson. The agent's MEMORY is the SMP brain (no private
scratchpad memory store): that is what makes G3 exercise the SMP integration.

The interface (run(task) -> result; tools with name/description/__call__) mirrors
HF ``smolagents.ToolCallingAgent`` so a public off-the-shelf framework can be
swapped in for the gated CSC run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from smp.agent.memory import SMPMemory
from smp.agent.gaia_scorer import extract_final_answer, tighten_gaia_answer

# Verified at source (Mialon et al. 2311.12983, Fig. 2): GAIA requires the model
# to finish with this template.
GAIA_SYSTEM = (
    "You are a general AI assistant. Answer the question. Report your thoughts, "
    "and finish your answer with the template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
    "YOUR FINAL ANSWER must be a number, OR as few words as possible, OR a "
    "comma-separated list of numbers and/or strings. It is NOT a sentence: give "
    "ONLY the answer itself, no explanation, no 'The answer is', no leading "
    "article, no units unless asked. If the question asks for a species, name, "
    "place, or day, answer with just that one word or phrase (e.g. 'dog', not "
    "'Alice owns a dog named Rex').\n"
    "Use these actions, one per turn, formatted exactly as 'Action: tool[arg]':\n"
    "  remember[fact]     -- store a fact you observed into memory\n"
    "  recall[query]      -- retrieve facts from memory\n"
    "  final_answer[ans]  -- give the final answer and stop\n"
)

# The 'Action:' prefix is OPTIONAL: open models frequently emit a bare
# 'final_answer[Rex]' / 'recall[...]' line without the prefix. Matching both
# shapes is the real-LLM contract (the MockLLM oracle always uses the prefix).
_ACTION_RE = re.compile(r"(?:Action:\s*)?([a-z_]+)\s*\[(.*?)\]\s*$",
                        re.IGNORECASE | re.DOTALL | re.MULTILINE)
_KNOWN_VERBS = {"remember", "recall", "final_answer", "calculator",
                "read_file", "web_search"}


@dataclass
class AgentResult:
    answer: Optional[str]
    n_steps: int
    transcript: List[str] = field(default_factory=list)
    final_raw: str = ""


def nonempty_answer(answer: Optional[str], task: str,
                    memory: SMPMemory) -> str:
    """Guarantee a NON-EMPTY final answer (the agent contract). If ``answer``
    tightens to empty (e.g. the model finalized with a template-placeholder echo
    like '[YOUR FINAL ANSWER]', caught on the HotpotQA ext-smoke), fall back to
    the best recalled fact, then a stable placeholder -- never an empty answer."""
    a = (answer or "").strip()
    if a:
        return a
    hits = memory.recall(task, k=1)
    return hits[0] if hits else "unknown"


class ReActAgent:
    def __init__(self, *, llm, memory: SMPMemory, tools: List[object],
                 max_steps: int = 8):
        self.llm = llm
        self.memory = memory
        self.tools: Dict[str, object] = {t.name: t for t in tools}
        self.max_steps = max_steps

    def _prompt(self, task: str, scratchpad: List[str], context: List[str]) -> str:
        ctx = ("\nRemembered facts:\n" + "\n".join(f"- {c}" for c in context)
               if context else "")
        pad = ("\n" + "\n".join(scratchpad)) if scratchpad else ""
        return f"{GAIA_SYSTEM}\nQuestion: {task}{ctx}{pad}\nThought:"

    def _parse_action(self, turn: str):
        """Return (tool, arg) for a recognized verb, else None. Only a KNOWN
        verb (or a wired tool) counts, so a stray 'word[..]' in prose is not
        mistaken for an action now that the 'Action:' prefix is optional."""
        for m in _ACTION_RE.finditer(turn or ""):
            tool = m.group(1).lower()
            if tool in _KNOWN_VERBS or tool in self.tools:
                return tool, m.group(2).strip()
        return None

    def _synthesize_final(self, task: str, scratchpad: List[str],
                          transcript: List[str]) -> AgentResult:
        """Forced terminal turn: ask the model for the FINAL ANSWER directly so
        the agent ALWAYS terminates with a parseable answer within max_steps
        instead of returning None. Falls back to the best recalled fact, then a
        stable placeholder (never an empty contract)."""
        context = self.memory.recall(task, k=3)
        prompt = self._prompt(task, scratchpad, context) + (
            "\n(You are out of steps. Do NOT call a tool. Reply with EXACTLY: "
            "FINAL ANSWER: <answer> in the GAIA short form.)\nFINAL ANSWER:")
        turn = self.llm.complete(prompt)
        transcript.append(turn)
        answer = extract_final_answer("FINAL ANSWER: " + turn) \
            or extract_final_answer(turn)
        if not answer:
            answer = tighten_gaia_answer(turn) or (context[0] if context
                                                   else "unknown")
        return AgentResult(answer=answer, n_steps=len(transcript),
                           transcript=transcript, final_raw=turn)

    def run(self, task: str, clues: Optional[List[str]] = None) -> AgentResult:
        """Run the ReAct loop, then GUARANTEE a non-empty, non-template final
        answer at the boundary (the contract); see ``nonempty_answer``."""
        res = self._run(task, clues=clues)
        ans = nonempty_answer(tighten_gaia_answer(res.answer), task, self.memory)
        if ans != res.answer:
            res = AgentResult(answer=ans, n_steps=res.n_steps,
                              transcript=res.transcript, final_raw=res.final_raw)
        return res

    def _run(self, task: str, clues: Optional[List[str]] = None) -> AgentResult:
        scratchpad: List[str] = []
        transcript: List[str] = []
        # Seed in-context clues as observations: each clue is WRITTEN through the
        # SMP merge path, so the brain receives merged knowledge before reasoning.
        for clue in (clues or []):
            self.memory.write(clue)

        # One turn is RESERVED for a forced final-answer synthesis so the agent
        # always terminates with a parseable answer within max_steps.
        usable = max(1, self.max_steps - 1)
        for step in range(1, usable + 1):
            context = self.memory.recall(task, k=3)
            prompt = self._prompt(task, scratchpad, context)
            turn = self.llm.complete(prompt)
            transcript.append(turn)

            # the model may emit the GAIA template directly (no Action: verb) --
            # honor an explicit FINAL ANSWER line so we never spin past it.
            fa = extract_final_answer(turn)
            parsed = self._parse_action(turn)
            if fa and (parsed is None or parsed[0] != "final_answer"):
                return AgentResult(answer=nonempty_answer(fa, task, self.memory),
                                   n_steps=step, transcript=transcript,
                                   final_raw=turn)
            if parsed is None:
                scratchpad.append("Observation: (no parseable action)")
                continue
            tool, arg = parsed

            if tool == "final_answer":
                ans = nonempty_answer(tighten_gaia_answer(arg), task, self.memory)
                return AgentResult(answer=ans, n_steps=step,
                                   transcript=transcript, final_raw=turn)
            if tool == "remember":
                patches = self.memory.write(arg)
                decisions = ",".join(p.decision for p in patches) or "none"
                scratchpad.append(f"Observation: remembered ({decisions}).")
                continue
            if tool in self.tools:
                obs = self.tools[tool](arg)
                scratchpad.append(f"Observation: {obs}")
                continue
            scratchpad.append(f"Observation: unknown tool {tool!r}.")

        # FORCED final-answer synthesis (the reserved turn): never return None.
        return self._synthesize_final(task, scratchpad, transcript)


__all__ = ["ReActAgent", "AgentResult", "GAIA_SYSTEM", "nonempty_answer"]
