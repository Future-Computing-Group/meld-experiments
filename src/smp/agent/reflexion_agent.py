"""A plan-then-execute / Reflexion-style tool-agent (public pattern).

This is a SECOND distinct in-repo design behind the same interface as
``ReActAgent`` (``run(task) -> AgentResult``; tools with name/description/
__call__), so the registry can swap it for the gated run with no tool rewrite.

Public, citable lineage:
  - Reflexion: Shinn et al., "Reflexion: Language Agents with Verbal
    Reinforcement Learning", arXiv 2303.11366.
  - Plan-and-Solve / plan-then-execute: Wang et al., arXiv 2305.04091.

How it differs from plain ReAct (so it is a genuinely DIFFERENT baseline, not a
rename): the loop runs an explicit PLAN turn first (the LLM proposes the steps),
then executes ReAct steps; if it reaches ``max_steps`` WITHOUT a final answer it
emits ONE reflection turn (a verbal critique) and runs a short second attempt.
The MEMORY is the SAME SMP brain (no private store): every ``remember`` and every
evidence-tool observation goes PUBLISH -> merge-on-receive -> Patch, exactly like
ReAct, so this baseline exercises the SMP merge integration identically.

The LLM contract stays a PLAIN ReAct text turn (Thought:/Action:), honoring the
open-weight-quirks lesson (no JSON-output prompt).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from smp.agent.react_agent import (AgentResult, GAIA_SYSTEM, _ACTION_RE,
                                    _KNOWN_VERBS, nonempty_answer)
from smp.agent.gaia_scorer import extract_final_answer, tighten_gaia_answer
from smp.agent.memory import SMPMemory

_PLAN_PREAMBLE = (
    "First, briefly PLAN the steps to answer the question, then act. "
)

_SYNTHESIS_PREAMBLE = (
    "You are out of steps. Do NOT call any tool. Using only the remembered "
    "facts above, SYNTHESIZE the answer now and reply with EXACTLY one line: "
    "FINAL ANSWER: <answer> (the GAIA short form -- a number, as few words as "
    "possible, or a comma-separated list; not a sentence). "
)


class ReflexionAgent:
    """Plan-then-execute with one reflection retry. Same interface + same SMP
    brain backend as ReActAgent."""

    def __init__(self, *, llm, memory: SMPMemory, tools: List[object],
                 max_steps: int = 8):
        self.llm = llm
        self.memory = memory
        self.tools: Dict[str, object] = {t.name: t for t in tools}
        self.max_steps = max_steps

    def _prompt(self, task: str, scratchpad: List[str], context: List[str],
                preamble: str = "") -> str:
        ctx = ("\nRemembered facts:\n" + "\n".join(f"- {c}" for c in context)
               if context else "")
        pad = ("\n" + "\n".join(scratchpad)) if scratchpad else ""
        return f"{GAIA_SYSTEM}\n{preamble}Question: {task}{ctx}{pad}\nThought:"

    def _parse_action(self, turn: str):
        """Return (tool, arg) for the first RECOGNIZED verb, else None. Gating on
        a known verb keeps a stray 'word[..]' in prose from being mistaken for an
        action now that the 'Action:' prefix is optional."""
        for m in _ACTION_RE.finditer(turn or ""):
            tool = m.group(1).lower()
            if tool in _KNOWN_VERBS or tool in self.tools:
                return tool, m.group(2).strip()
        return None

    def _dispatch_turn(self, turn: str, task: str,
                       scratchpad: List[str]) -> Optional[AgentResult]:
        """Parse and dispatch ONE turn. Returns an AgentResult on final_answer,
        else None. Shared by the plan/execute/reflection phases so an action
        proposed in ANY phase is executed (never silently discarded)."""
        parsed = self._parse_action(turn)
        if parsed is None:
            return None
        tool, arg = parsed
        if tool == "final_answer":
            ans = nonempty_answer(tighten_gaia_answer(arg), task, self.memory)
            return AgentResult(answer=ans, n_steps=0,
                               transcript=[], final_raw=turn)
        if tool == "remember":
            patches = self.memory.write(arg)
            decisions = ",".join(p.decision for p in patches) or "none"
            scratchpad.append(f"Observation: remembered ({decisions}).")
        elif tool in self.tools:
            obs = self.tools[tool](arg)
            scratchpad.append(f"Observation: {obs}")
        else:
            scratchpad.append(f"Observation: unknown tool {tool!r}.")
        return None

    def _execute(self, task: str, scratchpad: List[str], transcript: List[str],
                 budget: int) -> Optional[AgentResult]:
        for step in range(budget):
            context = self.memory.recall(task, k=3)
            turn = self.llm.complete(self._prompt(task, scratchpad, context))
            transcript.append(turn)
            # the model may emit the GAIA template directly (no Action:) -- treat
            # an explicit FINAL ANSWER as a final_answer so we never spin past it.
            fa = extract_final_answer(turn)
            parsed = self._parse_action(turn)
            if fa and (parsed is None or parsed[0] != "final_answer"):
                return AgentResult(answer=nonempty_answer(fa, task, self.memory),
                                   n_steps=len(transcript),
                                   transcript=transcript, final_raw=turn)
            if parsed is None:
                scratchpad.append("Observation: (no parseable action)")
                continue
            res = self._dispatch_turn(turn, task, scratchpad)
            if res is not None:
                return AgentResult(answer=res.answer, n_steps=len(transcript),
                                   transcript=transcript, final_raw=res.final_raw)
        return None

    def _synthesize_final(self, task: str, scratchpad: List[str],
                          transcript: List[str]) -> AgentResult:
        """Forced terminal turn: ask the model for a FINAL ANSWER directly (no
        tool), so the agent ALWAYS terminates with a parseable answer within
        max_steps instead of returning None. Falls back to the most relevant
        remembered fact if the model still gives no FINAL ANSWER line, and to a
        non-empty placeholder as the last resort (never an empty contract)."""
        context = self.memory.recall(task, k=3)
        turn = self.llm.complete(
            self._prompt(task, scratchpad, context,
                         preamble=_SYNTHESIS_PREAMBLE))
        transcript.append(turn)
        answer = extract_final_answer(turn)
        if not answer:
            # the synthesis turn itself produced no template; take its raw text,
            # then the best recalled fact, then a stable placeholder.
            answer = tighten_gaia_answer(turn) or (context[0] if context
                                                   else "unknown")
        return AgentResult(answer=answer, n_steps=len(transcript),
                           transcript=transcript, final_raw=turn)

    def run(self, task: str, clues: Optional[List[str]] = None) -> AgentResult:
        """Run the plan-execute-reflect loop, then GUARANTEE a non-empty,
        non-template final answer (the contract). A model can finalize with a
        template-placeholder payload through several routes; this single safety
        net at the boundary tightens the result and falls back to a recalled fact
        rather than leaking the placeholder (caught on the HotpotQA ext-smoke,
        reflexion/mistral)."""
        res = self._run(task, clues=clues)
        ans = nonempty_answer(tighten_gaia_answer(res.answer), task, self.memory)
        if ans != res.answer:
            res = AgentResult(answer=ans, n_steps=res.n_steps,
                              transcript=res.transcript, final_raw=res.final_raw)
        return res

    def _run(self, task: str, clues: Optional[List[str]] = None) -> AgentResult:
        scratchpad: List[str] = []
        transcript: List[str] = []
        for clue in (clues or []):
            self.memory.write(clue)

        # One turn is RESERVED for the forced final-answer synthesis so the agent
        # always terminates with a parseable answer within max_steps. The plan,
        # execute, and reflect phases share the remaining budget.
        synth_reserve = 1
        usable = max(1, self.max_steps - synth_reserve)

        def remaining() -> int:
            return usable - len(transcript)

        # PLAN turn (explicit; this is what distinguishes the baseline). If the
        # plan turn also proposed an action, EXECUTE it (do not discard work):
        # an open-weight model frequently emits "Thought: plan... Action: ..."
        # in one turn.
        if remaining() > 0:
            plan_turn = self.llm.complete(
                self._prompt(task, scratchpad, self.memory.recall(task, k=3),
                             preamble=_PLAN_PREAMBLE))
            transcript.append(plan_turn)
            scratchpad.append(f"Plan: {plan_turn.strip()[:200]}")
            fa = extract_final_answer(plan_turn)
            if fa:
                return AgentResult(answer=nonempty_answer(fa, task, self.memory),
                                   n_steps=len(transcript),
                                   transcript=transcript, final_raw=plan_turn)
            res = self._dispatch_turn(plan_turn, task, scratchpad)
            if res is not None:
                return AgentResult(answer=res.answer, n_steps=len(transcript),
                                   transcript=transcript,
                                   final_raw=res.final_raw)

        # EXECUTE phase: leave room for one reflection + at least one retry turn.
        first_budget = max(0, remaining() - 2)
        if first_budget > 0:
            res = self._execute(task, scratchpad, transcript, first_budget)
            if res is not None:
                return res

        # REFLECTION retry: one verbal critique, then a short second attempt,
        # all within the remaining usable budget.
        if remaining() > 1:
            reflection = self.llm.complete(
                self._prompt(task, scratchpad, self.memory.recall(task, k=3),
                             preamble="Reflect on why the previous attempt "
                                      "failed, then answer. "))
            transcript.append(reflection)
            scratchpad.append(f"Reflection: {reflection.strip()[:200]}")
            fa = extract_final_answer(reflection)
            if fa:
                return AgentResult(answer=nonempty_answer(fa, task, self.memory),
                                   n_steps=len(transcript),
                                   transcript=transcript, final_raw=reflection)
            res = self._execute(task, scratchpad, transcript, remaining())
            if res is not None:
                return res

        # FORCED final-answer synthesis (uses the reserved turn): the agent must
        # NOT return None / an empty answer on a real model that never finalizes.
        return self._synthesize_final(task, scratchpad, transcript)


__all__ = ["ReflexionAgent"]
