"""The G3 competence-gate agent harness.

This package wires a PUBLIC, citable open-agent pattern (a minimal ReAct
tool-agent; Yao et al., "ReAct: Synergizing Reasoning and Acting in Language
Models", arXiv 2210.03629) to the SMP brain as its MEMORY BACKEND, plus a GAIA
loader, a quasi-exact-match scorer, and the competence gate (GAIA L2 >= 60%).

HARD ARCHITECTURAL CONSTRAINT (why this exists): the competence gate must stand
on a public, open foundation -- NOT the unpublished FUSILLI firmament/RLMF agent
-- so the SMP evaluation stays self-standing and independent of FUSILLI's
publication. The agent loop interface mirrors HF ``smolagents.ToolCallingAgent``
(Apache-2.0) so an off-the-shelf public framework can be swapped in by config
for the gated CSC run; the in-repo ReAct loop keeps the local ``.venv`` minimal
and the vendored-proto protobuf 6.x pin safe (no heavy framework deps locally).

The agent's memory backend is an SMP ``Brain``: the agent READS via the brain's
graph-walk read and WRITES observations via PUBLISH -> merge-on-receive ->
Patch. So a G3 run exercises the SMP merge integration, not just a bare agent.

ROBUSTNESS + ACHIEVABILITY extension:
- ``registry.make_agent(name)`` is an agent-framework registry (the SMP analogue
  of DWM's encoder ladder): >= 2-3 swappable PUBLIC open baselines behind ONE
  interface, each driving the SAME GAIA gate through the SAME SMP brain --
  ``react`` (in-repo, Yao et al. 2210.03629), ``reflexion`` (in-repo plan-then-
  execute, Shinn et al. 2303.11366), and ``smolagents-toolcalling`` (the REAL HF
  ``ToolCallingAgent``, Apache-2.0, lazy-imported so the offline path + the
  protobuf-6.x venv pin stay unperturbed).
- The PUBLIC tool surface GAIA L2 needs (``smp.agent.tools``): an offline
  web-search/replay tool (CSC compute nodes have no internet), a file/document
  reader (PDF/CSV/text), and a sandboxed calculator. Evidence-reading tools feed
  their content THROUGH the SMP brain as merged observations (SMP-on-message),
  not only as raw tool strings.
- The open-model axis lives in ``llm.OpenWeightLLM`` (Qwen2.5-7B-Instruct default;
  Mistral-7B-Instruct / Gemma-2-9B-it fallbacks); the same models run on every
  registry baseline. Llama-3.1-8B is avoided (open-weight JSON-prompt quirk).
"""
