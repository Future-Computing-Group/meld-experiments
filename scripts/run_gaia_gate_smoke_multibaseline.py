"""G3 LOCAL multi-baseline competence-gate WIRING smoke (no GPU, no billable).

Usage:  python3 scripts/run_gaia_gate_smoke_multibaseline.py

Runs SEVERAL public, open agentic baselines from the registry
(react, reflexion, and the REAL HF smolagents ToolCallingAgent if installed),
each over the SAME tiny GAIA-shaped fixture, each through the SAME MELD brain as
its memory backend, with a deterministic ORACLE MockLLM. Scores with GAIA
quasi-exact-match and prints the per-baseline competence gate (GAIA L2 >= 60%).

This validates the registry + harness + MELD-brain integration is SWAPPABLE across
baselines and each writes through the MELD merge path; it is NOT a GAIA result
(the real per-model score is the gated CSC run, PI-gated G5).
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from meld.agent.run_gaia_gate import run_gate_smoke_multibaseline  # noqa: E402
from meld.agent.registry import AGENT_PROVENANCE  # noqa: E402


def _print_baseline(name, r):
    g = r["gate"]
    prov = AGENT_PROVENANCE.get(name, {})
    print(f"\n-- baseline: {name}  [{prov.get('kind','?')}; "
          f"license {prov.get('license','?')}] --")
    print("   per-level quasi-exact-match:", end=" ")
    print(" ".join(f"{lvl} {b['correct']}/{b['n']}={b['rate']:.2f}"
                   for lvl, b in sorted(r["scored"]["by_level"].items())))
    print(f"   overall   : {r['scored']['overall']['correct']}"
          f"/{r['scored']['overall']['n']} "
          f"= {r['scored']['overall']['rate']:.2f}")
    print(f"   GATE L2>=60%: {g['l2_correct']}/{g['l2_n']}={g['l2_rate']:.2f} "
          f"PASSED={g['passed']}  (wiring oracle; NOT a competence claim)")
    # non-degeneracy summary
    steps = [rec["n_steps"] for rec in r["integration"]]
    patches = [rec["n_patches"] for rec in r["integration"]]
    units = [rec["brain_units"] for rec in r["integration"]]
    print(f"   MELD integration: steps={min(steps)}-{max(steps)} "
          f"patches={min(patches)}-{max(patches)} "
          f"brain_units={min(units)}-{max(units)} "
          f"(every task stepped + wrote a Patch)")


if __name__ == "__main__":
    baselines = ["react", "reflexion"]
    if importlib.util.find_spec("smolagents") is not None:
        baselines.append("smolagents-toolcalling")
        note = "(smolagents present -> REAL HF ToolCallingAgent included)"
    else:
        note = "(smolagents absent -> in-repo baselines only; install to add it)"

    print("== G3 multi-baseline competence-gate WIRING smoke ==")
    print(f"   baselines: {baselines} {note}")
    res = run_gate_smoke_multibaseline(n=6, baselines=baselines)
    print(f"   mode     : {res['mode']}")
    print(f"   tasks    : {res['n_tasks']}")
    for name in baselines:
        _print_baseline(name, res["per_baseline"][name])
