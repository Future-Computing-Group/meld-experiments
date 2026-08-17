"""G3 LOCAL competence-gate WIRING smoke (no GPU, no remote, no billable).

Usage:  python3 scripts/run_gaia_gate_smoke.py
Runs the wired ReAct agent (MELD brain as memory) over a tiny GAIA-shaped fixture
with a deterministic ORACLE MockLLM, scores with GAIA quasi-exact-match, and
prints the competence gate (GAIA L2 >= 60%). This validates the harness + MELD-
brain integration; it is NOT a GAIA result (the real score is the gated CSC run).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from meld.agent.run_gaia_gate import run_gate_smoke  # noqa: E402

if __name__ == "__main__":
    res = run_gate_smoke(n=6)
    g = res["gate"]
    print(f"== G3 competence-gate WIRING smoke ({res['mode']}) ==")
    print(f"  tasks            : {res['n_tasks']}")
    print("  per-level quasi-exact-match:")
    for lvl, b in sorted(res["scored"]["by_level"].items()):
        print(f"    {lvl}: {b['correct']}/{b['n']} = {b['rate']:.2f}")
    print(f"  overall          : {res['scored']['overall']['correct']}"
          f"/{res['scored']['overall']['n']} "
          f"= {res['scored']['overall']['rate']:.2f}")
    print(f"== GATE: GAIA L2 >= {g['bar']:.0%} ==")
    print(f"  L2 rate          : {g['l2_correct']}/{g['l2_n']} = {g['l2_rate']:.2f}")
    print(f"  PASSED           : {g['passed']}  "
          f"(wiring oracle; NOT a model competence claim)")
    print("== MELD-brain integration (per task) ==")
    for rec in res["integration"]:
        print(f"  task {rec['task_id']}: steps={rec['n_steps']} "
              f"patches={rec['n_patches']} units={rec['brain_units']} "
              f"decisions={rec['decisions']}")
