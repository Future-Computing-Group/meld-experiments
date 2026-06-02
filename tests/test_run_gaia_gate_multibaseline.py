"""G3 wiring smoke across SEVERAL baselines: the registry drives the SAME GAIA
gate through the SAME SMP brain for each baseline, and each is non-degenerate.

WIRING smoke (oracle LLM), NOT a GAIA result: it proves the
registry <-> agent <-> SMP-brain <-> scorer harness runs end-to-end for >= 2
public baselines and the gate computes per baseline. The oracle recovers gold
(validates recall/answer wiring); it is NOT a model competence claim.
"""
from __future__ import annotations

from smp.agent.run_gaia_gate import run_gate_smoke_multibaseline


def test_multibaseline_smoke_runs_offline_baselines():
    res = run_gate_smoke_multibaseline(n=6, baselines=["react", "reflexion"])
    assert set(res["per_baseline"]) == {"react", "reflexion"}
    for name, r in res["per_baseline"].items():
        assert r["n_tasks"] == 6
        assert r["gate"]["l2_n"] >= 1
        assert 0.0 <= r["gate"]["l2_rate"] <= 1.0
        assert isinstance(r["gate"]["passed"], bool)


def test_multibaseline_smoke_is_nondegenerate_per_baseline():
    res = run_gate_smoke_multibaseline(n=6, baselines=["react", "reflexion"])
    for name, r in res["per_baseline"].items():
        for rec in r["integration"]:
            assert rec["n_steps"] >= 2, f"{name}: agent did not step"
            assert rec["n_patches"] >= 1, f"{name}: no Patch reached the brain"
            assert rec["brain_units"] >= 1, f"{name}: brain got no knowledge"


def test_multibaseline_oracle_recovers_gold_for_each_baseline():
    res = run_gate_smoke_multibaseline(n=6, baselines=["react", "reflexion"])
    for name, r in res["per_baseline"].items():
        assert r["scored"]["overall"]["rate"] == 1.0, \
            f"{name}: oracle wiring did not recover gold"
