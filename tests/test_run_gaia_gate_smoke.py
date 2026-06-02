"""G3 wiring smoke: the gate driver runs the agent over the GAIA fixture and
the SMP-brain integration is non-degenerate.

This is the WIRING smoke (oracle LLM), not a GAIA result: it proves the
agent <-> SMP-brain <-> scorer harness runs end-to-end and the gate computes.
"""
from __future__ import annotations

from smp.agent.run_gaia_gate import run_gate_smoke


def test_smoke_runs_end_to_end_and_gate_computes():
    res = run_gate_smoke(n=6)
    assert res["n_tasks"] == 6
    # the gate is computed and reports an L2 rate over real L2 tasks
    assert res["gate"]["l2_n"] >= 1
    assert 0.0 <= res["gate"]["l2_rate"] <= 1.0
    assert isinstance(res["gate"]["passed"], bool)


def test_smoke_path_is_non_degenerate():
    """Every task must have actually stepped the agent AND written through the
    SMP merge path (a Patch in the brain audit log) -- not a no-op shortcut."""
    res = run_gate_smoke(n=6)
    for rec in res["integration"]:
        assert rec["n_steps"] >= 2, "agent did not step"
        assert rec["n_patches"] >= 1, "no Patch reached the brain"
        assert rec["brain_units"] >= 1, "brain received no merged knowledge"
        assert all(d in ("MERGE", "RELATE", "CONFLICT", "REJECT")
                   for d in rec["decisions"])


def test_oracle_wiring_recovers_gold_answers():
    """With the ORACLE policy the wiring must recover the gold answers (proves
    the recall/answer path is connected). This validates WIRING, not GAIA."""
    res = run_gate_smoke(n=6)
    assert res["scored"]["overall"]["rate"] == 1.0
