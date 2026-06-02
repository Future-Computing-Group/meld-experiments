"""G3 — GAIA task loader (offline fixture stand-in; real corpus is gated).

The GAIA HF dataset (gaia-benchmark/GAIA) is GATED (login + agree-to-terms +
no-reshare); the real validation slice is wired only on the gated CSC run, after
the benchmark identity gate (G0) is green. For the offline WIRING smoke this
loader emits a tiny, deterministic, GAIA-SHAPED fixture (a question, a level, a
short single answer, and the in-context clue an agent must read+remember) so the
agent<->brain<->scorer harness runs end-to-end with NO network and NO GAIA data.
It is NOT a GAIA result.
"""
from __future__ import annotations

from smp.agent.gaia_loader import load_gaia_fixture, GaiaTask


def test_fixture_tasks_are_gaia_shaped():
    tasks = load_gaia_fixture(n=4)
    assert tasks
    for t in tasks:
        assert isinstance(t, GaiaTask)
        assert t.question and t.gold and t.level in {"L1", "L2", "L3"}
        # each fixture task carries the clue(s) the agent must remember to answer
        assert t.clues


def test_fixture_has_l2_tasks_for_the_gate():
    tasks = load_gaia_fixture(n=6)
    assert any(t.level == "L2" for t in tasks), "gate needs L2 tasks"


def test_fixture_is_deterministic():
    a = load_gaia_fixture(n=4)
    b = load_gaia_fixture(n=4)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert [t.gold for t in a] == [t.gold for t in b]
