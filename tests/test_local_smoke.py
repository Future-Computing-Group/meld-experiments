"""G2 — local smoke to completion (full SIM pipeline + a rendered figure).

The smoke runs the real merge engine on a small slice with real (local) hashing
embeddings: extract -> PUBLISH -> merge-on-receive -> answer, emitting an M-1
mini-table, an M-2 confusion mini-matrix, and a rendered PNG figure. Tests check
INVARIANTS (figure exists, tables sane, false-MERGE ~ 0); visual quality is
checked by a human reading the PNG (the figure-inspection discipline).
"""
from __future__ import annotations

import os

from meld.eval.smoke import run_smoke


def test_smoke_runs_to_completion_and_renders_figure(tmp_path):
    res = run_smoke(out_dir=str(tmp_path), n_conversations=8, n_agents=3)
    # M-1 invariants
    assert 0.0 <= res["m1"]["distributed_accuracy"] <= 1.0
    assert res["m1"]["distributed_accuracy"] >= res["m1"]["centralized_accuracy"] - 0.2
    assert res["m1"]["naive_union_units"] > res["m1"]["distributed_units"]
    # M-2 invariants: the dangerous error is ~0
    assert res["m2"]["false_merge_rate"] == 0.0
    # figure rendered
    assert os.path.exists(res["figure"])
    assert os.path.getsize(res["figure"]) > 0
