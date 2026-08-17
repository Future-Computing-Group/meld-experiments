"""Resumable campaign checkpoint (decision 6: connectivity is flaky -> the run must
survive a control-link drop / VM1 restart and resume without redoing completed cells).

Each finished cell is appended durably (JSONL) the moment it completes -- results live on
VM disk, never only in the laptop session. On resume a fresh instance reads the file and
skips done cells. Last write wins for a re-run cell.
"""
from __future__ import annotations

import json


def test_record_marks_cell_done(tmp_path):
    from meld.eval.campaign_checkpoint import CampaignCheckpoint
    cp = CampaignCheckpoint(str(tmp_path / "ckpt.jsonl"))
    assert cp.done("rtt50_N4_broadcast_seed3") is False
    cp.record("rtt50_N4_broadcast_seed3", {"p50_ms": 51.2})
    assert cp.done("rtt50_N4_broadcast_seed3") is True


def test_resume_reads_prior_completed_from_disk(tmp_path):
    from meld.eval.campaign_checkpoint import CampaignCheckpoint
    path = str(tmp_path / "ckpt.jsonl")
    cp = CampaignCheckpoint(path)
    cp.record("cellA", {"recall": 0.9})
    cp.record("cellB", {"recall": 0.8})
    resumed = CampaignCheckpoint(path)                 # fresh instance (simulates VM1 restart)
    assert resumed.completed_cells() == {"cellA", "cellB"}
    assert resumed.results()["cellA"] == {"recall": 0.9}


def test_record_is_durable_jsonl(tmp_path):
    from meld.eval.campaign_checkpoint import CampaignCheckpoint
    path = tmp_path / "ckpt.jsonl"
    cp = CampaignCheckpoint(str(path))
    cp.record("c1", {"x": 1})
    cp.record("c2", {"x": 2})
    lines = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    assert [r["cell"] for r in lines] == ["c1", "c2"]   # appended as each completes


def test_pending_filters_out_completed(tmp_path):
    from meld.eval.campaign_checkpoint import CampaignCheckpoint
    cp = CampaignCheckpoint(str(tmp_path / "ckpt.jsonl"))
    cp.record("c1", {})
    all_cells = ["c1", "c2", "c3"]
    assert cp.pending(all_cells) == ["c2", "c3"]        # c1 skipped on resume


def test_rerecord_last_wins(tmp_path):
    from meld.eval.campaign_checkpoint import CampaignCheckpoint
    path = str(tmp_path / "ckpt.jsonl")
    cp = CampaignCheckpoint(path)
    cp.record("c1", {"v": 1})
    cp.record("c1", {"v": 2})                           # re-run (e.g. after a partial)
    assert CampaignCheckpoint(path).results()["c1"] == {"v": 2}
