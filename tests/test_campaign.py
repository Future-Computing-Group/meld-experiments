"""Campaign orchestrator CORE: grid enumeration, resumable run loop, seed-as-unit
aggregation. The pure orchestration is TDD'd here with a fake cell-runner; the REAL cell
execution (HTTP to brains + WAN/partition tc on VM1) is a thin function exercised at the
1-seed dry cell, not unit-tested. A cell that errors is NOT checkpointed (stays pending ->
retried on resume); completed cells are skipped. Aggregation bootstraps ACROSS seeds
(seed = unit, STATS-C1), grouping by (rtt, N, routing).
"""
from __future__ import annotations


def test_enumerate_cells_is_the_full_grid_with_unique_ids():
    from meld.eval.campaign import enumerate_cells
    cells = enumerate_cells(rtts=[0, 25], ns=[2, 4], routings=["broadcast", "semantic"],
                            seeds=[1, 2, 3])
    assert len(cells) == 2 * 2 * 2 * 3
    assert len({c.cell_id for c in cells}) == len(cells)        # unique ids
    assert cells[0].cell_id.startswith("rtt0_N2_broadcast_seed1")


def test_run_campaign_runs_each_pending_once_and_records(tmp_path):
    from meld.eval.campaign import enumerate_cells, run_campaign
    from meld.eval.campaign_checkpoint import CampaignCheckpoint
    cells = enumerate_cells(rtts=[0], ns=[2], routings=["broadcast"], seeds=[1, 2])
    calls = []

    def run_cell(c):
        calls.append(c.cell_id)
        return {"p50_ms": 10.0 + c.seed}
    cp = CampaignCheckpoint(str(tmp_path / "c.jsonl"))
    run_campaign(cells, run_cell, cp)
    assert sorted(calls) == sorted(c.cell_id for c in cells)    # each run once
    assert cp.completed_cells() == {c.cell_id for c in cells}


def test_run_campaign_resumes_and_skips_completed(tmp_path):
    from meld.eval.campaign import enumerate_cells, run_campaign
    from meld.eval.campaign_checkpoint import CampaignCheckpoint
    cells = enumerate_cells(rtts=[0], ns=[2], routings=["broadcast"], seeds=[1, 2, 3])
    path = str(tmp_path / "c.jsonl")
    CampaignCheckpoint(path).record(cells[0].cell_id, {"p50_ms": 1.0})   # cell already done
    calls = []
    run_campaign(cells, lambda c: (calls.append(c.cell_id) or {"p50_ms": 2.0}),
                 CampaignCheckpoint(path))
    assert cells[0].cell_id not in calls                        # skipped on resume
    assert len(calls) == 2                                       # only the 2 pending ran


def test_failed_cell_is_not_recorded_so_it_retries(tmp_path):
    from meld.eval.campaign import enumerate_cells, run_campaign
    from meld.eval.campaign_checkpoint import CampaignCheckpoint
    cells = enumerate_cells(rtts=[0], ns=[2], routings=["broadcast"], seeds=[1, 2])

    def run_cell(c):
        if c.seed == 1:
            raise RuntimeError("transient cell failure")
        return {"p50_ms": 5.0}
    cp = CampaignCheckpoint(str(tmp_path / "c.jsonl"))
    run_campaign(cells, run_cell, cp)
    done = cp.completed_cells()
    assert any("seed2" in c for c in done)                      # good cell recorded
    assert not any("seed1" in c for c in done)                  # failed cell stays pending


def test_aggregate_by_condition_is_seed_as_unit():
    from meld.eval.campaign import enumerate_cells, aggregate_by_condition
    cells = enumerate_cells(rtts=[0], ns=[2], routings=["broadcast"], seeds=[1, 2, 3, 4])
    # one recall value per SEED; aggregation bootstraps across the 4 seeds
    results = {c.cell_id: {"recall": 0.80 + 0.01 * c.seed} for c in cells}
    agg = aggregate_by_condition(cells, results, metric="recall", n_boot=2000, seed=0)
    key = (0, 2, "broadcast")
    assert key in agg
    point, lo, hi = agg[key]
    assert lo <= point <= hi
    assert abs(point - sum(0.80 + 0.01 * s for s in [1, 2, 3, 4]) / 4) < 1e-6   # mean over seeds
