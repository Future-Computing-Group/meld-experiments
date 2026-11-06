"""R2: overlap/k recall-sweep aggregation (TDD).

The sweep ORCHESTRATION reuses the already-tested build_condition + recall_at_k_and_mrr;
the NEW logic is the per-cell aggregation, tested here as a pure function. The end-to-end
recall on real tasks is a smoke-ladder step. Run:
  PYTHONPATH=src python3 -m pytest tests/test_recall_sweep.py -q
"""
import os

from meld.eval.recall_sweep import aggregate_recall, recall_rows

from test_agent_meld_memory import agent_epoch


def test_recall_rows_structures_per_cell_and_consumes_the_inner_brain():
    """The orchestration contract: one row per (overlap, condition, task, k); recall in
    [0,1]. Semantic merge (under the epoch) consolidates paraphrase-duplicates that
    exact-dedup (no epoch) cannot, so the
    merged store holds no MORE live units than the exact-dedup store."""
    from meld.agent.hotpot_loader import load_hotpot_fixture
    epoch = agent_epoch("minilm", "lexical")
    tasks = load_hotpot_fixture(n=2)
    rows = recall_rows(tasks, overlaps=[0.5], ks=[5],
                       conditions=["distributed_merge", "exact_dedup"],
                       encoder="minilm", nli="lexical", epoch=epoch)
    assert len(rows) == 1 * 2 * 2 * 1                         # overlaps x conds x tasks x ks
    assert all(0.0 <= r["recall"] <= 1.0 for r in rows)
    assert all(r["live_units"] >= 0 for r in rows)
    merge_live = sum(r["live_units"] for r in rows if r["condition"] == "distributed_merge")
    dedup_live = sum(r["live_units"] for r in rows if r["condition"] == "exact_dedup")
    assert merge_live <= dedup_live


def test_recall_rows_emits_progress_per_task_for_mid_run_visibility():
    """A long sweep must emit per-batch progress so a wedge surfaces early. recall_rows
    calls the injected ``progress`` callback once per (cell, task) with counters."""
    from meld.agent.hotpot_loader import load_hotpot_fixture
    epoch = agent_epoch("minilm", "lexical")
    seen = []
    recall_rows(load_hotpot_fixture(n=2), overlaps=[0.5], ks=[5],
                conditions=["centralized"], encoder="minilm", nli="lexical", epoch=epoch,
                progress=lambda info: seen.append(info))
    assert len(seen) == 2                                  # one per task in the one cell
    assert {"cell", "n_cells", "task", "n_tasks", "elapsed_s"} <= set(seen[-1])
    assert seen[-1]["task"] == 2 and seen[-1]["n_tasks"] == 2


def test_aggregate_means_recall_and_live_units_per_cell():
    rows = [
        {"overlap": 0.5, "condition": "merge", "k": 5, "recall": 0.8, "live_units": 40},
        {"overlap": 0.5, "condition": "merge", "k": 5, "recall": 0.6, "live_units": 42},
        {"overlap": 0.5, "condition": "union", "k": 5, "recall": 0.7, "live_units": 50},
        {"overlap": 0.0, "condition": "merge", "k": 5, "recall": 1.0, "live_units": 38},
    ]
    agg = aggregate_recall(rows)
    assert agg[(0.5, "merge", 5)]["mean_recall"] == 0.7         # (0.8+0.6)/2
    assert agg[(0.5, "merge", 5)]["mean_live_units"] == 41.0
    assert agg[(0.5, "merge", 5)]["n"] == 2
    assert agg[(0.5, "union", 5)]["mean_recall"] == 0.7
    assert agg[(0.0, "merge", 5)]["mean_recall"] == 1.0
    # distinct cells are kept distinct (no cross-contamination)
    assert len(agg) == 3
