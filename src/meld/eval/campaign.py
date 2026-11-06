"""Campaign orchestrator core (rung-4): grid enumeration, resumable run loop, seed-as-
unit aggregation. PURE orchestration -- the real per-cell execution (HTTP to the brains +
WAN/partition tc on VM1) is injected as ``run_cell``, so the grid/resume/aggregation
logic is unit-tested with a fake while the integration lives in the VM1 cell-runner.

Resumability (decision 6): pending cells only (CampaignCheckpoint), each recorded the
instant it completes; a cell that errors is NOT recorded (stays pending -> retried), and
the loop continues (one bad cell never kills the grid). Aggregation bootstraps ACROSS
seeds grouped by (rtt, N, routing) -- seed = unit (STATS-C1), never pooled claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple

from meld.eval.campaign_stats import bootstrap_ci


@dataclass(frozen=True)
class CellSpec:
    rtt_ms: float
    n: int
    routing: str
    seed: int

    @property
    def cell_id(self) -> str:
        return f"rtt{self.rtt_ms}_N{self.n}_{self.routing}_seed{self.seed}"

    @property
    def condition(self) -> Tuple[float, int, str]:
        return (self.rtt_ms, self.n, self.routing)


def enumerate_cells(*, rtts: List[float], ns: List[int], routings: List[str],
                    seeds: List[int]) -> List[CellSpec]:
    """Full factorial grid. Order: rtt outer → N → routing → seed inner."""
    return [CellSpec(rtt, n, r, s)
            for rtt, n, r, s in product(rtts, ns, routings, seeds)]


def run_campaign(cells: List[CellSpec], run_cell: Callable[[CellSpec], dict],
                 checkpoint, *, progress: Optional[Callable[[str, dict], None]] = None,
                 on_error: Optional[Callable[[CellSpec, Exception], None]] = None) -> Dict[str, dict]:
    """Run every not-yet-done cell once; record each result durably as it completes.
    A cell whose run_cell raises is left unrecorded (retried on the next resume) and the
    loop continues. Returns all results accumulated in the checkpoint."""
    pending = set(checkpoint.pending([c.cell_id for c in cells]))
    for cell in cells:
        if cell.cell_id not in pending:
            continue
        try:
            result = run_cell(cell)
        except Exception as e:                       # one bad cell must not kill the grid
            if on_error is not None:
                on_error(cell, e)
            continue                                 # not recorded -> stays pending
        checkpoint.record(cell.cell_id, result)
        if progress is not None:
            progress(cell.cell_id, result)
    return checkpoint.results()


def aggregate_by_condition(cells: List[CellSpec], results: Dict[str, dict], *,
                           metric: str, agg: str = "mean", n_boot: int = 10000,
                           seed: int = 0) -> Dict[Tuple[float, int, str], Tuple[float, float, float]]:
    """Group cells by (rtt, N, routing); collect the per-SEED metric value; return the
    seed-as-unit bootstrap CI (point, lo, hi) per condition. Cells missing the metric or
    absent from results are skipped (a partial grid still aggregates what it has)."""
    by_cond: Dict[Tuple[float, int, str], List[float]] = {}
    for c in cells:
        res = results.get(c.cell_id)
        if not res or metric not in res:
            continue
        by_cond.setdefault(c.condition, []).append(float(res[metric]))
    return {cond: bootstrap_ci(vals, n_boot=n_boot, agg=agg, seed=seed)
            for cond, vals in by_cond.items()}


def score_routing(delivered: Dict[str, dict], needed: Dict[str, frozenset], *, sender: str,
                  url: Callable[[str], str]) -> Dict[str, int]:
    """Routing counts of one brain's /publish reply: ``delivered`` is id -> {kind, text, peers};
    only deltas (kind D) are scored, by text against the needed-oracle (brain ids, mapped to the
    peer URLs the brains publish to by ``url``; the sender itself never needs its own delta)."""
    num = den = tot = 0
    for o in delivered.values():
        if o.get("kind") != "D":
            continue
        got = set(o["peers"])
        need = {url(b) for b in needed.get(o["text"], ()) if b != sender}
        num, den, tot = num + len(got & need), den + len(need), tot + len(got)
    return {"recall_num": num, "recall_den": den, "delivered_total": tot}


__all__ = ["CellSpec", "enumerate_cells", "run_campaign", "aggregate_by_condition", "score_routing"]
