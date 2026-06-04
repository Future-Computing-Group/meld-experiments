"""C-1 in-process demonstration (Section 7-c, paragraph 1): order-independence.

Runs the fixed canonical contested-lifecycle scenario (CANONICAL_C1_SCENARIO) in
many random delivery orderings under both the status-CRDT and the last-writer-wins
baseline, and prints the order-independence numbers the manuscript reports:

  - modal-agreement fraction = size of the largest cluster of identical per-claim
    status maps / n_orderings (1.0 iff order-independent),
  - number of distinct status-map outcomes observed.

Pure simulation: no models, no network, deterministic in the seed. Calls the
unit-tested ``run_inprocess_c1``; the printed numbers are the asserted invariants.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_eval_c_inprocess.py
"""
from __future__ import annotations

import json

from smp.eval.crdt_convergence import run_inprocess_c1, CANONICAL_C1_SCENARIO


def main(n_orderings: int = 200, seed: int = 0) -> dict:
    rep = run_inprocess_c1(n_orderings=n_orderings, seed=seed)
    crdt, lww = rep["crdt"], rep["lww"]
    print("== C-1 in-process order-independence "
          f"(canonical contested-lifecycle scenario; "
          f"{rep['n_claims']} claims, {rep['n_events']} events incl. idempotent "
          f"re-delivery; n_orderings={rep['n_orderings']}, seed={rep['seed']}) ==")
    print(f"  status-CRDT     modal_agreement={crdt['modal_agreement']:.3f}  "
          f"distinct_outcomes={crdt['n_distinct']}")
    print(f"  last-writer-wins modal_agreement={lww['modal_agreement']:.3f}  "
          f"distinct_outcomes={lww['n_distinct']}")
    print(f"  CRDT converged status (lattice-join, all orders): "
          f"{json.dumps(crdt['status'])}")
    assert crdt["status"] == CANONICAL_C1_SCENARIO["expected_crdt_status"], \
        "CRDT converged status must equal the per-claim lattice joins"
    return rep


if __name__ == "__main__":
    main()
