# EXPERIMENT — SMP (Semantic Merge Protocol)

Read this before every action. Follow steps in order; fix the protocol before improvising.

## What this repo is

SMP is binding glue + a merge engine, not a new system. A "brain" is a memory-store
replica of a world-model graph; agents PUBLISH graph deltas; a receiving brain runs a
merge-on-receive decision procedure (the four signals kappa/sigma/epsilon/chi + a
Context gate) and emits an auditable Patch. Per-claim status converges via a
state-based status-CRDT. The wire types (MemoryUnit / MemoryLink / Claim / Endpoint /
StalenessTag) are a vendored links-first-class proto copy; the reference graph data
model, retention law, and CRDT correctness are cited, not re-derived here.

## Claims under test (this build covers Phases 0-1, simulation, laptop, 0 cost)

- **M-2** — the merge decision procedure classifies claim pairs correctly
  (precision/recall/F1 per class; the dangerous error is a false MERGE).
- **H / M-1** — distributed agents syncing into a global brain match a centralized
  single store on answer quality; a naive set-union ablation is strictly worse.
- **O-1** — the four sync/version/retrieval/re-fetch overhead term counts.
- **A-1/2/3** — gate ablations, echo-safety probe, single-agent token-cost data point.

## Honest claim discipline — load-bearing

- The merge thresholds (theta_merge, sigma_lo) are **calibrated, not derived**;
  M-2 reports sensitivity. Calibrated values are written to a gitignored config, never
  to the tracked default.
- Simulation suffices for accuracy / classification / overhead-counts; latency /
  throughput / convergence-under-partition need real distributed infra and are
  deferred to a later, gated phase. Anything not yet runnable here is marked as such.
- The CRDT gives convergence of **status, not truth**. The four-way classifier
  represents conflict (a first-class CONTRADICTS edge); it does not silently resolve it.

## Proto build env — load-bearing
SMP's proto build/runtime uses a local `.venv` (`python3 -m venv .venv`) with
`protobuf` 6.x + `grpcio-tools`; recompile the vendored proto and run pytest in
it (`.venv/bin/python -m pytest -q`). Do NOT compile the vendored proto against
the GLOBAL env: grpc_tools' `protobuf<7` pin downgrades the global protobuf and
breaks other proto-users (e.g. firmament). SMP's 6.x deps live ONLY in `.venv`.

## Smoke ladder — STRICT ascent

1. Unit/integration tests (`tests/`) — must be green (G1).
2. **LOCAL smoke** (this machine): `python3 scripts/run_local_smoke.py` — full SIM
   pipeline extract -> PUBLISH -> merge-on-receive -> answer on a small benchmark
   slice + local embeddings. Visually inspect the rendered PNG (G2).
3. Bounded remote ext-smoke (short, 2-4 nodes, one induced partition) — NOT run here.
4. Full distributed run — only after the GO/NO-GO gate clears + PI GO. NOT run here.

### GO/NO-GO gate ladder
| Gate | Condition | Gates what |
|---|---|---|
| **G0** | benchmark identity verified at source; identity test green | any data wiring |
| **G1** | all unit/integration tests green (fresh evidence) | smoke |
| **G2** | local smoke to completion: M-1 sane, M-2 false-MERGE ~ 0, figure clean | SIM experiments |
| **G3** | agent clears the competence gate on the real harness | real-infra investment |
| **G4** | remote ext-smoke clean (liveness + delta persistence on induced abort + cost ceiling) | full real runs |
| **G5** | PI GO | any billable/multi-node full run |

## Benchmark identity (G0)
Benchmark provenance (name, authors, corpus, levels, QA metric) is verified at source
and recorded in `data/benchmark_provenance.json` (gitignored). The expected hashes are
guarded by `tests/test_benchmark_identity.py`. NO benchmark data is wired before G0 is
green. LongMemEval (distinct from LongMemBench) is the headline distributed-vs-centralized
vehicle; GAIA (levels L1-L3) is the competence gate, deferred to a later phase.

## Reuse posture
- REUSED: the experiment-repo template + embedding/encoder plumbing + hygiene tests
  (cached per-text-hash embeddings); the memory store + reference wire types; the
  retention law for the staleness/overrule REJECT gate; the status enum/lifecycle.
- NEW: the wire verbs, the four-way merge decision procedure + Context gate, the Patch
  object, the OR-Set status-CRDT merge + lattice-join recompute, the transport binding,
  and the distributed-vs-centralized evaluation.

## Repo hygiene (public-bound from commit 1)
`data/`, `plan/`, `*-LOG.md`, ops/billing/infra strings are gitignored from commit 1;
`tests/test_repo_hygiene.py` enforces it; the pre-commit hook runs the guard per commit.
Install hooks in a clone with: `git config core.hooksPath scripts/git-hooks`.
