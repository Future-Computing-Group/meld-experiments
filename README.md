# MELD

Reference implementation and experiments for MELD, a wire protocol that lets many
sovereign agent memories *merge knowledge* rather than only exchange tools. A
"brain" is a versioned, typed-link agent-memory store; agents PUBLISH graph deltas;
a receiving brain runs a merge-on-receive decision procedure (three core signals,
namely claim-key identity, embedding similarity, and an inference verdict, plus a
Context gate and an optional evidence-overlap signal that is disabled by default)
and emits an auditable Patch. Per-claim status converges through a state-based
status conflict-free replicated data type (CRDT).

This repository contains the protocol implementation, the merge engine, the status
CRDT, and the drivers for the deterministic experiments and ablations. It accompanies
the paper; comments in the code reference the corresponding paper sections (e.g. the
evaluation and consistency sections) by topic.

## Install

Python 3.11. From the repository root:

```bash
pip install -r requirements.txt
```

The dense M-2 merge-classification rung (a MiniLM sentence encoder + a cross-encoder
NLI) is optional and not needed for the deterministic experiments, the ablations, or
the test suite; to enable it, additionally `pip install "sentence-transformers>=3.0"
"torch>=2.0"`.

## Layout

```
src/meld/
  proto/        vendored typed-link wire schema (MemoryUnit / MemoryLink / Claim / ...)
  merge/        the three-signal decision procedure (+ optional evidence overlap), the status CRDT, the M-2 evaluation
  sync/         binding onto a publish/subscribe transport; semantic-routing ablation
  eval/         convergence (CRDT order-independence, reject-gate) and robustness sweeps
  agent/        memory-delta metrics, token economics, the QA harness
  config/       calibrated thresholds (theta_merge, sigma_lo, R_min, tau)
scripts/        runnable drivers for the ablations and the in-process convergence demos
tests/          the full test suite (run with pytest)
```

## Reproducing the experiments

The deterministic, model-free experiments reproduce end-to-end from this repository.

```bash
# Full test suite (unit tests + the public-safety hygiene guard)
PYTHONPATH=src pytest -q

# Ablations: gate necessity (A-1), conflict/authority (T-1), echo-safety (A-2),
# token economics (A-3); plus the M-2 merge-classification benchmark + ROC + baselines
# + the partition-duration x reorder-rate and decision-procedure robustness sweeps.
PYTHONPATH=src python scripts/run_ablation_metrics.py

# In-process status-CRDT convergence: order-independence + reject-gate placement.
PYTHONPATH=src python scripts/run_eval_c_inprocess.py
```

Every reported number is deterministic in its seed. Thresholds are calibrated, not
derived (theta_merge = 0.90, sigma_lo = 0.12, R_min = 0.10 at the operating point used
throughout); calibrated values live in `src/meld/config/`.

### What is and is not reproducible from this repository

- **Reproducible here:** the merge-classification benchmark (M-2), the gate/conflict
  ablations, the order-independence and reject-gate demonstrations, and the robustness
  sweeps, all deterministic and local.
- **Requires external infrastructure:** the headline distributed-versus-centralized
  merge sweep, which runs five open-weight models on HPC GPUs, and the sync-latency /
  routing / partition-heal results, which run on a multi-host 5G edge testbed. The
  protocol logic they exercise is the code here, and the drivers, model tags, and run
  configurations needed to re-run them on comparable hardware are in `hpc/`,
  `cross-tier/`, and `edge/`.

## License

Apache License 2.0; see `LICENSE`.

## Citation

See `CITATION.cff`.
