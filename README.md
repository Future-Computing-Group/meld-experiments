# The Semantic Merge Protocol (SMP)

Reference implementation and experiments for the Semantic Merge Protocol, a wire
protocol that lets many sovereign agent memories *merge knowledge* rather than only
exchange tools. A "brain" is a versioned, typed-link agent-memory store; agents
PUBLISH graph deltas; a receiving brain runs a merge-on-receive decision procedure
(four signals — claim-key identity, embedding similarity, evidence overlap, and an
inference verdict, plus a Context gate) and emits an auditable Patch. Per-claim status
converges through a state-based status conflict-free replicated data type (CRDT).

This repository contains the protocol implementation, the merge engine, the status
CRDT, and the drivers for the deterministic experiments and ablations. It accompanies
the paper; section labels in the code (e.g. "Section 7-c") point to the corresponding
results.

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
src/smp/
  proto/        vendored typed-link wire schema (MemoryUnit / MemoryLink / Claim / ...)
  merge/        the four-signal decision procedure, the status CRDT, the M-2 evaluation
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
throughout); calibrated values live in `src/smp/config/`.

### What is and is not reproducible from this repository

- **Reproducible here:** the merge-classification benchmark (M-2), the gate/conflict
  ablations, the order-independence and reject-gate demonstrations, and the robustness
  sweeps — all deterministic and local.
- **Not in this repository (reproduced from archived run logs):** the headline
  distributed-versus-centralized merge sweep, which runs five open-weight models on
  HPC GPUs, and the sync-latency / routing / partition-heal results, which run on a
  multi-host 5G edge testbed. These require external infrastructure; the protocol logic
  they exercise is the code here, and the archived per-run logs back the reported
  numbers.

## License

See `LICENSE`.

## Citation

See `CITATION.cff`.
