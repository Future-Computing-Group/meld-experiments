# MELD

Reference implementation, experiment drivers and study archive for MELD, a knowledge-merge protocol for sovereign agent memories. Replicas exchange only what principals assert, as five replicated grow-only sets (signed claim deltas, authored status effects, recorded NLI evidence, calibration records and calibration epochs), and every replica derives consolidation, representatives and typed links by a deterministic fold in a canonical order. Replicas holding the same sets agree on authored status at once and on the derived view after one pass, and each pass emits a certificate that any holder of the sets can check by recomputation. Admission is risk-specified: a federation sets a tolerated false-merge rate, and MELD derives a threshold per store-size bin by split-conformal calibration on labelled tested pairs, abstains to RELATE when the test cannot certify a merge, and maintains the thresholds through signed recalibration epochs that replicate like any other object.

This repository accompanies the paper; comments in the code reference the corresponding paper sections by topic.

## Install

Python 3.11. From the repository root:

```bash
pip install -r requirements.txt
```

The dense signals (a MiniLM sentence encoder and a cross-encoder NLI) are needed for the admission and recall experiments but not for the deterministic experiments or the test suite; to enable them, additionally `pip install "sentence-transformers>=3.0" "torch>=2.0"`.

## Layout

```
src/meld/
  proto/        vendored typed-link wire schema (MemoryUnit / MemoryLink / Claim / ...)
  protocol/     replicated objects, the fold with PASS certificates, the admission path
  merge/        the fold's step, signals (int8 sigma_q, NLI), conformal epochs, authored status, pair metrics
  sync/         binding onto a publish/subscribe transport; semantic-routing ablation
  eval/         convergence (CRDT order-independence, reject-gate) and robustness sweeps
  agent/        memory-delta metrics, token economics, the QA harness
  config/       signed epoch files (the scope's calibrated thresholds travel as an epoch record)
scripts/        runnable drivers for the experiments in the paper (testbed drivers in edge/ and cross-tier/)
tests/          the full test suite (run with pytest); tests/INVARIANTS.md maps each protocol invariant to its tests
release/study-archive/   result rows, signed epochs, calibration records and annotation records (see its README)
```

## Reproducing the experiments

The deterministic, model-free experiments reproduce end-to-end from this repository.

```bash
# Full test suite (unit tests + the public-safety hygiene guard)
PYTHONPATH=src pytest -q

# In-process status-CRDT convergence: order-independence + reject-gate placement.
PYTHONPATH=src python scripts/run_eval_c_inprocess.py
```

Every reported number is deterministic in its seed. There is no default operating point: a replica merges only under its scope's epoch, a signed record of per-bin split-conformal thresholds (`merge/conformal.py`); without one it makes no similarity merge.

### What is and is not reproducible from this repository

- **Reproducible here:** the order-independence and reject-gate demonstrations and the test suite, all deterministic and local.
- **Checkable from the study archive:** every reported number. `release/study-archive/` holds the result rows with their provenance stamps (code commit, encoder and NLI identifiers, signal-model and policy identities, epoch), the group-signed calibration epochs with their calibration records, the four annotation rounds and the audit, and the benchmark manifest with its build script. Its README maps each experiment of the paper to its directory; `sha256sum -c SHA256SUMS` verifies it. Source texts of the benchmark datasets are not redistributed; the build script regenerates them from the original datasets.
- **Requires the models:** the admission, recall and baseline runs take the encoder and NLI as required arguments and stamp every output row with the code and model provenance; the baselines' LLM calls need a served model.
- **Requires external infrastructure:** the sync-latency, routing and partition-heal results run on a multi-host 5G edge testbed. The protocol logic they exercise is the code here, and the drivers and run configurations needed to re-run them on comparable hardware are in `cross-tier/` and `edge/`.

## License

Code: Apache License 2.0; see `LICENSE`. Study archive: per file, as listed in `release/study-archive/LICENSES.md` (the study's own material CC BY 4.0; files derived from SICK CC BY-NC-SA 3.0; other dataset-derived files under their source licences).

## Citation

See `CITATION.cff`. Version 2.0.0 and later: DOI [10.5281/zenodo.24009240](https://doi.org/10.5281/zenodo.24009240) (all versions), version 2.0.0 [10.5281/zenodo.24009241](https://doi.org/10.5281/zenodo.24009241). Versions 1.0.0 to 1.0.6, of the earlier protocol: DOI [10.5281/zenodo.21878274](https://doi.org/10.5281/zenodo.21878274).
