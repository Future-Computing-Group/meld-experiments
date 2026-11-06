# MELD study archive

This directory holds the results, records and signed epochs behind the evaluation of the MELD paper and its supplement. Its layout follows the experiment inventory of the supplement (Appendix M): every path in the Outputs column of that table is a path under `outputs/` here. The reference implementation, the runners (`scripts/`, `edge/`, `cross-tier/`) and the tests are in the repository root; the commit stamped on each result row names the code that produced it.

## Verify

From this directory, `sha256sum -c SHA256SUMS` checks every file (on macOS, `shasum -a 256 -c SHA256SUMS`). Files ending in `.gz` were compressed with `gzip -n -9` and decompress with `gzip -dk`; the file name without `.gz` is the name the runner wrote.

Two directories carry stdlib-only re-derivations that recompute their headline numbers from the raw rows: `python3 -I outputs/wp6c-nd13/rederive.py outputs/wp6c-nd13/wp6c-nd13-tables.md` (104 values, 0 mismatches expected) and `python3 -I outputs/testbed-archive/2026-10-r1.4/rederive.py outputs/testbed-archive/2026-10-r1.4`. The first rewrites `rederived.json` in its directory and the second prints the same JSON to standard output; the shipped `rederived.json` of each equals that output byte for byte, so `sha256sum -c` still passes after a run.

`TRANSFORMS.tsv` lists every file whose bytes differ from what the runner wrote (texts removed, run-root prefixes of recorded paths removed, a host name redacted, records joined, or compression only), with the sha256 of the runner's output. Hashes that rows and reports cite (`labels_sha256`, `annotations_sha256`, `rows_sha256`, `pairs_sha256`) refer to those runner outputs, which held the pair texts; they can be checked after the build script has regenerated the texts.

## Inventory

| Experiment (Appendix M) | Path under `outputs/` | Contents |
|---|---|---|
| Annotation panel, four rounds | `annotate/` | Round 1 in `full/`, rounds 2 to 4 in `round2/` to `round4/`: pair-level annotations, final labels, agreement report, item-to-pair map, per-annotator raw records (`records/A1` to `A3`, gzipped, with prompt hashes) and adjudication records (`adjudication/A4`). Top level: combined annotations and labels of all rounds (`combined-annotations*.jsonl`, `labels-final*.jsonl`), including the labels with the auditor's S3 labels (`combined-annotations-humanS3.jsonl`, `full/labels-humanS3.jsonl`). `audit/`: audit sample with inclusion probabilities and weights, audited population, audit labels (200 pairs and 60 repeats), the S3 top-up sample and its 40 labels, and the 55 S3 re-check labels. `round4/spec-h4.json`: the world design of the round-4 calibration pool. |
| Calibration epochs (benchmark and recall workloads) | `calibration-labelled/` | Group-signed epochs (key fingerprint 133ffccfdf7b4c4d), `group-epochs.manifest.json` (epoch ids, body hashes) and calibration reports: `a0.02/`, `a0.05/`, `a0.10/` (PAWS, SICK and VitaminC; the VitaminC epochs use the auditor's S3 labels, report `calibration-report-vitaminc.json`), `humanS3-a0.02/` to `humanS3-a0.10/` (the same VitaminC epochs with their own manifest and report), `substrate-r4-a0.05-hotpot/` and `substrate-r4-a0.05-lme/` (HotpotQA and LongMemEval). All eleven epochs of the supplement's epoch table are here. |
| Admission on test streams, redraws, training-conditional variant | `wp6b/b3/` | `<domain>-a<alpha>/strict/benchmark-test-report.json.gz` (per-cell report, 200 calibration redraws, training-conditional variant, `rebuilt_is_given_epoch`) and `strict.log` for the nine cells. |
| Item re-splits | `resplit/` | `resplit.json`, `resplit-summary.txt` and the per-domain split item lists. |
| Candidate-set size m | `wp6e/ph13/` | `per-m.json`, `per-m.txt` and the evaluation epochs per m (signed with a local evaluation key, not the group key). |
| Multi-target admission (ablation) | `wp6g/` | `multi-target.json`, `multi-target.txt`. |
| Predicate diagnostic, group purity | `wp6d/runs/pairs-meld/`, `wp6d/runs/purity-b3/` | Pair rows, analysis and run metadata; purity benchmark report. |
| Signal-model selection | `tournament/` | Per-model probabilities on the PAWS dev, PAWS validation and SICK dev pairs (by pair index), metrics, run summary and the scoring and analysis scripts. |
| Portability grid and transfer | `wp6b/p1/` | Per encoder and NLI cell: `strict/` report and rows; the three cells calibrated with disclosed unlabelled pairs also carry `disclosed-<n>/` and `UNRESOLVED_DISCLOSED.txt`. |
| Calibration burden | `wp6b/p2/`, `wp6e/ph10/` | Burden rows and reports per encoder and domain (gzipped); card criterion `card-mstar.json`, `card-mstar.txt`. |
| Recalibration under shift | `wp6b/b4/` | Shift and degraded-signal arms: report and rows. |
| Delivery order; loss and anti-entropy | `wp6d/runs/convergence/` | Order shuffles on gold and the running example, lossy delivery with anti-entropy, running-example values. |
| Cross-hardware determinism | `h4-roihu-20261010/` | `h4-report.json` (14 cells: x86, Grace, GH200 by batch and thread setting) the pass-2 decision hashes per platform, and `cells-env.json`, the run environment and commit of each cell. |
| Cross-hardware determinism, own issuance | `h4-own-roihu-20261105/` | `h4own-report.json` (the same 14 cells, each issuing its own deltas; per workload and cell the int8 rows, delta ids, tie swaps, decisions, live units and fingerprints that differ from the x86 reference, with a `_summary` per workload family), `cells-decisions.json.gz` (claim keys, canonical order, first-pass decisions and fingerprint per cell and workload) and `cells-env.json`. |
| Independently evolving memories, star and all-to-all | `wp6c-nd13/n1-*/` | `n1-hotpot-star/`, `n1-lme-star/`, `n1-hotpot-a2a/`, `n1-lme-a2a/`: per-world rows. `wp6c-nd13-tables.md`, `rederive.py`, `rederived.json` cover the whole `wp6c-nd13/` directory. |
| Reader-only answer-F1 (equivalence margin) | `csc-m1-archive-rerun/` | Per-model rows of CSC job 7192177 (five readers, 1,000 rows each) the job log, and the driver as it ran (`run_hotpot_memdelta.py`; it imports the package as `smp`, the name at 3a444a0, so it runs against that tree). The job stamps commit b263f9e, whose tree is identical to 3a444a0. |
| Overlap and retrieval budget | `wp6c-nd13/s1/`, `wp6c-nd13/r2-lme/` | Overlap by k on HotpotQA (summary and gzipped rows) and the LongMemEval retrieval budget. |
| Typed-link value | `wp6c-nd13/e2t2/` | HotpotQA and LongMemEval results. |
| Mem0 fidelity | `mem0-fidelity/` | `mem0-fidelity.json` and `mem0-fidelity-rescored.json`. |
| Baselines on labelled pairs | `wp6b/bl/` | `<policy>/pairs/shard-NN.json` for Mem0 (update and full) and Graphiti over the labelled pairs. |
| Baselines on recall workloads | `wp6c-nd13/bl/` | Mem0, Graphiti and Caura per world (`nb-hotpot/`, `nb-lme/`) and by overlap (`r2-overlap-k-n200.json` with gzipped rows). Graphiti on LongMemEval shard 5 stopped on a context-length error; the excerpt `graphiti/nb-lme/shard-5-failure.txt` shows where. |
| MELD on labelled pairs | `wp6d/runs/pairs-meld/` | See the predicate diagnostic row. |
| Index scaling and admission agreement | `wp6/index/` | `synthetic-1m.json`, `hotpot-1m.json`, `lme.json`. |
| Bytes per message class | `wire-bytes.json`; `testbed-archive/2026-10-r1.4/` | Single-host bytes per message class (commit 0138e74); multi-host bytes in the testbed rows. |
| Per-delta admission cost | `wp6e/ph18b/` | `admission-cost.json` and per-delta rows for each run: `laptop-cpu/` (reported), `laptop-cpu-run1/` (the second laptop run), `gh200-cuda/` and `x86-cpu/`. |
| Multi-host latency, partition heal, throughput, cross-tier | `testbed-archive/2026-10-r1.4/` | Broadcast and semantic grids with their pilot cells, interest-level blocks A to C, partition and heal runs, smoke cells, the throughput knee, and `cross-tier/` (WAN partition heal and Kafka tiers, with the helper script as run). Hosts are labelled A to D. |

`bench/manifest.json.gz` is the benchmark manifest: every pair with its id, source item ids, stratum, label, label provenance and split, with the pair texts removed. `PYTHONPATH=src python -m meld.bench.build --data data/bench` (see `src/meld/bench/build.py` for the downloads it expects) rebuilds the full manifest from the original datasets; its canonical hash (`manifest_hash` in `src/meld/bench/manifest.py`) is 60694a496287644c6020009ee2b699b3fe6ad9f2c5fbf5ee3d78881a959462b6, the `manifest_sha256` that benchmark rows record.

## Provenance stamps

Over this archive, 627 result files carry 232,185 stamped rows (rows or cells with a `git_sha`), no value is non-finite, and each directory stamps the commit of its inventory row: bdb44ea (`wp6/index/`), 2efb97d (`wp6b/`), 70287aa (`resplit/`), 9572922 (`wp6e/ph13/`), 89b3637 (`wp6e/ph18b/`, `h4-own-roihu-20261105/`), 69d7280 (`wp6e/ph10/`), 22331dc (`wp6g/`), 78f0725 (`wp6d/runs/`), 9be8fee (`wp6c-nd13/`), 6e25d50 (`h4-roihu-20261010/`), 0138e74 (`wire-bytes.json`), b588bd7, bdb44ea and 62005af (`calibration-labelled/`), 36092c0 (`testbed-archive/`). Two further stamps: the interest-level block rows of `testbed-archive/` stamp 0ccbe83 (driver with the replication setting; brains and protocol at 36092c0), and `wp6d/runs/convergence/ph11-running-values.json` stamps 36092c0. The 21,957 portability, burden and recalibration stamps record `git_dirty: true` because the deployment wrote one untracked file naming the commit; the tracked tree was unmodified.

Paths that the cluster runs recorded in fields such as `labels_file`, `epoch_file` and `config` are given relative to the run root (for example `runs/wp6b/labels/labels-final.jsonl`); the files they name are the ones under `outputs/` with the same tail.

## Not included

Source texts of PAWS, SICK, VitaminC, LongMemEval, SciFact and HotpotQA: text fields are removed from labels, records and rows (or, for the reader rows, replaced by their sha256), and files that are mostly source text are left out (the per-cell cross-hardware dumps and their reference object set, which carry the benchmark sentences; the input envelopes of the throughput knee; the PAWS validation file of the signal-model selection). Keys and blinding secrets: the federation group key, the local evaluation key, the testbed run key, the annotation blinding files and the audit draw secret. LLM response caches, model weights, vLLM server logs, GPU utilisation samples, per-job stdout logs, status and completion markers, and job command files. Dev-signed copies of the group-signed epochs. Testbed host names and addresses. Only the runs the paper reports are included.
