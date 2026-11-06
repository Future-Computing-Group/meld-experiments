# Licences of the study archive

Each file in this archive carries one of the licences below. The licence of the code in the rest of the repository (Apache-2.0, see `LICENSE` at the repository root) is unchanged.

## Code: Apache-2.0

The scripts shipped inside the archive are code of this repository and are released under Apache-2.0: `outputs/wp6c-nd13/rederive.py`, `outputs/testbed-archive/2026-10-r1.4/rederive.py`, `outputs/testbed-archive/2026-10-r1.4/cross-tier/kafka_tiers.sh.used`, `outputs/csc-m1-archive-rerun/run_hotpot_memdelta.py`, `outputs/tournament/analyse.py`, `outputs/tournament/grid.py`, `outputs/tournament/score.py` and `outputs/tournament/run_all.sh`.

## Study material: CC BY 4.0

Every other file is study material of the MELD evaluation (annotation labels, audit labels, annotator and adjudication records, result rows, signed epochs and their manifests, calibration and test reports, the benchmark manifest of identifiers and labels, re-derivations and tables) and is released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/), file by file, except the SICK-derived files listed at the end, which keep the SICK licence.

The prompt strings recorded in `outputs/mem0-fidelity/` are the Mem0 prompts vendored in this repository (Apache-2.0, as recorded in `src/meld/baselines/_mem0_prompts.py`); the sentences in those files are from the constructed set S0 of this repository (Apache-2.0).

## Source datasets

No source text of any dataset is included: pair texts are removed from every label, record and result file, and the build script `src/meld/bench/build.py` regenerates them from the original datasets. Texts so regenerated, and pairs derived from a source, carry that source's licence:

| Source | Licence of the source and of pairs derived from it |
|---|---|
| PAWS | PAWS licence: may be freely used for any purpose; acknowledgement of Google LLC as the data source is appreciated. This archive acknowledges Google LLC as the data source of PAWS. |
| VitaminC | Wikipedia licence terms, else CC BY-SA 3.0 (VitaminC DATA_LICENSE); derived pairs CC BY-SA 3.0. |
| SciFact | Claims CC BY 4.0 (abstracts ODC-By 1.0, not used); derived pairs CC BY 4.0. |
| LongMemEval (cleaned) | MIT; derived pairs MIT. |
| SICK | CC BY-NC-SA 3.0 (Marelli et al., LREC 2014): non-commercial, share-alike; derived pairs and the files listed below CC BY-NC-SA 3.0. |
| HotpotQA (distractor) | CC BY-SA 4.0. |
| Constructed set S0 (MELD gold) | Apache-2.0, with the reference implementation. |

Source item identifiers (for example VitaminC page titles and HotpotQA question ids) appear in the files as identifiers only.

## SICK-derived files: CC BY-NC-SA 3.0

The files below contain labels, records or results keyed to SICK pairs (a SICK pair identifier, a blinded annotation item that maps to one, or a SICK source item id), hold per-pair results on the SICK dev pairs, or are SICK cells by name. They are released under CC BY-NC-SA 3.0 (https://creativecommons.org/licenses/by-nc-sa/3.0/). A directory entry means every file directly in that directory (not its subdirectories, which are listed on their own).

- `bench/manifest.json.gz`
- `outputs/annotate/` (all 6 files)
- `outputs/annotate/audit/audit-draw.jsonl`
- `outputs/annotate/audit/audit-labels.jsonl`
- `outputs/annotate/audit/audit-population.jsonl`
- `outputs/annotate/full/annotations.jsonl.gz`
- `outputs/annotate/full/item-to-pair.json`
- `outputs/annotate/full/labels-humanS3.jsonl`
- `outputs/annotate/full/labels.jsonl`
- `outputs/annotate/full/adjudication/item-to-pair.json`
- `outputs/annotate/full/adjudication/A4/` (all 36 files)
- `outputs/annotate/full/records/A1/` (all 146 files)
- `outputs/annotate/full/records/A2/` (all 146 files)
- `outputs/annotate/full/records/A3/` (all 146 files)
- `outputs/annotate/round3/annotations.jsonl`
- `outputs/annotate/round3/item-to-pair.json`
- `outputs/annotate/round3/labels.jsonl`
- `outputs/annotate/round3/adjudication/item-to-pair.json`
- `outputs/annotate/round3/adjudication/A4/j0001.json`
- `outputs/annotate/round3/adjudication/A4/j0002.json`
- `outputs/annotate/round3/records/A1/` (all 58 files)
- `outputs/annotate/round3/records/A2/` (all 58 files)
- `outputs/annotate/round3/records/A3/` (all 58 files)
- `outputs/calibration-labelled/a0.02/sick-seed-a0.02.group.epoch.json`
- `outputs/calibration-labelled/a0.05/sick-seed-a0.05.group.epoch.json`
- `outputs/calibration-labelled/a0.10/sick-seed-a0.10.group.epoch.json`
- `outputs/resplit/resplit-pairs-sick.json`
- `outputs/tournament/metrics-paws_dev.json`
- `outputs/tournament/metrics-paws_val.json`
- `outputs/tournament/probs-FacebookAI_roberta-large-mnli.json`
- `outputs/tournament/probs-Intel_bert-base-uncased-mrpc.json`
- `outputs/tournament/probs-cross-encoder_nli-deberta-v3-base.json`
- `outputs/tournament/probs-cross-encoder_nli-deberta-v3-large.json`
- `outputs/tournament/probs-cross-encoder_nli-deberta-v3-small.json`
- `outputs/tournament/probs-cross-encoder_nli-roberta-base.json`
- `outputs/tournament/probs-cross-encoder_quora-distilroberta-base.json`
- `outputs/tournament/probs-cross-encoder_quora-roberta-base.json`
- `outputs/tournament/probs-cross-encoder_quora-roberta-large.json`
- `outputs/tournament/probs-domenicrosati_deberta-v3-large-finetuned-paws-paraphrase-detector.json`
- `outputs/tournament/probs-semindan_paws_x_xlm_r_only_en.json`
- `outputs/wp6b/b3/sick-a0.02/strict.log`
- `outputs/wp6b/b3/sick-a0.02/strict/benchmark-test-report.json.gz`
- `outputs/wp6b/b3/sick-a0.05/strict.log`
- `outputs/wp6b/b3/sick-a0.05/strict/benchmark-test-report.json.gz`
- `outputs/wp6b/b3/sick-a0.10/strict.log`
- `outputs/wp6b/b3/sick-a0.10/strict/benchmark-test-report.json.gz`
- `outputs/wp6b/b4/a0.02/degraded-sick/strict/report.json`
- `outputs/wp6b/b4/a0.02/degraded-sick/strict/rows.jsonl`
- `outputs/wp6b/b4/a0.02/shift-paws-sick/strict/report.json`
- `outputs/wp6b/b4/a0.02/shift-paws-sick/strict/rows.jsonl`
- `outputs/wp6b/b4/a0.02/shift-sick-paws/strict/report.json`
- `outputs/wp6b/b4/a0.02/shift-sick-paws/strict/rows.jsonl`
- `outputs/wp6b/b4/a0.02/shift-sick-vitaminc/strict/report.json`
- `outputs/wp6b/b4/a0.02/shift-sick-vitaminc/strict/rows.jsonl`
- `outputs/wp6b/b4/a0.02/shift-vitaminc-sick/strict/report.json`
- `outputs/wp6b/b4/a0.02/shift-vitaminc-sick/strict/rows.jsonl`
- `outputs/wp6b/bl/graphiti/pairs/` (all 16 files)
- `outputs/wp6b/bl/mem0-full/pairs/` (all 16 files)
- `outputs/wp6b/bl/mem0-update/pairs/` (all 16 files)
- `outputs/wp6b/p2/a0.02/minilm-l12-sick/strict/report.json.gz`
- `outputs/wp6b/p2/a0.02/minilm-l12-sick/strict/rows.jsonl.gz`
- `outputs/wp6b/p2/a0.02/minilm-sick/strict/report.json.gz`
- `outputs/wp6b/p2/a0.02/minilm-sick/strict/rows.jsonl.gz`
- `outputs/wp6b/p2/a0.02/mpnet-sick/strict/report.json.gz`
- `outputs/wp6b/p2/a0.02/mpnet-sick/strict/rows.jsonl.gz`
- `outputs/wp6d/runs/pairs-meld/rows.jsonl.gz`
- `outputs/wp6e/ph13/epochs/sick-a0.02-m1.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.02-m10.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.02-m3.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.02-m5.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.05-m1.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.05-m10.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.05-m3.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.05-m5.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.10-m1.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.10-m10.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.10-m3.epoch.json`
- `outputs/wp6e/ph13/epochs/sick-a0.10-m5.epoch.json`
