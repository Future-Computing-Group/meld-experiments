# HPC runner — M-1 memory-delta model-family sweep

Generic, site-agnostic SLURM scaffolding for the headline distributed-versus-centralized
merge sweep (five open-weight models served with Ollama). All site-specific values
(account, partition, scratch paths, modules) live in a gitignored `site.env`; this tree
ships only the generic logic and a sanitized `site.example.env`.

## Run

```bash
cd hpc
cp site.example.env site.env        # site.env is gitignored
$EDITOR site.env                    # fill in account, partition, paths
./submit.sh                         # submits run_memdelta.sbatch via SLURM
```

`submit.sh` reads `site.env` and passes the SLURM account/partition/resources on the
`sbatch` command line; `run_memdelta.sbatch` reads the on-cluster paths from `site.env`,
serves the weights with Ollama on the GPU, and runs `scripts/run_memdelta.py` (the
durable, resumable, env-driven runner) once per model. Each run prints a provenance
block and writes a durable JSONL under `$SMP_SCRATCH/outputs/`; re-submitting resumes
from already-scored rows. The runner aborts FATAL if more than 10% of rows carry an
error, so a serving failure cannot masquerade as all-zero data.

## Prerequisites on the cluster

- An Ollama install under `$SMP_OLLAMA_ROOT` (`bin/`, `lib/`) with the model tags in
  `SMP_MODELS` pulled into its model store.
- A Python (`SMP_PY`) with this repo's dependencies (the MiniLM sentence encoder + numpy);
  see the top-level `requirements.txt` plus `sentence-transformers`.
- The repository checked out at `$SMP_REPO`.
