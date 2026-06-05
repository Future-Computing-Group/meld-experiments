#!/bin/bash
#SBATCH --job-name=smp-m1
#SBATCH --output=%x-%j.out
#
# Generic M-1 memory-delta runner (the distributed-vs-centralized merge sweep).
# Submit via submit.sh, which passes --account/--partition/--gres/--time/--mem from
# site.env. This script reads the on-cluster paths from site.env and serves the open
# weights with Ollama (Q4, OpenAI-compatible endpoint); the runner talks to it over
# HTTP, so the LLM never enters the runner process. recall@k / footprint /
# contradiction-count are the model-independent headline; answer-F1 is the per-model
# corroboration. A provenance block is printed so the run reproduces from its own log.
set -uo pipefail

SITE_ENV="${SMP_SITE_ENV:-$(dirname "$0")/site.env}"
# shellcheck disable=SC1090
source "$SITE_ENV"
if [ -n "${SMP_SITE_PROFILE:-}" ]; then set +u; source "$SMP_SITE_PROFILE"; set -u; fi

R="$SMP_SCRATCH"
export OLLAMA_MODELS="$SMP_OLLAMA_ROOT/models"
export OLLAMA_HOST=127.0.0.1:11434
export LD_LIBRARY_PATH="$SMP_OLLAMA_ROOT/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$R/hf_home" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/tmp"
export PYTHONPATH="$SMP_REPO/src:${PYTHONPATH:-}"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$OLLAMA_MODELS"

export SMP_ST_DEVICE=cpu                  # encoder on CPU; the GPU is for serving
export SMP_TRANSPORT=ollama
export SMP_ENDPOINT=http://127.0.0.1:11434/v1/completions
export SMP_PER_LEVEL="${SMP_PER_LEVEL:-hard:200}"
export SMP_CONDITIONS="${SMP_CONDITIONS:-none,full_context,centralized,distributed_merge,naive_union}"
export SMP_MODES="${SMP_MODES:-fixed_recall}"
export SMP_BASELINE="${SMP_BASELINE:-react}"
export SMP_N_BRAINS="${SMP_N_BRAINS:-3}"
export SMP_OVERLAP="${SMP_OVERLAP:-0.5}"
export SMP_INJECT_CONFLICT="${SMP_INJECT_CONFLICT:-0}"
export SMP_MAX_STEPS="${SMP_MAX_STEPS:-8}"
export SMP_MAX_NEW_TOKENS="${SMP_MAX_NEW_TOKENS:-64}"   # HotpotQA answers are short
export SMP_K="${SMP_K:-5}"
export SMP_ENCODER="${SMP_ENCODER:-minilm}"
export OUT_DIR="$R/outputs"; mkdir -p "$OUT_DIR"

OLLAMA_BIN="$SMP_OLLAMA_ROOT/bin/ollama"
SMP_PY="${SMP_PY:-python}"

# ---- provenance (reproducible from the log) -------------------------------
echo "==== PROVENANCE jobid=${SLURM_JOB_ID:-NA} ===="
echo "[prov] node=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/[prov] gpu=/'
echo "[prov] ollama=$($OLLAMA_BIN --version 2>&1 | grep -oE 'version is [0-9.]+' | head -1)"
echo "[prov] git_sha=${SMP_GIT_SHA:-unknown}"
echo "[prov] models=${SMP_MODELS:-qwen2.5-7b mistral-7b} encoder=$SMP_ENCODER transport=$SMP_TRANSPORT"
echo "[prov] per_level=$SMP_PER_LEVEL conditions=$SMP_CONDITIONS N=$SMP_N_BRAINS overlap=$SMP_OVERLAP max_new=$SMP_MAX_NEW_TOKENS k=$SMP_K"
$SMP_PY -c "import sentence_transformers as s, numpy, sys; print(f'[prov] py={sys.version.split()[0]} sentence_transformers={s.__version__} numpy={numpy.__version__}')" 2>/dev/null
echo "========================================"

# ---- serve the open weights on the GPU ------------------------------------
$OLLAMA_BIN serve > "$OUT_DIR/ollama-serve-${SLURM_JOB_ID:-local}.log" 2>&1 &
OPID=$!
for _ in $(seq 1 40); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 3; done
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "[job] ERROR: serving endpoint did not come up"; kill $OPID 2>/dev/null; exit 1
fi
echo "[prov] models available:"; $OLLAMA_BIN list 2>&1 | sed 's/^/[prov]   /'

# ---- run the memory-delta grid per model (durable + resumable JSONL) ------
rc_total=0
for KEY in ${SMP_MODELS:-qwen2.5-7b mistral-7b}; do
  echo "[run] M-1 memory-delta grid for $KEY"
  SMP_MODEL_KEY=$KEY \
  SMP_OUT="$OUT_DIR/memdelta_${SMP_ENCODER}_$KEY.jsonl" \
  srun --overlap "$SMP_PY" "$SMP_REPO/scripts/run_memdelta.py"; rc=$?
  rc_total=$(( rc_total | rc ))
done

kill $OPID 2>/dev/null
echo "[job] done rc_total=$rc_total"
exit $rc_total
