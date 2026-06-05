#!/bin/bash
# Submit the M-1 memory-delta model-family sweep to SLURM.
# Prerequisite: copy site.example.env to site.env and fill it in for your site.
#
# The SLURM account/partition/resources come from site.env (passed on the sbatch
# command line); the runner reads the on-cluster paths from site.env via --export.
set -euo pipefail
cd "$(dirname "$0")"

SITE_ENV="${SMP_SITE_ENV:-site.env}"
if [ ! -f "$SITE_ENV" ]; then
  echo "ERROR: $SITE_ENV not found. Copy site.example.env to site.env and fill it in." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$SITE_ENV"

GIT_SHA="$(git -C "${SMP_REPO:-.}" rev-parse --short HEAD 2>/dev/null || echo unknown)"

sbatch \
  --account="$SMP_ACCOUNT" \
  --partition="$SMP_PARTITION" \
  --gres="$SMP_GRES" \
  --time="$SMP_TIME" \
  --mem="$SMP_MEM" \
  --cpus-per-task="$SMP_CPUS" \
  --export=ALL,SMP_SITE_ENV="$PWD/$SITE_ENV",SMP_GIT_SHA="$GIT_SHA" \
  run_memdelta.sbatch
