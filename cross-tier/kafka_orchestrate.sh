#!/usr/bin/env bash
# MELD cross-tier deployment over real Apache Kafka: latency probes + the running example.
#
# Laptop hub (a KRaft broker container on localhost:$KAFKA_PORT) + a local brain + an EDGE-tier
# brain + an optional HPC-tier brain (cross-tier/kafka_tiers.sh: a remote tier gets the bundle by
# tar pipe and reaches the broker through `ssh -R`; a tier whose host is `local` runs on the
# laptop). The driver (kafka_xtier.py drive) publishes the signed epoch first, then signed probe
# deltas and the running example, under the brains' encoder and NLI model.
#
# Env (site.env or the environment): EDGE_SSH (alias | local), EDGE_DIR, EDGE_ENV; HPC_SSH
# (optional), HPC_DIR, HPC_ENV; KAFKA_PORT (9092); KAFKA_CONTAINER (meld-kafka); ENCODER/NLI
# (hashing/lexical); EPOCH_FILE; KEY_FILE; LAT_K (100).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
[ -f "$HERE/site.env" ] && [ -z "${NO_SITE_ENV:-}" ] && source "$HERE/site.env"
: "${EDGE_SSH:?}" "${KEY_FILE:?}" "${EPOCH_FILE:?}"
HPC_SSH="${HPC_SSH:-}"
PORT="${KAFKA_PORT:-9092}"; KC="${KAFKA_CONTAINER:-meld-kafka}"
ENCODER="${ENCODER:-hashing}"; NLI="${NLI:-lexical}"
PY="${PY:-$REPO/.venv/bin/python}"
OUT="${OUT_DIR:-$REPO/outputs/cross-tier}"; mkdir -p "$OUT"
STAMP="$(date +%Y%m%d-%H%M%S)"; RESULT="$OUT/kafka-result-$STAMP.json"
RUN="$(date +%s)"; CLAIMS="meld-claims-$RUN"; ACKS="meld-acks-$RUN"
SMP_GROUP_KEY="$(tr -d '[:space:]' < "$KEY_FILE")"; export SMP_GROUP_KEY
TIERS=(edge local); [ -n "$HPC_SSH" ] && TIERS=(edge hpc local)
BRAIN_PIDS=(); TUN_PIDS=(); EDGE_TUN=""; EDGE_PID=""; CAFF_PID=""; LAUNCHED=()
source "$HERE/kafka_tiers.sh"

command -v caffeinate >/dev/null && { caffeinate -s & CAFF_PID=$!; }
docker ps --filter "name=$KC" --format '{{.Names}}' | grep -q "$KC" || { echo "FATAL: broker container $KC not running"; exit 2; }
echo "[1] tiers: ${TIERS[*]} (edge=$(host_of edge) hpc=${HPC_SSH:-none}) encoder=$ENCODER nli=$NLI broker=localhost:$PORT"
for t in "${TIERS[@]}"; do start_brain "$t"; done
for t in "${TIERS[@]}"; do wait_ready "$t" || { echo "FATAL: $t brain not ready"; exit 3; }; done
BRAINS="$(IFS=,; echo "${TIERS[*]}")"
echo "[2] driver over Kafka, brains=$BRAINS -> $RESULT"
SMP_KAFKA_BOOTSTRAP="localhost:$PORT" PYTHONPATH="$REPO/src:$HERE" "$PY" "$HERE/kafka_xtier.py" drive \
  --brains "$BRAINS" --claims-topic "$CLAIMS" --acks-topic "$ACKS" --k "${LAT_K:-100}" \
  --epoch "$EPOCH_FILE" --encoder "$ENCODER" --nli "$NLI" --out "$RESULT"
echo "[3] done -> $RESULT"
echo "$(date -u +%FT%TZ) kafka cross-tier run $STAMP brains=$BRAINS encoder=$ENCODER nli=$NLI result=$RESULT" >> "$OUT/run-log.txt"
