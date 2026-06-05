#!/usr/bin/env bash
# SMP cross-tier deployment over REAL Apache Kafka (the standard fabric).
#
# Same laptop-hub topology as the relay orchestrator, but the transport is Kafka
# (the smp-kafka KRaft container on the laptop, :9092) instead of the stand-in relay,
# and brains compute over the standard fabric. Each remote tier reaches the broker
# through an `ssh -R 9092:localhost:9092` tunnel; because the broker advertises
# PLAINTEXT://localhost:9092, that same address resolves to each tier's tunnel, so a
# single advertised listener serves laptop + all tiers. CPU-only, short, self-tearing.
#
# Real targets from cross-tier/site.env (GITIGNORED). Same hub topology + teardown
# discipline as the relay orchestrator (tmux remote brains, keepalive tunnels,
# keep-awake, run-log, trap).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
source "$HERE/site.env"
: "${HPC_SSH:?}" "${HPC_DIR:?}" "${HPC_ENV:?}"
OUT="${OUT_DIR:-$REPO/outputs/cross-tier}"; mkdir -p "$OUT"
STAMP="$(date +%Y%m%d-%H%M%S)"; RESULT="$OUT/kafka-result-$STAMP.json"
RUN="$(date +%s)"; CLAIMS="smp-claims-$RUN"; ACKS="smp-acks-$RUN"
PIDS=(); LAUNCHED=()
cleanup() {
  echo "[teardown]"
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  for h in "${LAUNCHED[@]:-}"; do ssh "$h" 'tmux kill-session -t smp-kbrain 2>/dev/null || true' 2>/dev/null || true; done
  [ -n "${CAFF_PID:-}" ] && kill "$CAFF_PID" 2>/dev/null || true
}
trap cleanup EXIT

deploy_kafka_brain() {   # SSH DIR ENV ID
  local sa="$1" dir="$2" env="${3:-true}" id="$4"
  [ -z "$env" ] && env=true
  echo "  [$id] rsync Kafka brain bundle + vendored kafka-python -> $sa:$dir"
  ssh "$sa" "mkdir -p '$dir/src' '$dir/cross-tier/vendor'"
  rsync -az --delete "$REPO/src/smp" "$sa:$dir/src/"
  rsync -az "$HERE/kafka_brain.py" "$HERE/kafka_transport.py" "$HERE/brain_core.py" "$sa:$dir/cross-tier/"
  rsync -az "$HERE/vendor/kafka" "$sa:$dir/cross-tier/vendor/"
  echo "  [$id] reverse tunnel -> $sa (remote :9092 -> laptop :9092)"
  ssh -N -R 9092:localhost:9092 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes "$sa" &
  PIDS+=($!); sleep 3
  echo "  [$id] launch brain in tmux (vendored kafka-python on PYTHONPATH; no remote pip)"
  local rcmd="cd '$dir'; set +u; $env; set -u; PYTHONPATH=src:cross-tier/vendor python3 cross-tier/kafka_brain.py --bootstrap localhost:9092 --id $id --claims-topic $CLAIMS --acks-topic $ACKS --log '$dir/kbrain-$id.jsonl' --run-seconds 150"
  ssh "$sa" "tmux kill-session -t smp-kbrain 2>/dev/null || true; tmux new-session -d -s smp-kbrain bash -c \"$rcmd\""
  LAUNCHED+=("$sa")
  local _i; for _i in $(seq 1 40); do
    ssh "$sa" "grep -q '\"ev\": \"ready\"' '$dir/kbrain-$id.jsonl' 2>/dev/null" && { echo "    $id brain ready (Kafka connected)"; return 0; }
    sleep 1
  done
  echo "    WARN: $id brain did not report ready (check kafka-python install / tunnel)"
}

echo "[1] caffeinate + ensure local Kafka"
command -v caffeinate >/dev/null && { caffeinate -s & CAFF_PID=$!; }
docker ps --filter name=smp-kafka --format '{{.Names}}' | grep -q smp-kafka || { echo "FATAL: smp-kafka container not running"; exit 2; }

echo "[2] HPC tier over Kafka"
deploy_kafka_brain "$HPC_SSH" "$HPC_DIR" "$HPC_ENV" "hpc"
BRAINS="local,hpc"

if [ -n "${EDGE_SSH:-}" ]; then
  echo "[3] EDGE tier over Kafka"
  deploy_kafka_brain "$EDGE_SSH" "${EDGE_DIR:?}" "${EDGE_ENV:-true}" "edge"
  BRAINS="local,hpc,edge"
fi

echo "[4] local brain over Kafka (venv python)"
PYTHONPATH="$REPO/src:$HERE" "$REPO/.venv/bin/python" "$HERE/kafka_brain.py" --bootstrap localhost:9092 --id local \
  --claims-topic "$CLAIMS" --acks-topic "$ACKS" --log "$OUT/kbrain-local-$STAMP.jsonl" --run-seconds 150 &
PIDS+=($!); sleep 3

echo "[5] driver over Kafka, brains=$BRAINS -> $RESULT"
PYTHONPATH="$REPO/src:$HERE" "$REPO/.venv/bin/python" "$HERE/kafka_xtier.py" drive --brains "$BRAINS" \
  --claims-topic "$CLAIMS" --acks-topic "$ACKS" --k "${LAT_K:-100}" --out "$RESULT"

echo "[6] done -> $RESULT"
echo "$(date -u +%FT%TZ) kafka cross-tier run $STAMP brains=$BRAINS result=$RESULT" >> "$OUT/run-log.txt"
