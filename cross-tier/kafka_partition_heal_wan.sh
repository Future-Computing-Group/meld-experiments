#!/usr/bin/env bash
# M3a: seeded MULTI-TRIAL cross-tier partition-heal over REAL Apache Kafka WAN.
#
# Laptop hub (meld-kafka KRaft :9092) + local brain + an HPC-tier brain + an edge-tier brain
# (real targets in gitignored site.env), each reaching the broker via `ssh -R 9092:localhost:9092`.
# Per trial: deprecate a fresh claim (all three converge) -> PARTITION by dropping the
# EDGE tunnel (no delivery to edge) -> overrule (local+hpc apply it; edge is cut off ->
# DIVERGED) -> HEAL by reopening the EDGE tunnel (the brain re-consumes the missed offset
# from its grow-only set -> RECONVERGED). N seeded trials => reconvergence-success
# fraction + heal-latency p50/p95 distribution (the quantitative form M3a asks for, vs the
# single-session existence proof). Self-tearing-down; remote brains also auto-exit
# (--run-seconds) so a hard kill self-heals. Real targets from site.env (GITIGNORED).
#
#   TRIALS=8 bash cross-tier/kafka_partition_heal_wan.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
source "$HERE/site.env"
: "${HPC_SSH:?}" "${HPC_DIR:?}" "${EDGE_SSH:?}" "${EDGE_DIR:?}"
N="${TRIALS:-8}"
PY="$REPO/.venv/bin/python"
OUT="${OUT_DIR:-$REPO/outputs/cross-tier}"; mkdir -p "$OUT"
STAMP="$(date +%Y%m%d-%H%M%S)"; RUN="$(date +%s)"
CLAIMS="meld-phwan-claims-$RUN"; ACKS="meld-phwan-acks-$RUN"
LOCAL_LOG="$OUT/phwan-local-$STAMP.jsonl"
HPC_LOG="$HPC_DIR/phwan-hpc.jsonl"; EDGE_LOG="$EDGE_DIR/phwan-edge.jsonl"
TRIALS_JSON="$OUT/m3a-wan-trials-$STAMP.jsonl"; : > "$TRIALS_JSON"
RESULT="$OUT/m3a-wan-$STAMP.json"
LOCAL_BRAIN=""; HPC_TUN=""; EDGE_TUN=""; CAFF_PID=""

cleanup() {
  echo "[teardown]"
  [ -n "$EDGE_TUN" ] && kill "$EDGE_TUN" 2>/dev/null || true
  [ -n "$HPC_TUN" ]  && kill "$HPC_TUN"  2>/dev/null || true
  [ -n "$LOCAL_BRAIN" ] && kill "$LOCAL_BRAIN" 2>/dev/null || true
  ssh "$HPC_SSH"  'tmux kill-session -t meld-phwan 2>/dev/null || true' 2>/dev/null || true
  ssh "$EDGE_SSH" 'tmux kill-session -t meld-phwan 2>/dev/null || true' 2>/dev/null || true
  [ -n "$CAFF_PID" ] && kill "$CAFF_PID" 2>/dev/null || true
}
trap cleanup EXIT

open_tun() {  # host  -> sets global TUN_PID (no sleep; caller waits on readiness)
  ssh -N -R 9092:localhost:9092 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes "$1" & TUN_PID=$!
}
deploy_brain() {  # host dir env id log
  local h="$1" d="$2" e="${3:-true}" id="$4" lg="$5"; [ -z "$e" ] && e=true
  echo "  [$id] deploy bundle -> $h:$d"
  ssh "$h" "mkdir -p '$d/src' '$d/cross-tier/vendor'"
  rsync -az --delete "$REPO/src/meld" "$h:$d/src/"
  rsync -az "$HERE/kafka_brain.py" "$HERE/kafka_transport.py" "$HERE/brain_core.py" "$h:$d/cross-tier/"
  rsync -az "$HERE/vendor/kafka" "$h:$d/cross-tier/vendor/"
  ssh "$h" "tmux kill-session -t meld-phwan 2>/dev/null || true; tmux new-session -d -s meld-phwan bash -c \"cd '$d'; set +u; $e; set -u; PYTHONPATH=src:cross-tier/vendor python3 cross-tier/kafka_brain.py --bootstrap localhost:9092 --id $id --claims-topic $CLAIMS --acks-topic $ACKS --log '$lg' --run-seconds 900\""
}
wait_ready_remote() {  # host log
  local i; for i in $(seq 1 40); do
    ssh "$1" "grep -q '\"ev\": \"ready\"' '$2' 2>/dev/null" && return 0; sleep 1; done; return 1
}
pub() {  # payload-json -> publish to the claims topic on the laptop broker
  CLAIMS_TOPIC="$CLAIMS" PAYLOAD="$1" PYTHONPATH="$REPO/src:$HERE" "$PY" - <<'PYEOF'
import os
from kafka_transport import KafkaTransport
t = KafkaTransport("localhost:9092")
t.publish(os.environ["CLAIMS_TOPIC"], os.environ["PAYLOAD"]); t.close()
PYEOF
}
last_status() {  # tier(local|hpc|edge) cid -> latest eff_status for cid
  local line
  case "$1" in
    local) line=$(grep "\"claim_id\": \"$2\"" "$LOCAL_LOG" 2>/dev/null | grep '"ev": "merge"' | tail -1 || true);;
    hpc)   line=$(ssh "$HPC_SSH"  "grep '\"claim_id\": \"$2\"' '$HPC_LOG'  2>/dev/null | grep '\"ev\": \"merge\"' | tail -1" 2>/dev/null || true);;
    edge)  line=$(ssh "$EDGE_SSH" "grep '\"claim_id\": \"$2\"' '$EDGE_LOG' 2>/dev/null | grep '\"ev\": \"merge\"' | tail -1" 2>/dev/null || true);;
  esac
  echo "$line" | sed -n 's/.*"eff_status": "\([A-Z]*\)".*/\1/p'
}
wait_status() {  # tier cid want timeout_s   (returns 0 if reached)
  local end=$(( $(date +%s) + $4 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    [ "$(last_status "$1" "$2")" = "$3" ] && return 0; sleep 0.4; done
  return 1
}

command -v caffeinate >/dev/null && { caffeinate -s & CAFF_PID=$!; }
docker ps --filter name=meld-kafka --format '{{.Names}}' | grep -q meld-kafka || { echo "FATAL: meld-kafka not running"; exit 2; }

echo "[setup] deploy HPC + EDGE brains over Kafka; start local brain; open tunnels"
open_tun "$HPC_SSH";  HPC_TUN="$TUN_PID";  sleep 3
deploy_brain "$HPC_SSH" "$HPC_DIR" "$HPC_ENV" "hpc" "$HPC_LOG"
open_tun "$EDGE_SSH"; EDGE_TUN="$TUN_PID"; sleep 3
deploy_brain "$EDGE_SSH" "$EDGE_DIR" "${EDGE_ENV:-true}" "edge" "$EDGE_LOG"
PYTHONPATH="$REPO/src:$HERE" "$PY" "$HERE/kafka_brain.py" --bootstrap localhost:9092 --id local \
  --claims-topic "$CLAIMS" --acks-topic "$ACKS" --log "$LOCAL_LOG" --run-seconds 900 & LOCAL_BRAIN=$!
wait_ready_remote "$HPC_SSH" "$HPC_LOG"  || { echo "FATAL: hpc brain not ready"; exit 3; }
wait_ready_remote "$EDGE_SSH" "$EDGE_LOG" || { echo "FATAL: edge brain not ready"; exit 3; }
sleep 3
echo "[setup] all brains ready"

ok=0
for i in $(seq 1 "$N"); do
  cid="c2-$i"
  echo "[trial $i/$N] claim=$cid"
  # [1] baseline: deprecate; all three converge
  pub "{\"nonce\": \"d-$i\", \"op\": \"deprecate\", \"claim_id\": \"$cid\", \"link_id\": \"Ld-$i\"}"
  wait_status local "$cid" DEPRECATED 20 || true
  wait_status hpc  "$cid" DEPRECATED 20 || true
  wait_status edge "$cid" DEPRECATED 20 || true
  # [2] partition: drop the EDGE tunnel
  kill "$EDGE_TUN" 2>/dev/null || true; EDGE_TUN=""; sleep 3
  # [3] overrule during partition: local+hpc apply, edge cut off
  pub "{\"nonce\": \"o-$i\", \"op\": \"overrule\", \"claim_id\": \"$cid\", \"link_id\": \"Lo-$i\"}"
  peers_ok=1
  wait_status local "$cid" OVERRULED 25 || peers_ok=0
  wait_status hpc  "$cid" OVERRULED 25 || peers_ok=0
  edge_now="$(last_status edge "$cid")"
  diverged_first=false
  [ "$peers_ok" = 1 ] && [ "$edge_now" = DEPRECATED ] && diverged_first=true
  echo "    diverged_first=$diverged_first (edge=$edge_now while peers OVERRULED)"
  # [4] heal: reopen the EDGE tunnel; measure to reconvergence
  t0=$(date +%s.%N)
  open_tun "$EDGE_SSH"; EDGE_TUN="$TUN_PID"
  if wait_status edge "$cid" OVERRULED 60; then converged=true; else converged=false; fi
  t1=$(date +%s.%N)
  heal_ms=$("$PY" -c "import sys;print(round((float(sys.argv[2])-float(sys.argv[1]))*1000,2))" "$t0" "$t1")
  $converged && ok=$((ok+1))
  echo "    converged=$converged heal_ms=$heal_ms"
  printf '{"seed": %d, "converged": %s, "heal_ms": %s, "diverged_first": %s}\n' \
    "$i" "$converged" "$heal_ms" "$diverged_first" >> "$TRIALS_JSON"
done

echo "[aggregate] $ok/$N reconverged -> $RESULT"
TRIALS_JSON="$TRIALS_JSON" RESULT="$RESULT" BOOT="laptop:9092 (WAN: edge + HPC tiers)" \
PYTHONPATH="$HERE" "$PY" - <<'PYEOF'
import json, os
from kafka_partition_heal import aggregate_trials
trials = [json.loads(l) for l in open(os.environ["TRIALS_JSON"]) if l.strip()]
agg = aggregate_trials(trials); agg["trials"] = trials; agg["transport"] = os.environ["BOOT"]
open(os.environ["RESULT"], "w").write(json.dumps(agg, indent=2))
print(json.dumps(agg, indent=2))
PYEOF
echo "$(date -u +%FT%TZ) m3a-wan $STAMP trials=$N reconverged=$ok result=$RESULT" >> "$OUT/run-log.txt"
