#!/usr/bin/env bash
# SMP cross-tier continuum run — laptop-hub orchestrator (reproducible artifact).
#
# Topology: the laptop hosts the broker + a local brain + the SSH reverse tunnels;
# each remote tier runs a torch-free brain that reaches the broker through its
# loopback tunnel port. CPU-only, no GPU, <=~15 min active. The whole run lives in
# ONE process tree with `trap cleanup EXIT`, fully self-tearing-down.
#
# Remote orchestration: the laptop is forced into the hub role by topology (the edge
# node accepts no inbound connections; the HPC node is firewalled). Mitigations:
# remote brains in tmux, tunnel keepalive, keep-awake, SHORT run, a per-host run-log
# line. Both remotes are driven over SSH (HPC key-based; edge after interactive auth).
#
# Real targets come from cross-tier/site.env (GITIGNORED). Never hardcode.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CONF="$HERE/site.env"
[ -f "$CONF" ] || { echo "FATAL: $CONF missing (cp site.example.env site.env; fill targets)"; exit 2; }
# shellcheck disable=SC1090
source "$CONF"
: "${BROKER_PORT:?}" "${TUN_PORT:?}" "${HPC_SSH:?}" "${HPC_DIR:?}" "${HPC_ENV:?}"
OUT="${OUT_DIR:-$REPO/outputs/cross-tier}"; mkdir -p "$OUT"
STAMP="$(date +%Y%m%d-%H%M%S)"; RESULT="$OUT/result-$STAMP.json"
PIDS=(); LAUNCHED=()
cleanup() {
  echo "[teardown]"
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  for h in "${LAUNCHED[@]:-}"; do ssh "$h" 'tmux kill-session -t smp-brain 2>/dev/null || true' 2>/dev/null || true; done
  [ -n "${CAFF_PID:-}" ] && kill "$CAFF_PID" 2>/dev/null || true
}
trap cleanup EXIT

# deploy_and_launch SSH DIR ENV ID — rsync the bundle, open a reverse tunnel, verify
# reachability, launch the brain in tmux, wait for it to register (SUB ready).
deploy_and_launch() {
  local sa="$1" dir="$2" env="${3:-true}" id="$4"
  [ -z "$env" ] && env=true
  echo "  [$id] rsync bundle -> $sa:$dir (code delivery = rsync)"
  ssh "$sa" "mkdir -p '$dir/src' '$dir/cross-tier'"
  rsync -az --delete "$REPO/src/smp" "$sa:$dir/src/"
  rsync -az "$HERE/brain.py" "$HERE/probe_client.py" "$sa:$dir/cross-tier/"
  echo "  [$id] reverse tunnel -> $sa (remote :$TUN_PORT -> laptop :$BROKER_PORT)"
  local opts=(-N -R "${TUN_PORT}:localhost:${BROKER_PORT}" -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes)
  if command -v autossh >/dev/null; then autossh "${opts[@]}" "$sa" & else ssh "${opts[@]}" "$sa" & fi
  PIDS+=($!); sleep 3
  echo "  [$id] verify tunnel (PING through loopback)"
  ssh "$sa" "set +u; $env; set -u; python3 '$dir/cross-tier/probe_client.py' ping 127.0.0.1 $TUN_PORT" \
    | grep -q PONG && echo "    $id tunnel OK" || { echo "FATAL: $id cannot reach the broker"; exit 3; }
  echo "  [$id] launch brain in tmux"
  local rcmd="cd '$dir'; set +u; $env; set -u; PYTHONPATH=src python3 cross-tier/brain.py --broker-host 127.0.0.1 --broker-port $TUN_PORT --id $id --log '$dir/brain-$id.jsonl'"
  ssh "$sa" "tmux kill-session -t smp-brain 2>/dev/null || true; tmux new-session -d -s smp-brain bash -c \"$rcmd\""
  LAUNCHED+=("$sa")
  local _i; for _i in $(seq 1 20); do
    ssh "$sa" "grep -q '\"ev\": \"ready\"' '$dir/brain-$id.jsonl' 2>/dev/null" && { echo "    $id brain ready"; return 0; }
    sleep 0.5
  done
  echo "    WARN: $id brain did not report ready in time"
}

echo "[1] caffeinate"
command -v caffeinate >/dev/null && { caffeinate -s & CAFF_PID=$!; }

echo "[2] broker on laptop loopback :$BROKER_PORT"
python3 "$HERE/probe_broker.py" "$BROKER_PORT" >"$OUT/broker-$STAMP.log" 2>&1 & PIDS+=($!)
sleep 0.5

echo "[3] HPC tier"
deploy_and_launch "$HPC_SSH" "$HPC_DIR" "$HPC_ENV" "hpc"
BRAINS="local,hpc"

if [ -n "${EDGE_SSH:-}" ]; then
  echo "[4] EDGE tier"
  deploy_and_launch "$EDGE_SSH" "${EDGE_DIR:?}" "${EDGE_ENV:-true}" "edge"
  BRAINS="local,hpc,edge"
fi

echo "[5] local brain on the laptop"
PYTHONPATH="$REPO/src" python3 "$HERE/brain.py" --broker-host 127.0.0.1 --broker-port "$BROKER_PORT" \
  --id local --log "$OUT/brain-local-$STAMP.jsonl" & PIDS+=($!)
sleep 2

echo "[6] driver (latency + running example), brains=$BRAINS -> $RESULT"
PYTHONPATH="$REPO/src" python3 "$HERE/driver.py" --broker-host 127.0.0.1 --broker-port "$BROKER_PORT" \
  --brains "$BRAINS" --k "${LAT_K:-100}" --out "$RESULT"

echo "[7] done -> $RESULT"
echo "$(date -u +%FT%TZ) cross-tier run $STAMP brains=$BRAINS result=$RESULT" >> "$OUT/run-log.txt"
