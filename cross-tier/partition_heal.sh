#!/usr/bin/env bash
# MELD cross-tier PARTITION-HEAL on real infrastructure (the C-1 result, upgraded
# from a homogeneous LAN to the real continuum). Laptop hub + local brain + the
# EDGE brain. Sequence: baseline -> drop the edge tunnel (partition) ->
# overrule (edge misses it: DIVERGED) -> re-open the tunnel (heal; the brain's
# reconnect loop re-SUBs, retaining its CRDT state) -> re-deliver -> RECONVERGED.
# Self-tearing-down. Real targets from cross-tier/site.env (GITIGNORED).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
source "$HERE/site.env"
: "${BROKER_PORT:?}" "${TUN_PORT:?}" "${EDGE_SSH:?}" "${EDGE_DIR:?}"
OUT="${OUT_DIR:-$REPO/outputs/cross-tier}"; mkdir -p "$OUT"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOCAL_LOG="$OUT/ph-brain-local-$STAMP.jsonl"
EDGE_LOG="$EDGE_DIR/brain-edge-ph.jsonl"
PIDS=(); TUN_PID=""
cleanup() {
  echo "[teardown]"
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  ssh "$EDGE_SSH" 'tmux kill-session -t meld-ph 2>/dev/null || true' 2>/dev/null || true
  [ -n "${CAFF_PID:-}" ] && kill "$CAFF_PID" 2>/dev/null || true
}
trap cleanup EXIT

pub() { python3 "$HERE/probe_client.py" pub 127.0.0.1 "$BROKER_PORT" claims "$1" >/dev/null; }
last_c2() {  # $1 = local|edge -> prints the last c2 effective status
  local line
  if [ "$1" = local ]; then
    line=$(grep '"claim_id": "c2"' "$LOCAL_LOG" 2>/dev/null | grep '"ev": "merge"' | tail -1 || true)
  else
    line=$(ssh "$EDGE_SSH" "grep '\"claim_id\": \"c2\"' '$EDGE_LOG' 2>/dev/null | grep '\"ev\": \"merge\"' | tail -1" || true)
  fi
  echo "$line" | sed -n 's/.*"eff_status": "\([A-Z]*\)".*/\1/p'
}
open_tunnel() {
  ssh -N -R "${TUN_PORT}:localhost:${BROKER_PORT}" -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$EDGE_SSH" & TUN_PID=$!
  PIDS+=("$TUN_PID"); sleep 3
}

command -v caffeinate >/dev/null && { caffeinate -s & CAFF_PID=$!; }
echo "[setup] broker + edge deploy + tunnel + brains"
python3 "$HERE/probe_broker.py" "$BROKER_PORT" >"$OUT/ph-broker-$STAMP.log" 2>&1 & PIDS+=($!); sleep 0.5
ssh "$EDGE_SSH" "mkdir -p '$EDGE_DIR/src' '$EDGE_DIR/cross-tier'"
rsync -az --delete "$REPO/src/meld" "$EDGE_SSH:$EDGE_DIR/src/"
rsync -az "$HERE/brain.py" "$HERE/probe_client.py" "$EDGE_SSH:$EDGE_DIR/cross-tier/"
open_tunnel
ssh "$EDGE_SSH" "tmux kill-session -t meld-ph 2>/dev/null || true; tmux new-session -d -s meld-ph bash -c \"cd '$EDGE_DIR'; set +u; ${EDGE_ENV:-true}; set -u; PYTHONPATH=src python3 cross-tier/brain.py --broker-host 127.0.0.1 --broker-port $TUN_PORT --id edge --log '$EDGE_LOG'\""
PYTHONPATH="$REPO/src" python3 "$HERE/brain.py" --broker-host 127.0.0.1 --broker-port "$BROKER_PORT" \
  --id local --log "$LOCAL_LOG" & PIDS+=($!)
sleep 4

echo "[1 baseline] deprecate c2"
pub '{"nonce": "d0", "op": "deprecate", "claim_id": "c2", "link_id": "L_dep"}'; sleep 1.5
echo "    local=$(last_c2 local)  edge=$(last_c2 edge)   (expect DEPRECATED / DEPRECATED)"

echo "[2 partition] drop the EDGE tunnel (kill $TUN_PID)"
kill "$TUN_PID" 2>/dev/null || true; sleep 4
echo "[3 overrule during partition] local applies it; edge is cut off"
pub '{"nonce": "o1", "op": "overrule", "claim_id": "c2", "link_id": "L_ovr"}'; sleep 1.5
L=$(last_c2 local); E=$(last_c2 edge)
echo "    local=$L  edge=$E"
[ "$L" = OVERRULED ] && [ "$E" = DEPRECATED ] && echo "    >>> DIVERGED (edge missed the overrule) ✓" || echo "    >>> divergence not as expected"

echo "[4 heal] re-open the EDGE tunnel; the brain's reconnect loop re-SUBs (state retained)"
open_tunnel
for _ in $(seq 1 24); do
  n=$(ssh "$EDGE_SSH" "grep -c '\"ev\": \"ready\"' '$EDGE_LOG' 2>/dev/null || echo 0")
  [ "${n:-0}" -ge 2 ] && { echo "    edge reconnected (ready x$n)"; break; }
  sleep 0.5
done

echo "[5 re-deliver] overrule again; grow-only set is idempotent -> edge converges"
pub '{"nonce": "o1", "op": "overrule", "claim_id": "c2", "link_id": "L_ovr"}'; sleep 1.5
L=$(last_c2 local); E=$(last_c2 edge)
echo "    local=$L  edge=$E"
if [ "$L" = OVERRULED ] && [ "$E" = OVERRULED ]; then
  echo "    >>> RECONVERGED (order-independent, no coordinator) ✓"; RC=PASS
else
  echo "    >>> RECONVERGENCE FAILED"; RC=FAIL
fi
echo "$(date -u +%FT%TZ) partition-heal $STAMP local=$L edge=$E -> $RC" >> "$OUT/run-log.txt"
