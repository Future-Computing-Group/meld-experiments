#!/usr/bin/env bash
# Seeded multi-trial cross-tier partition-heal over real Apache Kafka.
#
# Laptop hub (a KRaft broker container, published on localhost:$KAFKA_PORT) + a local brain + an
# EDGE-tier brain + an optional HPC-tier brain. A remote tier reaches the broker through
# `ssh -R $KAFKA_PORT:localhost:$KAFKA_PORT` (the broker advertises localhost:$KAFKA_PORT, so the
# same address resolves on every tier). A tier whose host is `local` runs on the laptop (the
# local rung of the smoke ladder; its partition is a SIGSTOP of that brain, healed by SIGCONT).
#
# Every message is a signed object (driver.signed_json under the group key): the epoch first,
# then per trial a fresh claim delta and two authored effects on it. Per trial: deprecate (all
# converge) -> PARTITION the edge (tunnel dropped, or SIGSTOP locally) -> overrule (local + hpc
# apply it; edge is cut off -> diverged) -> HEAL (tunnel reopened, or SIGCONT): the edge brain
# consumes the missed offset -> reconverged. N trials => reconvergence fraction + heal latency.
#
# Env (site.env, gitignored, or the environment): EDGE_SSH (host alias | local), EDGE_DIR,
# EDGE_ENV; HPC_SSH (optional; empty = two tiers), HPC_DIR, HPC_ENV; KAFKA_PORT (9092);
# KAFKA_CONTAINER (meld-kafka); ENCODER/NLI (hashing/lexical; the brains and the driver must
# agree); EPOCH_FILE (signed under the group key, for the brains' M_nli); KEY_FILE (the group key
# file; read on each tier, never put on a command line); TRIALS (8); PART_TIER (edge | hpc: the
# tier partitioned each trial; a remote edge loses its tunnel, a local or job-mode tier is frozen
# by SIGSTOP and healed by SIGCONT, see sig_tier in kafka_tiers.sh).
#
#   TRIALS=2 EDGE_SSH=local KAFKA_PORT=19092 EPOCH_FILE=... KEY_FILE=... bash cross-tier/kafka_partition_heal_wan.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
[ -f "$HERE/site.env" ] && [ -z "${NO_SITE_ENV:-}" ] && source "$HERE/site.env"
: "${EDGE_SSH:?}" "${KEY_FILE:?}" "${EPOCH_FILE:?}"
HPC_SSH="${HPC_SSH:-}"
N="${TRIALS:-8}"; PORT="${KAFKA_PORT:-9092}"; KC="${KAFKA_CONTAINER:-meld-kafka}"
ENCODER="${ENCODER:-hashing}"; NLI="${NLI:-lexical}"
PY="${PY:-$REPO/.venv/bin/python}"
OUT="${OUT_DIR:-$REPO/outputs/cross-tier}"; mkdir -p "$OUT"
STAMP="$(date +%Y%m%d-%H%M%S)"; RUN="$(date +%s)"
CLAIMS="meld-phwan-claims-$RUN"; ACKS="meld-phwan-acks-$RUN"
TRIALS_JSON="$OUT/m3a-wan-trials-$STAMP.jsonl"; : > "$TRIALS_JSON"
RESULT="$OUT/m3a-wan-$STAMP.json"
SMP_GROUP_KEY="$(tr -d '[:space:]' < "$KEY_FILE")"; export SMP_GROUP_KEY
TIERS=(edge local); [ -n "$HPC_SSH" ] && TIERS=(edge hpc local)
PT="${PART_TIER:-edge}"; [[ " ${TIERS[*]} " == *" $PT "* ]] && [ "$PT" != local ] || { echo "FATAL: PART_TIER=$PT is not a remote-capable tier of: ${TIERS[*]}"; exit 2; }
BRAIN_PIDS=(); TUN_PIDS=(); EDGE_TUN=""; EDGE_PID=""; CAFF_PID=""; LAUNCHED=()

source "$HERE/kafka_tiers.sh"

pub() {  # message-json (from a file) -> the claims topic
  CLAIMS_TOPIC="$CLAIMS" MSG_FILE="$1" BOOT="localhost:$PORT" PYTHONPATH="$REPO/src:$HERE" "$PY" - <<'PYEOF'
import os
from kafka_transport import KafkaTransport
t = KafkaTransport(os.environ["BOOT"])
t.publish(os.environ["CLAIMS_TOPIC"], open(os.environ["MSG_FILE"]).read()); t.close()
PYEOF
}
msgs() {  # trial i -> writes $MSG/{claim_id,delta,deprecate,overrule}
  I="$1" MSG="$MSG" ENC="$ENCODER" NL="$NLI" PYTHONPATH="$REPO/src:$HERE" "$PY" - <<'PYEOF'
import os
from driver import trial_messages
for k, v in trial_messages(int(os.environ["I"]), os.environ["ENC"], os.environ["NL"]).items():
    open(os.path.join(os.environ["MSG"], k), "w").write(v)
PYEOF
}
tun_part() { [ "$PT" = edge ] && [ "$(host_of edge)" != local ]; }   # a remote edge: drop and reopen its tunnel
partition() { if tun_part; then kill "$EDGE_TUN" 2>/dev/null || true; EDGE_TUN=""; else sig_tier "$PT" STOP; fi; }
heal() { if tun_part; then open_tun "$EDGE_SSH"; EDGE_TUN="$TUN_PID"; else sig_tier "$PT" CONT; fi; }

command -v caffeinate >/dev/null && { caffeinate -s & CAFF_PID=$!; }
docker ps --filter "name=$KC" --format '{{.Names}}' | grep -q "$KC" || { echo "FATAL: broker container $KC not running"; exit 2; }
MSG="$(mktemp -d)"
EPOCH_MSG="$MSG/epoch.msg"; F="$EPOCH_FILE" PYTHONPATH="$REPO/src:$HERE" "$PY" -c "
import os; from driver import epoch_message; from meld.config import load_epoch
open('$EPOCH_MSG', 'w').write(epoch_message(load_epoch(os.environ['F'])))"

echo "[setup] tiers: ${TIERS[*]} partitioned=$PT (edge=$(host_of edge) hpc=${HPC_SSH:-none}) encoder=$ENCODER nli=$NLI broker=localhost:$PORT"
for t in "${TIERS[@]}"; do start_brain "$t"; done
for t in "${TIERS[@]}"; do wait_ready "$t" || { echo "FATAL: $t brain not ready"; exit 3; }; done
pub "$EPOCH_MSG"; sleep 2
echo "[setup] all brains ready; epoch published"

ok=0
for i in $(seq 1 "$N"); do
  msgs "$i"; cid="$(cat "$MSG/claim_id")"
  echo "[trial $i/$N] claim=$cid"
  pub "$MSG/delta"; pub "$MSG/deprecate"                                 # [1] baseline: all converge
  for t in "${TIERS[@]}"; do wait_status "$t" "$cid" DEPRECATED 30 || true; done
  partition; sleep 3                                                     # [2] partition tier $PT
  pub "$MSG/overrule"                                                    # [3] overrule while cut off
  peers_ok=1
  for t in "${TIERS[@]}"; do [ "$t" = "$PT" ] || wait_status "$t" "$cid" OVERRULED 25 || peers_ok=0; done
  part_now="$(last_status "$PT" "$cid")"; diverged_first=false
  [ "$peers_ok" = 1 ] && [ "$part_now" = DEPRECATED ] && diverged_first=true
  echo "    diverged_first=$diverged_first ($PT=$part_now while peers OVERRULED)"
  t0=$("$PY" -c 'import time; print(time.time())')                     # [4] heal, to reconvergence
  heal
  if wait_status "$PT" "$cid" OVERRULED 90; then converged=true; ok=$((ok+1)); else converged=false; fi
  t1=$("$PY" -c 'import time; print(time.time())')
  heal_ms=$("$PY" -c "import sys;print(round((float(sys.argv[2])-float(sys.argv[1]))*1000,2))" "$t0" "$t1")
  echo "    converged=$converged heal_ms=$heal_ms"
  printf '{"seed": %d, "converged": %s, "heal_ms": %s, "diverged_first": %s}\n' \
    "$i" "$converged" "$heal_ms" "$diverged_first" >> "$TRIALS_JSON"
done

echo "[aggregate] $ok/$N reconverged -> $RESULT"
TRIALS_JSON="$TRIALS_JSON" RESULT="$RESULT" BOOT="localhost:$PORT tiers=${TIERS[*]} partitioned=$PT edge=$(host_of edge) encoder=$ENCODER nli=$NLI" \
PYTHONPATH="$HERE" "$PY" - <<'PYEOF'
import json, os
from kafka_partition_heal import aggregate_trials
trials = [json.loads(l) for l in open(os.environ["TRIALS_JSON"]) if l.strip()]
agg = aggregate_trials(trials); agg["trials"] = trials; agg["transport"] = os.environ["BOOT"]
open(os.environ["RESULT"], "w").write(json.dumps(agg, indent=2))
print(json.dumps(agg, indent=2))
PYEOF
echo "$(date -u +%FT%TZ) m3a-wan $STAMP trials=$N reconverged=$ok tiers=${TIERS[*]} partitioned=$PT result=$RESULT" >> "$OUT/run-log.txt"
[ "$ok" = "$N" ]
