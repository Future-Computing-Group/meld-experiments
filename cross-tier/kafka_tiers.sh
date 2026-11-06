# Shared tier helpers for the Kafka cross-tier scripts (sourced, not run). The caller sets REPO,
# HERE, PY, PORT, OUT, STAMP, CLAIMS, ACKS, ENCODER, NLI, EPOCH_FILE, EDGE_SSH/EDGE_DIR/EDGE_ENV,
# HPC_SSH/HPC_DIR/HPC_ENV and the arrays BRAIN_PIDS, TUN_PIDS, LAUNCHED. A tier whose host is
# `local` runs on the laptop; any other host gets the bundle by tar pipe (sha256-checked), a
# reverse tunnel to the broker, and its brain in tmux `meld-phwan` (the group key is read from
# the key file on that host, never put on a command line).
# Optional: EDGE_IMAGE runs the edge brain inside that container image (host network; $EDGE_DIR
# absolute). HPC_JOB (a batch script on the HPC host) runs the HPC brain as a scheduler job:
# `sbatch --parsable $HPC_JOB $PORT <brain args>` from $HPC_DIR, wait (HPC_QUEUE_WAIT s, 600)
# until it runs, then the reverse tunnel goes to its compute node through the login host
# (`ssh -J`, plus HPC_NODE_SSH_OPTS); the script waits for the port, then execs the brain.
host_of() { case "$1" in edge) echo "$EDGE_SSH";; hpc) echo "$HPC_SSH";; local) echo local;; esac; }
dir_of()  { case "$1" in edge) echo "${EDGE_DIR:-}";; hpc) echo "${HPC_DIR:-}";; *) echo "";; esac; }
env_of()  { case "$1" in edge) echo "${EDGE_ENV:-true}";; hpc) echo "${HPC_ENV:-true}";; *) echo true;; esac; }
log_of()  { local h; h="$(host_of "$1")"; if [ "$h" = local ]; then echo "$OUT/phwan-$1-$STAMP.jsonl"; else echo "$(dir_of "$1")/phwan-$1-$STAMP.jsonl"; fi; }

cleanup() {
  echo "[teardown]"
  [ -n "$EDGE_PID" ] && kill -CONT "$EDGE_PID" 2>/dev/null || true
  [ -n "${HPC_PID:-}" ] && kill -CONT "$HPC_PID" 2>/dev/null || true
  [ -n "${HPC_JOBID:-}" ] && { ssh "$HPC_SSH" "scancel $HPC_JOBID" 2>/dev/null || true; }
  for p in "${TUN_PIDS[@]:-}" "${BRAIN_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  for h in "${LAUNCHED[@]:-}"; do [ -n "$h" ] && ssh "$h" 'tmux kill-session -t meld-phwan 2>/dev/null || true; docker rm -f meld-phwan >/dev/null 2>&1 || true' 2>/dev/null || true; done
  [ -n "$CAFF_PID" ] && kill "$CAFF_PID" 2>/dev/null || true
}
trap cleanup EXIT

open_tun() {  # ssh target args (host, or -J login node) -> sets global TUN_PID
  ssh -N -R "$PORT:localhost:$PORT" -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes "$@" & TUN_PID=$!; TUN_PIDS+=("$TUN_PID")
}
ship() {  # host dir: the brain bundle by tar pipe (no rsync), then a sha256 check of the copy
  local h="$1" d="$2" files=(src/meld cross-tier/kafka_brain.py cross-tier/kafka_transport.py cross-tier/brain_core.py cross-tier/vendor/kafka)
  local sums='find "$@" -type f ! -name "*.pyc" -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-64'
  ssh "$h" "mkdir -p '$d' && rm -rf '$d/src/meld'"
  (cd "$REPO" && COPYFILE_DISABLE=1 tar -cf - --exclude='__pycache__' "${files[@]}") | ssh "$h" "tar -xf - -C '$d'"
  [ "$(cd "$REPO" && bash -c "$sums" _ "${files[@]}")" = "$(ssh "$h" "cd '$d' && bash -c '$sums' _ ${files[*]}")" ] \
    || { echo "FATAL: shipped bundle differs on $h"; exit 4; }
  scp -q "$EPOCH_FILE" "$h:$d/epoch.json"
}
start_job() {  # host dir args: the HPC brain as a batch job, then the tunnel to its node
  local h="$1" d="$2" args="$3" node="" end
  HPC_JOBID="$(ssh "$h" "cd '$d' && MELD_XTIER_KEY='${REMOTE_KEY_FILE:-$d/run.key}' sbatch --parsable '$HPC_JOB' $PORT $args")"
  HPC_JOBID="${HPC_JOBID%%;*}"; echo "    hpc job $HPC_JOBID submitted"
  end=$(( $(date +%s) + ${HPC_QUEUE_WAIT:-600} ))
  while [ "$(date +%s)" -lt "$end" ]; do
    node="$(ssh "$h" "squeue -h -j $HPC_JOBID -o '%T %N'" | sed -n 's/^RUNNING //p')"; [ -n "$node" ] && break; sleep 5
  done
  [ -n "$node" ] || { echo "FATAL: hpc job $HPC_JOBID not running after ${HPC_QUEUE_WAIT:-600}s"; exit 3; }
  echo "    hpc job $HPC_JOBID running on $node"
  open_tun ${HPC_NODE_SSH_OPTS:-} -J "$h" "$node"; sleep 3
  kill -0 "$TUN_PID" 2>/dev/null || { echo "FATAL: tunnel to $node failed (port $PORT taken there?)"; exit 3; }
}
start_brain() {  # tier
  local t="$1" h d e lg; h="$(host_of "$t")"; lg="$(log_of "$t")"
  local args="--bootstrap localhost:$PORT --id $t --claims-topic $CLAIMS --acks-topic $ACKS --encoder $ENCODER --nli $NLI --log '$lg' --run-seconds 1800"
  if [ "$h" = local ]; then
    bash -c "PYTHONPATH='$REPO/src:$HERE' exec '$PY' '$HERE/kafka_brain.py' $args" & BRAIN_PIDS+=($!)
    if [ "$t" = edge ]; then EDGE_PID=$!; fi
    if [ "$t" = hpc ]; then HPC_PID=$!; fi
  elif [ "$t" = hpc ] && [ -n "${HPC_JOB:-}" ]; then
    d="$(dir_of "$t")"; ship "$h" "$d"; start_job "$h" "$d" "$args"
  else
    d="$(dir_of "$t")"; e="$(env_of "$t")"; ship "$h" "$d"
    open_tun "$h"; if [ "$t" = edge ]; then EDGE_TUN="$TUN_PID"; fi; sleep 3
    if [ "$t" = edge ] && [ -n "${EDGE_IMAGE:-}" ]; then  # the brain inside the deployed-model image, host network; $d absolute
      ssh "$h" "tmux kill-session -t meld-phwan 2>/dev/null || true; docker rm -f meld-phwan >/dev/null 2>&1 || true; tmux new-session -d -s meld-phwan bash -c \"cd '$d'; export SMP_GROUP_KEY=\\\$(tr -d '[:space:]' < '${REMOTE_KEY_FILE:-$d/run.key}'); exec docker run --rm --name meld-phwan --network host -e SMP_GROUP_KEY -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=src:cross-tier/vendor -v '$d:$d' -w '$d' --entrypoint python3 $EDGE_IMAGE cross-tier/kafka_brain.py $args\""
    else
      ssh "$h" "tmux kill-session -t meld-phwan 2>/dev/null || true; tmux new-session -d -s meld-phwan bash -c \"cd '$d'; set +u; $e; set -u; SMP_GROUP_KEY=\\\$(tr -d '[:space:]' < '${REMOTE_KEY_FILE:-$d/run.key}') PYTHONPATH=src:cross-tier/vendor python3 cross-tier/kafka_brain.py $args\""
    fi
    LAUNCHED+=("$h")
  fi
}
sig_tier() {  # tier STOP|CONT: freeze or thaw that tier's brain (its consumer stops consuming = partition)
  local t="$1" s="$2"
  if [ "$(host_of "$t")" = local ] && [ "$t" = edge ]; then kill -"$s" "$EDGE_PID"
  elif [ "$(host_of "$t")" = local ] && [ "$t" = hpc ]; then kill -"$s" "$HPC_PID"
  elif [ "$t" = hpc ] && [ -n "${HPC_JOBID:-}" ]; then  # a step on the job's own allocation signals the brain there
    ssh "$HPC_SSH" "srun --overlap --jobid=$HPC_JOBID pkill -$s -u \$USER -f '[k]afka_brain.py --bootstrap [^ ]* --id $t '"
  else echo "FATAL: no signal path to the $t brain"; return 1; fi
}
log_lines() {  # tier pattern
  local h; h="$(host_of "$1")"
  if [ "$h" = local ]; then grep "$2" "$(log_of "$1")" 2>/dev/null || true
  else ssh "$h" "grep '$2' '$(log_of "$1")' 2>/dev/null" 2>/dev/null || true; fi
}
wait_ready() { local i; for i in $(seq 1 "${READY_WAIT:-60}"); do [ -n "$(log_lines "$1" '"ev": "ready"')" ] && return 0; sleep 1; done; return 1; }
last_status() {  # tier cid -> latest eff_status for cid
  log_lines "$1" "\"claim_id\": \"$2\"" | grep '"ev": "merge"' | tail -1 | sed -n 's/.*"eff_status": "\([A-Z]*\)".*/\1/p'
}
wait_status() {  # tier cid want timeout_s
  local end=$(( $(date +%s) + $4 ))
  while [ "$(date +%s)" -lt "$end" ]; do [ "$(last_status "$1" "$2")" = "$3" ] && return 0; sleep 0.4; done
  return 1
}
