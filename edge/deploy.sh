#!/bin/bash
# Deploy / control the MELD-brain node on every host in testbed.yaml.
#
# Generic + config-driven: hosts, IPs, ports, interface, and image tags are read
# from `testbed.yaml` (via config.py), so this script carries no host aliases.
# One brain per host, --network=host so peers are reachable at each host's IP on
# the configured iface. No docker --restart: a downed/injected node stays down
# (failure injection must be real).
#
# Usage: ./deploy.sh {push|start|stop|status} [routing] [encoder] [node]
#   push [smoke|campaign]      docker save the image locally, load on every host
#   start [broadcast|semantic] [hashing|minilm] [node]   run the brains (config.py render_start)
#   stop                       remove the brains + clear tc on every host
#   status                     show container + healthz per host
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="${PYTHON:-python3}"

# Load topology / ports / image tags from testbed.yaml via config.py.
read_cfg() {
  "$PYBIN" - <<PY
import sys
sys.path.insert(0, "$HERE")
from config import load_testbed
c = load_testbed()
print("IFACE=%s" % c["iface"])
print("PORT=%s" % int(c.get("brain_port", 8080)))
print("IMG_SMOKE=%s" % c.get("images", {}).get("smoke", "meld-brain:smoke"))
print("IMG_CAMPAIGN=%s" % c.get("images", {}).get("campaign", "meld-brain:minilm"))
print("NODES='%s'" % " ".join("%s|%s|%s" % (n["id"], n["host"], n["ip"]) for n in c["nodes"]))
PY
}
eval "$(read_cfg)"

ENC_DEFAULT=hashing
IMG="$IMG_SMOKE"
TAR="/tmp/meld-brain.tar.gz"

# Parse the NODES blob into parallel arrays (bash-3.2 safe; no assoc arrays).
IDS=(); HOSTS=(); IPS=()
for triple in $NODES; do
  IDS+=("${triple%%|*}"); rest="${triple#*|}"
  HOSTS+=("${rest%%|*}"); IPS+=("${rest##*|}")
done
N=${#IDS[@]}

case "${1:-}" in
  push)
    SEL="${2:-smoke}"; [ "$SEL" = "campaign" ] && IMG="$IMG_CAMPAIGN"
    echo "saving $IMG -> $TAR"; docker save "$IMG" | gzip > "$TAR"; ls -lh "$TAR"
    for ((i=0; i<N; i++)); do echo "load -> ${HOSTS[$i]}"; ssh "${HOSTS[$i]}" "docker load" < "$TAR"; done ;;
  start)
    # One start path: config.py render_start renders the docker run (deployed NLI, the run key
    # as a read-only file, the epoch read-only, SMP_HTTP_TIMEOUT; lexical only on the smoke rung).
    ROUTING="${2:-broadcast}"          # broadcast | semantic
    ENC="${3:-$ENC_DEFAULT}"           # hashing (smoke) | minilm (campaign)
    ONLY="${4:-}"                      # optional: start this node id only
    for ((i=0; i<N; i++)); do
      [ -n "$ONLY" ] && [ "${IDS[$i]}" != "$ONLY" ] && continue
      cmd=$("$PYBIN" "$HERE/config.py" start-cmd "${IDS[$i]}" "$ROUTING" "$ENC")
      ssh "${HOSTS[$i]}" "$cmd" >/dev/null
      echo "started ${IDS[$i]} @ ${HOSTS[$i]}  routing=$ROUTING enc=$ENC"
    done ;;
  status)
    for ((i=0; i<N; i++)); do
      echo -n "${HOSTS[$i]} (${IDS[$i]}): "
      ssh "${HOSTS[$i]}" "docker ps --filter name=meld-brain --format '{{.Status}}' | head -1; \
                curl -fsS --max-time 3 http://localhost:$PORT/healthz && echo || echo ' UNHEALTHY'"
    done ;;
  stop)
    for ((i=0; i<N; i++)); do
      ssh "${HOSTS[$i]}" "docker rm -f meld-brain >/dev/null 2>&1 || true; \
                sudo -n tc qdisc del dev $IFACE root >/dev/null 2>&1 || true"
      echo "stopped + tc-cleared ${HOSTS[$i]}"
    done ;;
  *) echo "usage: $0 {push|start|stop|status} [routing] [encoder] [node]"; exit 2 ;;
esac
