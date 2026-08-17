#!/bin/bash
# Deploy / control the MELD-brain node on every host in testbed.yaml.
#
# Generic + config-driven: hosts, IPs, ports, interface, and image tags are read
# from `testbed.yaml` (via config.py), so this script carries no host aliases.
# One brain per host, --network=host so peers are reachable at each host's IP on
# the configured iface. No docker --restart: a downed/injected node stays down
# (failure injection must be real).
#
# Usage: ./deploy.sh {push|start|stop|status} [routing] [encoder]
#   push                       docker save the image locally, load on every host
#   start [broadcast|semantic] [hashing|minilm]   run the brains
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

peers_for() {   # all host IP:PORT urls except index $1
  local self_i=$1 out=() j
  for ((j=0; j<N; j++)); do [ "$j" -ne "$self_i" ] && out+=("http://${IPS[$j]}:$PORT"); done
  local IFS=,; echo "${out[*]}"
}

case "${1:-}" in
  push)
    SEL="${2:-smoke}"; [ "$SEL" = "campaign" ] && IMG="$IMG_CAMPAIGN"
    echo "saving $IMG -> $TAR"; docker save "$IMG" | gzip > "$TAR"; ls -lh "$TAR"
    for ((i=0; i<N; i++)); do echo "load -> ${HOSTS[$i]}"; ssh "${HOSTS[$i]}" "docker load" < "$TAR"; done ;;
  start)
    ROUTING="${2:-broadcast}"          # broadcast | semantic
    ENC="${3:-$ENC_DEFAULT}"           # hashing (smoke) | minilm (campaign)
    [ "$ENC" = "minilm" ] && IMG="$IMG_CAMPAIGN"
    for ((i=0; i<N; i++)); do
      peers=$(peers_for "$i")
      ssh "${HOSTS[$i]}" "sudo tc qdisc del dev $IFACE root >/dev/null 2>&1 || true; \
        docker rm -f meld-brain >/dev/null 2>&1 || true; docker run -d --name meld-brain \
        --network=host -e SMP_NODE_ID=${IDS[$i]} -e SMP_PORT=$PORT -e SMP_ENCODER=$ENC \
        -e SMP_NLI=lexical -e SMP_ROUTING_MODE=$ROUTING -e SMP_PEERS='$peers' $IMG" >/dev/null
      echo "started ${IDS[$i]} @ ${HOSTS[$i]}  routing=$ROUTING enc=$ENC peers=$peers"
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
                sudo tc qdisc del dev $IFACE root >/dev/null 2>&1 || true"
      echo "stopped + tc-cleared ${HOSTS[$i]}"
    done ;;
  *) echo "usage: $0 {push|start|stop|status} [routing] [encoder]"; exit 2 ;;
esac
