#!/usr/bin/env python3
"""Edge-testbed eval-s (sync latency) + eval-r (routing) campaign orchestrator.

Generic, config-driven driver: the topology (nodes / SSH targets / IPs / ports /
image tags) and the netem RTT/routing grid are read from `testbed.yaml`; this file
carries no host aliases. It ties the tested core (smp.eval.campaign +
CampaignCheckpoint + bootstrap_ci) to live brains over HTTP and to `tc` (WanShaper).

Run it from a persistent session (e.g. tmux) on the orchestrator node named by
`orchestrator:` in the config, so it survives a client disconnect. The smp package
must be importable there (PYTHONPATH=src), the brains must be deployed (deploy.sh),
and a watchdog must listen on `watchdog_port` on each host.

SCOPE: the eval-s/eval-r grid only. eval-c (partition -> heal -> agreement) is a
separate driver (run_eval_c.py).

Modes (env): SMP_DRY_CELL=1 -> 1 cell; SMP_PILOT=1 -> small pilot; otherwise FULL.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smp.eval.campaign import enumerate_cells, run_campaign, aggregate_by_condition  # noqa: E402
from smp.eval.campaign_checkpoint import CampaignCheckpoint                          # noqa: E402
from smp.sync.wan import WanShaper                                                   # noqa: E402
from smp.sync.workload import build_workload, load_hotpot_claims                     # noqa: E402

from config import load_testbed                                                      # noqa: E402

CFG = load_testbed()
NODES = CFG["ids"]
IP = CFG["ip"]
HOST = CFG["host"]            # SSH target per node
URL = CFG["url"]
IFACE = CFG["iface"]
ORCH = CFG["orchestrator"]    # node co-located with this driver (commands run local)
WD_PORT = int(CFG["watchdog_port"])
CAMPAIGN_IMG = CFG.get("images", {}).get("campaign", "smp-brain:minilm")
GRID = CFG.get("grid", {})
HEARTBEAT_S = int(GRID.get("heartbeat_s", 30))
PUBLISH_TIMEOUT = int(GRID.get("publish_timeout_s", 120))
CKPT = os.environ.get("SMP_CKPT", "campaign_results.jsonl")


def log(msg):
    line = f"- {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} campaign: {msg}"
    print(line, flush=True)


def _get(url, timeout=8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url, data=b"", timeout=30):
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _ssh(node, cmd):
    """Run a shell command on a node: local for the orchestrator node, SSH otherwise."""
    full = cmd if node == ORCH else f"ssh {HOST[node]} {json.dumps(cmd)}"
    return subprocess.run(full, shell=True, capture_output=True, text=True)


def _tc_runner(node, argv):
    """tc on a node: local sudo for the orchestrator node, SSH sudo otherwise."""
    inner = "sudo " + " ".join(argv)
    cp = (subprocess.run(["sudo"] + list(argv), capture_output=True, text=True)
          if node == ORCH else _ssh(node, inner))

    class R:
        returncode = cp.returncode
        stdout = cp.stdout
        stderr = cp.stderr

    return R()


def _clear_all_tc():
    """Best-effort: clear tc on every node (startup pre-clean + campaign safety net)."""
    shaper = WanShaper(_tc_runner, iface=IFACE)
    for node in NODES:
        try:
            shaper.clear(node)
        except Exception as e:
            log(f"  WARN clear tc on {node}: {type(e).__name__}: {e}")


def _safe_healthz(node):
    try:
        return _get(URL[node] + "/healthz", timeout=3).get("status") == "ok"
    except Exception:
        return False


def _redeploy(routing, brains):
    """Redeploy brains with the given routing mode + campaign image (node-driven publish
    preserves node->peer latency per mode). Called once per routing level (routing-outer).
    semantic routes to the single highest-similarity peer (top_k=1); broadcast delivers
    to all (the wasted-delivery baseline)."""
    extra = "-e SMP_TOP_K=1 " if routing == "semantic" else ""
    log(f"  redeploy brains: routing={routing} image={CAMPAIGN_IMG} {extra.strip()}")
    for node in brains:
        peers = ",".join(URL[p] for p in brains if p != node)
        run = (f"docker rm -f smp-brain >/dev/null 2>&1 || true; docker run -d --name smp-brain "
               f"--network=host -e SMP_NODE_ID={node} -e SMP_PORT={URL[node].rsplit(':', 1)[1]} "
               f"-e SMP_ENCODER=minilm -e SMP_NLI=lexical -e SMP_ROUTING_MODE={routing} "
               f"{extra}-e SMP_PEERS='{peers}' {CAMPAIGN_IMG}")
        r = _ssh(node, run)
        if r.returncode != 0:
            raise RuntimeError(f"redeploy {node} failed: {r.stderr}")
    deadline = time.time() + 90
    while time.time() < deadline:
        if all(_safe_healthz(n) for n in brains):
            return
        time.sleep(2)
    raise RuntimeError("brains not healthy after redeploy")


def _heartbeat_loop(stop):
    while not stop.is_set():
        for node in NODES:
            try:
                _post(f"http://{IP[node]}:{WD_PORT}/heartbeat", timeout=3)
            except Exception:
                pass
        stop.wait(HEARTBEAT_S)


def _ping_rtt_ms(peer_ip):
    cp = subprocess.run(["ping", "-c", "3", peer_ip], capture_output=True, text=True)
    for tok in cp.stdout.split():
        if tok.count("/") >= 3:                # min/avg/max/mdev
            try:
                return float(tok.split("/")[1])
            except ValueError:
                pass
    return None


def _run_eval_sr_cell(cell):
    brains = NODES[:cell.n]
    shaper = WanShaper(_tc_runner, iface=IFACE)
    try:
        for h in brains:                               # per-cell isolation: fresh brain + latency
            _post(URL[h] + "/reset")
        if cell.rtt_ms > 0:
            for h in brains:
                for peer in brains:
                    if peer != h:
                        shaper.shape(h, IP[peer], delay_ms=cell.rtt_ms, jitter_ms=0)
            rtt = _ping_rtt_ms(IP[brains[-1]])
            if rtt is not None and rtt < cell.rtt_ms:      # netem must have taken
                log(f"  WARN rtt read-back {rtt}ms < configured {cell.rtt_ms}ms on {cell.cell_id}")
        else:
            rtt = 0.0
        claims = load_hotpot_claims(200)
        wl = build_workload(claims, brains, replication=2, seed=cell.seed)
        for h in brains:
            for text in wl.assignments[h]:
                _post(URL[h] + "/ingest", text.encode())
        delivered_ok, needed_total, delivered_total = 0, 0, 0
        for h in brains:
            resp = _post(URL[h] + "/publish", timeout=PUBLISH_TIMEOUT)
            for claim, peers in resp.get("delivered", {}).items():
                got = set(peers)
                need = {URL[b] for b in wl.needed.get(claim, ()) if b != h}
                delivered_ok += len(got & need)        # correct deliveries (recall numerator)
                needed_total += len(need)              # needed edges (recall denominator)
                delivered_total += len(got)            # ALL deliveries (for wasted-delivery)
        lat = _get(URL[brains[0]] + "/metrics").get("latency", {})
        wasted = (delivered_total - delivered_ok) / delivered_total if delivered_total else 0.0
        return {"p50_ms": lat.get("p50_ms"), "p95_ms": lat.get("p95_ms"),
                "recall_num": delivered_ok, "recall_den": needed_total,
                "delivered_total": delivered_total, "wasted_rate": wasted, "rtt_meas_ms": rtt}
    finally:
        for h in brains:                                   # per-host cleanup, never abort the rest
            try:
                shaper.clear(h)
            except Exception as e:
                log(f"  WARN clear {h} after {cell.cell_id}: {type(e).__name__}: {e}")


def main():
    mode = "DRY-CELL" if os.environ.get("SMP_DRY_CELL") == "1" else \
           "PILOT" if os.environ.get("SMP_PILOT") == "1" else "FULL"
    g_rtts = GRID.get("rtt_ms", [0, 10, 25, 50])
    g_ns = GRID.get("n_nodes", [2, 3, 4])
    g_routings = GRID.get("routings", ["broadcast", "semantic"])
    g_seeds = list(range(1, int(GRID.get("seeds", 15)) + 1))

    if mode == "DRY-CELL":                                  # 1 cell, validates the core path
        rtts, ns, routings, seeds = [0], [2], ["broadcast"], [1]
    elif mode == "PILOT":                                   # validates WAN+semantic + 5-seed variance
        rtts, ns, routings, seeds = [0, 50], [max(g_ns)], ["broadcast", "semantic"], [1, 2, 3, 4, 5]
    else:
        rtts, ns, routings, seeds = g_rtts, g_ns, g_routings, g_seeds

    log(f"=== campaign START (eval-s/r grid, {mode}) ===")
    cells = enumerate_cells(rtts=rtts, ns=ns, routings=routings, seeds=seeds)
    cells = sorted(cells, key=lambda c: (c.routing, c.rtt_ms, c.n, c.seed))   # routing-outer
    cp = CampaignCheckpoint(CKPT)
    stop = threading.Event()
    threading.Thread(target=_heartbeat_loop, args=(stop,), daemon=True).start()
    log(f"grid: {len(cells)} cells; {len(cp.pending([c.cell_id for c in cells]))} pending; heartbeat -> :{WD_PORT}")

    state = {"routing": None}

    def run_cell(cell):
        if cell.routing != state["routing"]:               # redeploy once per routing level
            _redeploy(cell.routing, NODES[:cell.n])
            state["routing"] = cell.routing
        return _run_eval_sr_cell(cell)

    try:
        _clear_all_tc()                                    # startup pre-clean
        run_campaign(cells, run_cell, cp,
                     progress=lambda cid, r: log(f"  done {cid}: p50={r.get('p50_ms')} "
                                                  f"recall={r.get('recall_num')}/{r.get('recall_den')}"),
                     on_error=lambda c, e: log(f"  CELL FAIL {c.cell_id}: {type(e).__name__}: {e}"))
        res = cp.results()
        agg = aggregate_by_condition(cells, res, metric="p50_ms", n_boot=10000, seed=0)
        for cond, (pt, lo, hi) in sorted(agg.items()):
            log(f"  agg p50_ms {cond}: {pt:.2f} [{lo:.2f},{hi:.2f}]")
        log("=== campaign DONE ===")
    finally:
        stop.set()
        _clear_all_tc()                                    # safety net on any exit


if __name__ == "__main__":
    main()
