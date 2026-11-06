#!/usr/bin/env python3
"""Edge-testbed eval-s (sync latency) + eval-r (routing) campaign orchestrator.

Generic, config-driven driver: the topology (nodes / IPs / ports / image tags) and the netem
RTT/routing grid are read from `testbed.yaml`; this file carries no host aliases. It ties the
tested core (meld.eval.campaign + CampaignCheckpoint + bootstrap_ci) to live brains over HTTP.

No ssh and no docker here. The brains of ONE routing phase are deployed from the laptop with
`deploy.sh start <routing> <encoder>` before this driver starts; the driver runs only that
phase's cells (SMP_ROUTING, checked against every brain's /metrics). Each host's watchdog is the
tc agent (POST /shape, /heal) and the control path to its brain (/brain/<path>); the driver
heartbeats every watchdog.

Per cell: POST /reset with the cell's peers to each brain of the cell (one deployment serves
every N), check the deployed epoch is held after the reset (result rungs), shape netem delay
toward the cell's other brains, ingest the topic-homed workload, POST /publish on every brain
(node-driven, node-to-peer), score routing on deltas against the needed-oracle, and record
latency, recall, waste, bytes per message class and the provenance stamps.

Modes (env): SMP_DRY_CELL=1 -> 1 cell; SMP_PILOT=1 -> small pilot; otherwise FULL.
SMP_ROUTING = the deployed phase (broadcast | semantic); SMP_CKPT = the checkpoint JSONL.
grid.replication (default 2) = brains homing each topic (a claim interests replication/N of a cell).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meld.eval.campaign import (aggregate_by_condition, enumerate_cells, run_campaign,  # noqa: E402
                                score_routing)
from meld.eval.campaign_checkpoint import CampaignCheckpoint                          # noqa: E402
from meld.eval.topology import stamps                                                 # noqa: E402
from meld.sync.workload import build_workload, load_hotpot_claims                     # noqa: E402

from config import load_testbed                                                      # noqa: E402


def log(msg):
    print(f"- {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} campaign: {msg}", flush=True)


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url, data=b"", timeout=60):
    if isinstance(data, (dict, list)):
        data = json.dumps(data).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _heartbeat_loop(cfg, stop, period):
    while not stop.is_set():
        for n in cfg["ids"]:
            try:
                _post(cfg["wd_url"][n] + "/heartbeat", timeout=3)
            except Exception:
                pass
        stop.wait(period)


def _heal_all(cfg, nodes):
    for n in nodes:
        try:
            _post(cfg["wd_url"][n] + "/heal", {}, timeout=30)
        except Exception as e:
            log(f"  WARN heal {n}: {type(e).__name__}: {e}")


def _ping_rtt_ms(peer_ip):
    cp = subprocess.run(["ping", "-c", "3", peer_ip], capture_output=True, text=True)
    for tok in cp.stdout.split():
        if tok.count("/") >= 3:                # min/avg/max/mdev
            try:
                return float(tok.split("/")[1])
            except ValueError:
                pass
    return None


def run_cell(cell, cfg, *, routing, publish_timeout, require_epoch, claims, replication=2):
    if cell.routing != routing:
        raise RuntimeError(f"cell {cell.cell_id} is {cell.routing}; deployed phase is {routing}")
    brains = cfg["ids"][:cell.n]
    url, ip = cfg["url"], cfg["ip"]
    try:
        held = {}
        for h in brains:                               # per-cell isolation + the cell's peers
            _post(url[h] + "/reset", {"peers": [cfg["peer_url"][p] for p in brains if p != h]})
            m = _get(url[h] + "/metrics")
            if m.get("routing_mode") != routing:
                raise RuntimeError(f"brain {h} runs {m.get('routing_mode')}, not {routing}")
            held[h] = all("Ep" in r for r in m["roots"].values()) and bool(m["roots"])
        if require_epoch and not all(held.values()):
            raise RuntimeError(f"deployed epoch missing after /reset: {held}")
        if cell.rtt_ms > 0:
            for h in brains:
                _post(cfg["wd_url"][h] + "/shape",
                      {"peer_ips": [ip[p] for p in brains if p != h], "delay_ms": cell.rtt_ms})
            rtt = _ping_rtt_ms(ip[brains[-1]])
            if rtt is not None and rtt < cell.rtt_ms:      # netem must have taken
                log(f"  WARN rtt read-back {rtt}ms < configured {cell.rtt_ms}ms on {cell.cell_id}")
        else:
            rtt = 0.0
        wl = build_workload(claims, brains, replication=replication, seed=cell.seed)
        for h in brains:
            for text in wl.assignments[h]:
                _post(url[h] + "/ingest", text.encode())
        sc, nbytes, lat = Counter(), {}, {}
        t0 = time.monotonic()
        for h in brains:
            resp = _post(url[h] + "/publish", timeout=publish_timeout)
            sc.update(score_routing(resp["delivered"], wl.needed, sender=h,
                                    url=lambda b: cfg["peer_url"][b]))
            for cls, r in resp["bytes"].items():
                acc = nbytes.setdefault(cls, {"count": 0, "bytes": 0})
                acc["count"] += r["count"]
                acc["bytes"] += r["bytes"]
            lat[h] = resp.get("latency") or {}
        wall = time.monotonic() - t0
        wasted = ((sc["delivered_total"] - sc["recall_num"]) / sc["delivered_total"]
                  if sc["delivered_total"] else 0.0)
        return {"p50_ms": lat[brains[0]].get("p50_ms"), "p95_ms": lat[brains[0]].get("p95_ms"),
                "latency_per_brain": lat, "recall_num": sc["recall_num"],
                "recall_den": sc["recall_den"], "delivered_total": sc["delivered_total"],
                "wasted_rate": wasted, "bytes": nbytes, "epoch_held": held,
                "publish_wall_s": round(wall, 3), "rtt_meas_ms": rtt}
    finally:
        _heal_all(cfg, brains)                         # per-host cleanup, never abort the rest


def main():
    cfg = load_testbed()
    grid = cfg.get("grid", {})
    mode = ("DRY-CELL" if os.environ.get("SMP_DRY_CELL") == "1" else
            "PILOT" if os.environ.get("SMP_PILOT") == "1" else "FULL")
    routing = os.environ.get("SMP_ROUTING", "")
    if routing not in ("broadcast", "semantic"):
        raise SystemExit("SMP_ROUTING must name the deployed phase: broadcast | semantic")
    smoke = cfg.get("smoke") is True
    ckpt = os.environ.get("SMP_CKPT", f"campaign_{routing}.jsonl")
    g_seeds = list(range(1, int(grid.get("seeds", 15)) + 1))
    if mode == "DRY-CELL":                                  # 1 cell, validates the core path
        rtts, ns, seeds = [0], [2], [1]
    elif mode == "PILOT":                                   # WAN + the largest N, 5-seed variance
        rtts, ns, seeds = [0, 50], [max(grid.get("n_nodes", [2, 3, 4]))], [1, 2, 3, 4, 5]
    else:
        rtts, ns, seeds = grid.get("rtt_ms", [0, 10, 25, 50]), grid.get("n_nodes", [2, 3, 4]), g_seeds

    m0 = _get(cfg["url"][cfg["ids"][0]] + "/metrics")
    stamp = {**stamps(m0["encoder"], m0["nli"], None, None), "routing": routing, "mode": mode,
             "smoke": smoke, "brain_m_nli": m0["m_nli"], "epoch_ids": m0["epoch_ids"],
             "replication": int(grid.get("replication", 2))}
    cp = CampaignCheckpoint(ckpt)
    old = {r.get("stamp", {}).get("git_sha") for r in cp.results().values()}
    if old - {stamp["git_sha"]}:
        raise SystemExit(f"{ckpt} holds rows from another commit {sorted(old)}; use a new SMP_CKPT")
    cells = sorted(enumerate_cells(rtts=rtts, ns=ns, routings=[routing], seeds=seeds),
                   key=lambda c: (c.rtt_ms, c.n, c.seed))
    claims = load_hotpot_claims(200)
    stop = threading.Event()
    threading.Thread(target=_heartbeat_loop, args=(cfg, stop, int(grid.get("heartbeat_s", 30))),
                     daemon=True).start()
    log(f"=== campaign START ({mode}, routing={routing}) {len(cells)} cells, "
        f"{len(cp.pending([c.cell_id for c in cells]))} pending, ckpt={ckpt} ===")
    try:
        _heal_all(cfg, cfg["ids"])                         # startup pre-clean
        run_campaign(cells, lambda c: {**run_cell(c, cfg, routing=routing,
                                                  publish_timeout=int(grid.get("publish_timeout_s", 120)),
                                                  require_epoch=not smoke, claims=claims,
                                                  replication=stamp["replication"]),
                                       "stamp": stamp},
                     cp,
                     progress=lambda cid, r: log(f"  done {cid}: p50={r.get('p50_ms')} recall="
                                                  f"{r.get('recall_num')}/{r.get('recall_den')} "
                                                  f"delivered={r.get('delivered_total')}"),
                     on_error=lambda c, e: log(f"  CELL FAIL {c.cell_id}: {type(e).__name__}: {e}"))
        agg = aggregate_by_condition(cells, cp.results(), metric="p50_ms", n_boot=10000, seed=0)
        for cond, (pt, lo, hi) in sorted(agg.items()):
            log(f"  agg p50_ms {cond}: {pt:.2f} [{lo:.2f},{hi:.2f}]")
        log("=== campaign DONE ===")
    finally:
        stop.set()
        _heal_all(cfg, cfg["ids"])                         # safety net on any exit


if __name__ == "__main__":
    main()
