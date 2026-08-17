#!/usr/bin/env python3
"""Edge-testbed partition-heal CRDT-convergence orchestrator (eval-c).

Generic, config-driven: the topology (nodes / SSH targets / IPs / iface) and the
partition cell (mode / seeds / claim / effects) are read from `testbed.yaml`; this
file carries no host aliases. It runs the status-CRDT partition-heal experiment
over REAL brains (HTTP) and a REAL, healable network partition (tc, via
PartitionController), across seeds, and aggregates CRDT vs last-writer-wins
reconvergence. Run from a persistent session (tmux) on the orchestrator node.

Per-seed protocol (one cell):
  1. POST /reset + POST /ingest the claim to every brain.
  2. REAL partition: split brains into side_a / side_b; shape every cross-side host
     pair to loss-100% via PartitionController (verifies the rule landed).
  3. During the partition: side_a applies link "La" (effect_a), side_b link "Lb"
     (effect_b); cross-side delivery is blocked.
  4. agreement_during = modal-status fraction (expect < 1.0).
  5. Optional: sleep partition_s before healing.
  6. HEAL, then resync: deliver BOTH links to EVERY brain in a per-brain seeded
     order (the CRDT join is order-independent).
  7. crdt_healed measured over the wire (expect 1.0); lww_healed is the analytical
     last-writer-wins counterfactual off the SAME delivery order.

Partition modes: "tc" = a real, healable tc partition between hosts (needs ssh+sudo
tc per host); "delivery" = the partition is the absence of cross-side delivery
(HTTP-only, no tc/ssh) — faithful because MELD status reconciliation is delivery-
driven. Set it under `partition.mode` in testbed.yaml.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meld.sync.eval_c_partition import (                   # noqa: E402
    LINK_A, LINK_B, agreement as _agreement, aggregate_results, partition_plan,
    per_seed_result, seeded_delivery_orders)
from meld.sync.partition import PartitionController        # noqa: E402

from config import load_testbed                           # noqa: E402


# ======================================================================================
# WIRE SEAMS (HTTP + tc)
# ======================================================================================

def _post(url: str, data: bytes = b"", timeout: float = 20.0) -> dict:
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _status_of(url: str, claim: str, timeout: float = 8.0) -> str:
    with urllib.request.urlopen(url + "/metrics", timeout=timeout) as r:
        return json.loads(r.read().decode())["status"].get(claim, "ACTIVE")


def make_ssh_runner(local_host: str) -> Callable:
    """Real tc exec seam: run `sudo tc ...` locally on the orchestrator host, over ssh sudo
    on any other host. ``host`` is the SSH target; ``local_host`` names the host co-located
    with this orchestrator. Returns a result with .returncode/.stdout/.stderr."""
    import subprocess
    from types import SimpleNamespace

    def runner(host: str, argv: List[str]):
        try:
            if host == local_host:
                cp = subprocess.run(["sudo", *argv], capture_output=True, text=True,
                                    timeout=30)
            else:
                inner = "sudo " + " ".join(argv)
                cp = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                     "-o", "StrictHostKeyChecking=no", host, inner],
                    capture_output=True, text=True, timeout=30)
            return SimpleNamespace(returncode=cp.returncode, stdout=cp.stdout,
                                   stderr=cp.stderr)
        except subprocess.TimeoutExpired as e:
            return SimpleNamespace(returncode=124, stdout="", stderr=f"ssh/tc timeout: {e}")

    return runner


def _deliver_status(url: str, claim: str, link_id: str, effect: str,
                    timeout: float = 20.0) -> None:
    _post(url + "/status",
          json.dumps({"claim": claim, "link_id": link_id, "effect": effect}).encode(),
          timeout=timeout)


def run_seed(*, seed: int, brains: List[str], url: Dict[str, str], ip: Dict[str, str],
             claim: str, effect_a: str, effect_b: str, partition_s: float,
             controller: PartitionController, ssh_hosts: Dict[str, str],
             partition_fn: Optional[Callable] = None,
             heal_fn: Optional[Callable] = None) -> Dict:
    """One cell over the wire. ``partition_fn``/``heal_fn`` are the injectable partition seam:
    the testbed default uses ``controller`` (real tc); the delivery mode withholds cross-side
    /status so no tc is needed. Returns the per-seed record dict (resumable JSONL row)."""
    half = len(brains) // 2
    side_a, side_b = brains[:half], brains[half:]
    links = [(LINK_A, effect_a), (LINK_B, effect_b)]
    plan = partition_plan([(b, ip[b]) for b in side_a], [(b, ip[b]) for b in side_b])

    for b in brains:
        _post(url[b] + "/reset")
        _post(url[b] + "/ingest", claim.encode())

    def _heal_all():
        if heal_fn is None:
            for b in brains:
                controller.heal_host(ssh_hosts[b])
        else:
            heal_fn(side_a, side_b)

    try:
        partition_verified = True
        if partition_fn is None:
            for host, peer_ips in plan:
                controller.isolate_egress_multi(ssh_hosts[host], peer_ips)
            for host, peer_ips in plan:
                partition_verified = partition_verified and all(
                    controller.verify_partition(ssh_hosts[host], pip) for pip in peer_ips)
        else:
            partition_verified = partition_fn(side_a, side_b)

        for a in side_a:
            _deliver_status(url[a], claim, LINK_A, effect_a)
        for c in side_b:
            _deliver_status(url[c], claim, LINK_B, effect_b)
        during_statuses = {b: _status_of(url[b], claim) for b in brains}
        agreement_during = _agreement(list(during_statuses.values()))

        if partition_s > 0:
            time.sleep(partition_s)
    finally:
        _heal_all()        # heal BEFORE the resync (and on any failure above)

    orders = seeded_delivery_orders(brains, links, seed=seed)
    for b in brains:
        for link_id, effect in orders[b]:
            _deliver_status(url[b], claim, link_id, effect)
    crdt_healed = _agreement([_status_of(url[b], claim) for b in brains])

    metrics = per_seed_result(during_statuses=during_statuses, delivery_orders=orders,
                              effect_a=effect_a, effect_b=effect_b)

    return {"seed": seed,
            "agreement_during": agreement_during,
            "crdt_healed": crdt_healed,
            "lww_healed": metrics["lww_healed"],
            "partition_verified": bool(partition_verified)}


# ======================================================================================
# RESUMABLE JSONL + config wiring
# ======================================================================================

def _existing_seeds(path: str) -> set:
    seeds = set()
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seeds.add(int(json.loads(line)["seed"]))
            except (ValueError, KeyError):
                continue
    return seeds


def _append_jsonl(path: str, record: Dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def _read_all(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def config_from_testbed() -> Dict:
    """Build the run config from testbed.yaml. The brain set is a prefix of the
    configured nodes sized to an even count (the partition needs two equal sides)."""
    cfg = load_testbed()
    nodes = cfg["ids"]
    n = (len(nodes) // 2) * 2 or len(nodes)
    brains = nodes[:n]
    part = cfg.get("partition", {})
    return {
        "brains": brains,
        "url": {b: cfg["url"][b] for b in brains},
        "ssh_hosts": {b: cfg["host"][b] for b in brains},
        "ip": {b: cfg["ip"][b] for b in brains},
        "claim": part.get("claim", "the sky is blue"),
        "effect_a": part.get("effect_a", "ACTIVE"),
        "effect_b": part.get("effect_b", "RETRACTED"),
        "n_seeds": int(part.get("seeds", 15)),
        "partition_s": float(part.get("partition_s", 0)),
        "iface": cfg["iface"],
        "local_host": cfg["host"][cfg["orchestrator"]],
        "out": part.get("out", "eval_c_partition_heal.jsonl"),
        "delivery": str(part.get("mode", "tc")).lower() == "delivery",
    }


def main() -> None:  # pragma: no cover - testbed entrypoint
    cfg = config_from_testbed()
    controller = PartitionController(make_ssh_runner(cfg["local_host"]), iface=cfg["iface"])
    pfn = (lambda sa, sb: True) if cfg["delivery"] else None
    hfn = (lambda sa, sb: None) if cfg["delivery"] else None
    done = _existing_seeds(cfg["out"])
    print(f"=== partition-heal convergence run: {cfg['n_seeds']} seed(s); mode="
          f"{'delivery' if cfg['delivery'] else 'tc'}; {len(done)} already in {cfg['out']} "
          f"(skipped) ===", flush=True)

    try:
        for seed in range(cfg["n_seeds"]):
            if seed in done:
                continue
            rec = run_seed(seed=seed, brains=cfg["brains"], url=cfg["url"], ip=cfg["ip"],
                           claim=cfg["claim"], effect_a=cfg["effect_a"],
                           effect_b=cfg["effect_b"], partition_s=cfg["partition_s"],
                           controller=controller, ssh_hosts=cfg["ssh_hosts"],
                           partition_fn=pfn, heal_fn=hfn)
            _append_jsonl(cfg["out"], rec)
            print(f"  seed {seed}: during={rec['agreement_during']:.2f} "
                  f"crdt_healed={rec['crdt_healed']:.2f} lww_healed={rec['lww_healed']:.2f} "
                  f"partition_verified={rec['partition_verified']}", flush=True)
    finally:
        if not cfg["delivery"]:
            for b in cfg["brains"]:
                try:
                    controller.heal_host(cfg["ssh_hosts"][b])
                except Exception as e:
                    print(f"  [cleanup] heal_host({b}) failed: {e}", flush=True)

    agg = aggregate_results(_read_all(cfg["out"]))
    print("\n=== aggregate (seed-as-unit, mean [95% bootstrap CI]) ===", flush=True)
    for metric in ("agreement_during", "crdt_healed", "lww_healed"):
        m = agg[metric]
        print(f"  {metric:18s}: {m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]", flush=True)
    print(f"  crdt_reconverges : {agg['crdt_reconverges']:.3f}  "
          f"lww_reconverges: {agg['lww_reconverges']:.3f}  (n={agg['n_seeds']})", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
