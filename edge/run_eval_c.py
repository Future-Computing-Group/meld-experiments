#!/usr/bin/env python3
"""Edge-testbed partition-heal convergence orchestrator (eval-c) on the current protocol.

Generic, config-driven: the topology and the partition cell (mode / seeds / claim / effects /
key) are read from `testbed.yaml`; this file carries no host aliases. Run it from a persistent
session (tmux) on the orchestrator host. No ssh: each host's watchdog installs and heals the
partition (POST /isolate, /heal; the rule drops only the brains' peer port toward the other
side's addresses) and is the driver's control path to its brain (/brain/<path>).

Per seed (one cell):
  1. POST /reset (all other brains as peers) + /ingest the claim on every brain, then /resync so
     every brain holds the same sets.
  2. Partition the brains 2|2 (N/2) through the watchdogs; each watchdog returns the installed
     filters, which must be exactly the other side's addresses.
  3. During the partition: one seeded author per side receives its side's signed status effect
     (side_a: link La, effect_a; side_b: link Lb, effect_b) on /patch; every brain /resyncs
     (within-side spread; cross-side peers must fail). Record statuses, roots and the effect ids
     each brain holds (DIGEST with empty roots).
  4. Heal: /heal every watchdog, then /resync rounds (seeded brain order) until every brain's
     Merkle roots agree (rounds and DIGEST/PULL/PATCH bytes counted), then /pass on every brain,
     re-running /resync + /pass while a pass misses (at most three passes; L = passes that missed).
  5. Record: agreement during, roots equal after, distinct post-pass fingerprints, L, the CRDT
     status agreement, and the last-writer-wins counterfactual off the ARRIVAL ORDER the brains
     observed (effects held during the partition first, then those that arrived at heal; a
     brain's survivor is the effect of the last link to arrive).

`partition.mode: delivery` is a labelled smoke (no tc: the partition is only the absence of
resync during the partition), never a result row.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meld.eval.runner_args import read_key_file             # noqa: E402
from meld.eval.topology import stamps                       # noqa: E402
from meld.protocol.objects import object_id, signed         # noqa: E402
from meld.sync.eval_c_partition import (                    # noqa: E402
    LINK_A, LINK_B, agreement, aggregate_results, partition_plan)
from meld.sync.node import status_effect, status_envelope   # noqa: E402
from meld.sync.wire_delta import group_key                  # noqa: E402

from config import load_testbed                            # noqa: E402

MAX_RESYNC_ROUNDS = 6


def _post(url: str, data=b"", timeout: float = 120.0) -> dict:
    if isinstance(data, (dict, list)):
        data = json.dumps(data).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _deliver_status(url: str, claim: str, link_id: str, effect: str, key: Optional[bytes] = None,
                    timeout: float = 20.0) -> None:
    """A status change PUBLISHed as a signed AuthoredEffect to /patch, signed under the run's
    group key; a refused envelope raises."""
    _post(url + "/patch", status_envelope(claim, link_id=link_id, effect=effect,
                                          key=key or group_key(os.environ)), timeout=timeout)


def _effect_id(claim: str, link_id: str, effect: str, key: bytes) -> str:
    return object_id(signed(status_effect(claim, link_id=link_id, effect=effect), key)[0]).hex()


def _held_effects(url: str, scope: str) -> List[str]:
    return _post(url + "/digest", {"scope": scope, "roots": {}})["ids"].get("A", [])


def _roots(url: str) -> str:
    return json.dumps(_get(url + "/metrics")["roots"], sort_keys=True)


def run_seed(*, seed: int, brains: List[str], url: Dict[str, str], wd_url: Dict[str, str],
             peer_url: Dict[str, str], ip: Dict[str, str], claim: str, effect_a: str,
             effect_b: str, partition_s: float, key: bytes, delivery: bool = False) -> Dict:
    rng = random.Random(seed)
    half = len(brains) // 2
    side_a, side_b = brains[:half], brains[half:]
    plan = partition_plan([(b, ip[b]) for b in side_a], [(b, ip[b]) for b in side_b])
    eff = {LINK_A: effect_a, LINK_B: effect_b}
    eid = {_effect_id(claim, lk, e, key): lk for lk, e in eff.items()}

    for b in brains:
        _post(url[b] + "/reset", {"peers": [peer_url[p] for p in brains if p != b]})
        _post(url[b] + "/ingest", claim.encode())
    for b in brains:
        _post(url[b] + "/resync")
    scope = sorted(json.loads(_roots(url[brains[0]])))[0]
    roots_equal_before = len({_roots(url[b]) for b in brains}) == 1

    verified, cross_failed = True, True
    try:
        if not delivery:
            for host, peer_ips in plan:
                got = _post(wd_url[host] + "/isolate", {"peer_ips": peer_ips})["filters"]
                verified = verified and sorted(got) == sorted(peer_ips)
        authors = {LINK_A: rng.choice(side_a), LINK_B: rng.choice(side_b)}
        for lk, b in authors.items():
            _deliver_status(url[b], claim, lk, eff[lk], key=key)
        if not delivery:
            for b in rng.sample(brains, len(brains)):
                failed = set(_post(url[b] + "/resync")["failed_peers"])
                other = side_b if b in side_a else side_a
                cross_failed = cross_failed and failed == {peer_url[p] for p in other}
        during = {b: _get(url[b] + "/metrics")["status"].get(claim, "ACTIVE") for b in brains}
        roots_equal_during = len({_roots(url[b]) for b in brains}) == 1
        held_during = {b: sorted(_held_effects(url[b], scope)) for b in brains}
        if partition_s > 0:
            time.sleep(partition_s)
    finally:
        if not delivery:
            for b in brains:
                _post(wd_url[b] + "/heal", {})

    ae = {"DIGEST": 0, "PULL": 0, "PATCH": 0}
    rounds = 0
    for rounds in range(1, MAX_RESYNC_ROUNDS + 1):
        for b in rng.sample(brains, len(brains)):
            r = _post(url[b] + "/resync")
            for k in ae:
                ae[k] += r["ae_bytes"][k]
        if len({_roots(url[b]) for b in brains}) == 1:
            break
    roots_equal_after = len({_roots(url[b]) for b in brains}) == 1

    misses = []
    for _ in range(3):           # Prop 3: L <= 1; a third pass makes an L >= 2 violation visible
        res = {b: _post(url[b] + "/pass")[scope] for b in brains}
        misses.append([res[b]["pass_time_misses"] for b in brains])
        if not any(misses[-1]):
            break
        for b in rng.sample(brains, len(brains)):
            _post(url[b] + "/resync")

    after = {b: _get(url[b] + "/metrics")["status"].get(claim, "ACTIVE") for b in brains}
    held_after = {b: _held_effects(url[b], scope) for b in brains}
    arrival = {b: [eid[i] for i in held_during[b] if i in eid] +
                  [eid[i] for i in sorted(set(held_after[b]) - set(held_during[b])) if i in eid]
               for b in brains}
    lww = [eff[order[-1]] if order else "ACTIVE" for order in arrival.values()]
    return {"seed": seed, "mode": "delivery-smoke" if delivery else "tc",
            "agreement_during": agreement(list(during.values())),
            "crdt_healed": agreement(list(after.values())),
            "lww_healed": agreement(lww),
            "partition_verified": bool(verified) and not delivery,
            "cross_side_failed_during": bool(cross_failed) and not delivery,
            "roots_equal_before": roots_equal_before, "roots_equal_during": roots_equal_during,
            "roots_equal_after": roots_equal_after, "resync_rounds": rounds, "ae_bytes": ae,
            "fingerprints_distinct": len({res[b]["fingerprint"] for b in brains}),
            "fingerprints": {b: res[b]["fingerprint"] for b in brains},
            "pass_misses": misses, "L": sum(map(any, misses)),
            "status_during": during, "status_after": after, "arrival": arrival,
            "authors": authors}


# ======================================================================================
# RESUMABLE JSONL + config wiring
# ======================================================================================

def _existing_seeds(path: str) -> set:
    seeds = set()
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
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
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def config_from_testbed() -> Dict:
    """The run config from testbed.yaml. The brain set is the even prefix of the configured
    nodes (the partition needs two equal sides)."""
    cfg = load_testbed()
    nodes = cfg["ids"]
    brains = nodes[:(len(nodes) // 2) * 2 or len(nodes)]
    part = cfg.get("partition", {})
    if not cfg.get("key_file"):
        raise SystemExit("key_file is required (the run's group key; the brains verify under it)")
    return {
        "brains": brains,
        "url": {b: cfg["url"][b] for b in brains},
        "wd_url": {b: cfg["wd_url"][b] for b in brains},
        "peer_url": {b: cfg["peer_url"][b] for b in brains},
        "ip": {b: cfg["ip"][b] for b in brains},
        "claim": part.get("claim", "the sky is blue"),
        "effect_a": part.get("effect_a", "DEPRECATED"),
        "effect_b": part.get("effect_b", "OVERRULED"),
        "n_seeds": int(part.get("seeds", 15)),
        "partition_s": float(part.get("partition_s", 0)),
        "out": part.get("out", "eval_c_partition_heal.jsonl"),
        "delivery": str(part.get("mode", "tc")).lower() == "delivery",
        "key": read_key_file(os.path.expanduser(cfg["key_file"])),
        "heartbeat_s": int(cfg.get("grid", {}).get("heartbeat_s", 30)),
    }


def main() -> None:  # pragma: no cover - testbed entrypoint
    import threading
    cfg = config_from_testbed()
    done = _existing_seeds(cfg["out"])
    m0 = _get(cfg["url"][cfg["brains"][0]] + "/metrics")
    stamp = {**stamps(m0["encoder"], m0["nli"], None, None), "brain_m_nli": m0["m_nli"],
             "epoch_ids": m0["epoch_ids"]}
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            for b in cfg["brains"]:
                try:
                    _post(cfg["wd_url"][b] + "/heartbeat", timeout=3)
                except Exception:
                    pass
            stop.wait(cfg["heartbeat_s"])
    threading.Thread(target=beat, daemon=True).start()
    print(f"=== partition-heal run: {cfg['n_seeds']} seed(s); mode="
          f"{'delivery (smoke)' if cfg['delivery'] else 'tc'}; {len(done)} already in {cfg['out']} ===",
          flush=True)
    run = {k: cfg[k] for k in ("brains", "url", "wd_url", "peer_url", "ip", "claim", "effect_a",
                               "effect_b", "partition_s", "key", "delivery")}
    try:
        for seed in range(cfg["n_seeds"]):
            if seed in done:
                continue
            rec = {**run_seed(seed=seed, **run), "stamp": stamp}
            _append_jsonl(cfg["out"], rec)
            print(f"  seed {seed}: during={rec['agreement_during']:.2f} crdt={rec['crdt_healed']:.2f} "
                  f"lww={rec['lww_healed']:.2f} roots_after={rec['roots_equal_after']} "
                  f"fp={rec['fingerprints_distinct']} L={rec['L']} rounds={rec['resync_rounds']} "
                  f"verified={rec['partition_verified']}", flush=True)
    finally:
        stop.set()
        if not cfg["delivery"]:
            for b in cfg["brains"]:
                try:
                    _post(cfg["wd_url"][b] + "/heal", {})
                except Exception as e:
                    print(f"  [cleanup] heal {b} failed: {e}", flush=True)

    agg = aggregate_results(_read_all(cfg["out"]))
    print("\n=== aggregate (seed-as-unit, mean [95% bootstrap CI]) ===", flush=True)
    for metric in ("agreement_during", "crdt_healed", "lww_healed"):
        m = agg[metric]
        print(f"  {metric:18s}: {m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]", flush=True)
    print(f"  crdt_reconverges : {agg['crdt_reconverges']:.3f}  "
          f"lww_reconverges: {agg['lww_reconverges']:.3f}  (n={agg['n_seeds']})", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
