"""edge/run_eval_c.py on the current protocol, over a fake network (no sockets, no tc).

Four in-process brain services; each host's watchdog is faked (POST /isolate drops the host's
egress to the listed peer addresses, /heal clears it, /brain/<path> reaches the local brain and
is never partitioned). One seed: partition 2|2 through the watchdogs, each side authors its
status effect, then heal = remove the rules, /resync rounds until the Merkle roots agree, then
/pass on every brain. Expected: divergence during, roots equal after, one post-pass fingerprint,
the CRDT status unanimous, and the last-writer-wins counterfactual read off the arrival order
the brains observed (each side saw its own effect first) split when the effects differ.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from meld.sync.node import BrainNode
from meld.sync.service import BrainService

EDGE = Path(__file__).resolve().parents[1] / "edge"
IDS = ["A", "B", "C", "D"]
IP = {n: f"10.0.0.{i + 1}" for i, n in enumerate(IDS)}


@pytest.fixture
def ev(monkeypatch):
    monkeypatch.syspath_prepend(str(EDGE))
    spec = importlib.util.spec_from_file_location("run_eval_c_probe", EDGE / "run_eval_c.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    blocked = set()                                   # (host, peer ip) egress drops
    svcs = {}

    def peer_call(src):
        def call(peer_url, path, body):
            dst = peer_url.split("//")[1].split(":")[0]
            if (src, dst) in blocked:
                raise ConnectionError("partitioned")
            return svcs[next(n for n in IDS if IP[n] == dst)].handle("POST", path, body)
        return call

    for n in IDS:
        svcs[n] = BrainService(BrainNode(n, encoder="hashing", nli="lexical"), call=peer_call(n))

    def route(method, url, data=b""):
        host = url.split("//")[1].split("/")[0]
        path = "/" + url.split("//")[1].split("/", 1)[1]
        n = host.replace("wd-", "")
        if path.startswith("/brain/"):
            return svcs[n].handle(method, path[len("/brain"):], data)
        if path == "/isolate":
            ips = json.loads(data)["peer_ips"]
            blocked.update((n, ip) for ip in ips)
            return 200, json.dumps({"filters": ips}).encode()
        if path == "/heal":
            blocked.difference_update({b for b in blocked if b[0] == n})
            return 200, b'{"healed": true}'
        if path == "/heartbeat":
            return 200, b"{}"
        raise AssertionError(url)

    def post(url, data=b"", timeout=None):
        status, body = route("POST", url, data if isinstance(data, bytes) else json.dumps(data).encode())
        if status != 200:
            raise RuntimeError((status, body))
        return json.loads(body)

    def get(url, timeout=None):
        return json.loads(route("GET", url)[1])

    monkeypatch.setattr(mod, "_post", post)
    monkeypatch.setattr(mod, "_get", get)
    cfg = {"brains": IDS, "url": {n: f"http://wd-{n}/brain" for n in IDS},
           "wd_url": {n: f"http://wd-{n}" for n in IDS},
           "peer_url": {n: f"http://{IP[n]}:8080" for n in IDS}, "ip": IP,
           "claim": "the sky is blue", "partition_s": 0, "key": svcs["A"].node.key, "delivery": False}
    return mod, cfg, svcs, blocked


def test_one_seed_partition_heal_by_anti_entropy_then_pass(ev):
    mod, cfg, svcs, blocked = ev
    rec = mod.run_seed(seed=0, effect_a="DEPRECATED", effect_b="OVERRULED", **cfg)
    assert rec["partition_verified"] is True
    assert rec["agreement_during"] < 1.0 and rec["roots_equal_during"] is False
    assert rec["cross_side_failed_during"] is True     # the partition held at the protocol level
    assert rec["roots_equal_after"] is True and rec["resync_rounds"] >= 1
    assert rec["ae_bytes"]["DIGEST"] > 0
    assert rec["crdt_healed"] == 1.0 and rec["fingerprints_distinct"] == 1
    assert rec["L"] <= 1
    assert rec["lww_healed"] < 1.0                     # each side saw its own effect first
    assert blocked == set()                            # healed


def test_identical_effects_never_diverge_and_lww_agrees(ev):
    mod, cfg, svcs, blocked = ev
    rec = mod.run_seed(seed=1, effect_a="OVERRULED", effect_b="OVERRULED", **cfg)
    assert rec["agreement_during"] == 1.0 and rec["crdt_healed"] == 1.0 and rec["lww_healed"] == 1.0


def test_shipped_defaults_are_effects_the_protocol_knows():
    from meld.sync.node import status_effect
    cfg = (EDGE / "testbed.example.yaml").read_text()
    import yaml
    part = yaml.safe_load(cfg)["partition"]
    for eff in (part["effect_a"], part["effect_b"]):
        status_effect("x", link_id="L", effect=eff)        # raises KeyError on an unknown effect
