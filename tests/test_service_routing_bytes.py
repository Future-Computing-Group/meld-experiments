"""Routing score and bytes on the wire over the brain service (multi-host harness).

/publish reports every published object by id with its kind and (deltas) text, so the campaign
scores routing on deltas only against the needed-oracle; /reset may set the cell's peers; /publish
and /resync carry bytes per message class (sync/wire_bytes.py classes, anti-entropy per path).
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

from meld.eval.campaign import score_routing
from meld.sync.node import BrainNode
from meld.sync.service import BrainService
from meld.sync.workload import build_workload

CLAIMS = [(f"topic {t} fact {i}: alpha{t} beta{t} gamma{i}", f"T{t}") for t in range(6) for i in range(3)]


def _mesh(ids, routing):
    svcs = {}

    def send(peer, payload):
        return SimpleNamespace(status_code=svcs[peer].handle("POST", "/patch", payload)[0])

    def live(peer):
        return sorted(svcs[peer].node.live_claim_texts())

    for i in ids:
        svcs[i] = BrainService(BrainNode(i, encoder="hashing", nli="lexical"), peers=[],
                               routing_mode=routing, top_k=1 if routing == "semantic" else None,
                               send=send, peer_live=live,
                               call=lambda p, path, body: svcs[p].handle("POST", path, body))
    return svcs


def _cell(routing, n=4, seed=3):
    ids = ["A", "B", "C", "D", "E"][:n]
    svcs = _mesh(ids + ["X"], routing)              # X: deployed, but not in this cell
    for i in ids:
        assert svcs[i].handle("POST", "/reset", json.dumps({"peers": [p for p in ids if p != i]}).encode())[0] == 200
    wl = build_workload(CLAIMS, ids, replication=2, seed=seed)
    for i in ids:
        for t in wl.assignments[i]:
            svcs[i].handle("POST", "/ingest", t.encode())
    resps = {i: json.loads(svcs[i].handle("POST", "/publish")[1]) for i in ids}
    return svcs, wl, resps


def test_publish_reports_kind_and_text_per_object_and_semantic_top1_hits_the_needed_peer():
    svcs, wl, resps = _cell("semantic")
    n_deltas = sum(len(v) for v in wl.assignments.values())
    sc = {k: 0 for k in ("recall_num", "recall_den", "delivered_total")}
    for i, r in resps.items():
        assert all(set(o) >= {"kind", "peers"} for o in r["delivered"].values())
        s = score_routing(r["delivered"], wl.needed, sender=i, url=lambda b: b)
        for k in sc:
            sc[k] += s[k]
    assert sc["delivered_total"] == n_deltas          # one peer per delta
    assert sc["recall_den"] == n_deltas               # replication 2: one other home per delta
    assert sc["recall_num"] == n_deltas               # identical text at the other home: sigma 1
    assert json.loads(svcs["X"].handle("GET", "/metrics")[1])["live_count"] == 0   # not in the cell


def test_broadcast_delivers_each_delta_to_the_cells_other_brains_only():
    svcs, wl, resps = _cell("broadcast")
    n_deltas = sum(len(v) for v in wl.assignments.values())
    tot = sum(score_routing(r["delivered"], wl.needed, sender=i, url=lambda b: b)["delivered_total"]
              for i, r in resps.items())
    assert tot == n_deltas * 3
    assert json.loads(svcs["A"].handle("GET", "/metrics")[1])["routing_mode"] == "broadcast"


def test_score_ignores_non_delta_objects():
    delivered = {"aa": {"kind": "Sig", "peers": ["B"]}, "bb": {"kind": "D", "text": "t", "peers": ["B", "C"]}}
    s = score_routing(delivered, {"t": frozenset({"A", "B"})}, sender="A", url=lambda b: b)
    assert (s["recall_num"], s["recall_den"], s["delivered_total"]) == (1, 1, 2)


def test_publish_bytes_per_class_match_the_wire_bytes_runner_on_the_running_example():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import run_wire_bytes
    from meld.eval.order_shuffle import running_example
    from meld.sync.experiment import node_factory
    pol, seed_objs, msgs = running_example("hashing")
    ref, _ = run_wire_bytes._held(node_factory(encoder="hashing", nli="lexical", pol=pol)(0),
                                  list(seed_objs) + list(msgs), "jur")
    from meld.sync.wire_bytes import meter
    want = meter(ref)
    sent = []
    a = BrainService(BrainNode("A", encoder="hashing", nli="lexical", pol=pol), peers=["B"],
                     send=lambda p, payload: sent.append(payload) or SimpleNamespace(status_code=200))
    for o in list(seed_objs) + list(msgs):
        a.node.publish_local(o)
    got = json.loads(a.handle("POST", "/publish")[1])["bytes"]
    for cls in ("DELTA", "EFFECT"):
        assert {k: got[cls][k] for k in ("count", "bytes")} == {k: want[cls][k] for k in ("count", "bytes")}
    assert sum(c["bytes"] for c in got.values()) == sum(map(len, sent))


def test_resync_reports_anti_entropy_bytes_per_path():
    svcs = _mesh(["A", "B"], "broadcast")
    for i, j in (("A", "B"), ("B", "A")):
        svcs[i].peers = [j]
    svcs["A"].node.ingest("Photosynthesis converts sunlight into chemical energy")
    r = json.loads(svcs["B"].handle("POST", "/resync")[1])
    assert r["pulled"] == 1 and set(r["ae_bytes"]) == {"DIGEST", "PULL", "PATCH"}
    assert r["ae_bytes"]["DIGEST"] > 0 and r["ae_bytes"]["PULL"] > 0 and r["ae_bytes"]["PATCH"] == 0
    assert r["ae_bytes"]["DIGEST"] == r["digest_bytes"]
