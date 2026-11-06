"""MELD-brain HTTP service (rung-2 containerized node), framework-free.

The request handling is a PURE function -- handle(method, path, body) -> (status, body)
-- so it is unit-testable with no socket; a thin stdlib ThreadingHTTPServer wraps it
(exercised at the docker smoke). Endpoints:

  * POST /patch    body = envelopes of signed objects (sync.wire_delta) -> the node
                   VERIFIES each MAC and inserts the object into its replicated sets
                   (merge happens in the admission step, T6); 403 if any does not verify.
  * GET  /metrics  -> the node's live claim set + count (the eval-c convergence probe
                   the orchestrator polls across nodes).
  * GET  /healthz  -> liveness (the orchestrator's peer-health + transport probe).

Delivery contract pairs with PatchTransport: only 2xx is delivery, so /patch returns
200 on success and the orchestrator's send treats non-2xx as non-delivery.
"""
from __future__ import annotations

import json




def _service(nid="A"):
    from meld.sync.node import BrainNode
    from meld.sync.service import BrainService
    return BrainService(BrainNode(nid, encoder="hashing", nli="lexical"))


def _signed(svc, text):
    """A signed Delta envelope for ``text`` in the default cell under the group key."""
    from meld.protocol.objects import attach_mac, canonical_bytes
    from meld.sync.wire_delta import pack
    from test_protocol_objects import _delta
    d = attach_mac(_delta(cell="agent.public", text=text, origin="peer", encoder="hashing"), svc.group_key)
    return pack([(canonical_bytes(d), d.mac, "agent")])


def _texts(svc):
    return {d.text for d in svc.node.objects.iter("D", "agent")}


def test_post_patch_inserts_signed_delta_into_the_replicated_set():
    svc = _service()
    status, body = svc.handle("POST", "/patch", _signed(svc, "Paris is the capital of France"))
    assert status == 200
    assert json.loads(body)["received"] == 1
    assert _texts(svc) == {"Paris is the capital of France"}


def test_post_patch_two_signed_deltas_both_inserted():
    svc = _service()
    for text in ("Paris is the capital of France", "Mount Everest is the tallest mountain"):
        status, body = svc.handle("POST", "/patch", _signed(svc, text))
        assert status == 200
        assert json.loads(body)["received"] == 1
    assert _texts(svc) == {"Paris is the capital of France",
                           "Mount Everest is the tallest mountain"}


def test_post_patch_rejects_forged_delta_before_admission():
    svc = _service()
    payload = _signed(svc, "Paris is the capital of France")
    tampered = payload.replace(b"Paris", b"Berlin")
    status, _ = svc.handle("POST", "/patch", tampered)
    assert status == 403                              # verification failed -> non-delivery
    assert _texts(svc) == set()                       # nothing inserted
    assert svc.node.live_claim_texts() == set()       # nothing merged


def test_get_metrics_reports_live_claims():
    svc = _service("node-7")
    svc.node.ingest("Paris is the capital of France")
    status, body = svc.handle("GET", "/metrics")
    assert status == 200
    m = json.loads(body)
    assert m["node_id"] == "node-7"
    assert m["live_count"] == 1
    assert any("Paris" in t for t in m["live"])


def test_post_status_is_retired_and_an_unsigned_status_write_changes_nothing():
    # /status was an unsigned local status write (04 finding 10); a status change now
    # enters only as a published, MAC-verified AuthoredEffect via /patch
    # (tests/test_wire_objects.py::test_status_endpoint_is_removed_and_overrule_...).
    svc = _service()
    svc.node.ingest("Paris is the capital of France")
    status, _ = svc.handle("POST", "/status", json.dumps(
        {"claim": "Paris is the capital of France", "link_id": "L1", "effect": "OVERRULED"}).encode())
    assert status == 404
    m = json.loads(svc.handle("GET", "/metrics")[1])
    assert m["status"]["Paris is the capital of France"] == "ACTIVE"


def test_get_healthz_ok():
    svc = _service()
    status, body = svc.handle("GET", "/healthz")
    assert status == 200
    assert json.loads(body)["status"] == "ok"


def test_unknown_route_is_404_not_a_silent_200():
    svc = _service()
    status, _ = svc.handle("GET", "/nope")
    assert status == 404            # never a silent success (delivery contract integrity)


def test_empty_or_malformed_patch_body_is_rejected_not_a_silent_merge():
    # an unsigned / malformed body fails verification -> 403, nothing merged
    # (never a silent 200 that pretends to have admitted a delta).
    svc = _service()
    status, _ = svc.handle("POST", "/patch", b"   \n  ")
    assert status == 403
    assert svc.node.live_claim_texts() == set()


def test_post_ingest_adds_local_observation():
    svc = _service()
    status, body = svc.handle("POST", "/ingest", b"Paris is the capital of France")
    assert status == 200
    assert json.loads(body)["ingested"] == 1
    assert any("Paris" in t for t in svc.node.live_claim_texts())


def test_publish_without_transport_is_503_not_silent_ok():
    # a node not configured with a send seam must not pretend it published (no silent 200)
    svc = _service()
    status, _ = svc.handle("POST", "/publish")
    assert status == 503


def test_post_reset_clears_node_state_and_latency():
    from meld.sync.latency import SyncLatencyMeter
    svc = _service()
    svc.meter = SyncLatencyMeter(clock=lambda: 0.0)
    svc.node.ingest("Paris is the capital of France")
    svc.meter.mark_published("p1"); svc.meter.mark_globally_consistent("p1")
    status, body = svc.handle("POST", "/reset")
    assert status == 200
    assert svc.node.live_claim_texts() == set()              # brain cleared
    assert _texts(svc) == set()                              # replicated sets cleared
    assert svc.handle("GET", "/metrics")[1] and json.loads(svc.handle("GET", "/metrics")[1])["live_count"] == 0
    assert svc.meter.summary()["n"] == 0                     # latency meter reset


def test_semantic_publish_fetches_peer_content_once_per_publish():
    # the eval-r fix: semantic routing must score against REAL peer content, fetched ONCE
    # per publish (not per claim, and not never). Previously peer_live was unwired -> empty
    # content + sigma_lo=0 -> every peer passed -> semantic degenerated to broadcast.
    from types import SimpleNamespace
    from meld.sync.node import BrainNode
    from meld.sync.service import BrainService
    calls = []

    def peer_live(p):
        calls.append(p)
        return ["a peer's topical content"]

    def send(peer, payload):
        return SimpleNamespace(status_code=200)

    node = BrainNode("A", encoder="hashing", nli="lexical")
    node.ingest("Paris is the capital of France")
    node.ingest("Mount Everest is the tallest mountain")     # 2 claims to publish
    svc = BrainService(node, peers=["p1", "p2"], routing_mode="semantic",
                       sigma_lo=0.0, send=send, peer_live=peer_live)
    svc.handle("POST", "/publish")
    assert calls.count("p1") == 1 and calls.count("p2") == 1  # once per peer, NOT per claim


def test_build_from_env_wires_peer_live_for_semantic_routing():
    from meld.sync.service import build_service_from_env
    svc, _port = build_service_from_env({
        "SMP_NODE_ID": "A", "SMP_PORT": "8080", "SMP_ENCODER": "hashing", "SMP_NLI": "lexical",
        "SMP_PEERS": "http://b:8080", "SMP_ROUTING_MODE": "semantic", "SMP_SIGMA_LO": "0.12"})
    assert svc._peer_live is not None         # a real peer-content fetch is wired (not None)
    assert svc.sigma_lo == 0.12               # calibrated threshold honored


def test_build_from_env_requires_explicit_core_config():
    # no-silent-default: NODE_ID / PORT / ENCODER / NLI must be set
    import pytest
    from meld.sync.service import build_service_from_env
    for missing in ("SMP_NODE_ID", "SMP_PORT", "SMP_ENCODER", "SMP_NLI"):
        env = {"SMP_NODE_ID": "A", "SMP_PORT": "8800",
               "SMP_ENCODER": "hashing", "SMP_NLI": "lexical"}
        del env[missing]
        with pytest.raises(ValueError):
            build_service_from_env(env)


def test_build_from_env_parses_peers_and_routing():
    from meld.sync.service import build_service_from_env
    env = {"SMP_NODE_ID": "node-2", "SMP_PORT": "8800",
           "SMP_ENCODER": "hashing", "SMP_NLI": "lexical",
           "SMP_PEERS": "http://b:8800,http://c:8800", "SMP_ROUTING_MODE": "semantic"}
    svc, port = build_service_from_env(env)
    assert port == 8800
    assert svc.node.node_id == "node-2"
    assert svc.peers == ["http://b:8800", "http://c:8800"]
    assert svc.routing_mode == "semantic"
    assert svc.transport is not None       # a real send seam is wired for the container


def test_build_from_env_loads_a_signed_epoch_and_refuses_a_tampered_one(tmp_path):
    import json as _json
    import pytest
    from meld.config import save_epoch
    from meld.sync.service import build_service_from_env
    from meld.sync.wire_delta import group_key
    from test_agent_meld_memory import agent_epoch
    env = {"SMP_NODE_ID": "A", "SMP_PORT": "8800", "SMP_ENCODER": "hashing", "SMP_NLI": "lexical",
           "SMP_EPOCH": str(tmp_path / "epoch.json")}
    save_epoch(agent_epoch(), group_key(env), env["SMP_EPOCH"])
    svc, _ = build_service_from_env(env)
    assert len(svc.node.objects.ids("Ep", "agent")) == 1
    raw = _json.loads((tmp_path / "epoch.json").read_text())
    raw["received"] = raw["received"].replace('"seq":1', '"seq":2')
    (tmp_path / "epoch.json").write_text(_json.dumps(raw))
    with pytest.raises(ValueError):
        build_service_from_env(env)


def test_build_from_env_refuses_an_epoch_signed_under_another_key_or_with_a_forged_mac(tmp_path):
    # the epoch file is a received object: its MAC is checked under the node's group key, and the
    # record is never re-signed locally (a parse-and-sign path would accept both files below)
    import json as _json
    import pytest
    from meld.config import save_epoch
    from meld.sync.service import build_service_from_env
    from test_agent_meld_memory import agent_epoch
    env = {"SMP_NODE_ID": "A", "SMP_PORT": "8800", "SMP_ENCODER": "hashing", "SMP_NLI": "lexical",
           "SMP_EPOCH": str(tmp_path / "epoch.json")}
    save_epoch(agent_epoch(), b"an-attacker-key", env["SMP_EPOCH"])
    with pytest.raises(ValueError, match="SMP_EPOCH refused"):
        build_service_from_env(env)
    raw = _json.loads((tmp_path / "epoch.json").read_text())
    raw["mac"] = "0" * 64
    (tmp_path / "epoch.json").write_text(_json.dumps(raw))
    with pytest.raises(ValueError, match="SMP_EPOCH refused"):
        build_service_from_env(env)


def test_reset_keeps_the_deployed_epoch(tmp_path):
    # F3: both testbed drivers POST /reset per cell; the boot epoch (SMP_EPOCH) must survive it,
    # else every result cell runs with no similarity merge
    from meld.config import save_epoch
    from meld.sync.service import build_service_from_env
    from meld.sync.wire_delta import group_key
    from test_agent_meld_memory import agent_epoch
    env = {"SMP_NODE_ID": "A", "SMP_PORT": "8800", "SMP_ENCODER": "hashing", "SMP_NLI": "lexical",
           "SMP_EPOCH": str(tmp_path / "epoch.json")}
    save_epoch(agent_epoch(), group_key(env), env["SMP_EPOCH"])
    svc, _ = build_service_from_env(env)
    svc.node.ingest("Paris is the capital of France")
    assert svc.handle("POST", "/reset")[0] == 200
    roots = json.loads(svc.handle("GET", "/metrics")[1])["roots"]["agent"]
    assert "Ep" in roots and "D" not in roots
    assert svc.node.outbox() == []        # the boot epoch is held, never re-published


def test_metrics_name_the_models_and_the_held_epoch_ids(tmp_path):
    from meld.config import load_epoch, save_epoch
    from meld.protocol.objects import object_id
    from meld.sync.service import build_service_from_env
    from meld.sync.wire_delta import group_key
    from test_agent_meld_memory import agent_epoch
    env = {"SMP_NODE_ID": "A", "SMP_PORT": "8800", "SMP_ENCODER": "hashing", "SMP_NLI": "lexical",
           "SMP_EPOCH": str(tmp_path / "epoch.json")}
    save_epoch(agent_epoch(), group_key(env), env["SMP_EPOCH"])
    svc, _ = build_service_from_env(env)
    m = json.loads(svc.handle("GET", "/metrics")[1])
    assert (m["encoder"], m["nli"], m["m_nli"]) == ("hashing", "lexical", svc.node.mem.m_nli.hex())
    assert m["epoch_ids"] == [object_id(load_epoch(env["SMP_EPOCH"])[0]).hex()]
