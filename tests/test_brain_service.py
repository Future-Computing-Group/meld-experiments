"""MELD-brain HTTP service (rung-2 containerized node), framework-free.

The request handling is a PURE function -- handle(method, path, body) -> (status, body)
-- so it is unit-testable with no socket; a thin stdlib ThreadingHTTPServer wraps it
(exercised at the docker smoke). Endpoints:

  * POST /patch    body = ONE signed, typed delta (a serialized MemoryUnit) -> the node
                   VERIFIES the signature, reconstructs the typed unit, and merges it via
                   merge-on-receive; 403 if the signature does not verify.
  * GET  /metrics  -> the node's live claim set + count (the eval-c convergence probe
                   the orchestrator polls across nodes).
  * GET  /healthz  -> liveness (the orchestrator's peer-health + transport probe).

Delivery contract pairs with PatchTransport: only 2xx is delivery, so /patch returns
200 on success and the orchestrator's send treats non-2xx as non-delivery.
"""
from __future__ import annotations

import json


def _cfg():
    from meld.agent.run_hotpot_gate import _calibrated_cfg
    return _calibrated_cfg("hashing")


def _service(nid="A"):
    from meld.sync.node import BrainNode
    from meld.sync.service import BrainService
    return BrainService(BrainNode(nid, encoder="hashing", nli="lexical", cfg=_cfg()))


def _signed(svc, text):
    """A signed, typed delta payload for ``text`` under the service's group key."""
    from meld.sync.wire_delta import serialize_signed_delta
    unit = svc.node.mem.build_unit(text)
    return serialize_signed_delta("peer", unit, svc.group_key)


def test_post_patch_merges_signed_typed_delta_into_node():
    svc = _service()
    status, body = svc.handle("POST", "/patch", _signed(svc, "Paris is the capital of France"))
    assert status == 200
    assert json.loads(body)["received"] == 1
    assert any("Paris" in t for t in svc.node.live_claim_texts())


def test_post_patch_two_signed_deltas_both_merge():
    svc = _service()
    for text in ("Paris is the capital of France", "Mount Everest is the tallest mountain"):
        status, body = svc.handle("POST", "/patch", _signed(svc, text))
        assert status == 200
        assert json.loads(body)["received"] == 1
    live = svc.node.live_claim_texts()
    assert any("Paris" in t for t in live) and any("Everest" in t for t in live)


def test_post_patch_rejects_forged_delta_before_admission():
    svc = _service()
    payload = _signed(svc, "Paris is the capital of France")
    tampered = payload.replace(b"Paris", b"Berlin")
    status, _ = svc.handle("POST", "/patch", tampered)
    assert status == 403                              # verification failed -> non-delivery
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


def test_post_status_overrules_claim_and_metrics_reports_it():
    svc = _service()
    svc.node.ingest("Paris is the capital of France")
    assert json.loads(svc.handle("GET", "/metrics")[1])["status"]["Paris is the capital of France"] == "ACTIVE"
    status, _ = svc.handle("POST", "/status", json.dumps(
        {"claim": "Paris is the capital of France", "link_id": "L1", "effect": "OVERRULED"}).encode())
    assert status == 200
    m = json.loads(svc.handle("GET", "/metrics")[1])
    assert m["status"]["Paris is the capital of France"] == "OVERRULED"   # status visible over the wire


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

    node = BrainNode("A", encoder="hashing", nli="lexical", cfg=_cfg())
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
