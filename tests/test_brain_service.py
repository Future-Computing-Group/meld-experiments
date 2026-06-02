"""SMP-brain HTTP service (rung-2 containerized node), framework-free.

The request handling is a PURE function -- handle(method, path, body) -> (status, body)
-- so it is unit-testable with no socket; a thin stdlib ThreadingHTTPServer wraps it
(exercised at the docker smoke). Endpoints:

  * POST /patch    body = newline-separated claim texts (a published delta) -> the node
                   merges each via merge-on-receive; returns the received count.
  * GET  /metrics  -> the node's live claim set + count (the eval-c convergence probe
                   the orchestrator polls across nodes).
  * GET  /healthz  -> liveness (the orchestrator's peer-health + transport probe).

Delivery contract pairs with PatchTransport: only 2xx is delivery, so /patch returns
200 on success and the orchestrator's send treats non-2xx as non-delivery.
"""
from __future__ import annotations

import json


def _cfg():
    from smp.agent.run_hotpot_gate import _calibrated_cfg
    return _calibrated_cfg("hashing")


def _service(nid="A"):
    from smp.sync.node import BrainNode
    from smp.sync.service import BrainService
    return BrainService(BrainNode(nid, encoder="hashing", nli="lexical", cfg=_cfg()))


def test_post_patch_merges_claim_into_node():
    svc = _service()
    status, body = svc.handle("POST", "/patch", b"Paris is the capital of France")
    assert status == 200
    assert json.loads(body)["received"] == 1
    assert any("Paris" in t for t in svc.node.live_claim_texts())


def test_post_patch_accepts_multi_claim_delta():
    svc = _service()
    payload = b"Paris is the capital of France\nMount Everest is the tallest mountain"
    status, body = svc.handle("POST", "/patch", payload)
    assert status == 200
    assert json.loads(body)["received"] == 2
    live = svc.node.live_claim_texts()
    assert any("Paris" in t for t in live) and any("Everest" in t for t in live)


def test_get_metrics_reports_live_claims():
    svc = _service("node-7")
    svc.node.ingest("Paris is the capital of France")
    status, body = svc.handle("GET", "/metrics")
    assert status == 200
    m = json.loads(body)
    assert m["node_id"] == "node-7"
    assert m["live_count"] == 1
    assert any("Paris" in t for t in m["live"])


def test_get_healthz_ok():
    svc = _service()
    status, body = svc.handle("GET", "/healthz")
    assert status == 200
    assert json.loads(body)["status"] == "ok"


def test_unknown_route_is_404_not_a_silent_200():
    svc = _service()
    status, _ = svc.handle("GET", "/nope")
    assert status == 404            # never a silent success (delivery contract integrity)


def test_empty_patch_body_receives_nothing():
    svc = _service()
    status, body = svc.handle("POST", "/patch", b"   \n  ")
    assert status == 200
    assert json.loads(body)["received"] == 0
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


def test_build_from_env_requires_explicit_core_config():
    # no-silent-default: NODE_ID / PORT / ENCODER / NLI must be set
    import pytest
    from smp.sync.service import build_service_from_env
    for missing in ("SMP_NODE_ID", "SMP_PORT", "SMP_ENCODER", "SMP_NLI"):
        env = {"SMP_NODE_ID": "A", "SMP_PORT": "8800",
               "SMP_ENCODER": "hashing", "SMP_NLI": "lexical"}
        del env[missing]
        with pytest.raises(ValueError):
            build_service_from_env(env)


def test_build_from_env_parses_peers_and_routing():
    from smp.sync.service import build_service_from_env
    env = {"SMP_NODE_ID": "node-2", "SMP_PORT": "8800",
           "SMP_ENCODER": "hashing", "SMP_NLI": "lexical",
           "SMP_PEERS": "http://b:8800,http://c:8800", "SMP_ROUTING_MODE": "semantic"}
    svc, port = build_service_from_env(env)
    assert port == 8800
    assert svc.node.node_id == "node-2"
    assert svc.peers == ["http://b:8800", "http://c:8800"]
    assert svc.routing_mode == "semantic"
    assert svc.transport is not None       # a real send seam is wired for the container
