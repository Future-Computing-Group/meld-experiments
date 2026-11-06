"""Wire integration smoke: real ThreadingHTTPServer + real HTTP send + the delivery
contract + skew-free latency, all in-process (no docker). Proves the socket wrapper and
the on-the-wire send seam behave exactly as the unit-tested handle()/PatchTransport do,
before containerizing. make_http_send() is the real transport seam the orchestrator uses
on the testbed; here it POSTs to a live local server.

Rewritten for the replication wire (design §2.2, WP2 T2): the convergence observable is
the replicated object set (the Merkle roots in /metrics), not the old brain's live
claims; received objects reach the merge step only through the admission hook (T6).
"""
from __future__ import annotations

import socket
import threading

import requests




def _node(nid):
    from meld.sync.node import BrainNode
    return BrainNode(nid, encoder="hashing", nli="lexical")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start(node, port):
    from meld.sync.service import serve
    httpd = serve(node, host="127.0.0.1", port=port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_publish_delivers_over_real_http_and_records_latency():
    from meld.sync.latency import SyncLatencyMeter
    from meld.sync.transport import PatchTransport
    from meld.sync.wire import make_http_send
    from meld.protocol.objects import attach_mac, canonical_bytes, object_id
    from meld.sync.wire_delta import group_key, pack
    from test_protocol_objects import _delta

    b = _node("B")
    port = _free_port()
    httpd = _start(b, port)
    try:
        url = f"http://127.0.0.1:{port}"
        # server is up: healthz responds
        assert requests.get(url + "/healthz", timeout=5).status_code == 200

        meter = SyncLatencyMeter()
        transport = PatchTransport(make_http_send(timeout=5), latency_meter=meter)
        # a signed object envelope on the wire (the same default group key both ends use)
        d = attach_mac(_delta(cell="agent.public", text="Paris is the capital of France",
                              encoder="hashing"),
                       group_key({}))
        delivered = transport.publish("p1", pack([(canonical_bytes(d), d.mac, "agent")]), [url])

        assert delivered[url] is True                       # 200 -> delivered
        assert b.objects.get("D", object_id(canonical_bytes(d))) == d   # verified into D
        m = requests.get(url + "/metrics", timeout=5).json()
        assert set(m["roots"]["agent"]) == {"D"}
        assert meter.latency_ms("p1") is not None           # latency recorded (all delivered)
        assert meter.latency_ms("p1") >= 0.0
    finally:
        httpd.shutdown()


def test_node_driven_publish_converges_three_brains_over_http():
    # the real §7-s path: each node publishes its OWN delta to its peers over HTTP
    # (node->peer, not orchestrator->peer), then all three converge.
    from meld.sync.latency import SyncLatencyMeter
    from meld.sync.service import BrainService, serve
    from meld.sync.wire import make_http_send

    ids = ["A", "B", "C"]
    ports = {i: _free_port() for i in ids}
    urls = {i: f"http://127.0.0.1:{ports[i]}" for i in ids}
    claims = {"A": "Photosynthesis converts sunlight into chemical energy",
              "B": "Tectonic plates drift across the molten mantle",
              "C": "Beethoven composed nine symphonies in Vienna"}

    httpds = []
    services = {}
    try:
        for i in ids:
            peers = [urls[j] for j in ids if j != i]          # full mesh
            svc = BrainService(_node(i), peers=peers, routing_mode="broadcast",
                               send=make_http_send(timeout=5),
                               latency_meter=SyncLatencyMeter())
            services[i] = svc
            from meld.sync.service import _make_handler
            from http.server import ThreadingHTTPServer
            httpd = ThreadingHTTPServer(("127.0.0.1", ports[i]), _make_handler(svc))
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            httpds.append(httpd)

        for i in ids:                                          # load local observations
            services[i].node.ingest(claims[i])
        for i in ids:                                          # node-driven publish
            r = requests.post(urls[i] + "/publish", timeout=5).json()
            assert r["published"] == 1

        roots = [requests.get(urls[i] + "/metrics", timeout=5).json()["roots"] for i in ids]
        assert roots[0] == roots[1] == roots[2]                # all three hold the same sets
        texts = {d.text for d in services["A"].node.objects.iter("D", "agent")}
        assert texts == set(claims.values())
    finally:
        for h in httpds:
            h.shutdown()


def test_resync_runs_digest_pull_over_http_and_a_behind_peer_catches_up():
    # anti-entropy (design §2.2): B is behind (missed everything, as after a partition).
    # A's /resync runs DIGEST + PULL against B over real HTTP and pushes what B lacks; B
    # ends with byte-identical sets. Replaces the old text re-send of live claims.
    from meld.sync.latency import SyncLatencyMeter
    from meld.sync.service import BrainService
    from meld.sync.wire import make_http_call, make_http_send

    b = _node("B")
    port = _free_port()
    httpd = _start(b, port)
    try:
        b_url = f"http://127.0.0.1:{port}"
        a = BrainService(_node("A"), peers=[b_url], routing_mode="broadcast",
                         send=make_http_send(timeout=5), latency_meter=SyncLatencyMeter(),
                         call=make_http_call(timeout=5))
        a.node.ingest("Photosynthesis converts sunlight into chemical energy")
        a.node.ingest("Tectonic plates drift across the molten mantle")
        assert requests.get(b_url + "/metrics", timeout=5).json()["roots"] == {"agent": {}}

        status, body = a.handle("POST", "/resync")
        assert status == 200
        import json as _j
        r = _j.loads(body)
        assert (r["pulled"], r["pushed"], r["failed_peers"]) == (0, 2, []) and r["digest_bytes"] > 0

        ids = a.node.objects.ids("D", "agent")
        assert b.objects.ids("D", "agent") == ids
        assert [b.objects.raw(i) for i in ids] == [a.node.objects.raw(i) for i in ids]
        r = _j.loads(a.handle("POST", "/resync")[1])
        assert (r["pulled"], r["pushed"]) == (0, 0)
    finally:
        httpd.shutdown()


def test_unknown_route_over_http_is_non_delivery():
    from meld.sync.transport import PatchTransport
    from meld.sync.wire import make_http_send

    b = _node("B")
    port = _free_port()
    httpd = _start(b, port)
    try:
        # point the send at a bad path by using a base URL whose /patch 404s:
        # the service only serves /patch, so hit a server with no such node by
        # sending to a route the handler 404s -> non-delivery (contract holds on the wire).
        bad = f"http://127.0.0.1:{port}/nonexistent-base"   # POST {bad}/patch -> 404
        transport = PatchTransport(make_http_send(timeout=5))
        delivered = transport.publish("p2", b"x", [bad])
        assert delivered[bad] is False                      # 404 is NOT delivery (NPS H3)
    finally:
        httpd.shutdown()
