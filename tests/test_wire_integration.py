"""Wire integration smoke: real ThreadingHTTPServer + real HTTP send + the delivery
contract + skew-free latency, all in-process (no docker). Proves the socket wrapper and
the on-the-wire send seam behave exactly as the unit-tested handle()/PatchTransport do,
before containerizing. make_http_send() is the real transport seam the orchestrator uses
on the testbed; here it POSTs to a live local server.
"""
from __future__ import annotations

import socket
import threading

import requests


def _cfg():
    from smp.agent.run_hotpot_gate import _calibrated_cfg
    return _calibrated_cfg("hashing")


def _node(nid):
    from smp.sync.node import BrainNode
    return BrainNode(nid, encoder="hashing", nli="lexical", cfg=_cfg())


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start(node, port):
    from smp.sync.service import serve
    httpd = serve(node, host="127.0.0.1", port=port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_publish_delivers_over_real_http_and_records_latency():
    from smp.sync.latency import SyncLatencyMeter
    from smp.sync.transport import PatchTransport
    from smp.sync.wire import make_http_send

    b = _node("B")
    port = _free_port()
    httpd = _start(b, port)
    try:
        url = f"http://127.0.0.1:{port}"
        # server is up: healthz responds
        assert requests.get(url + "/healthz", timeout=5).status_code == 200

        meter = SyncLatencyMeter()
        transport = PatchTransport(make_http_send(timeout=5), latency_meter=meter)
        delivered = transport.publish("p1", b"Paris is the capital of France", [url])

        assert delivered[url] is True                       # 200 -> delivered
        m = requests.get(url + "/metrics", timeout=5).json()
        assert any("Paris" in x for x in m["live"])         # merge-on-receive applied
        assert meter.latency_ms("p1") is not None           # latency recorded (all delivered)
        assert meter.latency_ms("p1") >= 0.0
    finally:
        httpd.shutdown()


def test_node_driven_publish_converges_three_brains_over_http():
    # the real §7-s path: each node publishes its OWN delta to its peers over HTTP
    # (node->peer, not orchestrator->peer), then all three converge.
    from smp.sync.latency import SyncLatencyMeter
    from smp.sync.service import BrainService, serve
    from smp.sync.wire import make_http_send

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
            from smp.sync.service import _make_handler
            from http.server import ThreadingHTTPServer
            httpd = ThreadingHTTPServer(("127.0.0.1", ports[i]), _make_handler(svc))
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            httpds.append(httpd)

        for i in ids:                                          # load local observations
            services[i].node.ingest(claims[i])
        for i in ids:                                          # node-driven publish
            r = requests.post(urls[i] + "/publish", timeout=5).json()
            assert r["published"] == 1

        for i in ids:                                          # all three converged
            live = requests.get(urls[i] + "/metrics", timeout=5).json()["live"]
            joined = " ".join(live)
            assert "Photosynthesis" in joined and "Tectonic" in joined and "Beethoven" in joined
    finally:
        for h in httpds:
            h.shutdown()


def test_resync_redelivers_full_state_to_a_behind_peer():
    # anti-entropy: B is behind (missed a claim, as after a partition). A's /resync sends
    # its FULL live set to B over the wire, so B catches up -- the convergence-after-heal
    # mechanism. A is a service object; B is a live server peer.
    from smp.sync.latency import SyncLatencyMeter
    from smp.sync.service import BrainService
    from smp.sync.wire import make_http_send

    b = _node("B")
    port = _free_port()
    httpd = _start(b, port)
    try:
        b_url = f"http://127.0.0.1:{port}"
        a = BrainService(_node("A"), peers=[b_url], routing_mode="broadcast",
                         send=make_http_send(timeout=5), latency_meter=SyncLatencyMeter())
        a.node.ingest("Photosynthesis converts sunlight into chemical energy")
        a.node.ingest("Tectonic plates drift across the molten mantle")
        # B starts empty (it missed everything)
        assert requests.get(b_url + "/metrics", timeout=5).json()["live_count"] == 0

        status, body = a.handle("POST", "/resync")
        assert status == 200
        import json as _j
        assert _j.loads(body)["resynced"] == 2

        live = " ".join(requests.get(b_url + "/metrics", timeout=5).json()["live"])
        assert "Photosynthesis" in live and "Tectonic" in live   # B caught up
    finally:
        httpd.shutdown()


def test_unknown_route_over_http_is_non_delivery():
    from smp.sync.transport import PatchTransport
    from smp.sync.wire import make_http_send

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
