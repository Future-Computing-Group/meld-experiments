"""SMP-brain HTTP service -- the containerized node for the multi-host sync runs.

Framework-free (stdlib http.server) to keep the public-bound repo dependency-lean.
The routing is a PURE method -- ``BrainService.handle(method, path, body) -> (status,
body)`` -- so it is unit-testable with no socket; ``serve()`` is the thin
ThreadingHTTPServer wrapper, exercised at the docker smoke. Endpoints:

  * POST /ingest   body = one local observation text -> merge into the own brain.
  * POST /patch    body = newline-separated claim texts (a delta) -> merge-on-receive.
  * POST /publish  this node extracts its delta, routes (semantic|broadcast), and pushes
                   to its peers over the wire (node->peer, the real §7-s path); 503 if no
                   send seam was configured (never a silent OK).
  * GET  /metrics  -> {node_id, live_count, live[], latency} (eval-c/eval-s probe).
  * GET  /healthz  -> liveness (orchestrator peer-health + transport probe).

Pairs with PatchTransport's 2xx-only delivery contract: success is 200, an unknown
route is 404 (never a silent 200), so non-delivery is never mistaken for delivery.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional, Tuple

from smp.sync.node import BrainNode
from smp.sync.router import broadcast_route, semantic_route
from smp.sync.transport import PatchTransport


class BrainService:
    def __init__(self, node: BrainNode, *, peers: Optional[List[str]] = None,
                 routing_mode: str = "broadcast", sigma_lo: float = 0.0,
                 top_k: Optional[int] = None, send: Optional[Callable] = None,
                 latency_meter=None,
                 peer_live: Optional[Callable[[str], List[str]]] = None):
        self.node = node
        self.peers = list(peers or [])
        self.routing_mode = routing_mode
        self.sigma_lo = sigma_lo
        self.top_k = top_k
        self.meter = latency_meter
        self.transport = PatchTransport(send, latency_meter=latency_meter) if send else None
        self._peer_live = peer_live          # for semantic routing over the wire

    def _targets(self, claim: str) -> List[str]:
        if self.routing_mode == "broadcast":
            return list(self.peers)
        if self.routing_mode == "semantic":
            brains = {p: (self._peer_live(p) if self._peer_live else []) for p in self.peers}
            return sorted(semantic_route(claim, brains, encoder=self.node.mem.encoder,
                                         sigma_lo=self.sigma_lo, top_k=self.top_k))
        raise ValueError(f"unknown routing_mode: {self.routing_mode!r}")   # no silent default

    def _publish(self) -> Tuple[int, bytes]:
        if self.transport is None:
            return 503, json.dumps({"error": "no send seam configured"}).encode("utf-8")
        delta = self.node.extract_delta()
        delivered = {}
        for claim in delta:
            d = self.transport.publish(f"{self.node.node_id}:{claim}",
                                       claim.encode("utf-8"), self._targets(claim))
            delivered[claim] = [p for p, ok in d.items() if ok]
        return 200, json.dumps({
            "published": len(delta), "delivered": delivered,
            "latency": self.meter.summary() if self.meter else None,
        }).encode("utf-8")

    def _resync(self) -> Tuple[int, bytes]:
        """Anti-entropy: re-send the FULL live set to every peer (idempotent merge-on-
        receive), so a peer behind after a partition heal catches up. Not routed and not
        latency-measured -- it is a catch-up round, not a steady-state publish."""
        if self.transport is None:
            return 503, json.dumps({"error": "no send seam configured"}).encode("utf-8")
        state = self.node.full_state()
        payload = "\n".join(state).encode("utf-8")
        delivered = {}
        for peer in self.peers:
            d = self.transport.publish(f"{self.node.node_id}:resync", payload, [peer])
            delivered[peer] = d.get(peer, False)
        return 200, json.dumps({"resynced": len(state), "delivered": delivered}).encode("utf-8")

    def handle(self, method: str, path: str, body: bytes = b"") -> Tuple[int, bytes]:
        if method == "POST" and path == "/resync":
            return self._resync()
        if method == "POST" and path == "/ingest":
            text = body.decode("utf-8").strip()
            if text:
                self.node.ingest(text)
            return 200, json.dumps({"ingested": 1 if text else 0}).encode("utf-8")
        if method == "POST" and path == "/patch":
            claims = [c for c in body.decode("utf-8").split("\n") if c.strip()]
            self.node.receive(claims)
            return 200, json.dumps({"received": len(claims)}).encode("utf-8")
        if method == "POST" and path == "/publish":
            return self._publish()
        if method == "GET" and path == "/metrics":
            live = sorted(self.node.live_claim_texts())
            return 200, json.dumps(
                {"node_id": self.node.node_id, "live_count": len(live), "live": live,
                 "latency": self.meter.summary() if self.meter else None}
            ).encode("utf-8")
        if method == "GET" and path == "/healthz":
            return 200, json.dumps({"status": "ok"}).encode("utf-8")
        return 404, json.dumps({"error": "not found"}).encode("utf-8")


def _make_handler(service: BrainService):
    class _Handler(BaseHTTPRequestHandler):
        def _respond(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            status, body = service.handle("GET", self.path)
            self._respond(status, body)

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(n) if n else b""
            status, resp = service.handle("POST", self.path, body)
            self._respond(status, resp)

        def log_message(self, *_a):  # silence default stderr access log
            return

    return _Handler


def serve(node: BrainNode, *, host: str = "0.0.0.0", port: int = 8800) -> ThreadingHTTPServer:
    """Start the stdlib HTTP server for ``node`` and return it (caller runs
    serve_forever / shutdown). Thin wrapper over the tested ``handle``."""
    httpd = ThreadingHTTPServer((host, port), _make_handler(BrainService(node)))
    return httpd


def build_service_from_env(env) -> Tuple["BrainService", int]:
    """Construct the node service for a container from env vars. Core config
    (node id / port / encoder / nli) has NO silent default -- a missing
    one raises. Experiment knobs (peers, routing, sigma_lo, top_k) carry documented
    baselines. Returns (service, port)."""
    from smp.config import load_thresholds
    from smp.sync.latency import SyncLatencyMeter
    from smp.sync.wire import make_http_send

    def _req(key):
        v = env.get(key)
        if not v:
            raise ValueError(f"{key} is required (no silent default)")
        return v

    node_id = _req("SMP_NODE_ID")
    port = int(_req("SMP_PORT"))
    encoder = _req("SMP_ENCODER")
    nli = _req("SMP_NLI")
    peers = [p for p in (env.get("SMP_PEERS", "").split(",")) if p.strip()]
    routing_mode = env.get("SMP_ROUTING_MODE", "broadcast")
    sigma_lo = float(env.get("SMP_SIGMA_LO", "0.0"))
    top_k = int(env["SMP_TOP_K"]) if env.get("SMP_TOP_K") else None
    timeout = float(env.get("SMP_HTTP_TIMEOUT", "10.0"))

    node = BrainNode(node_id, encoder=encoder, nli=nli, cfg=load_thresholds())
    svc = BrainService(node, peers=peers, routing_mode=routing_mode, sigma_lo=sigma_lo,
                       top_k=top_k, send=make_http_send(timeout=timeout),
                       latency_meter=SyncLatencyMeter())
    return svc, port


def serve_main(env=None) -> None:  # pragma: no cover - container entrypoint
    """Container entrypoint: build the node from env and serve until killed."""
    import os
    svc, port = build_service_from_env(env if env is not None else os.environ)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(svc))
    httpd.serve_forever()


__all__ = ["BrainService", "serve", "build_service_from_env", "serve_main"]


if __name__ == "__main__":  # pragma: no cover
    serve_main()
