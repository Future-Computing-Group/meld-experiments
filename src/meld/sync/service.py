"""MELD-brain HTTP service -- the containerized node for the multi-host sync runs.

Framework-free (stdlib http.server) to keep the public-bound repo dependency-lean.
The routing is a PURE method -- ``BrainService.handle(method, path, body) -> (status,
body)`` -- so it is unit-testable with no socket; ``serve()`` is the thin
ThreadingHTTPServer wrapper, exercised at the docker smoke. Endpoints:

  * POST /ingest   body = one local observation text -> a signed Delta in the own sets.
  * POST /patch    PUBLISH receive: body = envelopes (sync.wire_delta) of signed objects of
                   any kind; each is MAC-verified into the sets (design §2.2-2.3). 200 iff
                   every envelope was accepted, else 403 (non-delivery).
  * POST /publish  ships the stored signed bytes of every own object not yet published,
                   routed (semantic for deltas | broadcast) to the peers; 503 if no send seam.
                   Reply: per object id {kind, text (deltas), peers delivered} and the bytes
                   sent per message class (sync/wire_bytes.meter).
  * POST /digest   DIGEST: body {"scope", "roots": {kind: root}} -> {"roots", "ids"}.
  * POST /pull     PULL: body {"scope", "kind", "ids": [hex]} -> envelopes of original bytes.
  * POST /resync   runs one DIGEST/PULL anti-entropy round with every peer for every
                   subscribed scope (replaces the text re-send); 503 if no call seam. Reply
                   carries request + reply bytes per call path (DIGEST, PULL, PATCH).
  * POST /pass     one pass per scope on the node's own provider -> {scope: {fingerprint,
                   pass_time_misses, components}}; /metrics carries the last fingerprint.
  * POST /reset    per-cell isolation; body {"peers": [...]} optionally sets the cell's peers.
  * GET  /metrics  -> {node_id, live_count, live[], status, roots, latency}; ``roots`` is
                   the Merkle root per scope and kind, the replication convergence probe.
  * GET  /healthz  -> liveness (orchestrator peer-health + transport probe).

RETIRED: POST /status (an unsigned local status write; 04 finding 10). A status change
enters only as a published, MAC-verified AuthoredEffect through /patch. It now 404s.

Pairs with PatchTransport's 2xx-only delivery contract: success is 200, an unknown
route is 404 (never a silent 200), so non-delivery is never mistaken for delivery.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional, Tuple

import requests

from meld.config import load_epoch
from meld.eval.graph_fingerprint import components
from meld.merge.signals import EmbeddingSimilarity
from meld.protocol.fold import readjudicate
from meld.sync.anti_entropy import digest_response, objectset_round, pull_response, scope_digest
from meld.sync.latency import SyncLatencyMeter
from meld.sync.node import BrainNode
from meld.sync.objectset import KINDS
from meld.sync.router import semantic_route
from meld.sync.transport import PatchTransport
from meld.sync.wire import make_http_call, make_http_send
from meld.sync.wire_bytes import meter
from meld.sync.wire_delta import group_key, pack, unpack


class BrainService:
    def __init__(self, node: BrainNode, *, peers: Optional[List[str]] = None,
                 routing_mode: str = "broadcast", sigma_lo: float = 0.0,
                 top_k: Optional[int] = None, send: Optional[Callable] = None,
                 latency_meter=None,
                 peer_live: Optional[Callable[[str], List[str]]] = None,
                 group_key: Optional[bytes] = None,
                 call: Optional[Callable[[str, str, bytes], Tuple[int, bytes]]] = None,
                 boot: Optional[List[Tuple[bytes, str, Optional[str]]]] = None):
        self.node = node
        self.peers = list(peers or [])
        self.routing_mode = routing_mode
        self.sigma_lo = sigma_lo
        self.top_k = top_k
        self.meter = latency_meter
        self.transport = PatchTransport(send, latency_meter=latency_meter) if send else None
        self._peer_live = peer_live          # for semantic routing over the wire
        self._call = call                    # request/response seam for DIGEST/PULL
        # the group HMAC key is the node's: objects are signed and verified under it
        if group_key is not None and group_key != node.key:
            raise ValueError("group_key differs from the node's key")
        self.group_key = node.key
        # objects the node holds from boot (the deployed epoch): re-received after every /reset
        self.boot = list(boot or [])
        self._receive_boot()
        self._fingerprint = {}               # scope -> last PASS certificate fingerprint (hex)

    def _receive_boot(self) -> None:
        for e in self.boot:
            if not self.node.receive_object(*e):
                raise ValueError("SMP_EPOCH refused: bad MAC, scope or record")

    def _fetch_peer_content(self) -> dict:
        """Pull each peer's live claim set ONCE per publish (not per claim) -- the content
        semantic routing scores against. Empty if no peer_live is wired."""
        return {p: (self._peer_live(p) if self._peer_live else []) for p in self.peers}

    def _targets(self, claim: str, peer_content: Optional[dict] = None, es=None) -> List[str]:
        if self.routing_mode == "broadcast":
            return list(self.peers)
        if self.routing_mode == "semantic":
            brains = peer_content if peer_content is not None else self._fetch_peer_content()
            return sorted(semantic_route(claim, brains, encoder=self.node.mem.encoder,
                                         sigma_lo=self.sigma_lo, top_k=self.top_k, es=es))
        raise ValueError(f"unknown routing_mode: {self.routing_mode!r}")   # no silent default

    def _publish(self) -> Tuple[int, bytes]:
        if self.transport is None:
            return 503, json.dumps({"error": "no send seam configured"}).encode("utf-8")
        ids = self.node.outbox()
        peer_content = es = None
        if self.routing_mode == "semantic":
            peer_content = self._fetch_peer_content()
            es = EmbeddingSimilarity(encoder=self.node.mem.encoder)   # ONE cache across all claims
        delivered, sent = {}, []
        for oid in ids:
            delta = self.node.objects.get("D", oid)
            targets = (self._targets(delta.text, peer_content, es) if delta is not None
                       else list(self.peers))          # only deltas carry text to route on
            raw = self.node.objects.raw(oid)
            d = self.transport.publish(f"{self.node.node_id}:{oid.hex()}", pack([raw]), targets)
            sent += [raw] * len(targets)
            delivered[oid.hex()] = {"kind": self.node.objects.kind_of(oid),
                                    "text": delta.text if delta is not None else None,
                                    "peers": [p for p, ok in d.items() if ok]}
        return 200, json.dumps({
            "published": len(ids), "delivered": delivered, "bytes": meter(sent),
            "latency": self.meter.summary() if self.meter else None,
        }).encode("utf-8")

    def _resync(self) -> Tuple[int, bytes]:
        """Anti-entropy: one DIGEST/PULL round with every peer for every subscribed scope.
        A peer that fails (lost call, malformed reply, exception) costs only its own round."""
        if self._call is None:
            return 503, json.dumps({"error": "no call seam configured"}).encode("utf-8")
        pulled = pushed = digest_bytes = 0
        failed = []
        ae = {"DIGEST": 0, "PULL": 0, "PATCH": 0}       # request + reply bytes per call path
        for peer in self.peers:
            def call(path, body, peer=peer):
                status, reply = self._call(peer, path, body)
                ae[path.strip("/").upper()] += len(body) + len(reply)
                return status, reply
            try:
                for s in sorted(self.node.objects.scopes):
                    p, q, b = objectset_round(self.node.objects, s, call, self.node.receive_object)
                    pulled, pushed, digest_bytes = pulled + p, pushed + q, digest_bytes + b
            except Exception:                         # noqa: BLE001 - a lost round for that peer
                failed.append(peer)
        return 200, json.dumps({"pulled": pulled, "pushed": pushed, "digest_bytes": digest_bytes,
                                "ae_bytes": ae, "failed_peers": failed}).encode()

    def _pass(self) -> Tuple[int, bytes]:
        """One pass per subscribed scope (fold.readjudicate on the node's own provider), as the
        in-process settle runs it; the new Sig records enter the own sets and travel by /resync
        or /publish. Reply per scope: the PASS certificate fingerprint, the pass-time misses and
        the five-component view (eval/graph_fingerprint.py)."""
        m, out = self.node.mem, {}
        with self.node._lock:
            for s in sorted(self.node.objects.scopes):
                r = readjudicate(m.store, s, m_nli=m.m_nli, provider=m.admission.provider,
                                 issuer=m.origin, hw=m.admission.hw)
                out[s] = {"fingerprint": r.certificate.fingerprint.hex(),
                          "pass_time_misses": r.counters["pass_time_misses"],
                          "components": {c: v.hex() for c, v in components(m.store, s, r.decisions).items()}}
                self._fingerprint[s] = out[s]["fingerprint"]
        return 200, json.dumps(out).encode()

    def _patch(self, body: bytes) -> Tuple[int, bytes]:
        entries = unpack(body) or []
        ok = sum(self.node.receive_object(*e) for e in entries)
        status = 200 if entries and ok == len(entries) else 403
        return status, json.dumps({"received": ok, "dropped": len(entries) - ok}).encode()

    def _digest_or_pull(self, path: str, body: bytes) -> Tuple[int, bytes]:
        try:
            d = json.loads(body)
            scope = d["scope"]
            if type(scope) is not str or scope not in self.node.objects.scopes:
                raise ValueError(scope)
            if path == "/digest":
                roots = d.get("roots", {})
                if not isinstance(roots, dict):
                    raise ValueError(roots)
                return 200, json.dumps(digest_response(self.node.objects, scope, roots)).encode()
            if d["kind"] not in KINDS:
                raise ValueError(d)
            return 200, pack(pull_response(self.node.objects, scope, d["kind"], d["ids"]))
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            return 400, json.dumps({"error": "malformed request"}).encode()

    def _reset(self, body: bytes = b"") -> Tuple[int, bytes]:
        """Per-cell isolation: drop the node's brain state + the latency records, so a
        campaign cell starts fresh (independent seeds; each RTT regime re-measured). The boot
        objects (the deployed epoch) are received again. A body {"peers": [url, ...]} sets the
        cell's peers (a subset of the deployed brains), so one deployment serves every cell."""
        try:
            peers = json.loads(body)["peers"] if body.strip() else self.peers
            if type(peers) is not list or not all(type(p) is str for p in peers):
                raise ValueError(peers)
        except (ValueError, TypeError, KeyError):
            return 400, json.dumps({"error": "malformed request"}).encode()
        self.peers = list(peers)
        self._fingerprint = {}
        self.node.reset()
        self._receive_boot()
        if self.meter is not None:
            self.meter.reset()
        return 200, json.dumps({"reset": True, "peers": self.peers}).encode("utf-8")

    def handle(self, method: str, path: str, body: bytes = b"") -> Tuple[int, bytes]:
        if method == "POST" and path == "/reset":
            return self._reset(body)
        if method == "POST" and path == "/resync":
            return self._resync()
        if method == "POST" and path == "/pass":
            return self._pass()
        if method == "POST" and path == "/ingest":
            text = body.decode("utf-8").strip()
            if text:
                self.node.ingest(text)
            return 200, json.dumps({"ingested": 1 if text else 0}).encode("utf-8")
        if method == "POST" and path == "/patch":
            return self._patch(body)
        if method == "POST" and path == "/publish":
            return self._publish()
        if method == "POST" and path in ("/digest", "/pull"):
            return self._digest_or_pull(path, body)
        # RETIRED: POST /status (unsigned local status write) falls through to 404.
        if method == "GET" and path == "/metrics":
            live = sorted(self.node.live_claim_texts())
            # per-claim effective status over ALL claims (incl. overruled, which are not live)
            all_texts = {d.text for _, d in self.node.mem.store.nodes().values()}
            status = {t: self.node.status_of(t) for t in all_texts}
            roots = {s: {k: d.set_hash for k, d in scope_digest(self.node.objects, s).items()}
                     for s in sorted(self.node.objects.scopes)}
            return 200, json.dumps(
                {"node_id": self.node.node_id, "live_count": len(live), "live": live,
                 "status": status, "roots": roots, "routing_mode": self.routing_mode,
                 "encoder": self.node.mem.encoder, "nli": self.node.mem.nli,
                 "m_nli": self.node.mem.m_nli.hex(), "host_cpu": _cpu_model(),
                 "epoch_ids": sorted(i.hex() for s in self.node.objects.scopes
                                     for i in self.node.objects.ids("Ep", s)),
                 "fingerprint": self._fingerprint,
                 "latency": self.meter.summary() if self.meter else None}
            ).encode("utf-8")
        if method == "GET" and path == "/healthz":
            return 200, json.dumps({"status": "ok"}).encode("utf-8")
        return 404, json.dumps({"error": "not found"}).encode("utf-8")


def _cpu_model() -> str:
    """The host CPU model (the brain's service time depends on it)."""
    import platform
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


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
    one raises. SMP_POL names an M_pol JSON, SMP_EPOCH a signed epoch file (config.load_epoch;
    without one the node makes no similarity merge). Experiment knobs (peers, routing, sigma_lo, top_k) carry documented
    baselines. Returns (service, port)."""

    def _req(key):
        v = env.get(key)
        if not v:
            raise ValueError(f"{key} is required (no silent default)")
        return v

    if env.get("SMP_GROUP_KEY_FILE"):         # the run key as a mounted file, not an env value
        with open(env["SMP_GROUP_KEY_FILE"], "rb") as fh:
            key = fh.read().strip().decode("utf-8")
        if not key:                           # never fall back to the labelled test key
            raise ValueError("SMP_GROUP_KEY_FILE is empty")
        env = {**env, "SMP_GROUP_KEY": key}
    node_id = _req("SMP_NODE_ID")
    port = int(_req("SMP_PORT"))
    encoder = _req("SMP_ENCODER")
    nli = _req("SMP_NLI")
    peers = [p for p in (env.get("SMP_PEERS", "").split(",")) if p.strip()]
    routing_mode = env.get("SMP_ROUTING_MODE", "broadcast")
    sigma_lo = float(env.get("SMP_SIGMA_LO", "0.0"))
    top_k = int(env["SMP_TOP_K"]) if env.get("SMP_TOP_K") else None
    timeout = float(env.get("SMP_HTTP_TIMEOUT", "10.0"))

    def _peer_live(peer_url):
        """Fetch a peer's live claim set for semantic routing to score against."""
        try:
            r = requests.get(peer_url.rstrip("/") + "/metrics", timeout=timeout)
            return r.json().get("live", []) if r.status_code == 200 else []
        except Exception:
            return []

    pol = None
    if env.get("SMP_POL"):                    # M_pol as JSON: rank_map, scopes, ...
        with open(env["SMP_POL"], encoding="utf-8") as fh:
            pol = json.load(fh)
    node = BrainNode(node_id, encoder=encoder, nli=nli, pol=pol, key=group_key(env))
    # the calibration authority's signed epoch record, held from boot and across /reset
    boot = [(*load_epoch(env["SMP_EPOCH"]), None)] if env.get("SMP_EPOCH") else []
    svc = BrainService(node, boot=boot, peers=peers, routing_mode=routing_mode, sigma_lo=sigma_lo,
                       top_k=top_k, send=make_http_send(timeout=timeout),
                       latency_meter=SyncLatencyMeter(), peer_live=_peer_live,
                       call=make_http_call(timeout=timeout))
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
