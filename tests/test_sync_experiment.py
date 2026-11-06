"""In-process experiment driver: composes the sync primitives into one runnable
eval-s / eval-r round over BrainNodes, so the full experiment LOGIC is verifiable
end-to-end before any docker/edge-testbed. Models §7-s direct publish: each node publishes
its local delta to its peers via PatchTransport (skew-free SyncLatencyMeter), routed
either semantically (meld.sync.router) or by broadcast. Reports the sync-latency
summary (eval-s), the per-claim delivered-sets (eval-r, scored against a needed-oracle
via router.routing_metrics), and presence_agreement -- the fraction of nodes that hold
each live claim (full propagation == 1.0). The status-CRDT convergence story (eval-c
partition->heal) stays in eval.crdt_convergence; this driver does NOT redefine it.
"""
from __future__ import annotations




def _node(nid):
    from meld.sync.node import BrainNode
    return BrainNode(nid, encoder="hashing", nli="lexical")


def test_broadcast_full_mesh_converges_and_delivers_to_all_peers():
    from meld.sync.experiment import run_sync
    a, b, c = _node("A"), _node("B"), _node("C")
    nodes = {"A": a, "B": b, "C": c}
    a.ingest("Paris is the capital of France")
    b.ingest("Mount Everest is the tallest mountain")
    c.ingest("The Pacific is the largest ocean")
    peers = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
    res = run_sync(nodes, peers, routing_mode="broadcast",
                   encoder="hashing", clock=lambda: 0.0)
    assert res.presence_agreement == 1.0                 # every node holds every claim
    assert all(len(d) == 2 for d in res.delivered.values())   # each claim -> its 2 peers


def test_latency_recorded_per_published_claim():
    from meld.sync.experiment import run_sync
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")           # only A has anything to publish
    nodes = {"A": a, "B": b}
    peers = {"A": ["B"], "B": ["A"]}
    clk = [0.0]

    def clock():                                         # advance 10 ms per read
        v = clk[0]; clk[0] += 0.01; return v
    res = run_sync(nodes, peers, routing_mode="broadcast", encoder="hashing", clock=clock)
    assert res.latency["n"] == 1                         # one Patch published
    assert res.latency["p50_ms"] >= 0.0


def test_semantic_routes_to_most_relevant_peer_only():
    from meld.sync.experiment import run_sync
    src, paris, everest = _node("SRC"), _node("PARIS"), _node("EVEREST")
    paris.ingest("Paris France capital Europe Seine river")
    everest.ingest("Everest mountain Himalaya climbing altitude snow")
    src.ingest("The capital of France is Paris on the Seine")
    nodes = {"SRC": src, "PARIS": paris, "EVEREST": everest}
    peers = {"SRC": ["PARIS", "EVEREST"], "PARIS": [], "EVEREST": []}
    res = run_sync(nodes, peers, routing_mode="semantic",
                   encoder="hashing", sigma_lo=0.0, top_k=1, clock=lambda: 0.0)
    claim = "The capital of France is Paris on the Seine"
    assert res.delivered[claim] == {"PARIS"}             # top-1 relevant peer, not EVEREST


def test_semantic_uses_fewer_deliveries_than_broadcast():
    from meld.sync.experiment import run_sync

    def setup():
        src, paris, everest = _node("SRC"), _node("PARIS"), _node("EVEREST")
        paris.ingest("Paris France capital Europe Seine river")
        everest.ingest("Everest mountain Himalaya climbing altitude snow")
        src.ingest("The capital of France is Paris on the Seine")
        return ({"SRC": src, "PARIS": paris, "EVEREST": everest},
                {"SRC": ["PARIS", "EVEREST"], "PARIS": [], "EVEREST": []})

    n1, p1 = setup()
    bc = run_sync(n1, p1, routing_mode="broadcast", encoder="hashing", clock=lambda: 0.0)
    n2, p2 = setup()
    sm = run_sync(n2, p2, routing_mode="semantic",
                  encoder="hashing", sigma_lo=0.0, top_k=1, clock=lambda: 0.0)
    bc_total = sum(len(d) for d in bc.delivered.values())
    sm_total = sum(len(d) for d in sm.delivered.values())
    assert sm_total < bc_total                           # the routing win: fewer deliveries


def test_run_sync_records_every_delta_per_sender_and_forwards_rules():
    # two senders publish the same text: `delivered` (keyed by text) keeps one, `deliveries`
    # keeps both (src, text, peers) records; margin/gap reach the router.
    from meld.sync.experiment import run_sync
    a, b, c = _node("A"), _node("B"), _node("C")
    a.ingest("Paris is the capital of France")
    b.ingest("Paris is the capital of France")
    c.ingest("Everest mountain Himalaya climbing altitude snow")
    nodes = {"A": a, "B": b, "C": c}
    peers = {"A": ["B", "C"], "B": ["A", "C"], "C": []}
    res = run_sync(nodes, peers, routing_mode="semantic", encoder="hashing",
                   sigma_lo=0.0, margin=1e-6, clock=lambda: 0.0)
    recs = sorted((s, t, frozenset(p)) for s, t, p in res.deliveries)
    assert recs == [("A", "Paris is the capital of France", frozenset({"B"})),
                    ("B", "Paris is the capital of France", frozenset({"A"})),
                    ("C", "Everest mountain Himalaya climbing altitude snow", frozenset())]
