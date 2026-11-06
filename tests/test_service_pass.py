"""POST /pass over the brain service: one readjudication on the node's own provider, returning the
PASS certificate fingerprint, the pass-time misses and the five-component view per scope; /metrics
carries the last pass's fingerprint. Over the wire, Sig records travel by /resync (one of the five
kinds), so resync + pass reproduce the in-process settle.
"""
from __future__ import annotations

import json
import random

from meld.agent.memory import DEFAULT_POL
from meld.eval.anti_entropy_eval import with_effects
from meld.eval.graph_fingerprint import COMPONENTS
from meld.eval.order_shuffle import gold_workload
from meld.protocol.objects import signed
from meld.sync.node import BrainNode
from meld.sync.service import BrainService


def _three(drop=0.2, seed=0):
    deltas, roles = gold_workload("hashing", n_seeds=6)
    work = with_effects(deltas, roles, DEFAULT_POL)
    svcs = {}
    for i in "ABC":
        svcs[i] = BrainService(BrainNode(i, encoder="hashing", nli="lexical"), peers=[j for j in "ABC" if j != i],
                               call=lambda p, path, body: svcs[p].handle("POST", path, body))
    rng = random.Random(seed)
    for s in svcs.values():
        for o in work:
            if rng.random() >= drop:
                s.node.receive_object(*signed(o, s.node.key), None)
    return svcs


def _settle(svcs, rounds=3):
    """The driver's heal: resync rounds until the roots agree, then (resync + pass) until no pass misses."""
    for _ in range(5):
        for s in svcs.values():
            s.handle("POST", "/resync")
        if len({json.dumps(json.loads(s.handle("GET", "/metrics")[1])["roots"], sort_keys=True)
                for s in svcs.values()}) == 1:
            break
    for _ in range(rounds):
        res = {i: json.loads(s.handle("POST", "/pass")[1])["agent"] for i, s in svcs.items()}
        if not any(r["pass_time_misses"] for r in res.values()):
            break
        for s in svcs.values():
            s.handle("POST", "/resync")
    return res


def test_resync_then_pass_gives_one_fingerprint_and_metrics_carry_it():
    svcs = _three()
    res = _settle(svcs)
    assert len({r["fingerprint"] for r in res.values()}) == 1
    assert all(set(r["components"]) == set(COMPONENTS) for r in res.values())
    m = json.loads(svcs["A"].handle("GET", "/metrics")[1])
    assert m["fingerprint"]["agent"] == res["A"]["fingerprint"]
    svcs["A"].handle("POST", "/reset")
    assert json.loads(svcs["A"].handle("GET", "/metrics")[1])["fingerprint"] == {}


def test_a_planted_divergence_gives_two_fingerprints():
    svcs = _three()
    _settle(svcs)
    extra = BrainNode("Z", encoder="hashing", nli="lexical").mem.delta("a claim only replica C holds")
    svcs["C"].node.receive_object(*signed(extra, svcs["C"].node.key), None)
    res = {i: json.loads(s.handle("POST", "/pass")[1])["agent"] for i, s in svcs.items()}
    assert len({r["fingerprint"] for r in res.values()}) == 2
