"""A memory condition whose receive path applies a baseline's update policy instead of MELD
admission.

``BaselineMemory(policy, **SMPMemory kwargs)`` is an ``SMPMemory`` (same store, publish,
receive gate and graph-walk recall) whose ``_admit`` hook runs ``policy`` on each newly held
claim. The policy sees the claim text and the top ``policy.k`` live claims by the memory's
encoder similarity (both baselines retrieve before they decide) and returns a ``Verdict``:
which retrieved claims to retire, whether the arriving claim is dropped, which rewritten
texts to store, the protocol decision it maps to, and the baseline's own event tag. Retiring
a claim is a REVOKE effect (the claim stays in the store for audit, never recalled again);
a rewritten text is published as a new claim. No link is ever emitted: neither baseline
surfaces a contradiction. MELD's Admission is constructed by the base class and never called.

Each arrival logs one ``classifier.Decision`` (``gates = (policy.name, event)``), and
``admission.decisions(scope)`` returns that log, so ``mem_conditions._counts`` and the recall
sweep read a baseline brain exactly as they read a MELD one.

``pair_rows``/``summarize_pairs`` score a policy on labelled claim pairs (one held claim, one
arrival): the four-way prediction (INSERT and REJECT count as NO_RELATION), the false-merge
flag (a non-MERGE pair merged), and, on CONFLICT pairs, a silent overwrite (the held claim
retired without a surfaced contradiction) or a missed contradiction (the held claim still
live and no CONFLICT decision), and a lost claim (a REJECT: the arrival stored nothing and
was absorbed into nothing, e.g. no facts extracted or an unparsable reply).
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

from meld.agent.memory import DEFAULT_CELL, DEFAULT_SCOPE, SMPMemory
from meld.merge.classifier import Decision
from meld.merge.m2_eval import classification_metrics
from meld.protocol.objects import AuthoredEffect, canonical_bytes, object_id, signed


class Verdict(NamedTuple):
    decision: str                       # INSERT / MERGE / CONFLICT / REJECT
    event: str                          # the baseline's own outcome tag
    revoke: Tuple[int, ...] = ()        # indices into the retrieved claims to retire
    add: Tuple[str, ...] = ()           # texts to store (the arriving text = keep it)
    target: Optional[int] = None        # the retrieved claim the decision is about
    events: tuple = ()                  # the baseline's own event list, if it reports one


class _DecisionLog:
    """Where the harness reads ``mem.admission.decisions(scope)``."""

    def __init__(self, log: List[Decision]):
        self._log = log

    def decisions(self, scope: str) -> List[Decision]:
        return list(self._log)


class BaselineMemory(SMPMemory):
    def __init__(self, policy, **kw):
        super().__init__(**kw)
        self.policy = policy
        self.admission = _DecisionLog(self.decisions)

    def _revoke(self, key: bytes) -> None:
        e = AuthoredEffect(scope=DEFAULT_SCOPE, kind="REVOKE", c1=key, c2=b"",
                           target_cell=self.store.cell_of()[key], author=self.origin,
                           author_cell=DEFAULT_CELL, author_auth=0, created_at=self._t)
        self.store.apply(*signed(e, self.key))

    def _admit(self, d) -> List[Decision]:
        nodes = self.store.nodes()
        if nodes.get(d.claim_key, (None,))[0] != object_id(canonical_bytes(d)):
            return []       # an exact copy of a held claim (one node) or no node: no decision
        live = [k for k in self.store.live_nodes() if k != d.claim_key]
        sim = {k: self._es.sigma(d.text, nodes[k][1].text) for k in live}
        old = sorted(live, key=lambda k: (-sim[k], k))[:self.policy.k]
        v = self.policy(d.text, [nodes[k][1].text for k in old])
        for i in v.revoke:
            self._revoke(old[i])
        if d.text not in v.add:
            self._revoke(d.claim_key)
        for t in dict.fromkeys(v.add):
            if t != d.text:
                self.publish(t)
        dec = Decision(b=d.claim_key, decision=v.decision,
                       a_star=None if v.target is None else old[v.target], tested=None,
                       s_q=None, theta_q=None, gates=(self.policy.name, v.event), links=(),
                       n_prior=len(live), K=tuple(old))
        self.decisions.append(dec)
        return [dec]


PRED = {"MERGE": "MERGE", "RELATE": "RELATE", "CONFLICT": "CONFLICT",
        "INSERT": "NO_RELATION", "REJECT": "NO_RELATION"}


def pair_rows(pairs: Iterable[Tuple[str, str, str]],
              make_memory: Callable[[], BaselineMemory]) -> List[dict]:
    """One row per (gold, text_a, text_b): text_a is held (published, no policy call), text_b
    arrives through the policy."""
    rows = []
    for gold, a, b in pairs:
        mem = make_memory()
        ka = mem.publish(a).claim_key
        dec = mem.write(b)
        decision, event = (dec[0].decision, dec[0].gates[1]) if dec else ("MERGE", "EXACT_COPY")
        a_live = ka in set(mem.store.live_nodes())
        rows.append({"gold": gold, "decision": decision, "event": event, "pred": PRED[decision],
                     "a_live": a_live,
                     "false_merge": gold != "MERGE" and decision == "MERGE",
                     "silent_overwrite": gold == "CONFLICT" and not a_live,
                     "missed_contradiction": gold == "CONFLICT" and a_live and decision != "CONFLICT",
                     "claim_lost": decision == "REJECT"})
    return rows


def summarize_pairs(rows: List[dict]) -> dict:
    conflict = [r for r in rows if r["gold"] == "CONFLICT"]
    non_merge = [r for r in rows if r["gold"] != "MERGE"]

    def rate(xs, key):
        return sum(r[key] for r in xs) / len(xs) if xs else 0.0
    return {"n": len(rows),
            "metrics": classification_metrics([r["gold"] for r in rows], [r["pred"] for r in rows]),
            "false_merge_rate": rate(non_merge, "false_merge"),
            "silent_overwrite_rate": rate(conflict, "silent_overwrite"),
            "missed_contradiction_rate": rate(conflict, "missed_contradiction"),
            "claim_lost_rate": rate(rows, "claim_lost"),
            "events": dict(Counter(r["event"] for r in rows))}


__all__ = ["BaselineMemory", "Verdict", "PRED", "pair_rows", "summarize_pairs"]
