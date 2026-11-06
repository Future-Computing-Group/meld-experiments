"""The step of the fold (design §3.3, §4.3-4.4): the decision for one node b given the nodes
processed before it, and ``select_tested_pair``, the selector the fold and the calibration
builder share.

step(state, b, ctx), first match:

    REJECT    b not admissible (R1: status ACTIVE and retention >= r_min), before anything.
    INSERT    K(b) empty: b opens a group.
    CONFLICT  the canonical-direction mask on a* = K[0] fires (gate "mask"); or, after the test, a pair mask
              on (a_e, b) in either direction ("pair_mask"), or a collected CONTRADICTS(b, r)
              with r in group(a_e) ("group_contradiction").
    MERGE     s(a_e, b) > theta_{g(b)} strictly, and neither contradiction gate, the group
              cannot-link ("group_cannot_link": DISTINGUISHES between b and a member of
              group(a_e)) nor the Cal veto ("cal_veto": a NON_MERGEABLE Cal record for (a_e, b))
              fires: b joins group(a_e).
    RELATE    otherwise (abstention): b opens a group, ASSOCIATES(b, a*).

Selection is theta- and Sig-independent: K(b) = the top-m processed nodes by exact sigma_q with
sigma_q >= sigma_lo, ties by canonical order; a_e = the first r in K(b) with cell(r) = cell(b),
rank(b) <= rank(r) and no authorized DISTINGUISHES(b, r). It never reads vis or auth.

The step reads NLI evidence through ``ctx.sig`` only (``dir(premise, hyp)``, ``pair(x, y)``; the
fold's SigView over joined Sig records). The read set is fixed by the selector: the test pair
(both directions, read first so one record carries both), the mask on a*, and the canonical
direction of every r in K(b) minus a* with sigma_q >= sigma_c. Each is read whatever the
outcome, so a pass reads the same pairs under any Sig and any epoch table (Prop 3, Lemma 6).
A missing payload is bottom: it cannot pass the test and never contradicts.

``ctx.prefilter`` (None everywhere but the admission path's optional index flag) narrows the rows
the exact top-m ranks; the fold and the calibration builder never set it.

Without an epoch, or in a bin without a row, theta is INF16 and no similarity merge happens.

Decisions not covered by the text of §3.3, resolved here: the two contradiction gates send the
pair to CONFLICT whether or not the test passed (first-match order puts CONFLICT before MERGE
and RELATE; either way they only move away from MERGE, Lemma 5b), and a pair-mask CONFLICT adds
CONTRADICTS(b, a_e) when no collected link records it. CONSOLIDATED_FROM(b -> rep) names the
representative before b joins, so no raw link is a self-loop.

Decision.gates holds the gate names above that fired, in that vocabulary; "mask" (the a* mask
before the test) is recorded beside the four gates after the test.
"""
from __future__ import annotations

import math
from typing import AbstractSet, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np

from meld.brain.store import Link
from meld.merge.signals import sigma_q_cmp, topm_exact
from meld.proto import canon
from meld.protocol.objects import Delta, contradicts, score_s


class Node(NamedTuple):
    key: bytes
    cell: str
    order: Tuple[int, bytes]    # (created_at, delta_id): the canonical order
    q: np.ndarray               # int8
    text: str                   # canon(text), the NLI input
    status: str
    retention: int


def node_of(d: Delta, delta_id: bytes) -> Node:
    return Node(d.claim_key, d.cell, (d.created_at, delta_id), np.frombuffer(d.q, np.int8),
                canon(d.text), d.status, d.retention)


class FoldState:
    """Processed nodes in canonical order (``add`` refuses any other order), their int8 matrix
    with its float32 copy and int64 squared norms (kept incrementally for topm_exact), the
    partition into groups (group id = founding key) and the link multiset."""

    def __init__(self):
        self.keys: List[bytes] = []
        self.nodes: Dict[bytes, Node] = {}
        self.group_of: Dict[bytes, bytes] = {}
        self.members: Dict[bytes, List[bytes]] = {}
        self.links: List[Link] = []
        self._q = self._f = self._n = None

    def add(self, b: Node) -> None:
        if self.keys and b.order <= self.nodes[self.keys[-1]].order:
            raise ValueError("nodes must be added in strictly increasing canonical order")
        n = len(self.keys)
        if self._q is None or n == len(self._q):
            cap, d = max(16, 2 * n), len(b.q)
            grown = (np.zeros((cap, d), np.int8), np.zeros((cap, d), np.float32),
                     np.zeros(cap, np.int64))
            if n:
                for new, old in zip(grown, (self._q, self._f, self._n)):
                    new[:n] = old
            self._q, self._f, self._n = grown
        self._q[n] = b.q
        self._f[n] = b.q
        self._n[n] = int((b.q.astype(np.int64) ** 2).sum())
        self.keys.append(b.key)
        self.nodes[b.key] = b

    def matrices(self):
        """(Q int8, Q float32, squared norms int64) of the processed rows."""
        n = len(self.keys)
        return self._q[:n], self._f[:n], self._n[:n]


def select_tested_pair(state: FoldState, b: Node, sigma_lo: int, m: int,
                       rank_map: Mapping[bytes, int],
                       constraints: AbstractSet[frozenset],
                       rows: Optional[Iterable[int]] = None) -> Tuple[List[bytes], Optional[bytes]]:
    """(K(b), a_e(b)): a function of the processed prefix, node-level DISTINGUISHES
    (``constraints``: frozenset pairs of keys) and sigma_q only. Q's rows are in canonical order
    (FoldState.add), and b must follow them, so index ties are ties by canonical order.

    ``rows`` (admission-path pre-filter only, never the fold): rank only these row positions,
    exactly and in canonical order; K equals the full K iff the full top-m lies inside them."""
    if not state.keys:
        return [], None
    if b.order <= state.nodes[state.keys[-1]].order:
        raise ValueError("b does not follow the processed nodes in canonical order")
    Q, Qf, norms = state.matrices()
    if rows is None:
        top = topm_exact(b.q, Q, m, sigma_lo, Qf=Qf, norms=norms)
    else:
        r = np.array(sorted(set(rows)), dtype=np.int64)
        top = r[topm_exact(b.q, Q[r], m, sigma_lo, Qf=Qf[r], norms=norms[r])].tolist()
    K = [state.keys[i] for i in top]
    a_e = next((r for r in K if state.nodes[r].cell == b.cell and rank_map[b.key] <= rank_map[r]
                and frozenset((b.key, r)) not in constraints), None)
    return K, a_e


class StepContext(NamedTuple):
    sigma_lo: int
    sigma_c: int
    m: int
    theta: Callable[[int], int]                 # n_prior -> theta_q of b's bin (INF16 if none)
    rank: Mapping[bytes, int]
    distinguishes: AbstractSet[frozenset]       # authorized DISTINGUISHES pairs
    cal_pairs: AbstractSet[Tuple[bytes, bytes]]  # NON_MERGEABLE Cal pairs, (c_x < c_y)
    sig: object                                 # .dir(premise, hyp), .pair(x, y) -> payloads
    rep: Callable                               # group members -> representative (Store.rep)
    r_min: int
    prefilter: Optional[Callable] = None        # (state, b) -> row positions; admission flag only


class Decision(NamedTuple):
    b: bytes
    decision: str
    a_star: Optional[bytes]
    tested: Optional[Tuple[bytes, bytes]]       # (a_e, b)
    s_q: Optional[int]
    theta_q: Optional[int]
    gates: Tuple[str, ...]
    links: Tuple[Link, ...]
    n_prior: int
    K: Tuple[bytes, ...]                        # the candidate set K(b), a* first


def _cos(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.astype(np.int64), y.astype(np.int64)
    nn = int(x @ x) * int(y @ y)
    return int(x @ y) / math.sqrt(nn) if nn else 0.0


def step(state: FoldState, b: Node, ctx: StepContext) -> Decision:
    """One step of the fold; appends b to ``state`` unless REJECTed (see module docstring)."""
    n_prior = len(state.keys)
    if b.status != "ACTIVE" or b.retention < ctx.r_min:
        return Decision(b.key, "REJECT", None, None, None, None, (), (), n_prior, ())
    K, a_e = select_tested_pair(state, b, ctx.sigma_lo, ctx.m, ctx.rank, ctx.distinguishes,
                                ctx.prefilter(state, b) if ctx.prefilter else None)
    nodes = state.nodes
    group = None
    gates: List[str] = []
    s = theta = tested = None
    if K:
        a_star = K[0]
        if a_e is not None:
            tested, theta = (a_e, b.key), ctx.theta(n_prior)
            fwd, back = ctx.sig.pair(b, nodes[a_e])
            s = score_s(fwd, back)
            group = state.members[state.group_of[a_e]]
        mask = contradicts(ctx.sig.dir(b, nodes[a_star]))
        collected = [r for r in K[1:] if sigma_q_cmp(b.q, nodes[r].q, ctx.sigma_c) >= 0
                     and contradicts(ctx.sig.dir(b, nodes[r]))]
        contra = ([a_star] if mask else []) + collected
        decision = "RELATE"
        if mask:
            decision, gates = "CONFLICT", ["mask"]
        elif a_e is not None:
            if contradicts(fwd) or contradicts(back):
                gates.append("pair_mask")
                if a_e not in contra:
                    contra.append(a_e)
            if any(state.group_of[r] == state.group_of[a_e] for r in collected):
                gates.append("group_contradiction")
            if gates:
                decision = "CONFLICT"
            elif s is not None and s > theta:
                if any(frozenset((b.key, g)) in ctx.distinguishes for g in group):
                    gates.append("group_cannot_link")
                if tuple(sorted((a_e, b.key))) in ctx.cal_pairs:
                    gates.append("cal_veto")
                decision = "RELATE" if gates else "MERGE"
        links = [("CONTRADICTS", b.key, r, 1.0) for r in contra]
        if decision == "RELATE":
            links.append(("ASSOCIATES", b.key, a_star, _cos(b.q, nodes[a_star].q)))
    else:
        a_star, decision, links = None, "INSERT", []
    state.add(b)
    if decision == "MERGE":
        links.append(("CONSOLIDATED_FROM", b.key, ctx.rep(group), 1.0))
        group.append(b.key)
        state.group_of[b.key] = state.group_of[a_e]
    else:
        state.group_of[b.key] = b.key
        state.members[b.key] = [b.key]
    state.links.extend(links)
    return Decision(b.key, decision, a_star, tested, s, theta, tuple(gates), tuple(links), n_prior,
                    tuple(K))


__all__ = ["Node", "node_of", "FoldState", "select_tested_pair", "StepContext", "Decision",
           "step"]
