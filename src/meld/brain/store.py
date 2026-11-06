"""The brain store: a memory-store replica of a world-model graph.

``Store`` (design §2.1, §3.2, §3.4) holds the five replicated sets of its scopes in an
``ObjectSet`` and a ``Derived`` cache that the fold fills (representative map, link
multiset over node keys). Everything else is computed at read from those two:

  * ``apply(received, mac, scope)`` -- the only mutation: MAC-verified, idempotent insertion.
  * ``nodes()``   -- N in canonical order: key -> (delta_id, the <-minimal admissible delta);
                     D^adm = status ACTIVE, retention >= pol["r_min"] and q of the encoder's
                     length. Nothing is overwritten.
  * ``auth(k)``   -- the join over authorized, valid effects (merge/status_crdt.py).
  * ``cons(k)``   -- DEPRECATED iff k is not its group's representative (default ACTIVE).
  * ``vis(k)``    -- auth(k) join cons(k); ``live_nodes()`` are those with vis ACTIVE.
  * ``rep(G)``    -- §3.4 representative of a group, for the fold.
  * ``graph_walk_read`` -- live nodes only, links resolved to representatives, each unit
                     once (its best node), ties by unit id.

nodes and cell_of are kept incrementally by ``apply`` (grow-only sets: a canonical-order-last
node is appended, anything else re-sorts once at the next read); the auth map and the SUPERSEDE
index are recomputed only after an effect, or a delta for a key some effect targets, arrives;
cons reads ``derived`` at every call.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from meld.merge.signals import ENCODER_DIM, EmbeddingSimilarity, _tokenize
from meld.merge.status_crdt import auth_map, authorized, join
from meld.proto import canon
from meld.protocol.objects import AuthoredEffect, Delta, effect_valid, object_id, scope_partition
from meld.sync.objectset import ObjectSet

Link = Tuple[str, bytes, bytes, float]      # (type, src key, dst key, weight)


def _fuse(query: str, texts: Dict, edges: Iterable[Tuple], es: EmbeddingSimilarity,
          beta_lex: float, beta_dense: float, beta_graph: float) -> Dict:
    """Def. 2/3 fusion: lexical token overlap + dense cosine + one hop of link propagation
    (dst gains dense(src) * weight) over the given items; returns id -> score."""
    q_tokens = set(_tokenize(query))
    dense, lex = {}, {}
    for i, text in texts.items():
        dense[i] = es.sigma(query, text) if text else 0.0
        lex[i] = len(q_tokens & set(_tokenize(text))) / len(q_tokens) if q_tokens else 0.0
    graph = dict.fromkeys(texts, 0.0)
    for src, dst, w in edges:
        if src in dense and dst in graph:
            graph[dst] = max(graph[dst], dense[src] * w)
    return {i: beta_lex * lex[i] + beta_dense * dense[i] + beta_graph * graph[i] for i in texts}


@dataclass
class Derived:
    """The fold's output slot: node key -> its group's representative (absent = itself),
    and the link multiset over node keys."""

    rep: Dict[bytes, bytes] = field(default_factory=dict)
    links: List[Link] = field(default_factory=list)


class Store:
    def __init__(self, pol: dict, key: bytes, scopes: Optional[Iterable[str]] = None, *,
                 encoder: str):
        """``pol["r_min"]`` (part of M_pol) bounds admissibility; ``encoder`` is M_nli's encoder:
        a delta under another encoder is dropped at receipt, one whose q is not that encoder's
        length is held but is never a node."""
        r_min = pol["r_min"]
        if type(r_min) is not int or r_min < 1:
            raise ValueError("r_min is a positive fixed-point retention; R = 0 is below it")
        self.pol, self.key, self.r_min, self.encoder = pol, key, r_min, encoder
        self._q_len = ENCODER_DIM[encoder]
        self.objects = ObjectSet(scope_partition(pol), scopes, encoder=encoder)
        self.derived = Derived()
        self._es: Dict[str, EmbeddingSimilarity] = {}
        self._best: Dict[bytes, Tuple[int, bytes, object]] = {}   # key -> its <-minimal delta
        self._cell_of: Dict[bytes, str] = {}
        self._nodes: Optional[Dict[bytes, Tuple[bytes, object]]] = {}   # None: re-sort _best
        self._status: Optional[Tuple[Dict, Dict]] = None    # (auth, supersede_by_c2); None: stale
        self._c1s: set = set()      # c1 of held effects: a new such key can change validity

    def apply(self, received: bytes, mac: str, scope: Optional[str] = None):
        """Insert one received object; the stored object, or None if it was dropped."""
        n = len(self.objects)
        obj = self.objects.add(received, mac, self.key, scope)
        if len(self.objects) > n:
            self._index(obj, object_id(received))
        return obj

    def _index(self, o, oid: bytes) -> None:
        """Update the views for one newly held object (the sets are grow-only)."""
        if isinstance(o, AuthoredEffect):
            self._c1s.add(o.c1)
            self._status = None
        if not isinstance(o, Delta):
            return
        k = o.claim_key
        if k not in self._cell_of and k in self._c1s:
            self._status = None
        self._cell_of[k] = o.cell                    # a key binds its cell
        if not (o.status == "ACTIVE" and o.retention >= self.r_min and len(o.q) == self._q_len):
            return
        old, new = self._best.get(k), (o.created_at, oid, o)
        if old is not None and old[:2] <= new[:2]:
            return
        self._best[k] = new
        nodes = self._nodes
        if nodes is not None and (not nodes or self._best[next(reversed(nodes))][:2] < new[:2]):
            nodes[k] = (oid, o)          # canonical-order last (never a replaced key): append
        else:
            self._nodes = None

    def _auth_sup(self) -> Tuple[Dict, Dict]:
        if self._status is None:
            scope_of, pol, cell_of = self.objects.scope_of, self.pol, self._cell_of
            # auth per scope (defense in depth): an effect filed under scope s only reaches
            # nodes whose cell lies in s, whatever target_cell it names
            auth: Dict[bytes, str] = {}
            sup: Dict[bytes, List[bytes]] = {}
            for s in sorted(self.objects.scopes):
                effects = list(self.objects.iter("A", s))
                for k, st in auth_map(effects, pol, cell_of).items():
                    if k in cell_of and scope_of.get(cell_of[k]) == s:
                        auth[k] = st
                for e in effects:
                    if e.kind == "SUPERSEDE" and self.ac(e):
                        sup.setdefault(e.c2, []).append(e.c1)
            self._status = (auth, sup)
        return self._status

    def ac(self, e: AuthoredEffect) -> bool:
        """e is an authorized, valid A_c effect (SUPERSEDE, DISTINGUISHES): the fold and the
        representative rule read it (§2.1). The one predicate admission, fold and rep share."""
        so = self.objects.scope_of
        return (e.kind in ("SUPERSEDE", "DISTINGUISHES") and authorized(e, self.pol, so)
                and effect_valid(e, so, self._cell_of))

    # -- nodes ---------------------------------------------------------------------------
    def nodes(self) -> Dict[bytes, Tuple[bytes, object]]:
        """key -> (delta_id, delta) of the <-minimal admissible delta, in canonical order."""
        if self._nodes is None:
            self._nodes = {k: v[1:] for k, v in sorted(self._best.items(), key=lambda kv: kv[1][:2])}
        return self._nodes

    def cell_of(self) -> Dict[bytes, str]:
        """Held claim key -> cell (a key binds its cell, so every delta of a key agrees)."""
        return self._cell_of

    # -- status --------------------------------------------------------------------------
    def auth(self, k: bytes) -> str:
        return self._auth_sup()[0].get(k, "ACTIVE")

    def cons(self, k: bytes) -> str:
        return "ACTIVE" if self.derived.rep.get(k, k) == k else "DEPRECATED"

    def vis(self, k: bytes) -> str:
        return join(self.auth(k), self.cons(k))

    def vis_map(self) -> Dict[bytes, str]:
        auth = self._auth_sup()[0]
        return {k: join(auth.get(k, "ACTIVE"), self.cons(k)) for k in self.nodes()}

    def live_nodes(self) -> List[bytes]:
        return [k for k, v in self.vis_map().items() if v == "ACTIVE"]

    def rep(self, group: Iterable[bytes]) -> bytes:
        """§3.4: min over the members with no authorized, valid SUPERSEDE to another member
        of (auth rank, created_at, delta_id); else min over the group. Members are nodes."""
        nodes, g = self.nodes(), set(group)
        missing = sorted(k for k in g if k not in nodes)
        if missing:
            raise ValueError(f"rep: group member {missing[0].hex()} is not a node")
        sup = self._auth_sup()[1]
        superseded = {c1 for c2 in g for c1 in sup.get(c2, ()) if c1 in g}

        def order(k):
            did, d = nodes[k]
            return (d.auth, d.created_at, did)
        return min(g - superseded or g, key=order)

    # -- read ----------------------------------------------------------------------------
    def resolved_links(self) -> List[Link]:
        """Links with endpoints resolved to representatives. Only a CONSOLIDATED_FROM that
        resolves to a self-loop is dropped (it records the merge itself); any other self-loop
        is kept so it stays visible. The raw multiset is ``derived.links``."""
        r = self.derived.rep
        out = [(t, r.get(s, s), r.get(d, d), w) for t, s, d, w in self.derived.links]
        return [x for x in out if not (x[0] == "CONSOLIDATED_FROM" and x[1] == x[2])]

    def fingerprint_inputs(self):
        """(vis, groups, resolved links), each sorted: the replica-comparable derived view."""
        groups: Dict[bytes, List[bytes]] = {}
        for k in self.nodes():
            groups.setdefault(self.derived.rep.get(k, k), []).append(k)
        return (sorted(self.vis_map().items()),
                sorted((r, tuple(sorted(m))) for r, m in groups.items()),
                sorted(self.resolved_links()))

    def graph_walk_read(self, query: str, *, encoder: str, k: int = 5,
                        beta_lex: float = 0.3, beta_dense: float = 0.5,
                        beta_graph: float = 0.2) -> List[Tuple[bytes, float]]:
        """Fused top-k units over live nodes as (unit_id, score): a unit scores the max over
        its live nodes and appears once; sorted by (-score, unit_id)."""
        es = self._es.setdefault(encoder, EmbeddingSimilarity(encoder=encoder))
        nodes = self.nodes()
        live = self.live_nodes()
        scores = _fuse(query, {n: canon(nodes[n][1].text) for n in live},
                       ((s, d, w) for _, s, d, w in self.resolved_links()), es,
                       beta_lex, beta_dense, beta_graph)
        unit: Dict[bytes, float] = {}
        for n, sc in scores.items():
            uid = nodes[n][1].unit_id
            unit[uid] = max(unit.get(uid, sc), sc)
        return sorted(unit.items(), key=lambda x: (-x[1], x[0]))[:k]


__all__ = ["Store", "Derived"]
