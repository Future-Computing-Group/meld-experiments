"""The admission (latency) path, design §3.5, Lemma 2, Prop 3.

``Admission(store, ...).admit(delta)`` runs on receipt of a MAC-verified delta already
inserted into the store. It keeps, per scope, the fold's state over the scope's nodes (the
cache) together with the step context ``fold.setup`` built from the sets, and:

  * if nothing that changes the fold arrived in the scope since the last admission and the
    delta's node is canonical-order last, runs one ``classifier.step`` (Lemma 2: the result
    is the fold over the sets) and updates only the derived entries of the group it joined;
  * otherwise refolds the scope from the log start through ``fold.setup`` and the step, so
    admission and ``fold.readjudicate`` read the same inputs and decide alike. The refold
    triggers are those of §3.5: another delta (a node that is not canonical-order last at
    receipt is counted in ``counters["not_last"]``, a divergence source of Prop 3); an
    authorized, valid A_c effect (SUPERSEDE, DISTINGUISHES), or a node arriving for the c1
    of one in another cell than its target cell (that invalidates it); a Sig record whose join changes a direction a
    step read; a Cal record whose two keys are processed nodes; a change of the active epoch.
    Other objects (status effects, a Sig for an unread direction or joining to the same
    payload, a Cal for a pair not yet processed, an epoch that is not active) are folded into
    the cache without a refold. ``counters["refolds"]`` counts every rebuild, the first one
    per scope included;
  * publishes every Sig direction the step evaluated (``fold.publish_sig``: both directions
    of a pair in one MAC'd record, inserted into the store's Sig set, hence on the outbox).
    Admission never writes an authored effect (A holds authored effects only, §2.1).

Optional index flag (``index``: CandidateIndex parameters plus ``k``, the pre-filter size, default
4m; None, the default, is the exact path): on the latency step only, the scope's candidate index
(merge/candidate_index.py, rebuilt on every refold, extended by every step) names k rows and the
step ranks those exactly, so the decision equals the exact path's whenever the exact top-m lies
inside them (a miss changes K and the next pass repairs it). Refolds stay exact.
``counters["prefiltered"]`` counts the steps that used it.

The active epoch, or the policy's bootstrap selector without one (theta = +inf: no similarity
MERGE), comes from ``fold.setup``. A delta that is no node returns no decision, except an
inadmissible one (status not ACTIVE or retention below R_min), which returns REJECT; an
exact copy later than its key's node returns None. Non-delta objects are not admitted; the
next delta reads them as above.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from meld.merge import classifier
from meld.protocol import fold
from meld.protocol.objects import Delta, canonical_bytes, join_signal_payloads, object_id


class _Cache:
    def __init__(self, epoch, state, ctx, sig, decisions, ac_c1, seen):
        self.epoch, self.state, self.ctx, self.sig, self.decisions = epoch, state, ctx, sig, decisions
        self.ac_c1, self.seen = ac_c1, seen     # c1 -> target cells of counted A_c effects
        self.index = None


class Admission:
    def __init__(self, store, *, m_nli: bytes, provider: fold.Provider, issuer: str, hw: str,
                 index: Optional[dict] = None):
        self.store, self.m_nli = store, m_nli
        self.provider, self.issuer, self.hw = provider, issuer, hw
        self._cache: Dict[str, _Cache] = {}
        self.counters = {"not_last": 0, "refolds": 0}
        self.index = index
        if index is not None:
            self.counters["prefiltered"] = 0

    def _refold(self, scope: str) -> _Cache:
        store = self.store
        epoch, _, nodes, sig, ctx = fold.setup(store, scope, m_nli=self.m_nli, provider=self.provider,
                                               issuer=self.issuer, hw=self.hw)
        state, ac_c1 = classifier.FoldState(), {}
        for e in store.objects.iter("A", scope):
            if store.ac(e):
                ac_c1.setdefault(e.c1, set()).add(e.target_cell)
        c = self._cache[scope] = _Cache(epoch, state, ctx, sig,
                                        [classifier.step(state, b, ctx) for b in nodes], ac_c1,
                                        len(store.objects.arrivals(scope)))
        self.counters["refolds"] += 1
        fold.write_derived(store, state)
        if self.index is not None and state.keys:
            Q = state.matrices()[0]
            self._index_of(c, Q.shape[1]).add(Q)
        return c

    def _index_of(self, c: _Cache, dim: int):
        if c.index is None:
            from meld.merge.candidate_index import CandidateIndex     # faiss only with the flag
            c.index = CandidateIndex(dim, **{k: v for k, v in self.index.items() if k != "k"})
        return c.index

    def _step(self, c: _Cache, b: classifier.Node) -> classifier.Decision:
        """The latency step, through the scope's index when the flag is set."""
        if self.index is None:
            return classifier.step(c.state, b, c.ctx)
        ctx = c.ctx
        if c.index is not None:
            k = self.index.get("k") or 4 * ctx.m
            ctx = ctx._replace(prefilter=lambda st, x: c.index.search(x.q, k))
            self.counters["prefiltered"] += 1
        dec = classifier.step(c.state, b, ctx)
        if len(c.state.keys) > len(c.index or ()):
            self._index_of(c, len(b.q)).add(b.q)
        return dec

    def _changed(self, c: _Cache, scope: str, did: bytes) -> bool:
        """Read the scope's arrivals since the last admission into the cache; True iff one of
        them is a refold trigger (module docstring)."""
        objects = self.store.objects
        log = objects.arrivals(scope)
        new, c.seen = log[c.seen:], len(log)
        for oid in new:
            kind = objects.kind_of(oid)
            o = objects.get(kind, oid)
            if kind == "D":
                if oid != did:
                    return True
            elif kind == "A":
                if self.store.ac(o):
                    return True
            elif kind == "Sig":
                if o.m_nli != self.m_nli:
                    continue
                old = c.sig.joined.get((o.c_x, o.c_y), (None, None))
                for p, h, a, b in ((o.c_x, o.c_y, old[0], o.pay_xy), (o.c_y, o.c_x, old[1], o.pay_yx)):
                    if (p, h) in c.sig.read and join_signal_payloads(a, b) != a:
                        return True
                fold.join_into(c.sig.joined, o)
            elif kind == "Cal":
                if o.c_x in c.state.nodes and o.c_y in c.state.nodes:
                    return True
                c.ctx.cal_pairs.add((o.c_x, o.c_y))
            elif fold.scope_epoch(self.store, scope, self.m_nli) != c.epoch:    # Ep
                return True
        return False

    def admit(self, d: Delta) -> Optional[classifier.Decision]:
        """The decision for d's node, or None; see the module docstring."""
        store = self.store
        scope_of = store.objects.scope_of
        scope = scope_of[d.cell]
        did = object_id(canonical_bytes(d))
        nodes = store.nodes()
        held = nodes.get(d.claim_key)
        is_node = held is not None and held[0] == did
        if is_node:
            last = next(k for k in reversed(nodes) if scope_of[nodes[k][1].cell] == scope)
            self.counters["not_last"] += last != d.claim_key
        c = self._cache.get(scope)
        if (c is None or self._changed(c, scope, did)
                or (is_node and (last != d.claim_key or d.claim_key in c.state.nodes
                                 or c.ac_c1.get(d.claim_key, {d.cell}) != {d.cell}))):
            c = self._refold(scope)
            dec = next(x for x in c.decisions if x.b == d.claim_key) if is_node else None
        elif is_node:
            c.ctx.rank[d.claim_key] = d.auth
            dec = self._step(c, classifier.node_of(d, did))
            c.decisions.append(dec)
            fold.write_group(store, c.state, d.claim_key, dec.links)
        else:
            dec = None
        if not is_node and (d.status != "ACTIVE" or d.retention < store.r_min):
            dec = classifier.step(c.state, classifier.node_of(d, did), c.ctx)    # REJECT
        fold.publish_sig(store, c.sig)
        c.seen = len(store.objects.arrivals(scope))      # own Sig records are already joined
        return dec

    def decisions(self, scope: str) -> List[classifier.Decision]:
        """The cached decisions of the scope's nodes in canonical order, as of the last
        admission (stale once other objects have arrived; a pass or the next delta repairs)."""
        c = self._cache.get(scope)
        return list(c.decisions) if c else []


__all__ = ["Admission"]
