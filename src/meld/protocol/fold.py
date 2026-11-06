"""The fold over a scope's replicated sets (design §3.3-3.5, §2.4, Props 1 and 3, Lemmas 2b, 6):
``readjudicate`` replays every node of the scope in canonical order through the step
(merge/classifier.py) from the log start, writes the partition's representatives (Store.rep) and
the link multiset into ``store.derived``, and returns the decisions, the counters and a PASS
certificate that ``verify_pass`` recomputes from the sets under its roots.

Inputs the fold reads: D (nodes), authorized valid DISTINGUISHES and SUPERSEDE of A (the latter
through Store.rep), the Cal pairs, the joined Sig payloads under M_nli, and the active epoch:
the largest-seq Ep record of the scope whose author is the scope's calibration authority and
whose M_nli and M_pol are the replica's. The selector (sigma_lo, sigma_c, m) is the epoch's;
without an epoch it is the policy's ``bootstrap_selector`` (part of M_pol) and theta is INF16
everywhere, so a bare replica INSERTs, RELATEs and CONFLICTs but never similarity-MERGEs.
R_min is the policy's (Store reads ``pol["r_min"]``), so it is part of M_pol too.

Signals: a direction absent from Sig is a pass-time miss. With a ``provider`` (premise text,
hypothesis text) -> uint16 triple, the fold evaluates it, inserts the new Sig record into the
store (so the end-of-pass roots include it) and returns it for publication; without one the
payload stays bottom. ``nli_provider(model)`` wraps signals.nli_probs.

Certificate (PASS Patch, §2.4): scope, epoch id and seq, M_nli, the selector used, the Merkle
roots of the five sets of the scope at the end of the pass, h16 of the canonical JSON of the
decisions (each carries K, a*, the tested pair, s_q, theta_q, the gates and its links; keys as
hex, link weights as fixed point), the fingerprint (h16 of the canonical JSON of vis, groups
with representatives and raw links) and the count of directions still missing at the end of
the pass. ``verify_pass`` certifies only Sig-complete folds (no miss on refold). The stored id
list per certificate is the returned ``snapshot``. Link ids are not separately
content-addressed: the decision digest covers the link tuples. ``publish_pass`` ships the
certificate as an authored record with a MAC under the group key; ``receive_pass`` drops one
whose MAC fails. The MAC attributes the issuer; ``verify_pass`` attests the content only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from meld.brain.store import Derived, Store
from meld.merge import classifier
from meld.merge.conformal import active_epoch, bin_of, threshold
from meld.merge import signals
from meld.merge.signals import nli_probs, to_fixed16
from meld.merge.status_crdt import epoch_authorized
from meld.protocol.objects import (
    FIXED_ONE, INF16, Payload, SignalRecord, attach_mac, canonical_bytes, from_bytes,
    h16, join_signal_payloads, m_pol_of, object_id, signed, verify_mac,
)
from meld.sync.merkle import merkle_root
from meld.sync.objectset import KINDS

Provider = Callable[[str, str], Tuple[int, int, int]]


def nli_provider(model: str, *, own: bool = False) -> Provider:
    """(premise, hypothesis) -> uint16 triple. ``own=True`` gives a cross-encoder provider its
    own model instance, exposed as ``.model``, so replicas in one process never share one (the
    multi-replica harnesses require it); otherwise the module-level instance is shared and
    loaded on first use. The lexical rung has no model state (``.model`` None)."""
    if own and model != "lexical":
        inst = signals.new_cross_encoder(model)
        f = lambda a, b: to_fixed16(inst.probs(a, b))      # noqa: E731
    else:
        inst, f = None, (lambda a, b: to_fixed16(nli_probs(a, b, model=model)))
    f.model = inst
    return f


def join_into(joined, r) -> None:
    """Join one Sig record into the (c_x, c_y) -> (pay_xy, pay_yx) map, per direction."""
    old = joined.get((r.c_x, r.c_y), (None, None))
    joined[r.c_x, r.c_y] = (join_signal_payloads(old[0], r.pay_xy),
                            join_signal_payloads(old[1], r.pay_yx))


class SigView:
    """Joined Sig payloads of one scope under one M_nli, read by the step; counts misses and
    provider calls, keeps the records it evaluated and the directions it read."""

    def __init__(self, joined: Dict[Tuple[bytes, bytes], Tuple[Payload, Payload]],
                 provider: Optional[Provider], *, scope: str, m_nli: bytes, issuer: str, hw: str):
        if provider is not None and not (issuer and hw):
            raise ValueError("a signal provider needs a non-empty issuer and hw for its records")
        self.joined, self.provider = joined, provider
        self.scope, self.m_nli, self.issuer, self.hw = scope, m_nli, issuer, hw
        self.missed: set = set()
        self.read: set = set()          # directions (premise key, hypothesis key) the steps read
        self.calls = 0
        self.new: List[SignalRecord] = []

    def _get(self, p, h) -> Payload:
        pay = self.joined.get((min(p.key, h.key), max(p.key, h.key)), (None, None))
        return pay[0] if p.key < h.key else pay[1]

    def _read(self, dirs):
        self.read.update((p.key, h.key) for p, h in dirs)
        missing = [(p, h) for p, h in dirs if self._get(p, h) is None]
        self.missed.update((p.key, h.key) for p, h in missing)
        if missing and self.provider is not None:
            got = {}
            for p, h in missing:
                got[p.key, h.key] = tuple(self.provider(p.text, h.text))
                self.calls += 1
            x, y = sorted((dirs[0][0].key, dirs[0][1].key))
            rec = SignalRecord(scope=self.scope, c_x=x, c_y=y, m_nli=self.m_nli,
                               pay_xy=got.get((x, y)), pay_yx=got.get((y, x)),
                               issuer=self.issuer, hw=self.hw)
            self.new.append(rec)
            join_into(self.joined, rec)
        return [self._get(p, h) for p, h in dirs]

    def dir(self, p, h) -> Payload:
        return self._read([(p, h)])[0]

    def pair(self, x, y) -> Tuple[Payload, Payload]:
        """(x -> y, y -> x); both missing directions go into one record."""
        return tuple(self._read([(x, y), (y, x)]))


def scope_epoch(store: Store, scope: str, m_nli: bytes):
    pol = store.pol
    mp = m_pol_of(pol)
    return active_epoch([E for E in store.objects.iter("Ep", scope) if epoch_authorized(E, pol)
                         and E.m_nli == m_nli and E.m_pol == mp], scope)


def _nodes(store: Store, scope: str):
    """The scope's nodes in canonical order, and the rank map from their attributes."""
    scope_of, nodes, rank = store.objects.scope_of, [], {}
    for did, d in store.nodes().values():
        if scope_of.get(d.cell) == scope:
            nodes.append(classifier.node_of(d, did))
            rank[d.claim_key] = d.auth
    return nodes, rank


def _distinguishes(store: Store, scope: str) -> set:
    return {frozenset((e.c1, e.c2)) for e in store.objects.iter("A", scope)
            if e.kind == "DISTINGUISHES" and store.ac(e)}


def tested_pairs(store: Store, scope: str, *, sigma_lo: int, m: int) -> List[Tuple[bytes, bytes, int]]:
    """(a_e, b, n_prior) of every node with a tested pair, by the fold's selector: the draw a
    calibration builder labels (§4.3). Reads D, node-level A_c and sigma_q only."""
    nodes, rank = _nodes(store, scope)
    cons = _distinguishes(store, scope)
    state, out = classifier.FoldState(), []
    for b in nodes:
        _, a_e = classifier.select_tested_pair(state, b, sigma_lo, m, rank, cons)
        if a_e is not None:
            out.append((a_e, b.key, len(state.keys)))
        state.add(b)
    return out


@dataclass(frozen=True)
class PassCertificate:
    scope: str
    epoch_id: bytes                 # b"" without an epoch
    epoch_seq: int                  # -1 without an epoch
    m_nli: bytes
    selector: Tuple[int, int, int]  # (sigma_lo, sigma_c, m) used
    roots: Tuple[bytes, ...]        # Merkle roots of D, A, Sig, Cal, Ep at the end of the pass
    decisions_digest: bytes
    fingerprint: bytes
    issuer: str
    misses: int                     # directions still bottom at the end of the pass
    mac: str = field(default="", compare=False)   # attributes the issuer (publish_pass)


class FoldResult(NamedTuple):
    decisions: List[classifier.Decision]
    certificate: PassCertificate
    snapshot: Dict[str, Tuple[bytes, ...]]   # the id list per set under the roots
    new_sig: List[SignalRecord]              # evaluated at this pass, MAC'd, inserted, to publish
    counters: Dict[str, int]
    epoch: object


def _js(x):
    """JSON-ready form: bytes as hex, floats (link weights) as fixed point, tuples as lists."""
    if isinstance(x, bytes):
        return x.hex()
    if isinstance(x, float):
        return round(x * FIXED_ONE)
    if isinstance(x, (tuple, list)):
        return [_js(v) for v in x]
    return x


def canon_json(x) -> bytes:
    return json.dumps(_js(x), sort_keys=True, separators=(",", ":")).encode()


def decisions_json(decisions) -> bytes:
    return canon_json([list(d) for d in decisions])


def setup(store: Store, scope: str, *, m_nli: bytes, provider: Optional[Provider] = None,
          issuer: str = "", hw: str = ""):
    """(epoch, selector, nodes, sig, ctx): everything the step reads, from the scope's sets.
    The fold and the admission path both start here, so they read the same inputs."""
    epoch = scope_epoch(store, scope, m_nli)
    if epoch is not None:
        selector = (epoch.sigma_lo_q, epoch.sigma_c_q, epoch.m)
    elif "bootstrap_selector" in store.pol:
        selector = tuple(store.pol["bootstrap_selector"])
    else:
        raise ValueError("no epoch and the policy names no bootstrap_selector")
    nodes, rank = _nodes(store, scope)
    joined: Dict[Tuple[bytes, bytes], Tuple[Payload, Payload]] = {}
    for r in store.objects.iter("Sig", scope):
        if r.m_nli == m_nli:
            join_into(joined, r)
    sig = SigView(joined, provider, scope=scope, m_nli=m_nli, issuer=issuer, hw=hw)
    theta = ((lambda n: threshold(epoch, bin_of(n, epoch.bins[0]))) if epoch
             else (lambda n: INF16))
    ctx = classifier.StepContext(
        sigma_lo=selector[0], sigma_c=selector[1], m=selector[2], theta=theta, rank=rank,
        distinguishes=_distinguishes(store, scope),
        cal_pairs={(k.c_x, k.c_y) for k in store.objects.iter("Cal", scope)},
        sig=sig, rep=store.rep, r_min=store.r_min)
    return epoch, selector, nodes, sig, ctx


def publish_sig(store: Store, sig: SigView) -> List[SignalRecord]:
    """MAC every record ``sig`` evaluated since the last call and insert it into the store's
    Sig set (so it is on the replica's outbox and under its roots); return the records."""
    new, sig.new = sig.new, []
    out = [attach_mac(rec, store.key) for rec in new]
    for o in out:
        if store.apply(canonical_bytes(o), o.mac, o.scope) is None:
            raise ValueError("own Sig record refused by the store")
    return out


def write_group(store: Store, state: classifier.FoldState, k: bytes, links) -> None:
    """The derived effects of one admission step: k's group's representative entries, and its
    links (the per-group form of ``write_derived``; the representative rule is Store.rep)."""
    members = state.members[state.group_of[k]]
    r = store.rep(members)
    for m in members:
        if m == r:
            store.derived.rep.pop(m, None)
        else:
            store.derived.rep[m] = r
    store.derived.links.extend(links)


def write_derived(store: Store, state: classifier.FoldState) -> Dict[bytes, bytes]:
    """Record the fold state's derived effects (representatives, links) for its nodes in
    ``store.derived``, keeping other scopes' entries; returns group id -> representative."""
    reps = {gid: store.rep(ms) for gid, ms in state.members.items()}
    mine = set(state.keys)
    old = store.derived
    store.derived = Derived(
        rep={**{k: v for k, v in old.rep.items() if k not in mine},
             **{k: reps[g] for k, g in state.group_of.items() if reps[g] != k}},
        links=[l for l in old.links if l[1] not in mine] + state.links)
    return reps


def readjudicate(store: Store, scope: str, *, m_nli: bytes,
                 provider: Optional[Provider] = None, issuer: str = "", hw: str = "",
                 truth: Optional[Callable[[bytes, bytes], str]] = None) -> FoldResult:
    """Fold the scope from the log start; see the module docstring. ``truth(x, y)`` in
    {MERGEABLE, NON_MERGEABLE, CONTRADICTION} enables the purity counters."""
    epoch, selector, nodes, sig, ctx = setup(store, scope, m_nli=m_nli, provider=provider,
                                             issuer=issuer, hw=hw)
    state = classifier.FoldState()
    decisions = [classifier.step(state, b, ctx) for b in nodes]
    calls, missed = sig.calls, len(sig.missed)
    new_sig = publish_sig(store, sig)
    reps = write_derived(store, state)
    mine = set(state.keys)

    groups = sorted((reps[g], tuple(sorted(ms))) for g, ms in state.members.items())
    fingerprint = h16(canon_json((sorted((k, store.vis(k)) for k in mine), groups,
                                   sorted(state.links))))
    snapshot = {kind: store.objects.ids(kind, scope) for kind in KINDS}
    cert = PassCertificate(
        scope=scope, epoch_id=object_id(canonical_bytes(epoch)) if epoch else b"",
        epoch_seq=epoch.seq if epoch else -1, m_nli=m_nli, selector=selector,
        roots=tuple(merkle_root(snapshot[kind]) for kind in KINDS),
        decisions_digest=h16(decisions_json(decisions)), fingerprint=fingerprint, issuer=issuer,
        misses=0 if provider is not None else missed)

    counters = {"pass_time_misses": missed, "nli_calls": calls}
    if truth is not None:
        K = {d.b: set(d.K) for d in decisions}
        pairs = [(x, y) for _, ms in groups for i, x in enumerate(ms) for y in ms[i + 1:]]
        labels = [truth(x, y) for x, y in pairs]
        contra = [p for p, lab in zip(pairs, labels) if lab == "CONTRADICTION"]
        # §7.5: a contradiction is undetectable at the later node's step when the earlier
        # member was outside its K
        later = {k: i for i, k in enumerate(state.keys)}
        counters.update(group_pairs=len(pairs),
                        impure_pairs=sum(lab != "MERGEABLE" for lab in labels),
                        in_group_contradictions=len(contra),
                        undetected_outside_k=sum(
                            min(p, key=later.get) not in K[max(p, key=later.get)] for p in contra))
    return FoldResult(decisions, cert, snapshot, new_sig, counters, epoch)


def publish_pass(cert: PassCertificate, key: bytes) -> Tuple[bytes, str]:
    """The certificate as an authored record: (canonical bytes, MAC under the group key)."""
    return signed(cert, key)


def receive_pass(received: bytes, mac: str, key: bytes) -> Optional[PassCertificate]:
    """The certificate if its bytes parse canonically and its MAC verifies, else None."""
    try:
        cert = from_bytes(PassCertificate, received, mac)
    except ValueError:
        return None
    return cert if verify_mac(cert, key) else None


def verify_pass(cert: PassCertificate, snapshot: Dict[str, Tuple[bytes, ...]], source: Store) -> bool:
    """Obtain the objects under the certificate's roots from ``source`` (by the retained id
    lists), refold them in a fresh replica without a provider, and compare every field.

    This attests the certificate's content (the fold over the sets under its roots), not its
    issuer: the issuer field is echoed into the refold, so any issuer name verifies.
    Attribution comes from the record MAC (``receive_pass``)."""
    fresh = Store(source.pol, source.key, scopes=[cert.scope], encoder=source.encoder)
    for kind in KINDS:          # the refold recomputes the roots, so they are compared below
        for oid in snapshot.get(kind, ()):
            if oid not in source.objects:
                return False
            received, mac, scope = source.objects.raw(oid)
            if fresh.apply(received, mac, scope) is None:
                return False
    again = readjudicate(fresh, cert.scope, m_nli=cert.m_nli, issuer=cert.issuer)
    return again.counters["pass_time_misses"] == 0 and again.certificate == cert


__all__ = ["SigView", "nli_provider", "scope_epoch", "tested_pairs", "PassCertificate",
           "FoldResult", "decisions_json", "readjudicate", "verify_pass", "publish_pass", "receive_pass",
           "setup", "publish_sig", "write_derived", "write_group", "join_into", "canon_json"]
