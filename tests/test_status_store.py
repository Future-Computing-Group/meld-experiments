"""Status and store (design §2.1 authored effects, §3.2 nodes, §3.4 representatives and the
view, §3.6 versions, Prop. 4): authorized(e, policy) from carried fields only, auth as a join
over authorized valid effects keyed by eid, vis = auth join cons with cons from the derived
cache, no re-put overwrite, links resolved to representatives, read ties by unit id."""
from __future__ import annotations

import random

import pytest

from meld.brain.store import Derived, Store
from meld.merge.status_crdt import _RANK, StatusState, authorized, join
from meld.proto import claim_key
from meld.protocol.objects import (
    FIXED_ONE, AuthoredEffect, Delta, attach_mac, canonical_bytes, effect_valid, object_id,
    scope_partition,
)

KEY = b"t5-test-group-key"
POL = {"rank_map": {"PRECEDENT": 0, "OBSERVATION": 1, "SPECULATION": 2},
       "scopes": {"eu": ["eu.a", "eu.b"], "us": ["us.a"]},
       "auth_policy": {"OVERRULE": 0, "REVOKE": 0, "SUPERSEDE": 1, "DISTINGUISHES": 1},
       "step_version": 0, "r_min": 1}
POL_SCOPE = {"eu.a": "eu", "eu.b": "eu", "us.a": "us"}
STATUSES = ("ACTIVE", "DEPRECATED", "OVERRULED", "REVOKED")


def _d(text, cell="eu.a", version=0, auth=1, created_at=10, unit_id=None, **over):
    kw = dict(unit_id=unit_id or claim_key(text, cell, version), text=text, cell=cell,
              version=version, claim_key=claim_key(text, cell, version), auth=auth,
              status="ACTIVE", retention=FIXED_ONE, created_at=created_at, origin="A",
              encoder="hashing", q=b"\x00" * 256)
    kw.update(over)
    return Delta(**kw)


def _e(kind, c1, c2=b"", target_cell="eu.a", author_cell="eu.a", author_auth=0, scope="eu",
       created_at=0, author="C"):
    return AuthoredEffect(scope=scope, kind=kind, c1=c1, c2=c2, target_cell=target_cell,
                          author=author, author_cell=author_cell, author_auth=author_auth,
                          created_at=created_at)


def _wire(obj, key=KEY):
    o = attach_mac(obj, key)
    return canonical_bytes(o), o.mac


def _store(*objs):
    s = Store(POL, KEY, encoder="hashing")
    for o in objs:
        assert s.apply(*_wire(o)) is not None
    return s


A, B, C = (claim_key(t, "eu.a") for t in ("alpha", "beta", "gamma"))


# --------------------------------------------------------------------------- authorization
def test_authorized_effect_from_carried_fields_unauthorized_never_materializes():
    d = _d("alpha")
    ok = _e("OVERRULE", A)
    foreign = _e("OVERRULE", A, author_cell="us.a")               # US author on an EU claim
    weak = _e("OVERRULE", A, author_auth=1)                        # rank above the policy max
    assert authorized(ok, POL)
    assert not authorized(foreign, POL) and not authorized(weak, POL)
    # a kind absent from the policy is never authorized (fail closed)
    assert not authorized(ok, {**POL, "auth_policy": {"REVOKE": 0}})
    # the verdict reads e and the policy only: identical in a store that holds nothing else
    for e in (foreign, weak):
        s = _store(d, e)
        assert object_id(canonical_bytes(e)) in s.objects.ids("A", "eu")  # appended, auditable
        assert s.auth(A) == "ACTIVE" and s.vis(A) == "ACTIVE"           # never materializes
        assert A in s.live_nodes()
    s = _store(d, foreign, weak, ok)
    assert s.auth(A) == "OVERRULED" and A not in s.live_nodes()


def test_unsigned_or_forged_object_never_enters_the_store():
    s = Store(POL, KEY, encoder="hashing")
    b, mac = _wire(_e("REVOKE", A))
    assert s.apply(b, "") is None
    assert s.apply(*_wire(_e("REVOKE", A), key=b"other-key")) is None
    assert s.objects.ids("A", "eu") == () and s.auth(A) == "ACTIVE"


def test_auth_never_decreases_when_deltas_arrive_after_effects():
    """Validity reads e, M_pol and the held c1 only, so auth is monotone
    (Prop. 4). A SUPERSEDE to a key in another scope keeps deprecating when that key arrives;
    over random effect sets and arrival orders no node's auth ever decreases."""
    far = claim_key("alpha", "us.a")
    sup = _e("SUPERSEDE", A, c2=far)
    s = _store(_d("alpha"), sup)
    assert s.auth(A) == "DEPRECATED"
    assert s.apply(*_wire(_d("alpha", cell="us.a"))) is not None
    assert s.auth(A) == "DEPRECATED"
    rng = random.Random(3)
    cells = ("eu.a", "eu.b", "us.a")
    for _ in range(200):
        deltas = [_d(f"t{i}", cell=rng.choice(cells)) for i in range(6)]
        keys = [d.claim_key for d in deltas]
        effs = []
        for _ in range(rng.randrange(1, 10)):
            kind = rng.choice(("SUPERSEDE", "OVERRULE", "REVOKE"))
            c1, c2 = rng.sample(keys, 2)
            tc = rng.choice(cells)
            effs.append(_e(kind, c1, c2=c2 if kind == "SUPERSEDE" else b"", target_cell=tc,
                           author_cell=tc, scope=POL_SCOPE[tc]))
        st = _store(*effs)
        seen = {}
        for d in rng.sample(deltas, len(deltas)):
            st.apply(*_wire(d))
            for k in st.nodes():
                r = _RANK[st.auth(k)]
                assert r >= seen.get(k, 0)
                seen[k] = r
    # a held c1 in another cell than target_cell invalidates (monotone: c1's cell never changes)
    assert _store(_d("alpha"), _e("OVERRULE", A, target_cell="eu.b",
                                  author_cell="eu.b")).auth(A) == "ACTIVE"


def test_replicas_with_different_subscriptions_agree_on_auth_in_the_shared_scope():
    far = claim_key("new", "us.a")
    objs = [_d("old"), _d("new", cell="us.a"), _e("SUPERSEDE", claim_key("old", "eu.a"), c2=far)]
    eu_only = Store(POL, KEY, scopes=["eu"], encoder="hashing")
    both = Store(POL, KEY, encoder="hashing")
    for o in objs:
        eu_only.apply(*_wire(o))
        both.apply(*_wire(o))
    shared = list(eu_only.nodes())
    assert shared == [claim_key("old", "eu.a")]
    assert {k: eu_only.auth(k) for k in shared} == {k: both.auth(k) for k in shared} \
        == {shared[0]: "DEPRECATED"}
    assert {k: v for k, v in both.vis_map().items() if k in shared} == eu_only.vis_map()


def test_effect_from_another_scope_never_revokes_or_overrules_a_node():
    """M29: an effect authored in scope us cannot REVOKE or OVERRULE an EU node, whatever
    target_cell it names; effect_valid refuses it and auth is evaluated per scope."""
    for kind in ("REVOKE", "OVERRULE"):
        for tc in ("eu.a", "us.a"):
            e = _e(kind, A, target_cell=tc, author_cell="us.a", scope="us")
            assert authorized(e, POL)
            assert not effect_valid(e, scope_partition(POL), {A: "eu.a"})
            s = _store(_d("alpha"), e)
            assert s.auth(A) == "ACTIVE" and s.live_nodes() == [A]


# --------------------------------------------------------------------------- the join
def test_status_join_order_independent_over_200_permutations_with_duplicates():
    rng = random.Random(5)
    effs = [_e("OVERRULE", A), _e("REVOKE", B, created_at=1), _e("SUPERSEDE", A, c2=B),
            _e("SUPERSEDE", B, c2=C), _e("DISTINGUISHES", min(A, C), c2=max(A, C),
                                         target_cell="eu.a"),
            _e("REVOKE", C, author_auth=3), _e("OVERRULE", C, author_cell="us.a")]
    effs = effs + effs[:4]                                     # duplicates (redelivery)
    deltas = [_d("alpha"), _d("beta"), _d("gamma")]
    want = None
    for _ in range(200):
        objs = deltas + effs
        rng.shuffle(objs)
        s = _store(*objs)
        got = {k: s.auth(k) for k in (A, B, C)}
        st = StatusState()
        for e in rng.sample(effs, len(effs)):
            st = st.add(e)
        assert {k: st.auth(k, POL, s.cell_of()) for k in (A, B, C)} == got
        want = want or got
        assert got == want
    assert want == {A: "OVERRULED", B: "REVOKED", C: "ACTIVE"}


def test_status_state_merge_is_a_join():
    e1, e2, e3 = _e("OVERRULE", A), _e("SUPERSEDE", B, c2=C), _e("REVOKE", A)
    x, y = StatusState().add(e1).add(e2), StatusState().add(e2).add(e3)
    assert x.merge(y) == y.merge(x) == x.merge(y).merge(y)
    assert set(x.merge(y).effects) == {object_id(canonical_bytes(e)) for e in (e1, e2, e3)}
    assert x.merge(y).auth(A, POL, {}) == "REVOKED"


def test_vis_dominates_auth():
    rng = random.Random(9)
    keys = [claim_key(f"t{i}", "eu.a") for i in range(6)]
    kinds = ("OVERRULE", "REVOKE", "SUPERSEDE")
    for trial in range(50):
        effs = []
        for _ in range(rng.randrange(0, 8)):
            k = rng.choice(kinds)
            c1 = rng.choice(keys)
            c2 = rng.choice([x for x in keys if x != c1]) if k == "SUPERSEDE" else b""
            effs.append(_e(k, c1, c2=c2, author_auth=rng.randrange(0, 2)))
        s = _store(*[_d(f"t{i}") for i in range(6)], *effs)
        s.derived = Derived(rep={k: rng.choice(keys[:2]) for k in keys})
        for k in keys:
            assert join(s.vis(k), s.auth(k)) == s.vis(k)
            assert _RANK[s.vis(k)] >= _RANK[s.auth(k)]
            if s.auth(k) in ("OVERRULED", "REVOKED"):
                assert s.vis(k) not in ("ACTIVE", "DEPRECATED")
        assert set(s.live_nodes()) == {k for k in keys if s.vis(k) == "ACTIVE"}


def test_cons_defaults_to_active_and_reads_the_derived_cache():
    s = _store(_d("alpha"), _d("beta"))
    assert s.vis(A) == s.vis(B) == "ACTIVE"
    s.derived = Derived(rep={A: B, B: B})
    assert s.cons(A) == s.vis(A) == "DEPRECATED" and s.vis(B) == "ACTIVE"
    assert s.live_nodes() == [B]


# --------------------------------------------------------------------------- nodes
def test_no_reput_overwrite_node_attributes_are_the_canonical_min():
    first = _d("alpha", auth=0, created_at=10, origin="C")
    late = _d("alpha", auth=2, created_at=20, origin="X")         # same key, other metadata
    for order in ((first, late), (late, first), (first, late, first)):
        s = _store(*order)
        assert list(s.nodes()) == [A]
        did, d = s.nodes()[A]
        assert d == attach_mac(first, KEY) and did == object_id(canonical_bytes(first))
        assert len(s.objects.ids("D", "eu")) == 2                 # both kept, none replaced
    # admissibility before anything else: a non-ACTIVE or zero-retention delta is no node
    s = _store(_d("beta", status="OVERRULED"), _d("gamma", retention=0))
    assert s.nodes() == {}
    with pytest.raises(ValueError):
        Store({**POL, "r_min": 0}, KEY, encoder="hashing")


# --------------------------------------------------------------------------- versions
def test_supersede_version_reactivates_via_representative():
    """§3.6, store level: c overruled; reactivation publishes c' = v + 1 and SUPERSEDE(c -> c').
    Whether c' joins c's group (the fold's tested pair passed) or forms its own, c' is live and
    c stays OVERRULED; rep excludes the superseded c even when c ranks better."""
    c = _d("rule", auth=0, created_at=1)
    c2 = _d("rule", version=1, auth=1, created_at=2)
    kc, kc2 = c.claim_key, c2.claim_key
    over, sup = _e("OVERRULE", kc), _e("SUPERSEDE", kc, c2=kc2)
    s = _store(c, c2, over, sup)
    assert s.rep([kc, kc2]) == kc2                                # rep excludes c
    s.derived = Derived(rep={kc: kc2, kc2: kc2})                   # merged
    assert s.vis(kc2) == "ACTIVE" and s.vis(kc) == "OVERRULED" and s.live_nodes() == [kc2]
    s.derived = Derived()                                          # own groups
    assert s.vis(kc2) == "ACTIVE" and s.vis(kc) == "OVERRULED" and s.live_nodes() == [kc2]
    plain = _store(c, c2, sup)                                     # no overrule: lineage only
    assert plain.vis(kc) == "DEPRECATED" and plain.live_nodes() == [kc2]
    # an unauthorized SUPERSEDE does not exclude: rep is the (rank, created_at, id) min
    weak = _store(c, c2, _e("SUPERSEDE", kc, c2=kc2, author_auth=2))
    assert weak.rep([kc, kc2]) == kc


# --------------------------------------------------------------------------- read
def _read(s, q, **kw):
    return [u for u, _ in s.graph_walk_read(q, encoder="hashing", k=10, **kw)]


def test_links_resolve_to_representatives_and_change_graph_walk_rank():
    x = _d("Paris capital France", unit_id=b"\x0a" * 16)
    x2 = _d("Paris is the capital of France", unit_id=b"\x0b" * 16)   # member of x's group
    y = _d("orchard harvest apples", unit_id=b"\x0c" * 16)
    z = _d("orchard harvest pears", unit_id=b"\x0d" * 16)
    kx, kx2, ky = x.claim_key, x2.claim_key, y.claim_key
    s = _store(x, x2, y, z)
    s.derived = Derived(rep={kx2: kx})
    base = _read(s, "Paris capital France orchard harvest", beta_graph=1.0)
    # the link names the non-representative member: it must resolve to the representative
    s.derived = Derived(rep={kx2: kx}, links=[("ASSOCIATES", kx2, ky, 1.0)])
    linked = _read(s, "Paris capital France orchard harvest", beta_graph=1.0)
    assert x2.unit_id not in base and x2.unit_id not in linked     # deprecated member hidden
    assert linked.index(y.unit_id) < base.index(y.unit_id)
    assert s.resolved_links() == [("ASSOCIATES", kx, ky, 1.0)]
    # a link inside one group resolves to a self-loop and is not walked
    s.derived = Derived(rep={kx2: kx}, links=[("CONSOLIDATED_FROM", kx2, kx, 1.0)])
    assert s.resolved_links() == [] and _read(s, "Paris capital France orchard harvest",
                                              beta_graph=1.0) == base


def test_read_ties_broken_by_unit_id():
    # canonical order (created_at) puts hi first; the read must still order by unit id
    lo = _d("same text", cell="eu.a", unit_id=b"\x01" * 16, created_at=20)
    hi = _d("same text", cell="eu.b", unit_id=b"\x02" * 16, created_at=10)
    for order in ((lo, hi), (hi, lo)):
        assert _read(_store(*order), "same text") == [lo.unit_id, hi.unit_id]


def test_read_hides_overruled_nodes_and_fingerprint_inputs_are_order_free():
    objs = [_d("alpha"), _d("beta"), _e("OVERRULE", A)]
    s1, s2 = _store(*objs), _store(*reversed(objs))
    assert _read(s1, "alpha beta") == [_d("beta").unit_id]
    for s in (s1, s2):
        s.derived = Derived(rep={}, links=[("ASSOCIATES", B, A, 0.5)])
    assert s1.fingerprint_inputs() == s2.fingerprint_inputs()


# --------------------------------------------------------------------------- fix round
def test_memoized_view_refreshes_on_apply_and_a_10k_key_loop_is_fast():
    import time
    s = _store(_d("alpha"))
    assert s.vis(A) == "ACTIVE"
    s.apply(*_wire(_e("OVERRULE", A)))
    assert s.vis(A) == "OVERRULED" and s.live_nodes() == []
    s.apply(*_wire(_d("beta")))
    assert s.live_nodes() == [B]
    n = 10_000
    big = Store(POL, KEY, encoder="hashing")
    keys = []
    for i in range(n):
        d = _d(f"t{i}", created_at=i)
        keys.append(d.claim_key)
        big.apply(*_wire(d))
        kind = ("OVERRULE", "REVOKE", "SUPERSEDE")[i % 3]
        big.apply(*_wire(_e(kind, d.claim_key, created_at=i,
                            c2=claim_key(f"t{i + 1}", "eu.a") if kind == "SUPERSEDE" else b"")))
    big.vis(keys[0])
    t = time.perf_counter()
    for k in keys:
        big.vis(k), big.auth(k)
    big.rep(keys[:3])
    assert time.perf_counter() - t < 1.0


def test_resolved_links_drop_only_consolidation_self_loops_raw_links_kept():
    s = _store(_d("alpha"), _d("beta"), _d("gamma"))
    raw = [("CONSOLIDATED_FROM", B, A, 1.0), ("CONTRADICTS", B, A, 1.0),
           ("ASSOCIATES", B, C, 0.5)]
    s.derived = Derived(rep={B: A}, links=list(raw))
    assert s.resolved_links() == [("CONTRADICTS", A, A, 1.0), ("ASSOCIATES", A, C, 0.5)]
    assert s.derived.links == raw


def test_read_returns_each_unit_once_with_its_best_node_score():
    uid = b"\x05" * 16
    a = _d("Paris capital France", unit_id=uid, created_at=1)
    b = _d("orchard harvest", unit_id=uid, created_at=2)
    other = _d("Paris capital", unit_id=b"\x06" * 16, created_at=3)
    s = _store(a, b, other)
    hits = s.graph_walk_read("Paris capital France", encoder="hashing", k=3)
    assert [u for u, _ in hits] == [uid, other.unit_id]                # k counts units
    single = _store(a).graph_walk_read("Paris capital France", encoder="hashing", k=1)
    assert hits[0][1] == single[0][1]


def test_rep_of_a_group_with_a_non_node_member_raises():
    s = _store(_d("alpha"))
    with pytest.raises(ValueError, match=B.hex()):
        s.rep([A, B])


def test_retention_at_r_min_is_a_node():
    s = Store({**POL, "r_min": 7}, KEY, encoder="hashing")
    s.apply(*_wire(_d("alpha", retention=7)))
    s.apply(*_wire(_d("beta", retention=6)))
    assert list(s.nodes()) == [A]


def test_invalid_supersede_does_not_exclude_from_rep():
    c = _d("rule", auth=0, created_at=1)
    c2 = _d("rule", version=1, auth=1, created_at=2)
    lying = _e("SUPERSEDE", c.claim_key, c2=c2.claim_key, target_cell="eu.b", author_cell="eu.b")
    assert authorized(lying, POL)
    assert _store(c, c2, lying).rep([c.claim_key, c2.claim_key]) == c.claim_key


def test_more_authoritative_paraphrase_becomes_the_representative():
    a = _d("the rule", auth=1, created_at=1)
    b = _d("the rule restated", auth=0, created_at=2)
    assert _store(a, b).rep([a.claim_key, b.claim_key]) == b.claim_key


def test_fingerprint_groups_follow_derived_rep():
    s = _store(_d("alpha"), _d("beta"), _d("gamma"))
    s.derived = Derived(rep={B: A})
    groups = s.fingerprint_inputs()[1]
    assert sorted(groups) == sorted([(A, tuple(sorted((A, B)))), (C, (C,))])


def test_ranks_are_non_negative():
    with pytest.raises(ValueError):
        _e("OVERRULE", A, author_auth=-1)
    with pytest.raises(ValueError):
        _d("alpha", auth=-1)


def test_authorized_accepts_a_precomputed_partition():
    e = _e("OVERRULE", A)
    assert authorized(e, POL, scope_partition(POL)) == authorized(e, POL) is True


def test_auth_is_scoped_even_if_effect_validity_regresses(monkeypatch):
    """Defense in depth for M29: auth is evaluated per scope, so an effect filed under us
    cannot reach an EU node even when effect_valid is replaced by one that accepts all."""
    import meld.brain.store as store_mod
    import meld.merge.status_crdt as crdt_mod
    monkeypatch.setattr(crdt_mod, "effect_valid", lambda e, scope_of, cell_of: True)
    monkeypatch.setattr(store_mod, "effect_valid", lambda e, scope_of, cell_of: True)
    s = _store(_d("alpha"), _e("REVOKE", A, target_cell="eu.a", author_cell="us.a", scope="us"))
    assert s.auth(A) == "ACTIVE" and s.live_nodes() == [A]


def test_incremental_view_equals_a_view_built_from_the_final_sets():
    # nodes, cell_of, auth and rep are kept incrementally by apply; reading them mid-stream
    # (appends, earlier exact copies, inadmissible copies, an effect before its target) must
    # leave them equal to a store that receives the same sets at once, in sorted order
    rng = random.Random(3)
    texts = ["alpha", "beta", "gamma", "delta"]
    objs = [_d(t, created_at=rng.randint(1, 40), unit_id=bytes([i]) * 16,
               retention=0 if i % 7 == 6 else FIXED_ONE)
            for i, t in enumerate(texts * 3)]
    objs += [_d("alpha", cell="eu.b", created_at=5),                  # another key, other cell
             _e("SUPERSEDE", A, B, author_auth=1), _e("OVERRULE", C),
             _e("REVOKE", claim_key("alpha", "eu.b"))]                 # target_cell mismatch
    want = _store(*sorted(objs, key=repr))
    for _ in range(30):
        order = objs[:]
        rng.shuffle(order)
        s = Store(POL, KEY, encoder="hashing")
        for o in order:
            s.apply(*_wire(o))
            s.nodes(), s.vis_map()                                     # mid-stream reads
        assert list(s.nodes().items()) == list(want.nodes().items())
        assert s.cell_of() == want.cell_of() and s.vis_map() == want.vis_map()
        assert s.rep([A, B]) == want.rep([A, B])
