"""The fold and its step (design §3.3-3.6, §4.3-4.4, §5 Props 1-5, Lemmas 2b, 5b, 5c, 6, 7b):
the pure step with select_tested_pair, the group gates after the test, readjudicate over the
replicated sets, PASS certificates, counters. Hashing encoder and the lexical NLI rung (its
probability form) keep every test offline; scripted providers pin exact payloads."""
from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from meld.brain.store import Derived, Store
from meld.merge import classifier
from meld.merge.classifier import FoldState, node_of, step, StepContext
from meld.merge.conformal import bin_of
from meld.merge.signals import _make_encoder, quantize_int8
from meld.merge.status_crdt import join
from meld.proto import canon, claim_key
from meld.protocol import fold as fold_mod
from meld.protocol.fold import nli_provider, readjudicate, verify_pass
from meld.protocol.objects import (
    FIXED_ONE, INF16, AuthoredEffect, CalRecord, Delta, EpochRecord, SignalRecord, attach_mac,
    canonical_bytes, m_nli, m_pol, m_pol_of, object_id,
)
from meld.sync.merkle import merkle_root

KEY = b"t6-test-group-key"
LO, SC = round(0.3 * FIXED_ONE), round(0.3 * FIXED_ONE)
SEL = (LO, SC, 3)
POL = {"rank_map": {"PRECEDENT": 0, "OBSERVATION": 1, "SPECULATION": 2},
       "scopes": {"eu": ["eu.a", "eu.b"]},
       "auth_policy": {"OVERRULE": 0, "REVOKE": 0, "SUPERSEDE": 1, "DISTINGUISHES": 1,
                       "calibration_authority": {"eu": "cal-auth"}},
       "step_version": 0, "r_min": 1, "bootstrap_selector": list(SEL)}
MNLI = m_nli("hashing", "lexical", "int8", "canon-1")
MPOL = m_pol_of(POL)
TH = round(0.7 * FIXED_ONE)
LEX = nli_provider("lexical")
_ENC = _make_encoder("hashing")


def _q(text):
    return quantize_int8(_ENC.encode_one(canon(text))).tobytes()


def _d(text, t, cell="eu.a", auth=1, version=0, unit=None, **over):
    kw = dict(unit_id=unit or claim_key(text, cell, version), claim_key=claim_key(text, cell, version),
              text=text, cell=cell, version=version, auth=auth, status="ACTIVE",
              retention=FIXED_ONE, created_at=t, origin="A", encoder="hashing", q=_q(text))
    kw.update(over)
    return Delta(**kw)


def _k(d):
    return d.claim_key


def _e(kind, c1, c2=b"", author_auth=0, target_cell="eu.a"):
    if kind == "DISTINGUISHES":
        c1, c2 = sorted((c1, c2))
    return AuthoredEffect(scope="eu", kind=kind, c1=c1, c2=c2, target_cell=target_cell,
                          author="C", author_cell="eu.a", author_auth=author_auth, created_at=0)


def _ep(rows=((0, 100, 0, TH, INF16),), seq=1, author="cal-auth", sel=SEL, B0=256, **over):
    kw = dict(scope="eu", seq=seq, m_nli=MNLI, m_pol=MPOL, policy="SEED", alpha=20_000,
              alpha_c=20_000, mondrian_c=False, cal_root=b"\x09" * 16, cal_hw="cpu",
              bins=(B0, "doubling"), table=tuple(rows), sigma_lo_q=sel[0], sigma_c_q=sel[1],
              m=sel[2], author=author, created_at=0)
    kw.update(over)
    return EpochRecord(**kw)


def _cal(x, y, n_prior=1):
    cx, cy = sorted((x, y))
    return CalRecord(scope="eu", c_x=cx, c_y=cy, bin=bin_of(n_prior, 256), s_q=1, hw="cpu", obs=1,
                     label="NON_MERGEABLE", subtype="OTHER", source="AUDIT", item=cx.hex(),
                     author="cal-auth", created_at=0, sigma_lo_q=LO, m=3, m_pol=MPOL,
                     n_prior=n_prior)


def _sig(x, y, pxy, pyx, hw="gpu"):
    if x > y:
        x, y, pxy, pyx = y, x, pyx, pxy
    return SignalRecord(scope="eu", c_x=x, c_y=y, m_nli=MNLI, pay_xy=pxy, pay_yx=pyx,
                        issuer="r2", hw=hw)


def _wire(obj):
    o = attach_mac(obj, KEY)
    return canonical_bytes(o), o.mac


def _store(*objs):
    s = Store(POL, KEY, encoder="hashing")
    for o in objs:
        assert s.apply(*_wire(o)) is not None
    return s


def _add(s, *objs):
    for o in objs:
        assert s.apply(*_wire(o)) is not None


class Counting:
    """Signal provider spy: counts calls; ``table`` overrides payloads by (premise, hypothesis)."""

    def __init__(self, table=None):
        self.calls, self.seen, self.table = 0, [], table or {}

    def __call__(self, a, b):
        self.calls += 1
        self.seen.append((a, b))
        return self.table.get((a, b)) or LEX(a, b)


def _fold(s, provider=LEX, **kw):
    return readjudicate(s, "eu", m_nli=MNLI, provider=provider, issuer="r1", hw="cpu", **kw)


def _tested(res):
    return [x.tested for x in res.decisions if x.tested]


def _dec_n(res, k):
    return next(x.n_prior for x in res.decisions if x.b == k)


def _dec(res, d):
    return next(x for x in res.decisions if x.b == _k(d))


def _group(s, k):
    r = s.derived.rep.get(k, k)
    return {n for n in s.nodes() if s.derived.rep.get(n, n) == r}


# The running instance r < a < b (m = 3): a paraphrases r and merges with it; b paraphrases a
# (the tested pair passes) but contradicts r, a member of a's group outside the test.
R = _d("alpha bravo", 10)
A = _d("alpha bravo charlie", 20)
B = _d("alpha charlie", 30)


# --------------------------------------------------------------------------- R1 and nodes
def test_r1_reject_before_candidates_including_relevance_zero(monkeypatch):
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("a REJECTed delta read a signal")

    def no_select(*a, **k):
        raise AssertionError("candidates computed for a REJECTed delta")

    monkeypatch.setattr(classifier, "select_tested_pair", no_select)
    st = FoldState()
    ctx = StepContext(sigma_lo=0, sigma_c=0, m=5, theta=lambda n: 0, rank={}, distinguishes=set(),
                      cal_pairs=set(), sig=Boom(), rep=None, r_min=1)
    for bad in (_d("alpha bravo", 40, retention=0),          # relevance zero is below R_min
                _d("alpha bravo", 40, status="OVERRULED")):
        dec = step(st, node_of(bad, b"\x01" * 16), ctx)
        assert dec.decision == "REJECT" and dec.links == () and st.keys == []
    # in the fold: a non-admissible exact paraphrase is no node, no decision, no link
    monkeypatch.undo()
    s = _store(R, _d("alpha bravo", 40, version=1, retention=0))
    res = _fold(s)
    assert [x.b for x in res.decisions] == [_k(R)] and s.derived.links == []


def test_exact_key_is_one_node_attributes_from_canonical_min():
    late = _d("alpha bravo", 20, auth=2, unit=b"U" * 16)          # same key, later, weak
    early = _d("alpha bravo", 10, auth=0, unit=b"V" * 16)         # same key, earlier, strong
    para = _d("alpha bravo", 30, auth=1, version=1)               # identical text, new key
    s = _store(late, early, para, _ep())
    assert list(s.nodes()) == [_k(early), _k(para)]
    assert s.nodes()[_k(early)][1].unit_id == b"V" * 16
    res = _fold(s)
    assert [x.b for x in res.decisions] == [_k(early), _k(para)]
    # rank from the canonical-min delta (0): the rank-1 paraphrase is not eligible, so RELATE
    assert _dec(res, para).decision == "RELATE" and _dec(res, para).tested is None


def test_conflict_retains_incoming_as_live_node():
    a = _d("the valve is open", 10)
    b = _d("the valve is not open", 20)
    s = _store(a, b, _ep())
    res = _fold(s)
    assert _dec(res, b).decision == "CONFLICT" and _dec(res, b).gates == ("mask",)
    assert set(s.live_nodes()) == {_k(a), _k(b)}
    assert s.derived.links == [("CONTRADICTS", _k(b), _k(a), 1.0)]                  # exact multiset


# --------------------------------------------------------------------------- selector
def test_selector_is_theta_independent():
    ds = [R, A, B, _d("alpha delta", 40), _d("bravo charlie delta", 50, cell="eu.b")]
    pairs = None
    for ep in (None, _ep(), _ep(rows=((0, 9, 0, 0, INF16),)), _ep(rows=((0, 9, 0, INF16, INF16),))):
        s = _store(*ds, *([ep] if ep else []))
        res = _fold(s)
        assert _tested(res) == [(a, b) for a, b, _ in fold_mod.tested_pairs(s, "eu", sigma_lo=LO, m=3)]
        pairs = pairs or _tested(res)
        assert _tested(res) == pairs and pairs


def test_candidacy_never_reads_vis_or_auth(monkeypatch):
    base = [R, A, B, _d("alpha delta", 40)]
    plain = fold_mod.tested_pairs(_store(*base), "eu", sigma_lo=LO, m=3)
    s = _store(*base, _e("OVERRULE", _k(R)), _e("REVOKE", _k(A)))

    def boom(*a, **k):
        raise AssertionError("candidacy read vis/auth")
    for name in ("vis", "auth", "vis_map", "live_nodes", "cons"):
        monkeypatch.setattr(Store, name, boom)
    assert fold_mod.tested_pairs(s, "eu", sigma_lo=LO, m=3) == plain and plain


def test_ties_by_canonical_order_and_read_ties_by_unit_id():
    for first, second in ((10, 20), (20, 10)):
        x = _d("alpha bravo", first, version=0)
        y = _d("alpha bravo", second, version=1)                 # identical q: a sigma tie
        b = _d("alpha bravo delta", 30, auth=2)                   # ineligible: abstains to a*
        s = _store(x, y, b)
        res = _fold(s)
        earlier = _k(x) if first < second else _k(y)
        assert _dec(res, b).a_star == earlier
    # read ties: equal scores order by unit id, against canonical order
    u1 = _d("echo foxtrot", 10, unit=b"\xff" * 16)
    u2 = _d("echo foxtrot", 20, version=1, unit=b"\x00" * 16)
    s = _store(u1, u2)
    _fold(s)
    top = s.graph_walk_read("echo foxtrot", encoder="hashing", k=2, beta_graph=0.0)
    assert [u for u, _ in top] == [b"\x00" * 16, b"\xff" * 16] and top[0][1] == top[1][1]


# --------------------------------------------------------------------------- the step
def _scripted_pair(s_val, contra_rev=False):
    """a, b with exactly scripted payloads: p_e = s_val both ways; optionally a -> b contradicts."""
    a, b = _d("kilo lima", 10), _d("kilo lima mike", 20)
    ta, tb = canon(a.text), canon(b.text)
    fwd = (0, s_val, FIXED_ONE - s_val)
    rev = (FIXED_ONE, s_val, 0) if contra_rev else fwd
    return a, b, Counting({(tb, ta): fwd, (ta, tb): rev})


def test_merge_iff_score_exceeds_bin_theta_and_gates_only_move_away_from_merge():
    T = 40_000
    for s_val in (T - 1, T, T + 1):
        for contra in (False, True):
            a, b, prov = _scripted_pair(s_val, contra)
            s = _store(a, b, _ep(rows=((0, 9, 0, T, INF16),)))
            d = _dec(_fold(s, prov), b)
            assert d.s_q == s_val and d.theta_q == T
            want = "CONFLICT" if contra else ("MERGE" if s_val > T else "RELATE")
            assert d.decision == want, (s_val, contra, d)
            # a reverse-direction pair mask is recorded as a link, not only as a gate
            assert (("CONTRADICTS", _k(b), _k(a), 1.0) in d.links) == contra
    # Context: a node of another cell is never a_e, whatever its score
    a, b, prov = _scripted_pair(T + 1)
    a = replace(a, cell="eu.b", claim_key=claim_key(a.text, "eu.b"), unit_id=b"X" * 16)
    d = _dec(_fold(_store(a, b, _ep(rows=((0, 9, 0, T, INF16),))), prov), b)
    assert d.a_star == _k(a) and d.tested is None and d.decision == "RELATE"
    # the bin of n_prior picks the row: b has n_prior 1, B_0 = 1 puts it in bin 1
    for th0, th1, want in ((0, FIXED_ONE, "RELATE"), (FIXED_ONE, 0, "MERGE")):
        a, b, prov = _scripted_pair(T)
        s = _store(a, b, _ep(rows=((0, 9, 0, th0, INF16), (1, 9, 0, th1, INF16)), B0=1))
        d = _dec(_fold(s, prov), b)
        assert d.n_prior == 1 and d.theta_q == th1 and d.decision == want


def test_collected_contradiction_with_group_member_forces_conflict():
    s = _store(R, A, B, _ep())
    res = _fold(s)
    assert _dec(res, A).decision == "MERGE"
    d = _dec(res, B)
    assert d.a_star == _k(A) and d.tested == (_k(A), _k(B)) and d.s_q > d.theta_q
    assert d.decision == "CONFLICT" and "group_contradiction" in d.gates
    assert _k(B) in s.live_nodes() and _group(s, _k(B)) == {_k(B)}
    assert ("CONTRADICTS", _k(B), _k(R), 1.0) in s.derived.links
    live = set(s.live_nodes())
    assert {(t, x, y) for t, x, y, _ in s.resolved_links() if t == "CONTRADICTS"} == {
        ("CONTRADICTS", _k(B), s.derived.rep.get(_k(R), _k(R)))} and _k(B) in live
    # without collection (sigma_c above sigma(b, r)) b merges and the contradiction is hidden
    # in the group: the in-group-undetected counter sees it under ground truth
    # in group: with r in K(b) it is an in-group contradiction but not one outside K(b); with
    # m = 1, r is outside K(b) and counts as undetected-outside-K (design §7.5)
    truth = lambda x, y: "CONTRADICTION" if {x, y} == {_k(B), _k(R)} else "MERGEABLE"
    for sel, outside in (((LO, FIXED_ONE, 3), 0), ((LO, SC, 1), 1)):
        s2 = _store(R, A, B, _ep(sel=sel))
        res2 = _fold(s2, truth=truth)
        assert _dec(res2, B).decision == "MERGE"
        assert res2.counters["in_group_contradictions"] == 1
        assert res2.counters["undetected_outside_k"] == outside
        assert res2.counters["impure_pairs"] == 1 and res2.counters["group_pairs"] == 3
    assert res.counters["pass_time_misses"] > 0 and res.counters["nli_calls"] > 0


def _assert_no_live_self_loop(s):
    assert all(src != dst for _, src, dst, _ in s.derived.links)          # raw links
    assert all(src != dst for t, src, dst, _ in s.resolved_links() if t != "CONSOLIDATED_FROM")


def test_no_live_link_is_a_self_loop():
    s = _store(R, A, B, _ep())
    _fold(s)
    _assert_no_live_self_loop(s)
    rng = random.Random(7)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "not"]
    for trial in range(30):
        ds = [_d(" ".join(rng.sample(words, rng.randint(2, 4))), 10 * i + 1, auth=rng.randint(0, 2),
                 version=i) for i in range(8)]
        s = _store(*ds, _ep(sel=(LO, round(0.2 * FIXED_ONE), 4)))
        _fold(s)
        _assert_no_live_self_loop(s)


def test_bounded_collection_adds_contradicts_and_only_forces_conflict_in_group():
    r = _d("alpha bravo", 10)
    a = _d("alpha charlie delta", 20)
    b = _d("alpha charlie", 30)
    s = _store(r, a, b, _ep())
    res = _fold(s)
    assert _dec(res, a).decision == "RELATE"                              # r, a: two groups
    d = _dec(res, b)
    assert d.a_star == _k(a) and d.decision == "MERGE" and d.gates == ()
    assert ("CONTRADICTS", _k(b), _k(r), 1.0) in s.derived.links
    # bounded: sigma_c above sigma(b, r) reads nothing on (b, r) and records no link
    prov = Counting()
    s2 = _store(r, a, b, _ep(sel=(LO, round(0.6 * FIXED_ONE), 3)))
    _fold(s2, prov)
    assert (canon(b.text), canon(r.text)) not in prov.seen
    assert not any(t == "CONTRADICTS" for t, *_ in s2.derived.links)
    # in group: the running instance goes to CONFLICT (test above)


def test_group_cannot_link_veto_after_test():
    b = _d("alpha bravo charlie", 30, version=1)                          # identical to A's text
    s = _store(R, A, b, _e("DISTINGUISHES", _k(b), _k(R)), _ep())
    d = _dec(_fold(s), b)
    assert d.tested == (_k(A), _k(b)) and d.s_q > d.theta_q
    assert d.decision == "RELATE" and d.gates == ("group_cannot_link",)
    assert _group(s, _k(b)) == {_k(b)}
    # the veto runs after the test: a failing test RELATEs with no gate recorded
    s = _store(R, A, b, _e("DISTINGUISHES", _k(b), _k(R)), _ep(rows=((0, 9, 0, FIXED_ONE, INF16),)))
    d = _dec(_fold(s), b)
    assert d.decision == "RELATE" and d.gates == ()


def test_cal_veto_relates_labelled_nonmergeable_pair_under_retroactive_epoch():
    s = _store(R, A, _ep())
    assert _dec(_fold(s), A).decision == "MERGE"
    _add(s, _cal(_k(R), _k(A)), _ep(seq=2))
    d = _dec(_fold(s), A)
    assert d.decision == "RELATE" and d.gates == ("cal_veto",) and d.s_q > d.theta_q
    assert set(s.live_nodes()) == {_k(R), _k(A)}


def test_distinguishes_splits_group_at_pass():
    s = _store(R, A, _ep())
    _fold(s)
    assert _group(s, _k(A)) == {_k(R), _k(A)}
    _add(s, _e("DISTINGUISHES", _k(R), _k(A)))
    d = _dec(_fold(s), A)
    assert d.tested is None and d.decision == "RELATE"
    assert _group(s, _k(A)) == {_k(A)} and set(s.live_nodes()) == {_k(R), _k(A)}


def test_no_default_thresholds_without_epoch_similarity_merge_disabled():
    ds = [_d("alpha bravo", 10), _d("alpha bravo", 20, version=1), _d("alpha bravo", 30, version=2)]
    for extra in ([], [_ep(rows=((0, 9, 0, INF16, INF16),))], [_ep(B0=1)]):   # none; +inf; short bin
        s = _store(*ds, *extra)
        res = _fold(s)
        assert all(x.decision != "MERGE" for x in res.decisions)
        assert all(x.theta_q == INF16 for x in res.decisions if x.tested)
        assert len(s.live_nodes()) == 3
    bare = Store({k: v for k, v in POL.items() if k != "bootstrap_selector"}, KEY, encoder="hashing")
    with pytest.raises(ValueError, match="bootstrap_selector"):
        readjudicate(bare, "eu", m_nli=MNLI)                             # no default selector


def test_unauthorized_epoch_is_ignored():
    ds = [_d("alpha bravo", 10), _d("alpha bravo", 20, version=1)]
    good = _ep(rows=((0, 9, 0, INF16, INF16),))
    for bad in (_ep(rows=((0, 9, 0, 0, INF16),), seq=9, author="mallory"),
                _ep(rows=((0, 9, 0, 0, INF16),), seq=9, m_nli=b"\x07" * 16),
                _ep(rows=((0, 9, 0, 0, INF16),), seq=9, m_pol=b"\x07" * 16)):
        s = _store(*ds, good, bad)
        res = _fold(s)
        assert res.certificate.epoch_id == object_id(canonical_bytes(good))
        assert all(x.decision != "MERGE" for x in res.decisions)


# --------------------------------------------------------------------------- views
def test_vis_dominates_auth():
    ds = [R, A, B, _d("alpha delta", 40), _d("alpha bravo", 50, version=1, auth=0)]
    s = _store(*ds, _e("OVERRULE", _k(R)), _e("REVOKE", _k(B)),
               _e("SUPERSEDE", _k(A), _k(ds[-1]), author_auth=1), _ep())
    _fold(s)
    assert s.derived.rep
    for k in s.nodes():
        assert join(s.vis(k), s.auth(k)) == s.vis(k)


def test_overrule_then_paraphrase_hidden_iff_eligible_tested_and_passed():
    a = _d("the valve is open", 10, auth=1)
    cases = [(1, _ep(), "MERGE", False),          # eligible, tested, passes, same rank: hidden
             (2, _ep(), "RELATE", True),          # less authoritative: abstains, visible
             (0, _ep(), "MERGE", True),           # more authoritative: becomes the rep, visible
             (1, None, "RELATE", True),           # no epoch: every paraphrase relates
             (1, _ep(rows=((0, 9, 0, FIXED_ONE, INF16),)), "RELATE", True)]   # test fails
    for rank, ep, want, visible in cases:
        b = _d("valve open", 20, auth=rank)
        s = _store(a, b, _e("OVERRULE", _k(a)), *([ep] if ep else []))
        assert _dec(_fold(s), b).decision == want
        assert (s.vis(_k(b)) == "ACTIVE") == visible, (rank, ep)
        assert s.vis(_k(a)) == "OVERRULED"


def test_supersede_version_reactivates_via_representative():
    c = _d("alpha bravo", 10)
    c2 = _d("alpha bravo", 20, version=1)
    eff = [_e("OVERRULE", _k(c)), _e("SUPERSEDE", _k(c), _k(c2), author_auth=1)]
    s = _store(c, c2, *eff, _ep())
    assert _dec(_fold(s), c2).decision == "MERGE"
    assert s.derived.rep[_k(c)] == _k(c2)
    assert s.vis(_k(c2)) == "ACTIVE" and s.vis(_k(c)) == "OVERRULED"
    s = _store(c, c2, *eff)                                       # no epoch: own live group
    assert _dec(_fold(s), c2).decision == "RELATE"
    assert s.vis(_k(c2)) == "ACTIVE" and s.vis(_k(c)) == "OVERRULED"


def test_links_resolve_to_representatives_and_change_graph_walk_rank():
    c = _d("alpha charlie delta echo", 30, auth=2)                # ineligible: RELATE to a* = A
    decoy = _d("foxtrot golf", 40)
    s = _store(R, A, c, decoy, _ep())
    res = _fold(s)
    assert _dec(res, A).decision == "MERGE" and _dec(res, c).a_star == _k(A)
    rep = s.derived.rep.get(_k(A), _k(A))
    assert rep == _k(R) and ("ASSOCIATES", _k(c), _k(A)) in [l[:3] for l in s.derived.links]
    assert ("ASSOCIATES", _k(c), _k(R)) in [l[:3] for l in s.resolved_links()]
    q = "charlie delta echo foxtrot"
    with_links = s.graph_walk_read(q, encoder="hashing", k=2, beta_graph=0.9)
    s.derived = Derived(rep=s.derived.rep, links=[])
    without = s.graph_walk_read(q, encoder="hashing", k=2, beta_graph=0.9)
    assert with_links != without
    assert R.unit_id in [u for u, _ in with_links] and R.unit_id not in [u for u, _ in without]


# --------------------------------------------------------------------------- Sig and epochs
def test_more_sig_never_creates_a_merge_except_group_gate_release():
    # the release: under Sig b is CONFLICT by the group gate; more evidence on (r, a) splits
    # r from a's group, so the collected contradiction no longer targets a member and b merges
    s = _store(R, A, B, _ep())
    before = {x.b: x for x in _fold(s).decisions}
    assert before[_k(B)].gates == ("group_contradiction",)
    _add(s, _sig(_k(R), _k(A), (0, 0, FIXED_ONE), (0, 0, FIXED_ONE)))
    after = {x.b: x for x in _fold(s, provider=None).decisions}
    assert after[_k(A)].decision == "RELATE" and after[_k(B)].decision == "MERGE"
    # the property on random streams: a merge created by more evidence is always a release
    rng = random.Random(3)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "not"]
    for trial in range(40):
        ds = [_d(" ".join(rng.sample(words, rng.randint(2, 4))), 10 * i + 1, auth=rng.randint(0, 1),
                 version=i) for i in range(8)]
        s = _store(*ds, _ep(sel=(LO, round(0.2 * FIXED_ONE), 4)))
        r1 = _fold(s)
        sig = {x.b: x for x in r1.decisions}
        keys = list(s.nodes())
        for _ in range(4):                                    # extra observations, other hw
            x, y = rng.sample(keys, 2)
            bump = lambda: (rng.randint(0, FIXED_ONE), rng.randint(0, FIXED_ONE // 2), 0)
            _add(s, _sig(x, y, bump(), bump()))
        r2 = _fold(s, provider=None)
        assert r2.counters["pass_time_misses"] == 0 and _tested(r2) == _tested(r1)
        for x in r2.decisions:
            if x.decision == "MERGE" and sig[x.b].decision != "MERGE":
                assert {"group_contradiction", "group_cannot_link"} & set(sig[x.b].gates)
        if not any({"group_contradiction", "group_cannot_link"} & set(d.gates) for d in sig.values()):
            assert {d.b for d in r2.decisions if d.decision == "MERGE"} <= {
                d.b for d in r1.decisions if d.decision == "MERGE"}


def test_calibration_pairs_are_drawn_by_the_admission_selector(monkeypatch):
    calls = []
    real = classifier.select_tested_pair

    def spy(*a, **k):
        calls.append(a[1].key)
        return real(*a, **k)
    monkeypatch.setattr(classifier, "select_tested_pair", spy)
    s = _store(R, A, B, _d("alpha delta", 40), _ep())
    drawn = fold_mod.tested_pairs(s, "eu", sigma_lo=LO, m=3)
    n_cal = len(calls)
    res = _fold(s)
    assert n_cal == len(s.nodes()) and len(calls) == 2 * n_cal
    assert [(a, b) for a, b, _ in drawn] == _tested(res)
    assert [n for _, _, n in drawn] == [_dec_n(res, b) for _, b, _ in drawn]
    # the draws carry the n_prior a Cal record bins by
    cals = [_cal(a, b, n) for a, b, n in drawn]
    assert all(k.bin == bin_of(_dec_n(res, b), 256) for k, (_, b, _) in zip(cals, drawn))


def test_table_only_epoch_refolds_with_zero_nli_calls():
    ds = [R, A, B, _d("alpha delta", 40), _d("bravo delta", 50)]
    s = _store(*ds, _ep())
    p1 = Counting()
    r1 = _fold(s, p1)
    assert p1.calls > 0 and r1.counters["nli_calls"] == p1.calls
    _add(s, _ep(rows=((0, 9, 0, FIXED_ONE, INF16),), seq=2))              # table only
    p2 = Counting()
    r2 = _fold(s, p2)
    assert p2.calls == 0 and r2.counters["pass_time_misses"] == 0
    assert _tested(r2) == _tested(r1)
    assert [x.decision for x in r2.decisions] != [x.decision for x in r1.decisions]


def test_selector_epoch_counts_new_nli_calls_and_new_tested_pairs():
    ds = [R, A, B, _d("alpha delta", 40), _d("bravo delta echo", 50), _d("echo golf", 60)]
    s = _store(*ds, _ep(sel=(round(0.6 * FIXED_ONE), SC, 1)))
    r1 = _fold(s, Counting())
    _add(s, _ep(sel=(round(0.1 * FIXED_ONE), round(0.1 * FIXED_ONE), 5), seq=2))
    p2 = Counting()
    r2 = _fold(s, p2)
    assert p2.calls > 0 and r2.counters["nli_calls"] == p2.calls
    assert set(_tested(r2)) - set(_tested(r1))


# --------------------------------------------------------------------------- whole fold
def _corpus():
    ds = [R, A, B,
          _d("alpha bravo", 15, version=1, auth=0),
          _d("alpha delta echo", 20, auth=2),                 # created_at tie with A
          _d("the valve is open", 35), _d("the valve is not open", 45),
          _d("valve open", 55, cell="eu.b"),
          _d("alpha bravo", 60, auth=2, unit=b"W" * 16),      # exact copy of R's key
          _d("echo golf", 70, retention=0)]
    effects = [_e("DISTINGUISHES", _k(ds[5]), _k(ds[6])), _e("OVERRULE", _k(ds[4])),
               _e("SUPERSEDE", _k(R), _k(ds[3]), author_auth=1)]
    return ds + effects + [_cal(_k(A), _k(ds[4])), _ep()]


def test_fold_is_permutation_invariant_200():
    objs = _corpus()
    ref = None
    rng = random.Random(11)
    for trial in range(200):
        order = objs[:]
        rng.shuffle(order)
        s = Store(POL, KEY, encoder="hashing")
        cut = rng.randint(0, len(order))
        for i, o in enumerate(order):
            s.apply(*_wire(o))
            if i == cut and trial % 2:
                _fold(s)                                       # an intermediate pass
        res = _fold(s)
        got = (res.certificate.fingerprint, s.graph_walk_read("alpha valve open", encoder="hashing"),
               [(x.b, x.decision) for x in res.decisions], res.certificate.decisions_digest)
        ref = ref or got
        assert got == ref
    assert {"MERGE", "RELATE", "CONFLICT", "INSERT"} <= {d for _, d in ref[2]}


def test_pass_patch_certificate_recomputes_from_end_of_pass_roots():
    s = Store(POL, KEY, encoder="hashing")
    for o in _corpus():
        s.apply(*_wire(o))
    res = _fold(s)
    cert, snap = res.certificate, res.snapshot
    assert res.new_sig and all(object_id(canonical_bytes(r)) in snap["Sig"] for r in res.new_sig)
    assert cert.roots == tuple(merkle_root(s.objects.ids(k, "eu")) for k in ("D", "A", "Sig", "Cal", "Ep"))
    assert verify_pass(cert, snap, s)
    for bad in (replace(cert, decisions_digest=b"\x00" * 16), replace(cert, fingerprint=b"\x00" * 16),
                replace(cert, roots=(b"\x00" * 16,) + cert.roots[1:])):
        assert not verify_pass(bad, snap, s)
    assert not verify_pass(cert, {**snap, "Sig": snap["Sig"][1:]}, s)
    # retained snapshot: later arrivals do not invalidate the certificate
    _add(s, _d("alpha bravo charlie delta", 99))
    assert verify_pass(cert, snap, s)


def test_pass_certificate_is_published_with_a_mac_and_a_tampered_issuer_fails_on_receipt():
    s = Store(POL, KEY, encoder="hashing")
    for o in _corpus():
        s.apply(*_wire(o))
    res = _fold(s)
    received, mac = fold_mod.publish_pass(res.certificate, KEY)
    got = fold_mod.receive_pass(received, mac, KEY)
    assert got == res.certificate and verify_pass(got, res.snapshot, s)
    forged = received.replace(b'"issuer":"r1"', b'"issuer":"mallory"')
    assert forged != received and fold_mod.receive_pass(forged, mac, KEY) is None
    assert fold_mod.receive_pass(received, mac, b"another-group-key") is None
    assert fold_mod.receive_pass(received, "", KEY) is None
    # the certificate attests content, not the issuer: only the record MAC attributes it
    assert verify_pass(replace(res.certificate, issuer="mallory"), res.snapshot, s)


def test_receive_pass_refuses_a_mac_that_does_not_cover_the_received_bytes():
    s = Store(POL, KEY, encoder="hashing")
    for o in _corpus():
        s.apply(*_wire(o))
    first = _fold(s).certificate
    _add(s, _d("golf hotel", 99))
    second = _fold(s).certificate
    (b1, m1), (b2, m2) = fold_mod.publish_pass(first, KEY), fold_mod.publish_pass(second, KEY)
    assert fold_mod.receive_pass(b2, m2, KEY) == second
    assert fold_mod.receive_pass(b2, m1, KEY) is None               # a valid MAC of other bytes
    flipped = m2[:-1] + ("0" if m2[-1] != "0" else "1")
    assert fold_mod.receive_pass(b2, flipped, KEY) is None
    stale = b2.replace(b'"misses":0', b'"misses":1')                # a field other than issuer
    assert stale != b2 and fold_mod.receive_pass(stale, m2, KEY) is None


def test_redelivery_is_idempotent_at_decision_level():
    objs = _corpus()
    once = Store(POL, KEY, encoder="hashing")
    twice = Store(POL, KEY, encoder="hashing")
    for o in objs:
        once.apply(*_wire(o))
        twice.apply(*_wire(o))
        twice.apply(*_wire(o))
    r1, r2 = _fold(once), _fold(twice)
    for r in r1.new_sig:                                      # Sig redelivered to the twin
        twice.apply(*_wire(r))
    r3 = _fold(twice, provider=None)
    assert r1.decisions == r2.decisions == r3.decisions
    assert r1.certificate == r2.certificate
    assert r3.counters["pass_time_misses"] == 0


# --------------------------------------------------------------------------- policy-borne inputs
def test_foreign_encoder_delta_is_dropped_and_never_a_node():
    foreign = _d("alpha bravo delta", 30, encoder="minilm", q=bytes(768))
    short = _d("alpha bravo echo", 40, q=bytes(8))                 # right encoder, wrong length
    s = _store(R, A, _ep())
    assert s.apply(*_wire(foreign)) is None                         # dropped at receipt
    _add(s, short)
    assert _k(foreign) not in s.nodes() and _k(short) not in s.nodes()
    res = _fold(s)
    assert [x.b for x in res.decisions] == [_k(R), _k(A)]
    with pytest.raises((KeyError, ValueError)):
        Store(POL, KEY, encoder="no-such-encoder")


def test_r_min_comes_from_the_policy():
    weak = _d("alpha bravo charlie", 20, retention=50)
    objs = [R, weak, _ep()]
    low, high = _store(*objs), Store({**POL, "r_min": 100}, KEY, encoder="hashing")
    _add(high, *objs)
    assert _k(weak) in low.nodes() and _k(weak) not in high.nodes()
    assert m_pol_of({**POL, "r_min": 100}) != MPOL                  # r_min is part of M_pol
    # an honest certificate verifies on any replica holding the sets under the same policy
    res = _fold(low)
    other = _store(*objs)
    for r in res.new_sig:
        _add(other, r)
    assert verify_pass(res.certificate, res.snapshot, other)


def test_bare_replica_without_epoch_folds_with_bootstrap_selector():
    ds = [_d("alpha bravo", 10), _d("alpha bravo", 20, version=1), _d("alpha bravo charlie", 30)]
    s = _store(*ds)
    res = readjudicate(s, "eu", m_nli=MNLI)
    assert res.certificate.selector == SEL and res.certificate.epoch_seq == -1
    assert _tested(res) and all(x.decision != "MERGE" for x in res.decisions)


def test_verify_pass_certifies_only_sig_complete_folds():
    s = _store(R, A, B, _ep())
    res = _fold(s, provider=None)
    assert res.certificate.misses > 0
    assert not verify_pass(res.certificate, res.snapshot, s)
    res = _fold(s)
    assert res.certificate.misses == 0 and verify_pass(res.certificate, res.snapshot, s)


def test_provider_needs_issuer_and_hw():
    s = _store(R, A, _ep())
    for kw in (dict(issuer="", hw="cpu"), dict(issuer="r1", hw="")):
        with pytest.raises(ValueError):
            readjudicate(s, "eu", m_nli=MNLI, provider=LEX, **kw)


def test_decisions_digest_is_canonical_json():
    import json
    s = _store(R, A, B, _ep())
    res = _fold(s)
    body = fold_mod.decisions_json(res.decisions)
    assert json.loads(body) and body == json.dumps(json.loads(body), sort_keys=True,
                                                     separators=(",", ":")).encode()
    assert res.certificate.decisions_digest == fold_mod.h16(body)
    assert len(res.certificate.roots) == 5


# --------------------------------------------------------------------------- gate and input edges
def test_mask_on_ineligible_a_star_gives_conflict():
    x = _d("the valve is open", 10, cell="eu.b")
    y = _d("the valve is not open", 20)
    s = _store(x, y, _ep())
    d = _dec(_fold(s), y)
    assert d.a_star == _k(x) and d.tested is None
    assert d.decision == "CONFLICT" and d.gates == ("mask",)
    assert d.links == (("CONTRADICTS", _k(y), _k(x), 1.0),)


def test_sig_under_a_foreign_m_nli_is_ignored():
    s = _store(R, A, _ep())
    base = [x.decision for x in _fold(s).decisions]
    _add(s, replace(_sig(_k(R), _k(A), (FIXED_ONE, 0, 0), (FIXED_ONE, 0, 0)), m_nli=b"\x05" * 16))
    assert [x.decision for x in _fold(s, provider=None).decisions] == base == ["INSERT", "MERGE"]


def test_distinguishes_counts_only_if_authorized_and_valid():
    for e, merged in ((_e("DISTINGUISHES", _k(R), _k(A)), False),
                      (_e("DISTINGUISHES", _k(R), _k(A), author_auth=2), True),          # unauthorized
                      (_e("DISTINGUISHES", _k(R), _k(A), target_cell="eu.b"), True)):    # invalid
        s = _store(R, A, e, _ep())
        assert (_dec(_fold(s), A).decision == "MERGE") == merged
        assert bool(fold_mod.tested_pairs(s, "eu", sigma_lo=LO, m=3)) == merged    # calibration too


def test_fingerprint_moves_with_vis():
    s = _store(R, A, B, _ep())
    fp = _fold(s).certificate.fingerprint
    _add(s, _e("OVERRULE", _k(B)))
    res = _fold(s)
    assert [x.decision for x in res.decisions] == [x.decision for x in _fold(s).decisions]
    assert res.certificate.fingerprint != fp


def test_collection_bound_is_inclusive_at_sigma_c():
    # b = e1, a = e1 (cos 1, a*), r = (3, 4) / 5 (cos 3/5 = 39321 / FIXED_ONE exactly)
    def vec(*v):
        return bytes(np.array(list(v) + [0] * (256 - len(v)), np.int8).tobytes())
    r = _d("lima mike", 10, q=vec(3, 4))
    a = _d("lima november", 20, q=vec(1))
    b = _d("lima oscar", 30, q=vec(1))
    prov = Counting({(canon(b.text), canon(r.text)): (FIXED_ONE, 0, 0)})
    for sigma_c, collected in ((39321, True), (39322, False)):
        s = _store(r, a, b, _ep(sel=(LO, sigma_c, 3)))
        d = _dec(_fold(s, prov), b)
        assert (("CONTRADICTS", _k(b), _k(r), 1.0) in d.links) == collected, sigma_c


# --------------------------------------------------------------------------- ported at T6b
def test_below_sigma_lo_is_insert_and_reads_no_signal():
    # ported from the retired classify/merge_on_receive suites (below-floor pair, spurious
    # contradiction): a node with no candidate at or above sigma_lo INSERTs, even when the
    # NLI would call the pair a contradiction, and no signal is read for it.
    a, b = _d("kilo lima", 10), _d("zulu yankee xray", 20)
    prov = Counting({(canon(b.text), canon(a.text)): (FIXED_ONE, 0, 0)})
    s = _store(a, b, _ep(sel=(FIXED_ONE, FIXED_ONE, 3)))
    d = _dec(_fold(s, prov), b)
    assert d.decision == "INSERT" and d.K == () and prov.calls == 0
    assert s.derived.links == []


def test_ineligible_a_star_does_not_shadow_an_admissible_a_e():
    # ported from the retired gate-aware fallback tests: a* in another cell is not eligible,
    # the next candidate in b's cell is a_e and is tested (MERGE above theta).
    x = _d("alpha bravo charlie", 10, cell="eu.b")
    y = _d("alpha bravo", 20)
    b = _d("alpha bravo charlie", 30, version=1)
    s = _store(x, y, b, _ep())
    d = _dec(_fold(s), b)
    assert d.a_star == _k(x) and d.tested == (_k(y), _k(b)) and d.decision == "MERGE"
