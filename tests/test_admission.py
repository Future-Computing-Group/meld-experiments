"""The admission (latency) path, design §3.5, Lemma 2, Prop 3: on receipt of a verified delta the
replica runs the fold's step over its cache under the active epoch (or the policy's bootstrap
selector), publishes every Sig direction it evaluated as a MAC'd SignalRecord, records the
step's derived effects (representatives, links) and never writes an authored effect. Admission
decides what the fold decides over the same sets."""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from meld.brain.store import Store
from meld.protocol.admission import Admission
from meld.protocol.fold import readjudicate
from meld.protocol.objects import FIXED_ONE, INF16, canonical_bytes, m_pol_of, object_id, verify_mac

from test_step_fold import (
    A, B, KEY, LEX, MNLI, POL, R, _add, _cal, _corpus, _d, _e, _ep, _k, _sig, _store, _wire,
)


def _adm(s, provider=LEX, issuer="r1"):
    return Admission(s, m_nli=MNLI, provider=provider, issuer=issuer, hw="cpu")


def _receive(s, adm, obj):
    """The replica's receive gate: insert, then admit a newly held delta; (the decision, the
    Sig records admission inserted into the store, i.e. published on the outbox)."""
    stored = s.apply(*_wire(obj))
    assert stored is not None
    if type(stored).__name__ != "Delta":
        return None, []
    before = set(s.objects.ids("Sig", "eu"))
    dec = adm.admit(stored)
    return dec, [s.objects.get("Sig", i) for i in sorted(set(s.objects.ids("Sig", "eu")) - before)]


def _decisions(s):
    return readjudicate(s, "eu", m_nli=MNLI).decisions


def _deltas(objs):
    return sorted((o for o in objs if type(o).__name__ == "Delta"),
                  key=lambda d: (d.created_at, object_id(canonical_bytes(d))))


def test_lemma2_in_order_admission_equals_fold():
    objs = _corpus()
    s = _store(*[o for o in objs if type(o).__name__ != "Delta"])
    adm = _adm(s)
    got = {}
    for d in _deltas(objs):
        dec, _ = _receive(s, adm, d)
        if dec is not None:
            got[dec.b] = dec
    fold = {x.b: x for x in _decisions(s)}
    assert {k: v for k, v in got.items() if v.decision != "REJECT"} == fold
    assert adm.counters == {"not_last": 0, "refolds": 1}          # one setup, then the latency path
    assert {"MERGE", "RELATE", "CONFLICT", "INSERT"} <= {x.decision for x in fold.values()}


def test_admission_publishes_every_evaluated_sig_direction():
    s = _store(_ep())
    adm = _adm(s)
    published = []
    for d in (R, A, B):
        _, new = _receive(s, adm, d)
        published += new
    held = set(s.objects.ids("Sig", "eu"))
    assert published and all(verify_mac(r, KEY) for r in published)
    assert {object_id(canonical_bytes(r)) for r in published} <= held
    # both directions of a tested pair travel in one record (SigView batching)
    assert any(r.pay_xy is not None and r.pay_yx is not None for r in published)
    res = readjudicate(s, "eu", m_nli=MNLI)                        # no provider
    assert res.counters["pass_time_misses"] == 0
    assert res.decisions == adm.decisions("eu")


def test_admission_writes_no_authored_effect_and_records_derived_effects():
    s = _store(_ep())
    adm = _adm(s)
    for d in (R, A, B):
        _receive(s, adm, d)
    assert s.objects.ids("A", "eu") == ()
    after_admission = s.fingerprint_inputs()
    assert s.vis(_k(A)) == "DEPRECATED" and s.derived.rep[_k(A)] == _k(R)       # a merged into r
    readjudicate(s, "eu", m_nli=MNLI)
    assert s.fingerprint_inputs() == after_admission


def test_no_epoch_admission_uses_bootstrap_selector_and_makes_no_similarity_merge():
    s = _store()
    adm = _adm(s)
    decs = [_receive(s, adm, d)[0] for d in (R, A, B)]
    assert decs[1].tested == (_k(R), _k(A)) and decs[1].theta_q == INF16
    assert all(x.decision != "MERGE" for x in decs)
    assert decs == list(_decisions(s))
    bare = Store({k: v for k, v in POL.items() if k != "bootstrap_selector"}, KEY, encoder="hashing")
    with pytest.raises(ValueError):
        _receive(bare, _adm(bare), R)


def test_epoch_arriving_between_deltas_is_used_by_the_next_admission():
    s = _store()
    adm = _adm(s)
    _receive(s, adm, R)
    _add(s, _ep())                                                 # not a delta: no admission
    dec, _ = _receive(s, adm, A)
    assert dec.decision == "MERGE" and adm.counters["refolds"] == 2
    # the active epoch is the largest authorized seq: a later all-INF epoch disables merging
    _add(s, _ep(rows=((0, 9, 0, INF16, INF16),), seq=2))
    dec, _ = _receive(s, adm, _d("alpha bravo charlie", 40, version=1))
    assert dec.theta_q == INF16 and dec.decision != "MERGE"
    assert adm.decisions("eu") == list(_decisions(s))


def test_rejected_and_exact_copy_deltas():
    s = _store(_ep())
    adm = _adm(s)
    _receive(s, adm, R)
    _receive(s, adm, _d("echo foxtrot", 25))
    dec, new = _receive(s, adm, _d("echo golf", 20, retention=0))
    assert dec.decision == "REJECT" and new == [] and dec.links == ()
    copy = _d("alpha bravo", 30, unit=b"W" * 16)                   # R's key, later: one node
    assert _receive(s, adm, copy) == (None, [])
    assert adm.counters["not_last"] == 0
    earlier = _d("alpha bravo", 5, unit=b"V" * 16)                 # R's key, earlier: refold
    dec, _ = _receive(s, adm, earlier)
    assert adm.counters["not_last"] == 1 and adm.decisions("eu") == list(_decisions(s))


def _perturbed(a, b):
    """A second hardware class: the lexical payload with p_e lowered slightly."""
    c, e, n = LEX(a, b)
    return (c, max(0, e - 300), n)


def test_admission_decisions_equal_readjudicate_after_sig_exchange():
    rng = random.Random(5)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "not", "valve", "open"]
    for trial in range(25):
        ds = [_d(" ".join(rng.sample(words, rng.randint(2, 4))), 10 * i + 1, auth=rng.randint(0, 2),
                 version=i, cell=rng.choice(["eu.a", "eu.a", "eu.b"])) for i in range(10)]
        ep = _ep(rows=((0, 9, 0, rng.randint(20_000, 50_000), INF16),))
        last = _d("alpha bravo charlie", 1_000, version=99)
        reps = []
        for provider, issuer in ((LEX, "r1"), (_perturbed, "r2"), (LEX, "r3")):
            s = _store(ep)
            adm = _adm(s, provider, issuer)
            order = ds[:]
            rng.shuffle(order)
            for d in order:
                _receive(s, adm, d)
            reps.append((s, adm))
        for s, _ in reps:                                          # Sig exchange
            for t, _ in reps:
                for oid in t.objects.ids("Sig", "eu"):
                    s.apply(*t.objects.raw(oid)[:2])
        for s, adm in reps:
            _receive(s, adm, last)                                 # the latency path, after exchange
            assert adm.decisions("eu") == list(_decisions(s)), trial
        # replicas agree on every node the exchange covered; ``last`` was evaluated after it
        assert len({tuple(x for x in adm.decisions("eu") if x.b != _k(last))
                    for _, adm in reps}) == 1, trial


def test_out_of_order_or_late_sig_repaired_by_pass_after_sig_exchange():
    # Two replicas on two hardware classes (scripted providers: one reads every pair as
    # entailment, one as neutral) receive the same deltas, r2 in reverse canonical order (every
    # arrival but the first is out of order). Their admission states differ; after a Sig
    # exchange each replica's next admission and its pass agree with the other's.
    ds = [R, A, _d("alpha delta echo", 25), B, _d("alpha bravo echo", 35)]
    entail = lambda a, b: (0, FIXED_ONE, 0)
    neutral = lambda a, b: (0, 0, FIXED_ONE)
    reps = []
    for provider, issuer, order in ((neutral, "r1", ds), (entail, "r2", ds[::-1])):
        s = _store(_ep())
        adm = _adm(s, provider, issuer)
        for d in order:
            _receive(s, adm, d)
        reps.append((s, adm, provider, issuer))
    assert reps[0][1].counters["not_last"] == 0 and reps[1][1].counters["not_last"] == len(ds) - 1
    before = [s.fingerprint_inputs() for s, *_ in reps]
    assert before[0] != before[1]                                   # r2 merged, r1 did not
    assert any(x.decision == "MERGE" for x in reps[1][1].decisions("eu"))
    for s, *_ in reps:                                              # Sig exchange: late records
        for t, *_ in reps:
            for oid in t.objects.ids("Sig", "eu"):
                s.apply(*t.objects.raw(oid)[:2])
    certs = []
    for s, adm, provider, issuer in reps:
        refolds = adm.counters["refolds"]
        _receive(s, adm, _d("golf hotel", 99))                      # the next admission
        assert adm.counters["refolds"] == refolds + 1               # a late Sig changed a read
        res = readjudicate(s, "eu", m_nli=MNLI, provider=provider, issuer=issuer, hw="cpu")
        assert res.counters["pass_time_misses"] == 0 and res.decisions == adm.decisions("eu")
        certs.append(res.certificate)
    assert certs[0].fingerprint == certs[1].fingerprint
    assert certs[0].decisions_digest == certs[1].decisions_digest
    assert reps[0][0].fingerprint_inputs() == reps[1][0].fingerprint_inputs()


def test_an_earlier_exact_copy_of_the_last_node_refolds():
    # the key stays canonical-order last but its node now comes from another delta (the
    # canonical min): its attributes changed, so the cache must be rebuilt, not stepped.
    s = _store(_ep())
    adm = _adm(s)
    _receive(s, adm, R)
    x = _d("alpha bravo charlie", 40)
    _receive(s, adm, x)
    copy = _d("alpha bravo charlie", 30, auth=0, unit=b"V" * 16)  # x's key, earlier, after R
    dec, _ = _receive(s, adm, copy)
    assert adm.counters["not_last"] == 0 and adm.counters["refolds"] == 2
    assert dec.b == _k(x) and adm.decisions("eu") == list(_decisions(s))
    assert s.nodes()[_k(x)][1].unit_id == b"V" * 16


# --------------------------------------------------------------------------- refold triggers (§3.5)
def _primed():
    """R and A admitted under the epoch (A's step reads both directions of (R, A)): one refold."""
    s = _store(_ep())
    adm = _adm(s)
    for d in (R, A):
        _receive(s, adm, d)
    assert adm.counters["refolds"] == 1
    return s, adm


def _after(s, adm, *objs, nxt=None):
    """Add ``objs`` (not deltas: no admission), admit one canonical-last delta; the refolds it
    took. The cache must equal the fold over the sets either way."""
    before = adm.counters["refolds"]
    _add(s, *objs)
    _receive(s, adm, nxt or _d("alpha bravo delta", 90, version=5))
    assert adm.decisions("eu") == list(_decisions(s))
    return adm.counters["refolds"] - before


def _peer_sig(x, y, pxy, pyx):
    return _sig(_k(x), _k(y), pxy, pyx)


def test_authorized_ac_effect_refolds():
    for e in (_e("DISTINGUISHES", _k(R), _k(A)), _e("SUPERSEDE", _k(R), _k(A), author_auth=1)):
        s, adm = _primed()
        assert _after(s, adm, e) == 1


def test_unauthorized_ac_effect_and_status_effects_do_not_refold():
    for e in (_e("DISTINGUISHES", _k(R), _k(A), author_auth=2),     # above the policy's rank
              _e("SUPERSEDE", _k(R), _k(A), author_auth=2),
              _e("OVERRULE", _k(A)), _e("REVOKE", _k(R))):            # not in A_c
        s, adm = _primed()
        assert _after(s, adm, e) == 0


def test_sig_join_changing_a_read_direction_refolds():
    s, adm = _primed()
    assert _after(s, adm, _peer_sig(A, R, (FIXED_ONE, 0, 0), None)) == 1   # A -> R was read


def test_sig_for_an_unprocessed_pair_or_joining_to_the_same_payload_does_not_refold():
    s, adm = _primed()
    nxt = _d("alpha bravo delta", 90, version=5)
    # the pair (nxt, A) is not processed yet; nxt's step reads this payload from the cache, so
    # it evaluates (and publishes) no record for the pair
    assert _after(s, adm, _peer_sig(nxt, A, (FIXED_ONE, 0, 0), (FIXED_ONE, 0, 0)), nxt=nxt) == 0
    assert {(r.c_x, r.c_y) for r in s.objects.iter("Sig", "eu")} >= {tuple(sorted((_k(nxt), _k(R))))}
    assert [r.issuer for r in s.objects.iter("Sig", "eu")
            if {r.c_x, r.c_y} == {_k(nxt), _k(A)}] == ["r2"]
    s, adm = _primed()
    (oid,) = [i for i in s.objects.ids("Sig", "eu")]
    own = s.objects.get("Sig", oid)
    assert _after(s, adm, replace(own, issuer="r2", hw="gpu", mac="")) == 0  # same payloads


def test_cal_for_a_processed_pair_refolds_and_for_an_unprocessed_pair_does_not():
    s, adm = _primed()
    assert _after(s, adm, _cal(_k(R), _k(A))) == 1
    s, adm = _primed()
    nxt = _d("alpha bravo", 90, version=5)                         # paraphrases R: tested (R, nxt)
    assert _after(s, adm, _cal(_k(R), _k(nxt)), nxt=nxt) == 0
    assert "cal_veto" in adm.decisions("eu")[-1].gates


@pytest.mark.parametrize("over", [{"m_pol": b"\x07" * 16}, {"m_nli": b"\x07" * 16},
                                  {"author": "mallory"}])
def test_epoch_under_another_m_pol_m_nli_or_author_is_ignored_by_admission(over):
    s, adm = _primed()
    off = _ep(rows=((0, 9, 0, INF16, INF16),), seq=2, **over)       # would disable merging
    assert _after(s, adm, off) == 0
    assert adm.decisions("eu")[1].decision == "MERGE"
    assert _after(s, adm, _ep(rows=((0, 9, 0, INF16, INF16),), seq=3),
                  nxt=_d("alpha bravo delta", 95, version=6)) == 1   # the authorized one does
    assert all(x.decision != "MERGE" for x in adm.decisions("eu"))


def test_node_arriving_for_the_c1_of_an_ac_effect_in_another_cell_refolds():
    # DISTINGUISHES(x, R) counted while x is not held; x then arrives canonical-order last in
    # another cell than the effect's target cell: the effect is invalid from then on, so x's
    # selection must not exclude R (the cached constraint set would)
    x = next(d for d in (_d("alpha bravo", 90, version=v) for v in range(50)) if _k(d) < _k(R))
    for cell, refolds in (("eu.b", 1), ("eu.a", 0)):
        s, adm = _primed()
        _add(s, _e("DISTINGUISHES", _k(x), _k(R), target_cell=cell))
        _receive(s, adm, _d("alpha delta", 80))                     # counts the effect: refold
        assert _after(s, adm, nxt=x) == refolds


def test_a_delta_held_but_not_admitted_refolds_at_the_next_admission():
    s, adm = _primed()
    assert _after(s, adm, _d("alpha charlie", 85)) == 1             # inserted, never admitted


def test_fast_path_writes_only_the_changed_group(monkeypatch):
    # bookkeeping per admission stays O(group), not O(N): no full derived rewrite, one rep call
    from meld.protocol import fold as fold_mod
    s, adm = _primed()
    for i in range(30):
        _receive(s, adm, _d(f"golf hotel {i}", 100 + i))
    reps = []
    real_rep = s.rep
    monkeypatch.setattr(s, "rep", lambda g: reps.append(len(g)) or real_rep(g))
    monkeypatch.setattr(fold_mod, "write_derived", lambda *a: pytest.fail("full derived rewrite"))
    dec, _ = _receive(s, adm, _d("alpha bravo", 200, version=9))       # merges into R's group
    assert dec.decision == "MERGE" and adm.counters["refolds"] == 1
    assert reps == [3]                          # the merged group's new representative only
    monkeypatch.undo()
    assert adm.decisions("eu") == list(_decisions(s))
    after = s.fingerprint_inputs()
    readjudicate(s, "eu", m_nli=MNLI)
    assert s.fingerprint_inputs() == after


# --------------------------------------------------------------------------- running example
# The paper's running example (main text §4.3, fig-running-example; texts and cells from
# eval/order_shuffle.py, the harness's workload): the receiver holds c1 ACTIVE; message 1 is an EU
# paraphrase of c1, message 2 the same sentence scoped US, message 3 an intra-EU contradiction
# c2, message 4 the EU authority's overrule of c2. The four messages arrive in any order.
from meld.eval.order_shuffle import C1_CANONICAL, C1_PARAPHRASE, C2_CONTRADICTION, EU, US  # noqa: E402

RX_POL = {**POL, "scopes": {"jur": [EU, US]},
          "auth_policy": {**POL["auth_policy"], "calibration_authority": {"jur": "cal-auth"}}}


def _rx_delta(text, cell, t):
    return replace(_d(text, t), unit_id=_rx_key(text, cell), claim_key=_rx_key(text, cell), cell=cell)


def _rx_key(text, cell):
    from meld.proto import claim_key
    return claim_key(text, cell)


def _rx_nli(a, b):
    """The lexical rung, with the example's contradiction scripted (lexical overlap cannot see
    'retained seven years' against 'deleted after two years')."""
    from meld.proto import canon
    if canon(C2_CONTRADICTION) in (a, b):
        return (FIXED_ONE, 0, 0)
    return LEX(a, b)


def _running_example():
    from meld.protocol.objects import AuthoredEffect
    ep = _ep(scope="jur", m_pol=m_pol_of(RX_POL))
    seed = [ep, _rx_delta(C1_CANONICAL, EU, 10)]
    msgs = [_rx_delta(C1_PARAPHRASE, EU, 20), _rx_delta(C1_PARAPHRASE, US, 30),
            _rx_delta(C2_CONTRADICTION, EU, 40),
            AuthoredEffect(scope="jur", kind="OVERRULE", c1=_rx_key(C2_CONTRADICTION, EU), c2=b"",
                           target_cell=EU, author="C", author_cell=EU, author_auth=0, created_at=0)]
    return seed, msgs


def test_running_example_fingerprint_count_is_asserted():
    # The submitted text reported 91/200 orderings agreeing (a stale figure from the retired merge
    # path); under admission plus a pass every one of 200 shuffled orders gives one fingerprint.
    seed, msgs = _running_example()
    rng = random.Random(2026)
    fingerprints, decisions = [], set()
    for _ in range(200):
        s = Store(RX_POL, KEY, encoder="hashing")
        adm = Admission(s, m_nli=MNLI, provider=_rx_nli, issuer="B", hw="cpu")
        order = msgs[:]
        rng.shuffle(order)
        for o in seed + order:
            stored = s.apply(*_wire(o))
            if type(stored).__name__ == "Delta":
                adm.admit(stored)
        admitted = s.fingerprint_inputs()
        res = readjudicate(s, "jur", m_nli=MNLI)
        assert res.counters["pass_time_misses"] == 0 and s.fingerprint_inputs() == admitted
        fingerprints.append(res.certificate.fingerprint)
        decisions.add(tuple(x.decision for x in res.decisions))
    assert fingerprints.count(fingerprints[0]) == 200 and len(set(fingerprints)) == 1
    # the figure's outcomes: c1 inserted, the EU paraphrase merged, the US copy related across the
    # boundary, the contradiction kept live as a conflict and then overruled
    assert decisions == {("INSERT", "MERGE", "RELATE", "CONFLICT")}
    assert s.vis(_rx_key(C2_CONTRADICTION, EU)) == "OVERRULED"
    assert s.vis(_rx_key(C1_PARAPHRASE, EU)) == "DEPRECATED"
    assert s.vis(_rx_key(C1_PARAPHRASE, US)) == "ACTIVE"


def test_retention_below_a_policy_r_min_is_rejected_before_any_candidate_or_signal():
    # R1 with r_min above one unit: a delta with positive retention below it is REJECTed on
    # receipt, reads no signal, is never a node, and leaves the cache on the latency path
    pol = {**POL, "r_min": 100}
    s = Store(pol, KEY, encoder="hashing")
    _add(s, _ep(m_pol=m_pol_of(pol)))
    calls = []
    adm = Admission(s, m_nli=MNLI, provider=lambda a, b: calls.append((a, b)) or LEX(a, b),
                    issuer="r1", hw="cpu")
    _receive(s, adm, R)
    n_calls = len(calls)
    dec, new = _receive(s, adm, _d("alpha bravo", 20, version=1, retention=99))   # paraphrase of R
    assert dec.decision == "REJECT" and dec.K == () and new == [] and len(calls) == n_calls
    assert _d("alpha bravo", 20, version=1).claim_key not in s.nodes()
    dec, _ = _receive(s, adm, _d("alpha bravo", 30, version=2, retention=100))      # at r_min: admitted
    assert dec.decision != "REJECT" and adm.counters["refolds"] == 1
    assert adm.decisions("eu") == list(readjudicate(s, "eu", m_nli=MNLI).decisions)
