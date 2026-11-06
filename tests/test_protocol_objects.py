"""Replicated objects: canonical bytes, content-addressed ids, MACs, scope partition,
the Sig join, claim keys and the memory facade fields (protocol design §2.0-2.3, §3.2)."""
from __future__ import annotations

import base64
import json
import random
from dataclasses import replace

import pytest

from meld.agent.memory import _cid
from meld.proto import CANON_VERSION, Claim, MemoryUnit, canon, claim_key
from meld.protocol.objects import (
    ALPHA_ONE, FIXED_ONE, INF16, AuthoredEffect, CalRecord, Delta, EpochRecord, SignalRecord,
    attach_mac, cal_pair, canonical_bytes, contradicts, effect_valid, from_bytes, h16,
    join_signal_payloads, m_nli, m_pol, object_id, payload, scope_partition, score_s,
    sig_key, unit_to_delta, verify_mac,
)
from meld.trust.signing import sign_delta

KEY = b"t1-test-group-key"
K1, K2 = claim_key("alpha", "eu.a"), claim_key("beta", "eu.a")
CX, CY = min(K1, K2), max(K1, K2)


def _delta(**over):
    kw = dict(unit_id=b"\x01" * 16, claim_key=K1, text="alpha", cell="eu.a", version=0,
              auth=2, status="ACTIVE", retention=FIXED_ONE, created_at=1_700_000_000_000_000,
              origin="A", encoder="enc-v1", q=bytes(range(16)))
    kw.update(over)
    if "claim_key" not in over and {"text", "cell", "version"} & set(over):
        kw["claim_key"] = claim_key(kw["text"], kw["cell"], kw["version"])
    return Delta(**kw)


def _eff(**over):
    kw = dict(scope="eu", kind="DISTINGUISHES", c1=CX, c2=CY, target_cell="eu.a",
              author="A", author_cell="eu.a", author_auth=1, created_at=0)
    kw.update(over)
    return AuthoredEffect(**kw)


def _sig(**over):
    kw = dict(scope="eu", c_x=CX, c_y=CY, m_nli=b"\x07" * 16, pay_xy=(60000, 100, 5435),
              pay_yx=None, issuer="A", hw="x86")
    kw.update(over)
    return SignalRecord(**kw)


def _cal(**over):
    kw = dict(scope="eu", c_x=CX, c_y=CY, bin=0, s_q=40000, hw="x86", obs=1,
              label="NON_MERGEABLE", subtype="OTHER", source="SEED", item="hotpot:17",
              author="cal", created_at=9, sigma_lo_q=30000, m=3, m_pol=b"\x08" * 16,
              n_prior=17)
    kw.update(over)
    return CalRecord(**kw)


def _ep(**over):
    kw = dict(scope="eu", seq=3, m_nli=b"\x07" * 16, m_pol=b"\x08" * 16, policy="SEED",
              alpha=20_000, alpha_c=20_000, mondrian_c=False, cal_root=b"\x09" * 16,
              cal_hw="x86", bins=(256, "doubling"),
              table=((0, 60, 4, 51000, INF16), (1, 10, 0, INF16, INF16)),
              sigma_lo_q=30000, sigma_c_q=45000, m=3, author="cal", created_at=11)
    kw.update(over)
    return EpochRecord(**kw)


SAMPLES = [_delta(), _eff(kind="SUPERSEDE", c1=K1, c2=K2), _sig(), _cal(), _ep()]
BUILDERS = [_delta, _eff, _sig, _cal, _ep]


def _rewrite(body: bytes, **set_fields) -> bytes:
    d = json.loads(body)
    d.update(set_fields)
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


# --------------------------------------------------------------------------- #
# claim keys, unit ids and the facade
# --------------------------------------------------------------------------- #
def test_claim_key_includes_version_and_cell():
    k = claim_key("Paris is in France", "eu.a")
    assert len(k) == 16                                        # H = SHA-256 truncated to 16 B
    assert k == claim_key("Paris is in France", "eu.a", 0)     # version defaults to 0
    assert k != claim_key("Paris is in France", "eu.a", 1)     # version enters the key
    assert k != claim_key("Paris is in France", "eu.b")        # cell enters the key
    # the cell is length-prefixed: a cell cannot absorb the version bytes
    assert claim_key("t", "c", 1) != claim_key("t", "c" + "\x00" * 7 + "\x01", 0)
    assert claim_key("a\x1fb", "c") != claim_key("a", "b\x1fc")
    assert claim_key("A", "\0" * 4) != claim_key("A\0\0\0\x04", "")
    assert claim_key("ab", "c") != claim_key("a", "bc")
    assert canon("x y") == "x y" and isinstance(CANON_VERSION, str) and CANON_VERSION


def test_unit_id_is_h16_over_length_prefixed_text_and_cell():
    assert len(_cid("t", "c")) == 16
    assert _cid("t", "c") != _cid("t", "d") and _cid("ab", "c") != _cid("a", "bc")
    assert _cid("t", "c") != claim_key("t", "c")               # tag separates ids from keys


def test_facade_carries_version_created_at_encoder_q_and_required_retention():
    c = Claim(claim_key=b"k", text="t", version=2)
    assert Claim.from_dict(c.to_dict()) == c and c.version == 2
    assert not hasattr(c, "valid_from") and not hasattr(c, "valid_until")
    u = MemoryUnit(id=b"u", retention=0.5, created_at=1_700_000_000_000_001,
                   encoder="enc-v1", q=b"\x80\x7f\x00", claims=[c])
    assert MemoryUnit.from_dict(u.to_dict()) == u
    assert isinstance(u.created_at, int)
    with pytest.raises(TypeError):
        MemoryUnit(id=b"u")                                    # retention has no default
    d = u.to_dict()
    del d["retention"]
    with pytest.raises(KeyError):
        MemoryUnit.from_dict(d)                                # nor on the wire


# --------------------------------------------------------------------------- #
# ids, canonical bytes, MACs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("obj", SAMPLES, ids=lambda o: type(o).__name__)
def test_objects_ids_are_content_addressed_over_received_bytes_and_mac_excluded(obj):
    signed = attach_mac(obj, KEY)
    body = canonical_bytes(signed)
    assert signed.mac and body == canonical_bytes(obj)            # MAC excluded from bytes
    assert b'"mac"' not in body
    assert json.loads(body)["t"] == type(obj).__name__            # type tag in the body
    assert object_id(body) == h16(body) and len(object_id(body)) == 16
    assert verify_mac(signed, KEY) and not verify_mac(signed, b"other-key")
    assert not verify_mac(obj, KEY)                               # unsigned never verifies
    back = from_bytes(type(obj), body, signed.mac)
    assert back == signed and verify_mac(back, KEY)
    # a forged body under the old MAC: refused at parse or fails the MAC
    d = json.loads(body)
    k = next(f for f in sorted(d) if f != "t")
    v = d[k]
    d[k] = (v + 1 if type(v) is int else "x" + v if type(v) is str else not v)
    tampered = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    assert object_id(tampered) != object_id(body)
    try:
        t_obj = from_bytes(type(obj), tampered, signed.mac)
    except ValueError:
        pass
    else:
        assert not verify_mac(t_obj, KEY)
    # a re-serialisation of the same content has another id and is refused
    pretty = json.dumps(json.loads(body), indent=1).encode()
    assert object_id(pretty) != object_id(body)
    with pytest.raises(ValueError):
        from_bytes(type(obj), pretty, signed.mac)


def test_bytes_fields_are_standard_padded_base64():
    d = _delta(q=b"\xff\xfe\x00")
    body = json.loads(canonical_bytes(d))
    assert body["q"] == base64.b64encode(b"\xff\xfe\x00").decode() == "//4A"
    assert body["claim_key"] == base64.b64encode(K1).decode()


def test_type_tag_mismatch_across_classes_is_refused():
    for obj in SAMPLES:
        body = canonical_bytes(obj)
        for other in {type(o) for o in SAMPLES} - {type(obj)}:
            with pytest.raises(ValueError):
                from_bytes(other, body)
        with pytest.raises(ValueError):                           # tag rewritten to the class
            from_bytes(type(obj), _rewrite(body, t="Other"))
        untagged = {k: v for k, v in json.loads(body).items() if k != "t"}
        with pytest.raises(ValueError):
            from_bytes(type(obj), json.dumps(untagged, sort_keys=True,
                                             separators=(",", ":")).encode())


def test_legacy_untagged_mac_is_not_accepted_and_empty_key_is_refused():
    d = _delta()
    untagged = {k: v for k, v in json.loads(canonical_bytes(d)).items() if k != "t"}
    legacy = sign_delta(untagged, KEY)["sig"]
    assert not verify_mac(replace(d, mac=legacy), KEY)
    with pytest.raises(ValueError):
        attach_mac(d, b"")
    with pytest.raises(ValueError):
        verify_mac(attach_mac(d, KEY), b"")


def test_canonical_bytes_deterministic_across_dict_insertion_orders():
    kw = dict(unit_id=b"\x01" * 16, claim_key=K1, text="alpha", cell="eu.a", version=0,
              auth=2, status="ACTIVE", retention=FIXED_ONE, created_at=1, origin="A",
              encoder="enc-v1", q=b"\x00\x01")
    rng = random.Random(0)
    seen = set()
    for _ in range(20):
        items = list(kw.items())
        rng.shuffle(items)
        seen.add(canonical_bytes(Delta(**dict(items))))
    assert len(seen) == 1


@pytest.mark.parametrize("field,bad", [
    ("created_at", 1.0), ("retention", 0.5), ("auth", True), ("version", False),
    ("text", 5), ("origin", b"A"), ("q", "AAE="), ("unit_id", bytearray(b"\x01" * 16)),
])
def test_exact_field_types_at_construction(field, bad):
    kw = {field: bad}
    if field == "text":
        kw["claim_key"] = K1
    with pytest.raises(ValueError):
        _delta(**kw)


def test_mondrian_c_is_exactly_bool_and_one_object_has_one_encoding():
    with pytest.raises(ValueError):
        _ep(mondrian_c=0)
    with pytest.raises(ValueError):
        _ep(mondrian_c="on")
    body = canonical_bytes(_ep())
    for alt in (body.replace(b'"mondrian_c":false', b'"mondrian_c":0'),
                body.replace(b'"seq":3', b'"seq":3.0'),
                body.replace(b'"seq":3', b'"seq":true')):
        assert alt != body
        with pytest.raises(ValueError):
            from_bytes(EpochRecord, alt)
    db = canonical_bytes(_delta())
    for alt in (db.replace(b'"auth":2', b'"auth":true'),
                db.replace(b'"text":"alpha"', b'"text":5'),
                db.replace(b'"created_at":1700000000000000', b'"created_at":NaN')):
        with pytest.raises(ValueError):
            from_bytes(Delta, alt)


def test_payload_lists_are_normalised_to_tuples():
    s = _sig(pay_xy=[1, 2, 3], pay_yx=[4, 5, 6])
    assert s.pay_xy == (1, 2, 3) and s == _sig(pay_xy=(1, 2, 3), pay_yx=(4, 5, 6))
    e = _ep(table=[[0, 60, 4, 51000, INF16]], bins=[256, "doubling"])
    assert e.table == ((0, 60, 4, 51000, INF16),) and e.bins == (256, "doubling")


def test_from_bytes_normalises_every_parse_failure_to_value_error():
    db = canonical_bytes(_delta())
    bad = [
        b"not json", b"[]", b"{}", db.replace(b'"auth":2,', b''),              # json, attr, key
        _rewrite(db, q=[1]), _rewrite(db, q="!!!"), _rewrite(db, q=5),          # type, b64
        _rewrite(db, aaa=1), b" " + db, db[:-1],
    ]
    sb = canonical_bytes(_sig())
    bad += [_rewrite(sb, pay_xy=[]), _rewrite(sb, pay_xy=[True, 2, 3]),
            _rewrite(sb, pay_xy=5)]
    for b in bad[:-3]:
        with pytest.raises(ValueError):
            from_bytes(Delta, b)
    for b in bad[-3:]:
        with pytest.raises(ValueError):
            from_bytes(SignalRecord, b)


# --------------------------------------------------------------------------- #
# field validation and ranges
# --------------------------------------------------------------------------- #
def test_delta_binds_its_claim_key():
    with pytest.raises(ValueError):
        _delta(text="evil", claim_key=K1)
    with pytest.raises(ValueError):
        _delta(cell="eu.b", claim_key=K1)
    with pytest.raises(ValueError):
        _delta(version=1, claim_key=K1)
    assert _delta(version=1).claim_key == claim_key("alpha", "eu.a", 1)
    forged = _rewrite(canonical_bytes(_delta()), text="evil")
    with pytest.raises(ValueError):
        from_bytes(Delta, forged)


@pytest.mark.parametrize("build,over", [
    (_delta, dict(retention=-1)), (_delta, dict(retention=FIXED_ONE + 1)),
    (_delta, dict(created_at=-1)), (_delta, dict(version=-1, claim_key=K1)),
    (_eff, dict(kind="MERGE")), (_eff, dict(kind="MERGE", c2=b"")),
    (_eff, dict(kind="REVOKE")), (_eff, dict(kind="OVERRULE")),                # carry c2
    (_eff, dict(kind="DISTINGUISHES", c2=b"")), (_eff, dict(kind="SUPERSEDE", c2=b"")),
    (_eff, dict(c1=CY, c2=CX)),                                                # reversed
    (_eff, dict(kind="SUPERSEDE", c1=K1, c2=K1)), (_eff, dict(kind="REVOKE", c1=b"", c2=b"")),
    (_sig, dict(c_y=CX)), (_sig, dict(c_x=CY, c_y=CX)),
    (_sig, dict(pay_xy=(-1, 0, 0))), (_sig, dict(pay_xy=(True, 0, 0))),
    (_sig, dict(pay_xy=(FIXED_ONE + 1, 0, 0))), (_sig, dict(pay_xy=(1, 2))),
    (_sig, dict(pay_yx=(1, 2, 3, 4))), (_sig, dict(pay_xy="abc")), (_sig, dict(c_x=b"a")),
    (_cal, dict(label="MERGEABLE")), (_cal, dict(subtype="X")),
    (_cal, dict(source="CORRECTION")), (_cal, dict(s_q=-1)), (_cal, dict(s_q=FIXED_ONE + 1)),
    (_cal, dict(bin=-1)), (_cal, dict(obs=0)), (_cal, dict(c_x=CY, c_y=CX)),
    (_cal, dict(c_y=CX)),
    (_cal, dict(sigma_lo_q=-1)), (_cal, dict(sigma_lo_q=FIXED_ONE + 1)),
    (_cal, dict(sigma_lo_q=True)), (_cal, dict(m=0)), (_cal, dict(m=1.0)),
    (_cal, dict(m_pol=b"\x08" * 15)), (_cal, dict(m_pol="AAAA")), (_cal, dict(n_prior=-1)),
    (_cal, dict(n_prior=False)), (_sig, dict(scope=1)), (_sig, dict(scope=b"eu")),
    (_ep, dict(policy="ALL")), (_ep, dict(table=())),
    (_ep, dict(table=((1, 1, 0, 1, 1), (0, 1, 0, 1, 1)))),
    (_ep, dict(table=((0, 1, 0, 1, 1), (0, 1, 0, 1, 1)))),
    (_ep, dict(table=((0, 1, 0, 1),))), (_ep, dict(table=((0, 1, 0, 1, 1, 1),))),
    (_ep, dict(table=((),))), (_ep, dict(table=((False, 1, 0, 1, 1),))),
    (_ep, dict(table=((0, 1, 2, 1, 1),))), (_ep, dict(table=((-1, 1, 0, 1, 1),))),
    (_ep, dict(table=((0, 1, 0, INF16 + 1, 1),))), (_ep, dict(table=((0, 1, 0, 1, -1),))),
    (_ep, dict(table=((0, 1, 0, FIXED_ONE + 2, 1),))),
    (_ep, dict(alpha=0)), (_ep, dict(alpha=ALPHA_ONE)), (_ep, dict(alpha_c=0)),
    (_ep, dict(alpha_c=ALPHA_ONE)),
    (_ep, dict(bins=(0, "doubling"))), (_ep, dict(bins=(256, "tripling"))),
    (_ep, dict(bins=(256,))),
])
def test_objects_reject_malformed_fields(build, over):
    with pytest.raises(ValueError):
        build(**over)


def test_valid_edges_are_accepted():
    _delta(retention=0), _delta(created_at=0)
    _sig(pay_xy=(0, 0, FIXED_ONE), pay_yx=(FIXED_ONE, 0, 0))
    _cal(s_q=0, obs=2), _cal(s_q=FIXED_ONE)
    _ep(table=((0, 0, 0, INF16, INF16),), alpha=1, alpha_c=ALPHA_ONE - 1, mondrian_c=True)
    _ep(table=((0, 5, 5, 0, FIXED_ONE),))


# --------------------------------------------------------------------------- #
# scope partition and cross-scope effects
# --------------------------------------------------------------------------- #
def test_scope_partition_assigns_every_cell_once_and_cross_scope_effect_is_invalid():
    pol = {"scopes": {"eu": ["eu.a", "eu.b"], "us": ["us.a"]}}
    scope_of = scope_partition(pol)
    assert scope_of == {"eu.a": "eu", "eu.b": "eu", "us.a": "us"}
    for bad in ({"scopes": {"eu": ["eu.a"], "us": ["eu.a"]}}, {"scopes": {"eu": ["eu.a", "eu.a"]}},
                {"scopes": {"eu": "eu.a"}}, {"scopes": {"eu": ("eu.a",)}}):
        with pytest.raises(ValueError):
            scope_partition(bad)

    k_eu_b, k_us = claim_key("gamma", "eu.b"), claim_key("delta", "us.a")
    cell_of = {K1: "eu.a", K2: "eu.a", k_eu_b: "eu.b", k_us: "us.a"}

    def dist(a, b, **over):
        lo, hi = min(a, b), max(a, b)
        return _eff(c1=lo, c2=hi, target_cell=cell_of[lo], **over)

    assert effect_valid(dist(K1, k_eu_b), scope_of, cell_of)          # cross-cell, one scope
    # a held c2 never changes the verdict, in any scope
    far = dist(K1, k_us)
    assert effect_valid(far, scope_of, cell_of) == effect_valid(far, scope_of,
                                                              {far.c1: cell_of[far.c1]})
    assert not effect_valid(dist(K1, k_eu_b, scope="us"), scope_of, cell_of)
    sup = _eff(kind="SUPERSEDE", c1=K1, c2=K2)
    assert effect_valid(sup, scope_of, cell_of)
    assert not effect_valid(replace(sup, target_cell="eu.b"), scope_of, cell_of)  # held c1
    assert not effect_valid(_eff(kind="OVERRULE", c1=K1, c2=b"", target_cell="nowhere"),
                            scope_of, {})                              # unknown cell
    assert effect_valid(_eff(kind="SUPERSEDE", c1=K1, c2=k_us), scope_of,
                        {K1: "eu.a", k_us: "nowhere"})                 # c2 is never read
    # keys not held yet: validity is decided by the carried fields alone
    assert effect_valid(_eff(kind="REVOKE", c1=K1, c2=b""), scope_of, {})
    assert effect_valid(_eff(kind="SUPERSEDE", c1=K1, c2=k_us), scope_of, {K1: "eu.a"})
    assert effect_valid(_eff(kind="SUPERSEDE", c1=K1, c2=k_us), scope_of, {})


# --------------------------------------------------------------------------- #
# Sig join, helpers
# --------------------------------------------------------------------------- #
def _rand_payload(rng):
    if rng.random() < 0.15:
        return None
    return tuple(rng.randint(0, FIXED_ONE) for _ in range(3))


def test_sig_join_is_semilattice_and_contradicts_is_monotone():
    rng = random.Random(20261009)
    j = join_signal_payloads
    for _ in range(3000):
        a, b, c = _rand_payload(rng), _rand_payload(rng), _rand_payload(rng)
        assert j(a, a) == a                                         # idempotent
        assert j(a, b) == j(b, a)                                   # commutative
        assert j(j(a, b), c) == j(a, j(b, c))                       # associative
        assert j(a, None) == a and j(None, a) == a                  # bottom
        if contradicts(a):
            assert contradicts(j(a, b))                             # monotone
    assert j(None, None) is None
    assert j((10, 20, 30), (15, 5, 40)) == (15, 5, 30)
    assert contradicts((3, 2, 2)) and not contradicts((3, 3, 2)) and not contradicts(None)
    assert not contradicts((5, 1, 9))                               # p_n > p_c > p_e
    assert not contradicts((0, 0, 0))


def test_sig_helpers():
    s = _sig(pay_xy=(1, 200, 3), pay_yx=(4, 100, 6))
    assert sig_key(s) == (CX, CY, b"\x07" * 16)
    assert payload(s, CX) == (1, 200, 3) and payload(s, CY) == (4, 100, 6)
    with pytest.raises(ValueError):
        payload(s, b"\x00" * 16)
    assert score_s(s.pay_xy, s.pay_yx) == 100
    assert score_s(s.pay_xy, None) is None and score_s(None, s.pay_yx) is None


def test_m_nli_and_m_pol_hash_canonical_json():
    a = m_nli("enc@w1", "nli@w2", "int8-sym", CANON_VERSION)
    assert len(a) == 16 and a == m_nli("enc@w1", "nli@w2", "int8-sym", CANON_VERSION)
    assert a != m_nli("enc@w1", "nli@w2", "int8-sym", "canon-v2")
    p = m_pol({"OBSERVATION": 2, "POLICY": 0}, {"eu.a": "eu"}, {"eu": "cal"}, 1)
    assert p == m_pol({"POLICY": 0, "OBSERVATION": 2}, {"eu.a": "eu"}, {"eu": "cal"}, 1)
    assert p != m_pol({"OBSERVATION": 2, "POLICY": 0}, {"eu.a": "eu"}, {"eu": "cal"}, 2)


def test_unit_to_delta_round_trips_a_facade_unit():
    from meld.agent.memory import _cid
    from meld.proto import Claim, MemoryUnit

    u = MemoryUnit(id=_cid("Paris is in France", "eu.a"), cell="eu.a", retention=0.5,
                   created_at=7, encoder="enc-v1", q=b"\x01\x02", owner_did=b"\xab",
                   claims=[Claim(claim_key=claim_key("Paris is in France", "eu.a"),
                                 text="Paris is in France")])
    d = unit_to_delta(u, u.claims[0], {"OBSERVATION": 3})
    assert d.claim_key == claim_key("Paris is in France", "eu.a") and d.unit_id == u.id
    assert d.retention == round(0.5 * FIXED_ONE) and d.auth == 3
    assert (d.created_at, d.encoder, d.q, d.cell, d.version, d.status) == (
        7, "enc-v1", b"\x01\x02", "eu.a", 0, "ACTIVE")
    assert from_bytes(Delta, canonical_bytes(d)) == d
    with pytest.raises(KeyError):
        unit_to_delta(u, u.claims[0], {})                           # authority not ranked


def test_signal_scope_and_cal_selector_fields_enter_bytes_and_id():
    a, b = _sig(), _sig(scope="us")
    assert object_id(canonical_bytes(a)) != object_id(canonical_bytes(b))
    assert json.loads(canonical_bytes(a))["scope"] == "eu"
    c = attach_mac(_cal(), KEY)
    body = json.loads(canonical_bytes(c))
    assert (body["sigma_lo_q"], body["m"], body["n_prior"]) == (30000, 3, 17)
    assert base64.b64decode(body["m_pol"]) == b"\x08" * 16
    assert from_bytes(CalRecord, canonical_bytes(c), c.mac) == c and verify_mac(c, KEY)
    for over in (dict(sigma_lo_q=1), dict(m=4), dict(m_pol=b"\x01" * 16), dict(n_prior=18)):
        assert object_id(canonical_bytes(_cal(**over))) != object_id(canonical_bytes(_cal()))
    # bin is carried as given; consistency with n_prior is the epoch builder's check
    assert _cal(bin=5, n_prior=0).bin == 5
    _cal(sigma_lo_q=0, m=1, n_prior=0), _cal(sigma_lo_q=FIXED_ONE)
    assert cal_pair(c) == (CX, CY)
