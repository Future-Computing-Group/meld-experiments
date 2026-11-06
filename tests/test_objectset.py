"""The per-scope replicated grow-only sets of design §2.1-2.3 (sync/objectset.py): MAC
verified before insertion, parsed with from_bytes, scope checked against the partition,
idempotent by object id; malformed or unverifiable objects are dropped, never raised."""
from __future__ import annotations

import random

import pytest

from meld.protocol.objects import attach_mac, canonical_bytes, object_id
from meld.sync.objectset import KINDS, ObjectSet
from test_protocol_objects import KEY, SAMPLES, _cal, _delta, _eff, _ep, _sig

SCOPE_OF = {"eu.a": "eu", "eu.b": "eu", "us.a": "us"}
KIND_OF = dict(zip(("D", "A", "Sig", "Cal", "Ep"), SAMPLES))


def _signed(obj, key=KEY):
    o = attach_mac(obj, key)
    return canonical_bytes(o), o.mac


def _add(s, obj, scope="eu", key=KEY):
    b, mac = _signed(obj, key)
    return s.add(b, mac, KEY, scope)


def test_add_accepts_each_kind_and_get_returns_the_parsed_object():
    s = ObjectSet(SCOPE_OF)
    for kind, obj in KIND_OF.items():
        got = _add(s, obj)
        oid = object_id(canonical_bytes(obj))
        assert got == attach_mac(obj, KEY)
        assert s.get(kind, oid) == got and oid in s
        assert s.ids(kind, "eu") == (oid,)
        assert list(s.iter(kind, "eu")) == [got]
        assert s.raw(oid) == (canonical_bytes(obj), got.mac, "eu")
    assert set(KINDS) == set(KIND_OF)


def test_add_is_idempotent_by_object_id():
    s = ObjectSet(SCOPE_OF)
    for obj in SAMPLES:
        first = _add(s, obj)
        assert _add(s, obj) == first
    assert sum(len(s.ids(k, "eu")) for k in KINDS) == len(SAMPLES)


@pytest.mark.parametrize("obj", SAMPLES, ids=list(KIND_OF))
def test_mac_failed_object_is_dropped_never_raised(obj):
    s = ObjectSet(SCOPE_OF)
    b, mac = _signed(obj)
    assert s.add(b, mac, b"another-group-key", "eu") is None      # wrong key
    assert s.add(b, "", KEY, "eu") is None                         # unsigned
    assert s.add(b, None, KEY, "eu") is None                       # mac of the wrong type
    assert s.add(b, mac[:-1] + ("0" if mac[-1] != "0" else "1"), KEY, "eu") is None
    assert all(s.ids(k, "eu") == () for k in KINDS)


@pytest.mark.parametrize("bad", [
    b"", b"not json", b"\xff\xfe", b"[]", b"{}", b'{"t":"Nope"}', b'{"t":"Delta"}', None, "str",
    b'{"t": "Delta"}'])
def test_malformed_bytes_are_dropped_never_raised(bad):
    s = ObjectSet(SCOPE_OF)
    assert s.add(bad, "00" * 32, KEY, "eu") is None


def test_non_canonical_encoding_of_a_signed_object_is_dropped():
    s = ObjectSet(SCOPE_OF)
    b, mac = _signed(_delta())
    assert s.add(b + b" ", mac, KEY, "eu") is None
    assert s.add(b.replace(b",", b", ", 1), mac, KEY, "eu") is None


def test_cross_scope_objects_are_refused():
    s = ObjectSet(SCOPE_OF)
    assert _add(s, _delta(), scope="us") is None                   # cell eu.a is in scope eu
    assert _add(s, _delta(cell="nowhere")) is None                 # cell outside the partition
    assert _add(s, _delta(cell="nowhere"), scope=None) is None
    assert _add(s, _eff(), scope="us") is None                     # envelope names another scope
    assert _add(s, _eff(scope="mars", target_cell="eu.a")) is None  # scope not in the partition
    assert _add(s, _cal(scope="us"), scope="eu") is None
    assert _add(s, _ep(scope="us"), scope="eu") is None
    assert _add(s, _sig(scope="us"), scope="eu") is None           # relabelled Sig envelope
    assert _add(s, _sig(), scope="us") is None
    assert _add(s, _sig(scope="mars"), scope="mars") is None
    assert _add(s, _sig(), scope=["eu"]) is None                   # unhashable envelope scope
    assert all(s.ids(k, sc) == () for k in KINDS for sc in ("eu", "us"))
    # the record's own scope stands when the envelope names none
    assert _add(s, _delta(cell="us.a"), scope=None) is not None
    assert _add(s, _eff(scope="us", target_cell="us.a"), scope=None) is not None
    assert _add(s, _sig(scope="us"), scope=None) is not None


def test_an_id_held_in_one_scope_is_never_acknowledged_under_another():
    s = ObjectSet(SCOPE_OF)
    for obj in SAMPLES:
        assert _add(s, obj, scope="eu") is not None
        b, mac = _signed(obj)
        assert s.add(b, mac, KEY, "us") is None
    assert all(s.ids(k, "us") == () for k in KINDS)


def test_delta_under_another_encoder_is_dropped_when_the_set_names_one():
    s = ObjectSet(SCOPE_OF, encoder="enc-v1")
    assert _add(s, _delta(encoder="enc-v2")) is None
    assert _add(s, _delta()) is not None                          # _delta's encoder is enc-v1
    assert ObjectSet(SCOPE_OF).add(*_signed(_delta(encoder="any")), KEY, "eu") is not None


def test_mac_is_checked_on_the_received_bytes_before_any_parse(monkeypatch):
    import meld.sync.objectset as om
    parsed = []
    real = om.json.loads
    monkeypatch.setattr(om.json, "loads", lambda b: (parsed.append(b), real(b))[1])
    s = ObjectSet(SCOPE_OF)
    b, mac = _signed(_delta())
    assert s.add(b, "ab" * 32, KEY, "eu") is None and parsed == []
    assert s.add(b"[" * 100_000, "ab" * 32, KEY, "eu") is None and parsed == []
    assert s.add(b, mac, KEY, "eu") is not None and parsed


def test_deeply_nested_signed_bytes_are_dropped_never_raised():
    import hashlib
    import hmac
    deep = b"[" * 100_000
    mac = hmac.new(KEY, deep, hashlib.sha256).hexdigest()
    assert ObjectSet(SCOPE_OF).add(deep, mac, KEY, "eu") is None


def test_unsubscribed_scope_is_refused():
    s = ObjectSet(SCOPE_OF, scopes=["us"])
    assert _add(s, _delta()) is None
    assert _add(s, _delta(cell="us.a"), scope="us") is not None


def test_ids_are_sorted_and_stable_across_insertion_orders():
    objs = [_delta(text=f"t{i}", cell=("eu.a", "eu.b")[i % 2]) for i in range(30)]
    objs += [_sig(issuer=f"n{i}") for i in range(10)]
    objs += [_cal(item=f"i{i}", n_prior=i) for i in range(5)]
    views = []
    for seed in range(5):
        order = objs[:]
        random.Random(seed).shuffle(order)
        s = ObjectSet(SCOPE_OF)
        for o in order:
            assert _add(s, o) is not None
        views.append((s.ids("D", "eu"), s.ids("Sig", "eu"), [x.text for x in s.iter("D", "eu")]))
    assert all(v == views[0] for v in views)
    assert list(views[0][0]) == sorted(views[0][0]) and len(views[0][0]) == 30
    assert ObjectSet(SCOPE_OF).ids("D", "eu") == ()
