"""The wire of design §2.2-2.3: PUBLISH ships an object's stored signed bytes in one generic
envelope for all five types; DIGEST carries a Merkle root per scope and kind; PULL returns
original bytes; every object is MAC-verified before it enters a set; /status is gone.
Class-level tests over every object kind, run through BrainService.handle."""
from __future__ import annotations

import json
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from meld.protocol.objects import attach_mac, canonical_bytes, object_id
from meld.sync.merkle import merkle_root
from meld.sync.objectset import KINDS
from meld.sync.wire_delta import pack, unpack
from test_protocol_objects import KEY, _cal, _delta, _eff, _ep, _sig

POL = {"rank_map": {"PRECEDENT": 0, "OBSERVATION": 1}, "auth_policy": {}, "step_version": 0,
       "scopes": {"eu": ["eu.a", "eu.b"], "us": ["us.a"]}, "r_min": 1,
       "bootstrap_selector": [32768, 32768, 3]}




def _svc(nid, admitted=None, **kw):
    from meld.sync.node import BrainNode
    from meld.sync.service import BrainService
    admit = (lambda kind, obj: admitted.append((kind, obj))) if admitted is not None else None
    node = BrainNode(nid, encoder="hashing", nli="lexical", pol=POL, key=KEY,
                     admit=admit)
    return BrainService(node, group_key=KEY, **kw)


def _five(i=0, scope="eu"):
    """One object of every kind in ``scope``, distinct per ``i``."""
    cell = {"eu": "eu.a", "us": "us.a"}[scope]
    return [("D", _delta(text=f"claim {i}", cell=cell, encoder="hashing")),
            ("A", _eff(scope=scope, target_cell=cell, created_at=i)),
            ("Sig", _sig(scope=scope, issuer=f"n{i}")),
            ("Cal", _cal(scope=scope, item=f"item:{i}")),
            ("Ep", _ep(scope=scope, seq=i))]


def _env(obj, scope="eu", key=KEY):
    o = attach_mac(obj, key)
    return pack([(canonical_bytes(o), o.mac, scope)])


# --------------------------------------------------------------------------- #
# envelope and Merkle root
# --------------------------------------------------------------------------- #
def test_envelope_round_trips_exact_bytes_and_refuses_malformed():
    entries = [(canonical_bytes(o), "ab" * 32, s) for s in ("eu", "us") for _, o in _five(1, s)]
    assert unpack(pack(entries)) == entries
    for bad in (b"", b"x", b"{}", b"[1]", b'[{"s":"eu","o":"x"}]', b'[{"s":1,"o":"x","m":""}]',
                b'[{"s":"eu","o":"\\u00e9","m":""}]', b"\xff", None):
        assert unpack(bad) is None


def test_merkle_root_is_order_free_and_separates_sets():
    ids = [bytes([i]) * 16 for i in range(7)]
    roots = {merkle_root(random.Random(s).sample(ids, len(ids))) for s in range(10)}
    assert len(roots) == 1
    subsets = {merkle_root(ids[:k]) for k in range(len(ids) + 1)}
    assert len(subsets) == len(ids) + 1                      # empty, singletons, prefixes distinct
    assert merkle_root(ids[:1]) != ids[0] and len(merkle_root([])) == 16


# --------------------------------------------------------------------------- #
# named tests of §7.2
# --------------------------------------------------------------------------- #
def test_wire_carries_signed_bytes_cell_authority_id_preserved():
    sent = []
    a = _svc("A", peers=["B"], send=lambda p, body: (sent.append(body), SimpleNamespace(status_code=200))[1])
    b = _svc("B")
    objs = [replace(_delta(cell="us.a", text="us claim", encoder="hashing"), auth=0, origin="A"),
            _delta(cell="eu.b", text="eu claim", auth=3, encoder="hashing")]
    objs += [o for _, o in _five(2, "us")[1:]]
    ids = [a.node.publish_local(o) for o in objs]
    status, body = a.handle("POST", "/publish")
    assert status == 200 and json.loads(body)["published"] == len(objs)
    shipped = [e for payload in sent for e in unpack(payload)]
    assert sorted(object_id(r) for r, _, _ in shipped) == sorted(ids)
    for received, mac, scope in shipped:
        assert (received, mac, scope) == a.node.objects.raw(object_id(received))   # stored bytes
        assert b.handle("POST", "/patch", pack([(received, mac, scope)]))[0] == 200
    for oid, obj in zip(ids, objs):
        kind = next(k for k, c in KINDS.items() if isinstance(obj, c))
        got = b.node.objects.get(kind, oid)
        assert got == attach_mac(obj, KEY)                     # cell, auth, unit id, all fields
        assert b.node.objects.raw(oid) == a.node.objects.raw(oid)
    d = b.node.objects.get("D", ids[0])
    assert d.cell == "us.a" and d.auth == 0 and d.unit_id == objs[0].unit_id
    assert a.handle("POST", "/publish")[1] and json.loads(a.handle("POST", "/publish")[1])["published"] == 0


def test_ingest_publishes_a_signed_delta_in_the_node_cell():
    from meld.agent.memory import DEFAULT_CELL
    from meld.sync.node import BrainNode
    n = BrainNode("A", encoder="hashing", nli="lexical", key=KEY)
    n.ingest("Paris is the capital of France")
    (d,) = list(n.objects.iter("D", n.scope_of[DEFAULT_CELL]))
    assert d.text == "Paris is the capital of France" and d.cell == DEFAULT_CELL
    assert d.origin == "A" and d.encoder == "hashing" and d.created_at > 0
    n.ingest("   ")
    assert len(n.objects.ids("D", n.scope_of[DEFAULT_CELL])) == 1


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_published_object_is_mac_verified_including_effects(kind):
    admitted = []
    b = _svc("B", admitted)
    obj = dict(_five(3))[kind]
    good = attach_mac(obj, KEY)
    received = canonical_bytes(good)
    forged = [pack([(received, "", "eu")]),                                   # unsigned
              _env(obj, key=b"some-other-group-key"),                         # wrong key
              pack([(received, attach_mac(replace(obj, created_at=99) if kind != "Sig"
                                          else replace(obj, issuer="evil"), KEY).mac, "eu")]),
              b"   \n", b"not json"]
    for payload in forged:
        assert b.handle("POST", "/patch", payload)[0] == 403
    assert admitted == [] and object_id(received) not in b.node.objects
    assert b.handle("POST", "/patch", _env(obj))[0] == 200
    assert admitted == [(kind, good)]
    assert b.node.objects.get(kind, object_id(received)) == good


@pytest.mark.parametrize("kind", list(KINDS))
def test_redelivery_is_idempotent_at_the_set_level(kind):
    admitted = []
    b = _svc("B", admitted)
    obj = dict(_five(4))[kind]
    for _ in range(3):
        assert b.handle("POST", "/patch", _env(obj))[0] == 200
    assert b.node.objects.ids(kind, "eu") == (object_id(canonical_bytes(obj)),)
    assert len(admitted) == 1


def test_received_objects_are_not_republished():
    sent = []
    b = _svc("B", peers=["C"], send=lambda p, body: (sent.append(body), SimpleNamespace(status_code=200))[1])
    for _, obj in _five(5):
        b.handle("POST", "/patch", _env(obj))
    assert json.loads(b.handle("POST", "/publish")[1])["published"] == 0 and sent == []


def test_status_endpoint_is_removed_and_overrule_enters_only_as_a_verified_effect():
    b = _svc("B")
    assert b.handle("POST", "/status", json.dumps(
        {"claim": "x", "link_id": "L1", "effect": "OVERRULED"}).encode())[0] == 404
    ovr = _eff(kind="OVERRULE", c1=_delta().claim_key, c2=b"")
    assert b.handle("POST", "/patch", _env(ovr, key=b"forger"))[0] == 403
    assert b.node.objects.ids("A", "eu") == ()
    assert b.handle("POST", "/patch", _env(ovr))[0] == 200
    assert b.node.objects.ids("A", "eu") == (object_id(canonical_bytes(ovr)),)


def test_digest_and_pull_refuse_malformed_requests():
    b = _svc("B")
    for path in ("/digest", "/pull"):
        for body in (b"", b"[]", b'{"scope": 3}', b'{"scope": "mars"}', b"\xff"):
            assert b.handle("POST", path, body)[0] == 400
    assert b.handle("POST", "/pull", b'{"scope": "eu", "kind": "D", "ids": ["zz"]}')[0] == 400
    assert b.handle("POST", "/pull", b'{"scope": "eu", "kind": "Q", "ids": []}')[0] == 400
    assert b.handle("POST", "/pull", b'{"scope": "eu", "kind": "D", "ids": "ab"}')[0] == 400
    assert b.handle("POST", "/pull", b'{"scope": "eu", "kind": "D", "ids": ["AB" ]}')[0] == 400


def test_digest_pull_reconciles_five_sets_under_loss():
    """N replicas each author objects of all five kinds in two scopes; PUBLISH and every
    anti-entropy call are lost at random; DIGEST+PULL rounds converge to equal id sets per
    scope and kind, equal to the union of everything authored."""
    rng = random.Random(7)
    n, p_loss = 5, 0.4
    services = {}

    def call(peer, path, body):
        return (0, b"") if rng.random() < p_loss else services[peer].handle("POST", path, body)

    def send(peer, body):
        status = 0 if rng.random() < p_loss else services[peer].handle("POST", "/patch", body)[0]
        return SimpleNamespace(status_code=status)

    names = [f"R{i}" for i in range(n)]
    for nm in names:
        services[nm] = _svc(nm, peers=[x for x in names if x != nm], send=send, call=call)
    authored = set()
    for i, nm in enumerate(names):
        for scope in ("eu", "us"):
            for j in range(3):
                for kind, obj in _five(100 * i + 10 * j + (scope == "us"), scope):
                    authored.add((scope, kind, services[nm].node.publish_local(obj)))
        services[nm].handle("POST", "/publish")

    def view(nm):
        o = services[nm].node.objects
        return {(s, k): o.ids(k, s) for s in ("eu", "us") for k in KINDS}

    assert len({tuple(sorted(view(nm).items())) for nm in names}) > 1   # loss left them apart
    for rounds in range(1, 40):
        for nm in names:
            assert services[nm].handle("POST", "/resync")[0] == 200
        if len({tuple(sorted(view(nm).items())) for nm in names}) == 1:
            break
    else:
        pytest.fail("anti-entropy did not converge")
    union = {(s, k, i) for (s, k), ids in view(names[0]).items() for i in ids}
    assert union == authored
    assert rounds > 1                                                     # loss hit the rounds too
    m = json.loads(services[names[0]].handle("GET", "/metrics")[1])
    assert all(json.loads(services[nm].handle("GET", "/metrics")[1])["roots"] == m["roots"]
               for nm in names)


def test_one_round_pulls_and_pushes_the_symmetric_difference():
    services = {}
    call = lambda peer, path, body: services[peer].handle("POST", path, body)   # noqa: E731
    a, b = _svc("A", peers=["B"], call=call), _svc("B", peers=["A"], call=call)
    services.update(A=a, B=b)
    for i, (kind, obj) in enumerate(_five(6)):
        (a if i % 2 else b).node.publish_local(obj)
    a.node.publish_local(_delta(cell="us.a", encoder="hashing"))
    r = json.loads(a.handle("POST", "/resync")[1])
    assert (r["pulled"], r["pushed"]) == (3, 3) and r["digest_bytes"] > 0
    assert all(a.node.objects.ids(k, s) == b.node.objects.ids(k, s)
               for k in KINDS for s in ("eu", "us"))
    for svc in (a, b):
        r = json.loads(svc.handle("POST", "/resync")[1])
        assert (r["pulled"], r["pushed"]) == (0, 0)


# --------------------------------------------------------------------------- #
# fix round: scope relabelling, liveness, filtering, concurrency, encoder, hostile JSON
# --------------------------------------------------------------------------- #
def test_relabelled_sig_envelope_is_refused():
    b = _svc("B")
    sig = attach_mac(_sig(scope="us"), KEY)
    assert b.handle("POST", "/patch", pack([(canonical_bytes(sig), sig.mac, "eu")]))[0] == 403
    assert b.node.objects.ids("Sig", "eu") == b.node.objects.ids("Sig", "us") == ()
    assert b.handle("POST", "/patch", pack([(canonical_bytes(sig), sig.mac, "us")]))[0] == 200


def test_mixed_batch_with_one_forged_entry_is_non_delivery():
    b = _svc("B")
    good, bad = (attach_mac(o, k) for o, k in ((_delta(encoder="hashing"), KEY),
                                               (_delta(text="x", encoder="hashing"), b"forger")))
    payload = pack([(canonical_bytes(good), good.mac, "eu"), (canonical_bytes(bad), bad.mac, "eu")])
    status, body = b.handle("POST", "/patch", payload)
    assert status == 403 and json.loads(body) == {"received": 1, "dropped": 1}


def test_delta_under_another_encoder_is_dropped_at_receipt():
    admitted = []
    b = _svc("B", admitted)
    assert b.handle("POST", "/patch", _env(_delta(encoder="enc-other")))[0] == 403
    assert admitted == [] and b.node.objects.ids("D", "eu") == ()
    with pytest.raises(ValueError):
        b.node.publish_local(_delta(encoder="enc-other"))


@pytest.mark.parametrize("body", [b"[" * 100_000, b"\xff\xfe", b'{"scope": "eu", "roots": []}'],
                         ids=["deep", "non-utf8", "roots-not-dict"])
def test_hostile_json_is_refused_on_every_receive_path(body):
    b = _svc("B")
    assert b.handle("POST", "/patch", body)[0] == 403
    assert b.handle("POST", "/digest", body)[0] == 400
    assert b.handle("POST", "/pull", body)[0] == 400


def test_cross_scope_pull_is_refused():
    b = _svc("B")
    oid = b.node.publish_local(_delta(cell="us.a", encoder="hashing"))
    req = json.dumps({"scope": "eu", "kind": "D", "ids": [oid.hex()]}).encode()
    assert b.handle("POST", "/pull", req) == (200, b"[]")


_BAD_DIGESTS = [b'{"roots": {"D": "x"}, "ids": {"D": [1, "a"]}}', b"[" * 100_000, b"\xff",
                b'{"roots": {"Q": "' + b"0" * 32 + b'"}, "ids": {}}', b'{"roots": [], "ids": {}}',
                b'{"roots": {}, "ids": {"D": ["zz"]}}', b'{"roots": {}, "ids": {"D": "ab"}}', b"null"]


@pytest.mark.parametrize("bad", _BAD_DIGESTS + [RuntimeError("boom")],
                         ids=[f"bad{i}" for i in range(len(_BAD_DIGESTS) + 1)])
def test_resync_survives_a_bad_peer_and_pulls_from_the_good_one(bad):
    good = _svc("G")
    oid = good.node.publish_local(_delta(text="from good", encoder="hashing"))

    def call(peer, path, body):
        if peer == "bad":
            if isinstance(bad, Exception):
                raise bad
            return 200, bad
        return good.handle("POST", path, body)
    me = _svc("M", peers=["bad", "good"], call=call)
    status, body = me.handle("POST", "/resync")
    assert status == 200 and json.loads(body)["failed_peers"] == ["bad"]
    assert me.node.objects.ids("D", "eu") == (oid,)


def test_a_lost_call_is_not_a_failed_peer_and_converged_digests_carry_no_ids():
    from meld.sync.anti_entropy import digest_response, objectset_round
    me = _svc("M", peers=["P"], call=lambda peer, path, body: (0, b""))
    me.node.publish_local(_delta(encoder="hashing"))
    assert objectset_round(me.node.objects, "eu", lambda p, b: (0, b""), None)[:2] == (0, 0)
    assert json.loads(me.handle("POST", "/resync")[1])["failed_peers"] == []
    roots = json.loads(me.handle("GET", "/metrics")[1])["roots"]["eu"]
    assert digest_response(me.node.objects, "eu", roots) == {"roots": roots, "ids": {}}


def test_pull_reply_is_filtered_and_counters_count_only_accepted_objects():
    want, extra, forged = (attach_mac(_delta(text=t, encoder="hashing"), k)
                           for t, k in (("wanted", KEY), ("unrequested", KEY), ("forged", b"x")))
    from meld.sync.anti_entropy import objectset_round
    peer_ids = sorted(object_id(canonical_bytes(o)).hex() for o in (want, forged))

    def call(path, body):
        if path == "/digest":
            return 200, json.dumps({"roots": {"D": "0" * 32}, "ids": {"D": peer_ids}}).encode()
        if path == "/pull":
            return 200, pack([(canonical_bytes(o), o.mac, "eu") for o in (want, extra, forged)])
        return 403, b""
    me = _svc("M")
    pulled, pushed, _ = objectset_round(me.node.objects, "eu", call, me.node.receive_object)
    assert (pulled, pushed) == (1, 0)
    assert me.node.objects.ids("D", "eu") == (object_id(canonical_bytes(want)),)
    mine = me.node.publish_local(_delta(text="mine", encoder="hashing"))
    _, pushed, _ = objectset_round(me.node.objects, "eu", call, me.node.receive_object)
    assert pushed == 0 and mine in me.node.objects                # peer refused the push


def test_concurrent_redeliveries_admit_exactly_once(monkeypatch):
    import threading
    import time
    import meld.sync.objectset as om
    admitted = []
    b = _svc("B", admitted)
    real = om.ObjectSet.add

    def slow(self, *a, **k):
        time.sleep(0.05)
        return real(self, *a, **k)
    monkeypatch.setattr(om.ObjectSet, "add", slow)
    body = _env(_delta(text="race", encoder="hashing"))
    ts = [threading.Thread(target=b.handle, args=("POST", "/patch", body)) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(admitted) == 1


def test_merkle_inner_nodes_are_separated_from_leaves():
    from meld.proto import h16
    a, b = b"\x01" * 16, b"\x02" * 16
    inner = h16(b"\x00" + a) + h16(b"\x00" + b)
    assert merkle_root([inner]) != merkle_root([a, b])


def _script(monkeypatch, *parts):
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1].joinpath(*parts)
    monkeypatch.syspath_prepend(str(path.parent))          # a sibling config module
    spec = importlib.util.spec_from_file_location(path.stem + "_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("parts,writer", [(("edge", "run_eval_c.py"), "_deliver_status"),
                                          (("scripts", "eval_c_docker_smoke.py"), "_status_write")])
def test_eval_c_status_writers_publish_a_signed_effect_through_patch(monkeypatch, parts, writer):
    # the eval-c drivers (testbed and docker smoke) change status only by PUBLISHing a signed
    # AuthoredEffect to /patch; the observable is vis from /metrics. A refused envelope raises.
    from meld.sync.node import BrainNode
    from meld.sync.service import BrainService
    mod = _script(monkeypatch, *parts)
    svc = BrainService(BrainNode("n", encoder="hashing", nli="lexical"))
    claim = getattr(mod, "CLAIM", "the canonical policy is X")
    svc.handle("POST", "/ingest", claim.encode())
    sent = []

    def post(url, data=b"", timeout=None):
        status, body = svc.handle("POST", url[url.index("/", 8):], data)
        sent.append(url)
        if status != 200:
            raise RuntimeError(status)
        return json.loads(body)
    monkeypatch.setattr(mod, "_post", post)
    if writer == "_deliver_status":
        write = lambda lid, eff: mod._deliver_status("http://n:1", claim, lid, eff)      # noqa: E731
    else:
        monkeypatch.setattr(mod, "URL", {"n": "http://n:1"})
        write = lambda lid, eff: mod._status_write("n", lid, eff)                       # noqa: E731
    write("La", "DEPRECATED")
    assert json.loads(svc.handle("GET", "/metrics")[1])["status"][claim] == "DEPRECATED"
    write("Lb", "OVERRULED")
    write("La", "DEPRECATED")                                  # redelivery: idempotent
    assert json.loads(svc.handle("GET", "/metrics")[1])["status"][claim] == "OVERRULED"
    assert sent and all(u.endswith("/patch") for u in sent)
    monkeypatch.setenv("SMP_GROUP_KEY", "another-key")          # unverifiable: refused, loudly
    with pytest.raises(RuntimeError):
        write("Lc", "REVOKED")
