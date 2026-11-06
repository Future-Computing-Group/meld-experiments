#!/usr/bin/env python3
"""Unit tests for the shared receive step (brain_core.process_delta).

``process_delta`` is the one source of truth both cross-tier brains (relay and Kafka) call,
so the two transports cannot diverge in semantics. These tests pin its contract directly,
transport-free and broker-free: real claim deltas are admitted by the protocol's step
(distinct content in => distinct decision out), authored effects join order-independently,
redelivery is idempotent, and nothing unverifiable is read.

Run:  PYTHONPATH=src:cross-tier python3 -m pytest cross-tier/test_brain_core.py -q
"""
import json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from brain_core import process_delta, replica, signed_json   # noqa: E402
import driver                                                # noqa: E402



def epoch_file(theta=0.5):
    """A signed test epoch for the brains' M_nli and M_pol (what the calibration authority
    would distribute), as a config.save_epoch file; returns its path."""
    from meld.agent.memory import DEFAULT_POL
    from meld.config import save_epoch
    from meld.proto import CANON_VERSION
    from meld.protocol.objects import FIXED_ONE, INF16, EpochRecord, m_nli, m_pol_of
    from brain_core import XTIER_POL, KEY as GROUP_KEY
    sel = DEFAULT_POL["bootstrap_selector"]
    ep = EpochRecord(scope="agent", seq=1, m_nli=m_nli(driver.ENCODER, driver.NLI, "int8", CANON_VERSION),
                     m_pol=m_pol_of(XTIER_POL), policy="SEED", alpha=20_000, alpha_c=20_000,
                     mondrian_c=False, cal_root=b"\x00" * 16, cal_hw="cpu", bins=(256, "doubling"),
                     table=((0, 100, 0, round(theta * FIXED_ONE), INF16),), sigma_lo_q=sel[0],
                     sigma_c_q=sel[1], m=sel[2], author="calibration", created_at=0)
    fd, path = tempfile.mkstemp(suffix=".epoch.json"); os.close(fd)
    save_epoch(ep, GROUP_KEY, path)
    return path


def _rep(bid="b"):
    return replica(bid, encoder=driver.ENCODER, nli=driver.NLI)


def _run(msgs, rep=None):
    rep = rep or _rep()
    return {n: process_delta(json.loads(m), rep, "b", 0.0) for n, m in msgs}, rep


def test_running_example_decides_each_class_distinctly_under_an_epoch():
    from meld.config import load_epoch
    acks, _ = _run([("epoch", driver.epoch_message(load_epoch(epoch_file())))]
                   + driver.running_example())
    assert {n: acks[n]["decision"] for n in ("seed", "merge", "relate", "conflict")} == {
        "seed": "INSERT", "merge": "MERGE", "relate": "RELATE", "conflict": "CONFLICT"}
    assert acks["deprecate"]["eff_status"] == "DEPRECATED"
    assert acks["overrule"]["eff_status"] == "OVERRULED" and acks["overrule"]["claim_id"] == driver.C2_ID
    # without the epoch the paraphrase abstains: no default operating point
    bare, _ = _run(driver.running_example())
    assert bare["merge"]["decision"] == "RELATE"


def test_ack_shape_carries_identity_and_a_publish_writes_no_authored_effect():
    (n, m), = driver.running_example()[:1]
    acks, rep = _run([(n, m)])
    ack = acks[n]
    assert ack["nonce"] == "seed" and ack["brain"] == "b" and ack["op"] == "publish"
    assert ack["eff_status"] == "ACTIVE" and ack["t_recv"] == 0.0
    assert rep.store.objects.ids("A", "agent") == ()


def test_authored_effects_join_order_independently():
    dep, ovr = driver.effect_message("d", "SUPERSEDE"), driver.effect_message("o", "OVERRULE")
    c2 = driver.c2_message()
    a1, _ = _run([("c2", c2), ("d", dep), ("o", ovr)])
    a2, _ = _run([("c2", c2), ("o", ovr), ("d", dep)])
    assert a1["o"]["eff_status"] == a2["d"]["eff_status"] == "OVERRULED"
    assert a1["d"]["op"] == "supersede" and a1["o"]["op"] == "overrule"


def test_idempotent_redelivery_decides_once():
    msgs = driver.running_example()[:2]
    rep = _rep()
    first, _ = _run(msgs, rep)
    again, _ = _run(msgs, rep)
    assert first["seed"]["decision"] == "INSERT" and again["seed"]["decision"] is None
    assert len(rep.store.objects.ids("D", "agent")) == 2 and len(rep.decisions) == 2


def test_brain_core_refuses_an_unverified_or_malformed_message():
    # 04 finding 10: unsigned, forged, tampered and wrong-key objects are dropped (None, no
    # ack) before anything is read; MAC verification precedes admission.
    good = json.loads(driver.effect_message("o", "OVERRULE"))
    forged = json.loads(signed_json(driver._effect("OVERRULE"), "o", key=b"another-key"))
    tampered = dict(good, o=good["o"].replace('"author":"driver"', '"author":"mallory"'))
    for bad in (dict(good, m=""), dict(good, m="00" * 32), forged, tampered,
                {"nonce": "o"}, dict(good, o=1), [1], None, "x"):
        rep = _rep()
        assert process_delta(bad, rep, "b", 0.0) is None
        assert len(rep.store.objects) == 0
    rep = _rep()
    process_delta(json.loads(driver.c2_message()), rep, "b", 0.0)
    assert process_delta(good, rep, "b", 0.0)["eff_status"] == "OVERRULED"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, "->", repr(e)[:200])
    print("=== %d failure(s) ===" % fails)
    sys.exit(1 if fails else 0)


def test_wan_trial_messages_are_signed_objects_the_brains_admit_and_join():
    # the partition-heal WAN script publishes, per trial, a fresh claim delta and two signed
    # authored effects on it (never the retired unsigned {"op": ...} payloads, which are dropped)
    t = driver.trial_messages(3)
    acks, rep = _run([("d", t["delta"]), ("dep", t["deprecate"])])
    assert acks["d"]["decision"] == "INSERT" and acks["d"]["claim_id"] == t["claim_id"]
    assert acks["dep"]["claim_id"] == t["claim_id"] and acks["dep"]["eff_status"] == "DEPRECATED"
    ack = process_delta(json.loads(t["overrule"]), rep, "b", 0.0)
    assert ack["eff_status"] == "OVERRULED"
    assert driver.trial_messages(4)["claim_id"] != t["claim_id"]
    assert process_delta({"nonce": "x", "op": "deprecate", "claim_id": t["claim_id"]}, rep, "b", 0.0) is None
