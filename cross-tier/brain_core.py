#!/usr/bin/env python3
"""Shared receive step for the cross-tier brains (relay and Kafka).

Each brain is one MELD replica (``replica``: an SMPMemory over the five replicated sets with
the admission path of src/meld/protocol/admission.py), so the two transports cannot diverge
in semantics. A message on the claims topic carries one signed object of any kind in its
original canonical bytes (``signed_json``): a claim delta, an authored effect, a Sig, Cal or
epoch record. ``process_delta`` is transport-independent: it verifies the MAC, inserts the
object, admits a newly held delta (decision INSERT / MERGE / RELATE / CONFLICT / REJECT under
the active epoch; without one, no similarity merge) and returns the ack dict with the
claim's visible status. An unverifiable or malformed message is dropped (None, no ack)
before anything is read (design §2.3; 04 finding 10).

The epoch is a replicated object too: the driver publishes the calibration authority's
signed record (config.load_epoch) on the same topic. There is no threshold config.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from meld.agent.memory import DEFAULT_CELL, DEFAULT_POL, SMPMemory   # noqa: E402
from meld.protocol.objects import (                                  # noqa: E402
    AuthoredEffect, Delta, signed,
)
from meld.sync.wire_delta import group_key                           # noqa: E402

KEY = group_key(os.environ)   # SMP_GROUP_KEY, else the labelled non-secret test key
OTHER_CELL = "agent.other"     # a second cell of the scope: cross-cell pairs never merge
XTIER_POL = {**DEFAULT_POL, "scopes": {"agent": [DEFAULT_CELL, OTHER_CELL]}}


def replica(brain_id: str, *, encoder: str, nli: str, key: bytes = KEY) -> SMPMemory:
    return SMPMemory(encoder=encoder, nli=nli, pol=XTIER_POL, key=key, origin=brain_id)


def signed_json(obj, nonce, key: bytes = KEY) -> str:
    """The wire form a publisher puts on the claims topic: one object's canonical bytes,
    its MAC under the group key, and the publisher's nonce."""
    received, mac = signed(obj, key)
    return json.dumps({"nonce": nonce, "o": received.decode("ascii"), "m": mac})


def process_delta(d, rep: SMPMemory, brain_id: str, t_recv: float):
    """Return the ack dict for one delivered message (None if it is malformed or its object
    does not verify under the replica's key). ``op`` is "publish" for a delta, else the
    object's kind; ``claim_id`` is the hex claim key (c1 for an effect)."""
    if not isinstance(d, dict) or type(d.get("o")) is not str or type(d.get("m")) is not str:
        return None
    received = d["o"].encode("utf-8")
    decs = rep.receive(received, d["m"])
    obj = rep.store.apply(received, d["m"])     # idempotent: the held object, None if dropped
    if obj is None:
        return None
    decision = decs[0].decision if decs else None
    key = obj.claim_key if isinstance(obj, Delta) else getattr(obj, "c1", b"")
    return {"nonce": d.get("nonce"), "brain": brain_id,
            "op": "publish" if isinstance(obj, Delta) else
                  (obj.kind.lower() if isinstance(obj, AuthoredEffect) else type(obj).__name__),
            "claim_id": key.hex(), "decision": decision,
            "eff_status": rep.store.vis(key) if key else None, "t_recv": t_recv}
