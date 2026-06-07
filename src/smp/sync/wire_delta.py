"""Signed, typed graph-delta on the wire (architecture A.1/§7-s).

A PUBLISH carries a SERIALIZED TYPED DELTA -- a ``MemoryUnit`` (its ``Claim``s,
status, relevance) plus the originating node id -- not bare claim text. The
receiver reconstructs the typed objects and runs the SAME merge-on-receive path
the in-process write uses, so the wire path and the in-process path are
behaviourally identical.

The envelope is HMAC-signed over a shared GROUP KEY (benign-fault, keyed-group
integrity/attribution -- NOT a PKI / Byzantine model). The receiver VERIFIES the
signature BEFORE reading any merge signal: a forged, tampered, or wrong-key
delta fails verification and is dropped before admission (returns None). HMAC is
the experiment stand-in; the specific public-key suite is a deployment choice
scoped out of the paper.

The key is read from config/env (``SMP_GROUP_KEY``); no secret is hardcoded in
source -- the fallback is a clearly-labelled non-secret TEST key for in-process
runs, exactly as the no-silent-default config discipline elsewhere.
"""
from __future__ import annotations

import json
from typing import Optional, Tuple

from smp.proto import MemoryUnit
from smp.trust.signing import sign_delta, verify_delta

# A clearly non-secret default for in-process/test runs. Real deployments set
# SMP_GROUP_KEY; this fallback is never used as a production secret.
_DEFAULT_TEST_KEY = b"smp-default-test-group-key-not-a-secret"


def group_key(env) -> bytes:
    """The shared group HMAC key: ``SMP_GROUP_KEY`` from env, else the labelled
    non-secret test key. Never empty (so signing/verification always run)."""
    v = env.get("SMP_GROUP_KEY")
    if v:
        return v.encode("utf-8") if isinstance(v, str) else v
    return _DEFAULT_TEST_KEY


def serialize_signed_delta(node_id: str, unit: MemoryUnit, key: bytes) -> bytes:
    """Serialize a typed delta (node id + MemoryUnit) and HMAC-sign it over the
    group ``key``. Returns the JSON bytes the transport carries to a peer."""
    body = {"node_id": node_id, "unit": unit.to_dict()}
    signed = sign_delta(body, key)            # adds a "sig" over the canonical body
    return json.dumps(signed, sort_keys=True, separators=(",", ":")).encode("utf-8")


def deserialize_verified_delta(
    payload: bytes, key: bytes
) -> Optional[Tuple[str, MemoryUnit]]:
    """Verify the envelope's signature, then reconstruct the typed delta. Returns
    ``(node_id, MemoryUnit)`` iff the signature verifies under ``key``; otherwise
    ``None`` (forged / tampered / wrong-key / malformed -> dropped before any
    merge signal is read)."""
    try:
        env = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(env, dict) or not verify_delta(env, key):
        return None
    try:
        unit = MemoryUnit.from_dict(env["unit"])
        node_id = env["node_id"]
    except (KeyError, TypeError, ValueError):
        return None
    return node_id, unit


__all__ = [
    "group_key",
    "serialize_signed_delta",
    "deserialize_verified_delta",
]
