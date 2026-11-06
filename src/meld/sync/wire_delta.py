"""The generic signed envelope of design §2.2: one wire form for every object type.

A payload is a JSON array of envelopes ``{"s": scope, "o": canonical bytes, "m": mac}``.
The canonical bytes carry the object's type tag (member ``t``) and are the stored received
bytes, never re-serialised; they are ASCII JSON, so they travel as a JSON string and come
back byte-identical. ``unpack`` validates shape only and returns None on any malformed
payload; MAC verification and parsing happen at ``ObjectSet.add`` (verify before insert).

The group HMAC key is read from config/env (``SMP_GROUP_KEY``); the fallback is a clearly
labelled non-secret TEST key for in-process runs. Benign-fault keyed-group integrity, not
a PKI.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

_DEFAULT_TEST_KEY = b"meld-default-test-group-key-not-a-secret"

Entry = Tuple[bytes, str, str]       # (canonical bytes, mac, scope)


def group_key(env) -> bytes:
    """The shared group HMAC key: ``SMP_GROUP_KEY`` from env, else the labelled
    non-secret test key. Never empty (so signing/verification always run)."""
    v = env.get("SMP_GROUP_KEY")
    if v:
        return v.encode("utf-8") if isinstance(v, str) else v
    return _DEFAULT_TEST_KEY


def pack(entries: List[Entry]) -> bytes:
    return json.dumps([{"s": s, "o": b.decode("ascii"), "m": m} for b, m, s in entries],
                      separators=(",", ":")).encode("ascii")


def unpack(payload: bytes) -> Optional[List[Entry]]:
    try:
        envs = json.loads(payload)
        if type(envs) is not list:
            return None
        out = []
        for e in envs:
            if not all(type(e[k]) is str for k in ("s", "o", "m")):
                return None
            out.append((e["o"].encode("ascii"), e["m"], e["s"]))
        return out
    except (ValueError, TypeError, KeyError, RecursionError):
        return None


__all__ = ["group_key", "pack", "unpack"]
