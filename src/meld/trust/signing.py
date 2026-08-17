"""Signature verification gate for the merge-on-receive path.

The manuscript signs every unit/link/Patch and verifies the signature before admission, so
the audit trail is attributable and a delta cannot be silently forged by the transport
(Sections 4.2, 5.4); the specific public-key suite is a deployment choice scoped out of the
paper. This module is the experiment's stand-in: HMAC-SHA256 over the canonical bytes of the
delta (excluding the signature field), with ``admit_signed`` the gate the receive path runs
*before* any merge signal is read. A forged or tampered delta fails verification and is
dropped (returns None).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Optional


def _canonical(delta: dict) -> bytes:
    """Deterministic canonical bytes of the delta EXCLUDING its own signature field."""
    body = {k: v for k, v in delta.items() if k != "sig"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_delta(delta: dict, key: bytes) -> dict:
    """Return a copy of ``delta`` carrying a ``sig`` (HMAC-SHA256 over its canonical bytes)."""
    sig = hmac.new(key, _canonical(delta), hashlib.sha256).hexdigest()
    return {**delta, "sig": sig}


def verify_delta(delta: dict, key: bytes) -> bool:
    """True iff ``delta`` carries a ``sig`` that matches the HMAC of its current content
    under ``key`` (constant-time compare). Unsigned, forged, tampered, or wrong-key -> False."""
    sig = delta.get("sig")
    if not isinstance(sig, str):
        return False
    expected = hmac.new(key, _canonical(delta), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def admit_signed(delta: dict, key: bytes) -> Optional[dict]:
    """The receive-path gate: return ``delta`` for processing iff its signature verifies,
    else None (dropped before any merge signal is read)."""
    return delta if verify_delta(delta, key) else None


__all__ = ["sign_delta", "verify_delta", "admit_signed"]
