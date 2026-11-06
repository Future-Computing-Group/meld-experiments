"""Wire bytes per message class, over the real serializers.

Every replicated object, and a PASS certificate (fold.publish_pass), travels as one signed envelope
``{"s": scope, "o": canonical bytes, "m": MAC}`` (sync/wire_delta.pack): ``envelope_bytes`` is the
length of a one-entry pack, i.e. what a single PUBLISH body of that object costs. The class is the
object's type tag: DELTA, EFFECT (authored effect), SIG, CAL, EPOCH, PASS. Anti-entropy traffic
(DIGEST request and reply, PULL request and reply, and the PATCH push of what the peer lacks) is
metered per call path by eval/anti_entropy_eval.reconcile; ``CLASSES`` names all of them.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, Tuple

from meld.sync.wire_delta import pack

_TAG = {"Delta": "DELTA", "AuthoredEffect": "EFFECT", "SignalRecord": "SIG", "CalRecord": "CAL",
        "EpochRecord": "EPOCH", "PassCertificate": "PASS"}
CLASSES = tuple(_TAG.values()) + ("DIGEST", "PULL", "PATCH")


def envelope_bytes(received: bytes, mac: str, scope: str) -> int:
    return len(pack([(received, mac, scope)]))


def message_class(received: bytes) -> str:
    return _TAG[json.loads(received)["t"]]


def meter(entries: Iterable[Tuple[bytes, str, str]]) -> Dict[str, Dict[str, float]]:
    """class -> count, total envelope bytes, mean, min, max, and the canonical body and MAC
    bytes inside them, over (received, mac, scope) entries."""
    out: Dict[str, Dict[str, float]] = {}
    for received, mac, scope in entries:
        n = envelope_bytes(received, mac, scope)
        r = out.setdefault(message_class(received),
                           {"count": 0, "bytes": 0, "min": n, "max": n, "mac": 0, "body": 0})
        r["count"] += 1
        r["bytes"] += n
        r["min"], r["max"] = min(r["min"], n), max(r["max"], n)
        r["mac"] += len(mac)
        r["body"] += len(received)
    for r in out.values():
        r["mean"] = r["bytes"] / r["count"]
    return out


__all__ = ["CLASSES", "envelope_bytes", "message_class", "meter"]
