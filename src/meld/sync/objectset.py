"""The per-scope replicated grow-only sets of design §2.1-2.3: D, A, Sig, Cal, Ep.

``add`` is the receive gate for every object type: it checks the MAC (HMAC-SHA256 under the
group key) over the received bytes BEFORE parsing them, parses with ``from_bytes`` (the type
tag ``t`` names the class; non-canonical encodings are refused, so the MAC'd bytes are the
object's canonical bytes), checks the object's own scope against the PUBLISH envelope's,
the partition and the subscribed scopes, and inserts idempotently by object id. An
unverifiable, malformed or cross-scope object is dropped (None), never raised. Pure
replication: no derived state is kept here.

Scope of an object: a Delta's is its cell's scope under the partition; every other kind
names its own. The envelope scope, when given, must equal it. Since the scope is a function
of the bytes (and the fixed partition), one id is only ever held under one scope.

With ``encoder`` set, a Delta under another encoder is dropped at receipt (design §2.1: enc
must equal M_nli's encoder). Anti-entropy then re-offers that delta every round, since the
peer still holds it; tolerable in a homogeneous federation, where it does not arise.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Dict, Iterable, Iterator, Optional, Tuple

from meld.protocol.objects import (
    AuthoredEffect, CalRecord, Delta, EpochRecord, SignalRecord, from_bytes, object_id,
)

KINDS = {"D": Delta, "A": AuthoredEffect, "Sig": SignalRecord, "Cal": CalRecord,
         "Ep": EpochRecord}
_KIND_OF_TAG = {cls.__name__: kind for kind, cls in KINDS.items()}


def _mac_ok(received, mac, key: bytes) -> bool:
    if not key:
        raise ValueError("empty HMAC key")
    return (type(received) is bytes and type(mac) is str and hmac.compare_digest(
        mac, hmac.new(key, received, hashlib.sha256).hexdigest()))


class ObjectSet:
    def __init__(self, scope_of: Dict[str, str], scopes: Optional[Iterable[str]] = None,
                 encoder: Optional[str] = None):
        self.scope_of = scope_of
        self.scopes = frozenset(scope_of.values() if scopes is None else scopes)
        self.encoder = encoder
        self._by_id: Dict[bytes, Tuple[str, str, bytes, object]] = {}  # id -> kind, scope, bytes, obj
        self._index: Dict[Tuple[str, str], set] = {}
        self._log: Dict[str, list] = {}       # scope -> ids in arrival order

    def add(self, received: bytes, mac: str, key: bytes, scope: Optional[str] = None):
        """Verify, parse, scope-check and insert; the stored object, or None if dropped."""
        if not _mac_ok(received, mac, key):
            return None
        try:
            kind = _KIND_OF_TAG[json.loads(received)["t"]]
            obj = from_bytes(KINDS[kind], received, mac)
        except (ValueError, KeyError, TypeError, RecursionError):
            return None
        own = self.scope_of.get(obj.cell) if kind == "D" else obj.scope
        if scope is not None and scope != own or own not in self.scopes:
            return None
        if kind == "D" and self.encoder is not None and obj.encoder != self.encoder:
            return None
        oid = object_id(received)
        if oid not in self._by_id:
            self._by_id[oid] = (kind, own, received, obj)
            self._index.setdefault((kind, own), set()).add(oid)
            self._log.setdefault(own, []).append(oid)
        return self._by_id[oid][3]

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, oid: bytes) -> bool:
        return oid in self._by_id

    def kind_of(self, oid: bytes) -> Optional[str]:
        e = self._by_id.get(oid)
        return e[0] if e else None

    def get(self, kind: str, oid: bytes):
        e = self._by_id.get(oid)
        return e[3] if e and e[0] == kind else None

    def ids(self, kind: str, scope: str) -> Tuple[bytes, ...]:
        return tuple(sorted(self._index.get((kind, scope), ())))

    def iter(self, kind: str, scope: str) -> Iterator:
        return (self._by_id[i][3] for i in self.ids(kind, scope))

    def arrivals(self, scope: str) -> list:
        """The scope's ids in arrival order (a live, grow-only list: callers keep an offset)."""
        return self._log.setdefault(scope, [])

    def raw(self, oid: bytes) -> Tuple[bytes, str, str]:
        """(received bytes, mac, scope): what PUBLISH and PULL ship, never re-serialised."""
        kind, scope, received, obj = self._by_id[oid]
        return received, obj.mac, scope


__all__ = ["KINDS", "ObjectSet"]
