"""Anti-entropy over the five replicated sets (design §2.2; it replaces the status-only digest).

A digest and its symmetric difference over TAGGED id sets: per scope, one ``ClaimDigest`` per object kind (D, A, Sig, Cal, Ep)
whose ``set_hash`` is the kind's Merkle root. ``digest_response``/``pull_response`` serve
DIGEST and PULL from an ``ObjectSet``; ``objectset_round`` runs one round against a peer:
DIGEST with own roots, PULL what is missing here, PUBLISH (push) what the peer lacks.
Pulled objects re-enter through the receive gate (MAC verified again), and only the
requested (scope, id) pairs are accepted from a PULL reply.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Tuple

from meld.protocol.objects import object_id
from meld.sync.merkle import merkle_root
from meld.sync.objectset import KINDS
from meld.sync.wire_delta import pack, unpack


@dataclass(frozen=True)
class ClaimDigest:
    """Digest entry of one (scope, kind): the sorted ids and their Merkle root (hex)."""

    link_ids: Tuple[str, ...]
    set_hash: str


_HEX_ID = re.compile(r"[0-9a-f]{32}")


def _hex_ids(x) -> list:
    """A validated list of 32-hex-char ids, or ValueError."""
    if type(x) is not list or not all(type(i) is str and _HEX_ID.fullmatch(i) for i in x):
        raise ValueError("ids must be a list of 32-hex-char strings")
    return x


def scope_digest(objs, scope: str) -> Dict[str, ClaimDigest]:
    """kind -> (sorted hex ids, Merkle root hex) for the non-empty kinds of ``scope``."""
    return {k: ClaimDigest(tuple(i.hex() for i in ids), merkle_root(ids).hex())
            for k in KINDS if (ids := objs.ids(k, scope))}


def digest_response(objs, scope: str, roots: Dict[str, str]) -> dict:
    """DIGEST reply: every root, plus the id list of each kind whose root differs from
    the requester's ``roots``."""
    mine = scope_digest(objs, scope)
    return {"roots": {k: d.set_hash for k, d in mine.items()},
            "ids": {k: list(d.link_ids) for k, d in mine.items() if roots.get(k) != d.set_hash}}


def pull_response(objs, scope: str, kind: str, hex_ids) -> list:
    """PULL reply: the original (bytes, mac, scope) of each requested id held in (kind, scope)."""
    held = set(objs.ids(kind, scope))
    return [objs.raw(i) for i in map(bytes.fromhex, _hex_ids(hex_ids)) if i in held]


def _their_ids(resp, mine: Dict[str, ClaimDigest]) -> Dict[str, set]:
    """The peer's id set per kind from a DIGEST reply (ValueError if malformed): sent ids
    where its root differs from ours, else equal to ours."""
    roots, ids = resp["roots"], resp["ids"]
    if type(roots) is not dict or type(ids) is not dict or not set(roots) | set(ids) <= set(KINDS):
        raise ValueError("malformed DIGEST reply")
    return {k: set(_hex_ids(ids[k])) if k in ids else
            set(mine[k].link_ids) if k in roots and k in mine else set() for k in KINDS}


def objectset_round(objs, scope: str, call, receive) -> Tuple[int, int, int]:
    """One anti-entropy round against a peer. ``call(path, body) -> (status, bytes)`` talks
    to the peer; ``receive(bytes, mac, scope) -> bool`` is the local receive gate. Returns
    (objects pulled and accepted here, objects pushed and acknowledged by the peer, DIGEST
    bytes on the wire). A lost call leaves its part for a later round; a malformed reply
    raises ValueError (the caller treats the round with that peer as lost)."""
    mine = scope_digest(objs, scope)
    req = json.dumps({"scope": scope, "roots": {k: d.set_hash for k, d in mine.items()}}).encode()
    status, body = call("/digest", req)
    if status != 200:
        return 0, 0, len(req)
    try:
        theirs = _their_ids(json.loads(body), mine)
    except (TypeError, KeyError, AttributeError, RecursionError) as e:
        raise ValueError("malformed DIGEST reply") from e
    pulled, push = 0, []
    for kind in KINDS:
        own = set(mine[kind].link_ids) if kind in mine else set()
        want = theirs[kind] - own
        push += [objs.raw(bytes.fromhex(i)) for i in sorted(own - theirs[kind])]
        if not want:
            continue
        status, reply = call("/pull", json.dumps(
            {"scope": scope, "kind": kind, "ids": sorted(want)}).encode())
        for received, mac, s in (unpack(reply) or []) if status == 200 else []:
            if s == scope and object_id(received).hex() in want:    # only what was asked for
                pulled += bool(receive(received, mac, s))
    pushed = len(push) if push and call("/patch", pack(push))[0] == 200 else 0
    return pulled, pushed, len(req) + len(body)


__all__ = ["scope_digest", "digest_response", "pull_response", "objectset_round", "ClaimDigest"]
