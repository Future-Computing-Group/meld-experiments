"""S3: a canonical graph-state fingerprint (order-shuffle experiment).

The status-CRDT (manuscript Sec. 5.1) is order-independent by construction:
replicas that receive the same set of status links compute the same per-claim
status regardless of delivery order. The merge GRAPH built by the per-pair
decision procedure (Sec. 4) against already-held state is explicitly NOT
claimed order-independent (05-consistency-trust.tex:36-39). This module gives
that disclaimer a measurable object: a deterministic, order-invariant-BY-
CONSTRUCTION serialization of a Brain's semantically relevant state, so two
Brains that received the SAME deltas in DIFFERENT orders can be compared.

Four components, each sorted so the serialization does not itself introduce
order-sensitivity:
  (a) live_claims       -- claim_key (hex) of every claim whose effective
                            status is ACTIVE.
  (b) status_map         -- claim_key (hex) -> effective status, for every
                            claim the brain has ever put.
  (c) links               -- the typed link MULTISET: (type, src, dst), each
                            endpoint normalized to its claim_key when the
                            link carries one (the semantic identity used
                            throughout the merge graph), else its unit id.
  (d) representatives     -- per SUPERSEDES-consolidated group (connected
                            component under the SUPERSEDES relation): the
                            group's members and which member(s) are still
                            ACTIVE (the surfaced canonical claim).

Each component is hashed separately (so a driver can report which component
diverged) and the four together form the joint digest. ``readable`` keeps the
sorted, decoded data so a diff is inspectable, not just detectable.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from smp.brain.store import Brain
from smp.merge.status_crdt import StatusState, effective_status
from smp.proto import Endpoint

_COMPONENTS = ("live_claims", "status_map", "links", "representatives")


def _hex(b: bytes) -> str:
    return b.hex() if isinstance(b, (bytes, bytearray)) else str(b)


def _endpoint_key(ep: Endpoint) -> str:
    """Normalize an Endpoint to one comparable string: the claim_key (the
    semantic identity the merge graph reasons over) when present, else the
    unit/node id."""
    if ep.claim_key:
        return f"claim:{_hex(ep.claim_key)}"
    if ep.node_id:
        return f"unit:{_hex(ep.node_id)}"
    return "empty:"


def _status_map(brain: Brain) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ck in brain._claims:
        state = brain._status.get(ck, StatusState())
        out[_hex(ck)] = effective_status(state)
    return out


def _live_claims(status_map: Dict[str, str]) -> List[str]:
    return sorted(k for k, st in status_map.items() if st == "ACTIVE")


def _links(brain: Brain) -> List[Tuple[str, str, str]]:
    return sorted((ln.type, _endpoint_key(ln.src), _endpoint_key(ln.dst))
                  for ln in brain.links)


def _representatives(brain: Brain, status_map: Dict[str, str]
                     ) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Connected components under SUPERSEDES (undirected), each reported as
    (sorted member endpoint-keys, sorted ACTIVE member claim-keys)."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    members: set = set()
    for ln in brain.links:
        if ln.type != "SUPERSEDES":
            continue
        sk, dk = _endpoint_key(ln.src), _endpoint_key(ln.dst)
        members.add(sk)
        members.add(dk)
        union(sk, dk)

    groups: Dict[str, List[str]] = {}
    for m in members:
        groups.setdefault(find(m), []).append(m)

    out: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
    for mem in groups.values():
        mem_sorted = tuple(sorted(mem))
        active = tuple(sorted(
            k[len("claim:"):] for k in mem
            if k.startswith("claim:") and status_map.get(k[len("claim:"):]) == "ACTIVE"
        ))
        out.append((mem_sorted, active))
    return sorted(out)


def _digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Fingerprint:
    component_digests: Dict[str, str]
    joint_digest: str
    readable: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"component_digests": dict(self.component_digests),
                "joint_digest": self.joint_digest, "readable": self.readable}


def fingerprint(brain: Brain) -> Fingerprint:
    """The canonical, order-invariant-by-construction fingerprint of ``brain``'s
    current semantically relevant state."""
    status_map = _status_map(brain)
    readable: Dict[str, Any] = {
        "live_claims": _live_claims(status_map),
        "status_map": sorted(status_map.items()),
        "links": [list(t) for t in _links(brain)],
        "representatives": [[list(m), list(a)] for m, a in
                            _representatives(brain, status_map)],
    }
    component_digests = {c: _digest(readable[c]) for c in _COMPONENTS}
    joint_digest = _digest(readable)
    return Fingerprint(component_digests=component_digests,
                       joint_digest=joint_digest, readable=readable)


def diff_fingerprints(a: Fingerprint, b: Fingerprint) -> Dict[str, Any]:
    """Concrete, inspectable diff of every component whose digest differs
    between ``a`` and ``b``. Empty dict iff the two fingerprints are identical."""
    out: Dict[str, Any] = {}

    if a.component_digests["live_claims"] != b.component_digests["live_claims"]:
        sa, sb = set(a.readable["live_claims"]), set(b.readable["live_claims"])
        out["live_claims"] = {"only_in_a": sorted(sa - sb), "only_in_b": sorted(sb - sa)}

    if a.component_digests["status_map"] != b.component_digests["status_map"]:
        ma, mb = dict(a.readable["status_map"]), dict(b.readable["status_map"])
        out["status_map"] = {k: [ma.get(k), mb.get(k)]
                             for k in sorted(set(ma) | set(mb)) if ma.get(k) != mb.get(k)}

    if a.component_digests["links"] != b.component_digests["links"]:
        la = Counter(tuple(x) for x in a.readable["links"])
        lb = Counter(tuple(x) for x in b.readable["links"])
        out["links"] = {"only_in_a": [list(x) for x in sorted((la - lb).elements())],
                        "only_in_b": [list(x) for x in sorted((lb - la).elements())]}

    if a.component_digests["representatives"] != b.component_digests["representatives"]:
        ra = {tuple(m): tuple(act) for m, act in a.readable["representatives"]}
        rb = {tuple(m): tuple(act) for m, act in b.readable["representatives"]}
        changed = {}
        for k in sorted(set(ra) | set(rb), key=lambda t: str(t)):
            if ra.get(k) != rb.get(k):
                changed[" | ".join(k)] = {"a": list(ra.get(k, ())), "b": list(rb.get(k, ()))}
        out["representatives"] = changed

    return out


__all__ = ["Fingerprint", "fingerprint", "diff_fingerprints"]
