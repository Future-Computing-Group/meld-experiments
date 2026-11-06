"""Python facade over the MELD-vendored links-first-class proto.

VENDORED -- REUSE-PENDING-RESTRUCTURE. MELD owns this copy of the
links-first-class memory model; the upstream firmament proto is upgraded later.
This module is a thin dataclass facade over the generated bindings in
``meld.proto._generated.fusilli.v1.memory_pb2`` (the same wrapper pattern the
firmament ``memory.types`` module uses), so the merge engine reasons over plain
Python objects with string enum names and bytes round-trips.

Exposes: MemoryUnit, Claim, MemoryLink, Endpoint, plus the
LINK_TYPES / LINK_PROVENANCES / PRECEDENT_STATUSES / ORIGIN_TAGS /
AUTHORITY_LEVELS name sets.

Freshness is a derived property of the per-claim/unit ``status`` field
(``is_fresh = status == "ACTIVE"``), so there is no standalone staleness type:
status is read directly (R1 admissibility in brain/store.py).
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import List

# Make the generated package importable as a top-level `fusilli.v1` path,
# matching how the bindings were compiled (`-I proto`).
_GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_generated")
if _GEN not in sys.path:
    sys.path.insert(0, _GEN)

from fusilli.v1 import memory_pb2 as _pb  # noqa: E402

# --- enum name<->number maps (string names are the public surface) ---
LINK_TYPES = {k: v for k, v in _pb.LinkType.items()}
LINK_PROVENANCES = {k: v for k, v in _pb.LinkProvenance.items()}
PRECEDENT_STATUSES = {k: v for k, v in _pb.PrecedentStatus.items()}
ORIGIN_TAGS = {k: v for k, v in _pb.OriginTag.items()}
AUTHORITY_LEVELS = {k: v for k, v in _pb.AuthorityLevel.items()}


def _b2h(b: bytes) -> str:
    return b.hex()


def _h2b(s: str) -> bytes:
    return bytes.fromhex(s) if s else b""


def h16(b: bytes) -> bytes:
    """H of design §2.1: SHA-256 truncated to 16 bytes."""
    return hashlib.sha256(b).digest()[:16]


def _lp(s: str) -> bytes:
    b = s.encode("utf-8")
    return len(b).to_bytes(4, "big") + b


CANON_VERSION = "identity-v1"   # enters M_nli; bump when canon() changes


def canon(text: str) -> str:
    """The canonicaliser the claim key and the NLI input share (identity; callers strip)."""
    return text


def claim_key(text: str, cell: str = "agent.public", version: int = 0) -> bytes:
    """c = H(canon(text) || cell || version) (design §2.1). Text and cell are length-prefixed
    and the version is fixed-width, so no two (text, cell, version) triples share a byte
    string. Only the discrete cell enters the key."""
    return h16(b"C\x1f" + _lp(canon(text)) + _lp(cell) + version.to_bytes(8, "big"))


@dataclass
class Endpoint:
    node_id: bytes = b""
    claim_key: bytes = b""

    def to_dict(self) -> dict:
        return {"node_id": _b2h(self.node_id), "claim_key": _b2h(self.claim_key)}

    @classmethod
    def from_dict(cls, d: dict) -> "Endpoint":
        return cls(node_id=_h2b(d.get("node_id", "")),
                   claim_key=_h2b(d.get("claim_key", "")))


@dataclass
class Claim:
    claim_key: bytes = b""
    text: str = ""
    confidence: float = 0.0
    claim_status: str = "ACTIVE"
    # Grounding-evidence endpoints (e.g. "doc:sentence" provenance) the claim is
    # derived from; the source of the OPTIONAL epsilon (evidence-overlap) signal,
    # which is disabled by default (eps_hi=0). Empty on ungrounded corpora
    # (HotpotQA) -> epsilon=0 -> the epsilon path is inert regardless.
    grounding: tuple = ()
    version: int = 0

    def to_dict(self) -> dict:
        return {"claim_key": _b2h(self.claim_key), "text": self.text,
                "confidence": self.confidence, "claim_status": self.claim_status,
                "grounding": list(self.grounding), "version": self.version}

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        return cls(claim_key=_h2b(d.get("claim_key", "")), text=d.get("text", ""),
                   confidence=d.get("confidence", 0.0),
                   claim_status=d.get("claim_status", "ACTIVE"),
                   grounding=tuple(d.get("grounding", ())),
                   version=d.get("version", 0))


@dataclass
class MemoryLink:
    type: str = "ASSOCIATES"
    src: Endpoint = field(default_factory=Endpoint)
    dst: Endpoint = field(default_factory=Endpoint)
    weight: float = 0.0
    provenance: str = "INFERRED"
    version: int = 0
    signature: bytes = b""
    origin: str = "L0_ECHO_FREE"
    authority: str = "OBSERVATION"
    status: str = "ACTIVE"
    entailment_score: float = 0.0
    id: bytes = b""

    def to_bytes(self) -> bytes:
        pb = _pb.MemoryLink(
            id=self.id,
            type=LINK_TYPES[self.type],
            weight=self.weight,
            provenance=LINK_PROVENANCES[self.provenance],
            version=self.version,
            signature=self.signature,
            origin=ORIGIN_TAGS[self.origin],
            authority=AUTHORITY_LEVELS[self.authority],
            status=PRECEDENT_STATUSES[self.status],
            entailment_score=self.entailment_score,
            src=_pb.Endpoint(node_id=self.src.node_id, claim_key=self.src.claim_key),
            dst=_pb.Endpoint(node_id=self.dst.node_id, claim_key=self.dst.claim_key),
        )
        return pb.SerializeToString()

    @classmethod
    def from_bytes(cls, blob: bytes) -> "MemoryLink":
        pb = _pb.MemoryLink()
        pb.ParseFromString(blob)
        return cls(
            id=pb.id,
            type=_pb.LinkType.Name(pb.type),
            weight=pb.weight,
            provenance=_pb.LinkProvenance.Name(pb.provenance),
            version=pb.version,
            signature=pb.signature,
            origin=_pb.OriginTag.Name(pb.origin),
            authority=_pb.AuthorityLevel.Name(pb.authority),
            status=_pb.PrecedentStatus.Name(pb.status),
            entailment_score=pb.entailment_score,
            src=Endpoint(node_id=pb.src.node_id, claim_key=pb.src.claim_key),
            dst=Endpoint(node_id=pb.dst.node_id, claim_key=pb.dst.claim_key),
        )

    def to_dict(self) -> dict:
        return {"id": _b2h(self.id), "type": self.type,
                "src": self.src.to_dict(), "dst": self.dst.to_dict(),
                "weight": self.weight, "provenance": self.provenance,
                "version": self.version, "signature": _b2h(self.signature),
                "origin": self.origin, "authority": self.authority,
                "status": self.status, "entailment_score": self.entailment_score}

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryLink":
        return cls(
            id=_h2b(d.get("id", "")), type=d.get("type", "ASSOCIATES"),
            src=Endpoint.from_dict(d.get("src", {})),
            dst=Endpoint.from_dict(d.get("dst", {})),
            weight=d.get("weight", 0.0), provenance=d.get("provenance", "INFERRED"),
            version=d.get("version", 0), signature=_h2b(d.get("signature", "")),
            origin=d.get("origin", "L0_ECHO_FREE"),
            authority=d.get("authority", "OBSERVATION"),
            status=d.get("status", "ACTIVE"),
            entailment_score=d.get("entailment_score", 0.0))


@dataclass
class MemoryUnit:
    id: bytes = b""
    owner_did: bytes = b""
    confidence: float = 0.0
    origin: str = "L0_ECHO_FREE"
    authority: str = "OBSERVATION"
    status: str = "ACTIVE"
    contradictions_received: int = 0
    access_count: int = 0
    signature: bytes = b""
    # The Context cell/scope frame the unit's claims live in. The Context gate
    # (architecture A.2) reads this on the live federation path: two claims in
    # DIFFERENT cells are frame-incompatible and RELATE rather than MERGE, even at
    # high similarity. Defaults to the single shared frame, preserving existing
    # behavior (M-2/headline use one cell).
    cell: str = "agent.public"
    claims: List[Claim] = field(default_factory=list)
    created_at: int = 0          # integer microseconds since the Unix epoch, set by the publisher
    encoder: str = ""            # encoder id of q
    q: bytes = b""               # int8 quantised embedding under encoder (may be empty)
    retention: float = field(kw_only=True)   # carried retention R; required, no default

    def to_dict(self) -> dict:
        return {"id": _b2h(self.id), "owner_did": _b2h(self.owner_did),
                "confidence": self.confidence, "origin": self.origin,
                "authority": self.authority, "status": self.status,
                "retention": self.retention,
                "created_at": self.created_at, "encoder": self.encoder,
                "q": _b2h(self.q),
                "contradictions_received": self.contradictions_received,
                "access_count": self.access_count,
                "signature": _b2h(self.signature),
                "cell": self.cell,
                "claims": [c.to_dict() for c in self.claims]}

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryUnit":
        return cls(
            id=_h2b(d.get("id", "")), owner_did=_h2b(d.get("owner_did", "")),
            confidence=d.get("confidence", 0.0), origin=d.get("origin", "L0_ECHO_FREE"),
            authority=d.get("authority", "OBSERVATION"),
            status=d.get("status", "ACTIVE"),
            retention=d["retention"],
            created_at=d.get("created_at", 0), encoder=d.get("encoder", ""),
            q=_h2b(d.get("q", "")),
            contradictions_received=d.get("contradictions_received", 0),
            access_count=d.get("access_count", 0),
            signature=_h2b(d.get("signature", "")),
            cell=d.get("cell", "agent.public"),
            claims=[Claim.from_dict(c) for c in d.get("claims", [])])


__all__ = [
    "MemoryUnit", "Claim", "MemoryLink", "Endpoint", "claim_key", "h16", "canon", "CANON_VERSION",
    "LINK_TYPES", "LINK_PROVENANCES", "PRECEDENT_STATUSES", "ORIGIN_TAGS",
    "AUTHORITY_LEVELS",
]
