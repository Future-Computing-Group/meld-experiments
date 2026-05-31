"""Python facade over the SMP-vendored links-first-class proto.

VENDORED -- REUSE-PENDING-RESTRUCTURE. SMP owns this copy of the
links-first-class memory model; the upstream firmament proto is upgraded later.
This module is a thin dataclass facade over the generated bindings in
``smp.proto._generated.fusilli.v1.memory_pb2`` (the same wrapper pattern the
firmament ``memory.types`` module uses), so the merge engine reasons over plain
Python objects with string enum names and bytes round-trips.

Exposes: MemoryUnit, Claim, MemoryLink, Endpoint, StalenessTag, plus the
LINK_TYPES / LINK_PROVENANCES / PRECEDENT_STATUSES / ORIGIN_TAGS /
AUTHORITY_LEVELS name sets and the deprecated Citation alias.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

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


@dataclass
class Endpoint:
    node_id: bytes = b""
    claim_key: bytes = b""


@dataclass
class StalenessTag:
    content_hash: bytes = b""
    phi: float = 0.0          # source_volatility (1/s)
    is_latest: bool = False


@dataclass
class Claim:
    claim_key: bytes = b""
    text: str = ""
    confidence: float = 0.0
    claim_status: str = "ACTIVE"


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


@dataclass
class Citation:
    """DEPRECATED node-centric alias kept for migration; first-class path is
    a GROUNDS MemoryLink. Do not author new code against this."""
    cited_memory_id: bytes = b""
    extracted_claim: str = ""
    entailment_score: float = 0.0


@dataclass
class MemoryUnit:
    id: bytes = b""
    owner_did: bytes = b""
    confidence: float = 0.0
    origin: str = "L0_ECHO_FREE"
    authority: str = "OBSERVATION"
    status: str = "ACTIVE"
    relevance_magnitude: float = 0.0
    contradictions_received: int = 0
    access_count: int = 0
    signature: bytes = b""
    claims: List[Claim] = field(default_factory=list)
    staleness: Optional[StalenessTag] = None
    citations: List[Citation] = field(default_factory=list)  # deprecated alias


__all__ = [
    "MemoryUnit", "Claim", "MemoryLink", "Endpoint", "StalenessTag", "Citation",
    "LINK_TYPES", "LINK_PROVENANCES", "PRECEDENT_STATUSES", "ORIGIN_TAGS",
    "AUTHORITY_LEVELS",
]
