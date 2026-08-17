"""The six MELD protocol primitives and the Patch wire object (architecture A.1.2).

Five primitives REUSE the vendored proto types:
    Concept  -> MemoryUnit (node)
    Claim    -> Claim (addressable assertion)
    Evidence -> a GROUNDS MemoryLink (+ grounded unit)
    Context  -> Endpoint + status-derived freshness (the frame a claim is valid in)
    Mapping  -> a typed MemoryLink (ASSOCIATES/ENTITY_LINK/SUPERSEDES/...)

The sixth, Patch (a MergeResult), is the NEW wire object: the reified output of
the merge decision procedure. It is append-only (frozen), signed, versioned, and
records exactly which gates fired, so a merge is auditable and replayable. A
Patch -- not the raw incoming delta -- is the only thing that mutates state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from meld.proto import Endpoint

# the merge decision procedure's outputs (architecture A.2 / manuscript Sec. 4.2)
# NO_RELATION is deliberately NOT here: it is a pair-adjudication CLASSIFIER
# verdict (meld.merge.classifier), never a Patch decision -- the protocol layer
# (merge_on_receive) maps an unreached NO_RELATION to INSERT before a Patch is
# ever constructed, and __post_init__ below enforces exactly these five.
DECISIONS = ("INSERT", "MERGE", "RELATE", "CONFLICT", "REJECT", "STATUS")
# STATUS: the append of an already-authored status link (overrule/supersede/
# revoke) into the per-claim CRDT; not a claim-admission outcome (Sec. 3.1).

# the five primitive names that REUSE proto types (Patch is the sixth, NEW)
PRIMITIVE_PROTO_MAP = {
    "Concept": "MemoryUnit",
    "Claim": "Claim",
    "Evidence": "MemoryLink(type=GROUNDS)",
    "Context": "Endpoint+status-freshness",
    "Mapping": "MemoryLink(typed)",
}


@dataclass(frozen=True)
class Patch:
    """Signed, append-only MergeResult. The verb's response; what mutates state.

    ``gates_that_fired`` records the justification (e.g. ["kappa","context"],
    ["chi"]) so other brains can audit/replay the merge. ``version`` is the
    Patch SCHEMA/format version (a wire-format constant, NOT a per-claim CRDT
    version: per-claim status convergence is owned by the status-CRDT, and the
    emitted MemoryLinks carry their own per-link versions). Provenance is NOT a
    Patch-level field -- each emitted MemoryLink carries its own ``provenance``.
    """

    decision: str
    target: Endpoint
    emitted: List[Any] = field(default_factory=list)        # units/links/status UPDATEs
    gates_that_fired: List[str] = field(default_factory=list)
    version: int = 0                                          # schema/format version (constant)
    signature: bytes = b""

    def __post_init__(self):
        if self.decision not in DECISIONS:
            raise ValueError(
                f"decision must be one of {DECISIONS}, got {self.decision!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "target": {"node_id": self.target.node_id, "claim_key": self.target.claim_key},
            "emitted": list(self.emitted),
            "gates_that_fired": list(self.gates_that_fired),
            "version": self.version,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Patch":
        t = d["target"]
        return cls(
            decision=d["decision"],
            target=Endpoint(node_id=t["node_id"], claim_key=t["claim_key"]),
            emitted=list(d.get("emitted", [])),
            gates_that_fired=list(d.get("gates_that_fired", [])),
            version=d.get("version", 0),
            signature=d.get("signature", b""),
        )


__all__ = ["Patch", "DECISIONS", "PRIMITIVE_PROTO_MAP"]
