"""SMPMemory: an MELD brain used as the agent's memory backend.

This is the load-bearing G3 integration. The agent does NOT keep a private
scratchpad; every observation it writes goes through the MELD wire path:

    text -> MemoryUnit (Concept + Claim)  [PUBLISH a delta]
         -> merge_on_receive(incoming, local)  [the three-signal (+ optional
            evidence overlap) + Context-gate decision procedure]
         -> Patch  [INSERT / MERGE / RELATE / CONFLICT / REJECT, auditable]
         -> brain.apply(Patch)  [the ONLY thing that mutates the brain]

and every recall is the brain's graph-walk read (lexical + dense + one-hop graph
fusion). So running a GAIA task against this memory exercises the MELD merge
integration, not just a bare agent. A contradicting write yields a CONFLICT
(a first-class CONTRADICTS edge), never a silent overwrite.
"""
from __future__ import annotations

import hashlib
from typing import List

from meld.brain.store import Brain
from meld.config import Thresholds, load_thresholds
from meld.merge.signals import EmbeddingSimilarity
from meld.protocol.merge_on_receive import merge_on_receive
from meld.protocol.messages import Patch
from meld.proto import Claim, MemoryUnit


DEFAULT_CELL = "agent.public"
_SEP = b"\x1f"


def _digest(tag: bytes, text: str, cell: str) -> bytes:
    """Content address over (text, cell). Length-prefixing the text makes the
    concatenation unambiguous, so ("a\x1fb", "c") and ("a", "b\x1fc") cannot collide."""
    t = text.encode("utf-8")
    return hashlib.sha256(tag + _SEP + len(t).to_bytes(4, "big") + t
                          + cell.encode("utf-8")).digest()[:12]


def _cid(text: str, cell: str = DEFAULT_CELL) -> bytes:
    """Concept (MemoryUnit) id."""
    return _digest(b"U", text, cell)


def claim_key(text: str, cell: str = DEFAULT_CELL) -> bytes:
    """Claim key = H(canonical content || discrete scope) (paper §4.1). Only the
    DISCRETE cell enters the key; validity interval and freshness stay in the
    Context gate, since hashing a continuous field would break key identity on any
    timestamp delta."""
    return _digest(b"C", text, cell)


def _unit_from_text(text: str, cell: str = DEFAULT_CELL) -> MemoryUnit:
    """A content-and-scope-addressed Concept (MemoryUnit) carrying one Claim."""
    return MemoryUnit(
        id=_cid(text, cell),
        cell=cell,
        relevance_magnitude=1.0,
        status="ACTIVE",
        claims=[Claim(claim_key=claim_key(text, cell), text=text,
                      confidence=0.9, claim_status="ACTIVE")],
    )


class SMPMemory:
    """Agent memory backed by an MELD ``Brain``.

    ``encoder``/``nli`` are required (no silent default), matching the MELD
    no-silent-default discipline. ``cfg`` defaults to the tracked thresholds;
    a calibrated config may be passed for the gated run.
    """

    def __init__(self, *, encoder: str, nli: str,
                 cfg: Thresholds | None = None, cache_dir: str | None = None):
        self.encoder = encoder
        self.nli = nli
        self.cfg = cfg or load_thresholds()
        self.brain = Brain(encoder=encoder, cache_dir=cache_dir)
        self._es = EmbeddingSimilarity(encoder=encoder, cache_dir=cache_dir)

    # -- write: PUBLISH -> merge-on-receive -> Patch -> apply ----------------
    def write(self, text: str) -> List[Patch]:
        """Publish one observation into the brain via the merge decision
        procedure. Returns the emitted Patch(es)."""
        text = (text or "").strip()
        if not text:
            return []
        return self.write_unit(_unit_from_text(text))

    def write_unit(self, incoming: MemoryUnit) -> List[Patch]:
        """Merge an already-typed incoming MemoryUnit through the SAME decision
        procedure ``write`` uses. The wire path deserializes a typed delta and
        calls this, so a received typed delta merges identically to a local
        text write (the only difference is where the unit was constructed)."""
        # §5.3/§5.5: only STATUS-LIVE local units are admission targets. extract_delta()
        # returns every unit regardless of CRDT status; merge-deprecation is recorded in
        # Brain._status, not on the unit/claim fields, so the field-level filter inside
        # merge_on_receive alone cannot see it. _unit_is_live is the SAME predicate the
        # graph-walk read already uses (store.py), so candidacy and recall agree by
        # construction, and a DEPRECATED/OVERRULED unit cannot be revived by a peer.
        local = [u for u in self.brain.extract_delta()
                if isinstance(u, MemoryUnit) and self.brain._unit_is_live(u)]
        patches = merge_on_receive(
            incoming, local, cfg=self.cfg, es=self._es, nli=self.nli)
        for p in patches:
            self.brain.apply(p)
        return patches

    def build_unit(self, text: str) -> MemoryUnit:
        """The typed Concept (MemoryUnit) a local observation publishes -- the
        same unit ``write`` would merge, exposed so the sync layer can serialize
        a TYPED delta on the wire instead of bare text."""
        return _unit_from_text((text or "").strip())

    # -- read: graph-walk fusion --------------------------------------------
    def recall(self, query: str, k: int = 3) -> List[str]:
        """Return up to ``k`` recalled claim texts, ranked by the brain's
        graph-walk read (lexical + dense + one-hop graph)."""
        hits = self.brain.graph_walk_read(query, encoder=self.encoder, k=k)
        texts: List[str] = []
        for uid, _score in hits:
            unit = self.brain.get_unit(uid)
            if unit and unit.claims:
                texts.append(" ".join(c.text for c in unit.claims))
        return texts


__all__ = ["SMPMemory"]
