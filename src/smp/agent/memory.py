"""SMPMemory: an SMP brain used as the agent's memory backend.

This is the load-bearing G3 integration. The agent does NOT keep a private
scratchpad; every observation it writes goes through the SMP wire path:

    text -> MemoryUnit (Concept + Claim)  [PUBLISH a delta]
         -> merge_on_receive(incoming, local)  [the three-signal (+ optional
            evidence overlap) + Context-gate decision procedure]
         -> Patch  [MERGE / RELATE / CONFLICT / REJECT, auditable]
         -> brain.apply(Patch)  [the ONLY thing that mutates the brain]

and every recall is the brain's graph-walk read (lexical + dense + one-hop graph
fusion). So running a GAIA task against this memory exercises the SMP merge
integration, not just a bare agent. A contradicting write yields a CONFLICT
(a first-class CONTRADICTS edge), never a silent overwrite.
"""
from __future__ import annotations

import hashlib
from typing import List

from smp.brain.store import Brain
from smp.config import Thresholds, load_thresholds
from smp.merge.signals import EmbeddingSimilarity
from smp.protocol.merge_on_receive import merge_on_receive
from smp.protocol.messages import Patch
from smp.proto import Claim, MemoryUnit


def _cid(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()[:12]


def _unit_from_text(text: str) -> MemoryUnit:
    """A content-addressed Concept (MemoryUnit) carrying one Claim."""
    return MemoryUnit(
        id=_cid(text),
        relevance_magnitude=1.0,
        status="ACTIVE",
        claims=[Claim(claim_key=_cid(text + "::claim"), text=text,
                      confidence=0.9, claim_status="ACTIVE")],
    )


class SMPMemory:
    """Agent memory backed by an SMP ``Brain``.

    ``encoder``/``nli`` are required (no silent default), matching the SMP
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
        incoming = _unit_from_text(text)
        local = [u for u in self.brain.extract_delta() if isinstance(u, MemoryUnit)]
        patches = merge_on_receive(
            incoming, local, cfg=self.cfg, es=self._es, nli=self.nli)
        for p in patches:
            self.brain.apply(p)
        return patches

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
