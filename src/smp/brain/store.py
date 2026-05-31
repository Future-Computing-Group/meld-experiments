"""The brain store: a memory-store replica of a world-model graph.

A "brain" holds Concepts (MemoryUnits with addressable Claims) and Mappings
(first-class MemoryLinks), plus a per-claim status CRDT state. The ONLY thing
that mutates a brain is applying a Patch (the merge decision procedure's output):
a raw incoming delta applied directly is a no-op. The brain can extract a delta
(the version set) for PUBLISH, and answer a graph-walk read that fuses lexical,
dense, and graph-relevance channels (Def. 2/3 of the restructure note).

REUSE posture: this mirrors the firmament LocalEncryptedStore semantics
(content-addressed units, append-only, origin-immutable) over the SMP-vendored
proto. SMP keeps its own in-process replica so it is not coupled to the
firmament generated bindings' protobuf version; the wire types are the vendored
links-first-class copy. NEW (thin): delta apply/extract + graph-walk read.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from smp.merge.signals import EmbeddingSimilarity, _tokenize
from smp.merge.status_crdt import StatusState
from smp.proto import Claim, MemoryLink, MemoryUnit
from smp.protocol.messages import Patch


class Brain:
    """In-process world-model graph replica."""

    def __init__(self, encoder: str = "hashing", cache_dir: Optional[str] = None):
        self._units: Dict[bytes, MemoryUnit] = {}
        self._claims: Dict[bytes, Claim] = {}
        self.links: List[MemoryLink] = []
        self._status: Dict[bytes, StatusState] = {}
        self.audit_log: List[Patch] = []
        self.changed = False
        self._encoder_name = encoder
        self._es = EmbeddingSimilarity(encoder=encoder, cache_dir=cache_dir)

    # -- read ------------------------------------------------------------
    def get_unit(self, unit_id: bytes) -> Optional[MemoryUnit]:
        return self._units.get(unit_id)

    def claim_status(self, claim_key: bytes) -> StatusState:
        return self._status.setdefault(claim_key, StatusState())

    # -- mutation: ONLY via Patch ---------------------------------------
    def apply(self, patch: Patch) -> None:
        """Apply a Patch. REJECT is a recorded no-op; MERGE/RELATE/CONFLICT
        emit their deltas. The Patch (not the raw delta) mutates state."""
        self.audit_log.append(patch)
        if patch.decision == "REJECT":
            return
        for item in patch.emitted:
            if isinstance(item, MemoryUnit):
                self._put_unit(item)
            elif isinstance(item, MemoryLink):
                self.links.append(item)
            elif isinstance(item, tuple) and len(item) == 3 and item[0] == "STATUS":
                # ("STATUS", claim_key, (link_id, effect))
                _, ck, (lid, eff) = item
                self._status[ck] = self.claim_status(ck).add(lid, eff)

    def receive_raw_delta(self, units) -> None:
        """A raw delta on the wire does NOT mutate state -- it must go through
        the merge decision procedure and be applied as a Patch. No-op here."""
        return None

    def _put_unit(self, unit: MemoryUnit) -> None:
        # content-addressed, append-only-ish: re-put of same id updates in place
        self._units[unit.id] = unit
        for c in unit.claims:
            self._claims[c.claim_key] = c
            self._status.setdefault(c.claim_key, StatusState())
        self.changed = True

    # -- extract for PUBLISH --------------------------------------------
    def extract_delta(self) -> List[MemoryUnit]:
        """The current version set to PUBLISH (units + links)."""
        return list(self._units.values()) + list(self.links)

    # -- graph-walk read (Def. 2/3 fusion) ------------------------------
    def graph_walk_read(self, query: str, *, encoder: str, k: int = 5,
                        beta_lex: float = 0.3, beta_dense: float = 0.5,
                        beta_graph: float = 0.2) -> List[Tuple[bytes, float]]:
        """Fuse lexical (token overlap), dense (cosine), and graph-relevance
        channels into a ranked list of (unit_id, score). The graph channel adds
        one hop of typed-link propagation from seed units (active links only)."""
        es = self._es if encoder == self._encoder_name else EmbeddingSimilarity(encoder=encoder)
        q_tokens = set(_tokenize(query))
        dense: Dict[bytes, float] = {}
        lex: Dict[bytes, float] = {}
        for uid, unit in self._units.items():
            text = " ".join(c.text for c in unit.claims)
            dense[uid] = es.sigma(query, text) if text else 0.0
            u_tokens = set(_tokenize(text))
            lex[uid] = (len(q_tokens & u_tokens) / len(q_tokens)) if q_tokens else 0.0
        # one-hop graph propagation over ACTIVE links
        graph: Dict[bytes, float] = {uid: 0.0 for uid in self._units}
        for ln in self.links:
            if ln.status != "ACTIVE":
                continue
            src, dst = ln.src.node_id, ln.dst.node_id
            if src in dense and dst in graph:
                graph[dst] = max(graph[dst], dense[src] * ln.weight)
        scored = []
        for uid in self._units:
            score = (beta_lex * lex[uid] + beta_dense * dense[uid]
                     + beta_graph * graph.get(uid, 0.0))
            scored.append((uid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


__all__ = ["Brain"]
