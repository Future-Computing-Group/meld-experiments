"""SMPMemory: a MELD replica used as the agent's memory backend.

The agent keeps no private scratchpad. Every observation it writes is a signed claim delta
that enters the replica's sets through the MAC-verified receive gate and is admitted by the
latency path (protocol/admission.py: the fold's step under the active epoch, or the policy's
bootstrap selector without one, which makes no similarity merge):

    text -> Delta (claim key, int8 embedding, created_at)  [PUBLISH]
         -> Store.apply                                    [MAC-verified insertion into D]
         -> Admission.admit                                [INSERT / MERGE / RELATE / CONFLICT / REJECT]

and every recall is the store's graph-walk read over live nodes. A contradicting write yields
CONFLICT (a CONTRADICTS link, both claims live), never a silent overwrite.

``DEFAULT_POL`` is the M_pol of a single-cell replica (the agent memory and the default
BrainNode): one scope holding the default cell, the proto authority levels as ranks, every
effect kind authorized at any rank, the calibration authority ``"calibration"``, R_min one
fixed-point unit, and a bootstrap selector. The selector only decides which pairs are read
before an epoch exists; theta is +inf without an epoch, so it is no merge operating point.
"""
from __future__ import annotations

import platform
import time
from typing import Callable, List, Optional

from meld.brain.store import Store
from meld.merge.classifier import Decision
from meld.merge.signals import EmbeddingSimilarity, quantize_int8
from meld.proto import AUTHORITY_LEVELS, CANON_VERSION, _lp, canon, claim_key, h16
from meld.protocol.admission import Admission
from meld.protocol.fold import nli_provider
from meld.protocol.objects import (
    EFFECT_KINDS, FIXED_ONE, Delta, EpochRecord, m_nli, object_id, signed,
)
from meld.sync.wire_delta import group_key

DEFAULT_CELL = "agent.public"
DEFAULT_SCOPE = "agent"
DEFAULT_POL = {
    "rank_map": dict(AUTHORITY_LEVELS), "scopes": {DEFAULT_SCOPE: [DEFAULT_CELL]},
    "auth_policy": {**{k: max(AUTHORITY_LEVELS.values()) for k in EFFECT_KINDS},
                    "calibration_authority": {DEFAULT_SCOPE: "calibration"}},
    "step_version": 0, "r_min": 1,
    "bootstrap_selector": [round(0.5 * FIXED_ONE), round(0.5 * FIXED_ONE), 3],
}


def _cid(text: str, cell: str = DEFAULT_CELL) -> bytes:
    """Concept (unit) id: H over length-prefixed text and cell, tagged apart from keys."""
    return h16(b"U\x1f" + _lp(text) + _lp(cell))


class SMPMemory:
    """Agent memory backed by one MELD replica (``store``) and its admission path.

    ``encoder``/``nli`` are required (no silent default). ``epoch`` is the scope's epoch
    record (author ``"calibration"``, M_nli ``self.m_nli``, M_pol of ``pol``); without one
    the memory makes no similarity merge. ``provider`` is the signal provider (default
    ``nli_provider(nli)``; a multi-replica harness passes one with its own model instance).
    ``decisions`` is the admission audit log, one
    entry per arrival; ``admission.decisions(scope)`` is every node's current decision
    (a later out-of-order arrival can refold earlier ones)."""

    def __init__(self, *, encoder: str, nli: str, epoch: Optional[EpochRecord] = None,
                 cache_dir: Optional[str] = None, pol: Optional[dict] = None,
                 key: Optional[bytes] = None, origin: str = "agent",
                 clock: Optional[Callable[[], int]] = None, provider=None):
        self.encoder, self.nli, self.origin = encoder, nli, origin
        self.pol = pol or DEFAULT_POL
        self.key = key or group_key({})
        self.m_nli = m_nli(encoder, nli, "int8", CANON_VERSION)
        self.store = Store(self.pol, self.key, encoder=encoder)
        self.admission = Admission(self.store, m_nli=self.m_nli, provider=provider or nli_provider(nli),
                                   issuer=origin, hw=platform.machine() or "unknown")
        self._es = EmbeddingSimilarity(encoder=encoder, cache_dir=cache_dir)
        self._clock = clock or (lambda: time.time_ns() // 1000)
        self._t = 0
        self.decisions: List[Decision] = []
        if epoch is not None and self.store.apply(*signed(epoch, self.key)) is None:
            raise ValueError("epoch refused by the memory's sets")

    def delta(self, text: str, cell: str = DEFAULT_CELL) -> Delta:
        """The claim delta a local observation publishes. created_at is ``clock()`` (default
        the wall clock in microseconds; a shared counter makes a build reproducible), made
        strictly increasing per memory, so local writes arrive in order."""
        self._t = max(self._clock(), self._t + 1)
        return Delta(unit_id=_cid(text, cell), claim_key=claim_key(text, cell), text=text,
                     cell=cell, version=0, auth=self.pol["rank_map"]["OBSERVATION"],
                     status="ACTIVE", retention=FIXED_ONE, created_at=self._t,
                     origin=self.origin, encoder=self.encoder,
                     q=quantize_int8(self._es._embed(canon(text))).tobytes())

    def publish(self, text: str) -> Optional[Delta]:
        """Insert a local observation into D without admission (the naive-union ablation)."""
        text = (text or "").strip()
        return self.store.apply(*signed(self.delta(text), self.key)) if text else None

    # -- write: PUBLISH -> receive gate -> admission ----------------------------------
    def _admit(self, d) -> List[Decision]:
        dec = self.admission.admit(d)
        if dec is None:
            return []
        self.decisions.append(dec)
        return [dec]

    def write(self, text: str) -> List[Decision]:
        """Publish one observation and admit it; the admission decision (none for an exact
        copy of a held claim)."""
        d = self.publish(text)
        return self._admit(d) if d is not None else []

    def receive(self, received: bytes, mac: str) -> List[Decision]:
        """The receive gate for a peer's object: MAC-verified insertion; a newly held delta
        is admitted. Redelivery and dropped objects decide nothing."""
        new = object_id(received) not in self.store.objects
        obj = self.store.apply(received, mac)
        return self._admit(obj) if new and isinstance(obj, Delta) else []

    # -- read: graph-walk fusion ------------------------------------------------------
    def texts(self) -> dict:
        """unit id -> claim text of every node."""
        return {d.unit_id: d.text for _, d in self.store.nodes().values()}

    def recall(self, query: str, k: int = 3) -> List[str]:
        """Up to ``k`` claim texts ranked by the store's graph-walk read over live nodes."""
        texts = self.texts()
        return [texts[u] for u, _ in self.store.graph_walk_read(query, encoder=self.encoder, k=k)]

    def footprint(self) -> dict:
        """Stored nodes and live (recallable) nodes."""
        return {"stored": len(self.store.nodes()), "live": len(self.store.live_nodes())}


__all__ = ["SMPMemory", "DEFAULT_CELL", "DEFAULT_SCOPE", "DEFAULT_POL", "signed"]
