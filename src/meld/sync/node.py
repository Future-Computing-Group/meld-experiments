"""BrainNode: the integration kernel deployed per host in the multi-host sync runs.

A node is one MELD replica (``mem``: an SMPMemory over a Store holding the five replicated
sets of its subscribed scopes, design §2.1) plus the publication bookkeeping of §2.2:

  * ``publish_local(obj)``  -- sign a locally authored object (any of the five kinds) and
                               insert it into the own sets, admitting it like a received one;
                               ``ingest(text)`` does this for a Delta in the node's cell.
  * ``outbox()``            -- ids to PUBLISH: own objects not yet published (Sig records
                               the admission path evaluated included). Received ids are
                               marked published on receipt (echo guard keyed by id).
  * ``receive_object(...)`` -- the receive gate: MAC-verified insertion into the sets, then
                               the admission hook ``admit(kind, obj)`` for a newly inserted
                               object. The default hook admits a delta through
                               protocol/admission.py (the active epoch, or the policy's
                               bootstrap selector: no similarity merge without an epoch).
                               Redelivery is idempotent at the set level and never re-admits.

The epoch is a replicated object: pass ``epoch`` (signed with the node's key) or receive the
calibration authority's signed record (config.load_epoch). There is no threshold config.

Read side: ``live_claim_texts`` (live nodes), ``status_of`` (vis of a claim
key). ``apply_status`` authors a signed effect (OVERRULE, REVOKE, or SUPERSEDE to the next
version for DEPRECATED) through ``publish_local``: a status change is an authored, MAC'd
object, never an unsigned local write.
"""
from __future__ import annotations

import threading
from typing import Callable, List, Optional, Set

from meld.agent.memory import DEFAULT_CELL, DEFAULT_POL, DEFAULT_SCOPE, SMPMemory
from meld.proto import claim_key
from meld.protocol.objects import (
    AuthoredEffect, EpochRecord, object_id, signed,
)
from meld.sync.objectset import KINDS
from meld.sync.wire_delta import group_key, pack

_EFFECT_KIND = {"OVERRULED": "OVERRULE", "REVOKED": "REVOKE", "DEPRECATED": "SUPERSEDE"}


def status_effect(claim_text: str, *, link_id: str, effect: str,
                  scope: str = DEFAULT_SCOPE) -> AuthoredEffect:
    """A status change on the claim (default cell, version 0) as an AuthoredEffect authored by
    ``link_id``; DEPRECATED is SUPERSEDE to the claim's next version."""
    kind = _EFFECT_KIND[effect]
    return AuthoredEffect(
        scope=scope, kind=kind, c1=claim_key(claim_text, DEFAULT_CELL),
        c2=claim_key(claim_text, DEFAULT_CELL, 1) if kind == "SUPERSEDE" else b"",
        target_cell=DEFAULT_CELL, author=link_id, author_cell=DEFAULT_CELL,
        author_auth=0, created_at=0)


def status_envelope(claim_text: str, *, link_id: str, effect: str, key: bytes) -> bytes:
    """The /patch body that PUBLISHes ``status_effect`` signed under the group key (the eval-c
    drivers' status write; POST /status is retired)."""
    e = status_effect(claim_text, link_id=link_id, effect=effect)
    return pack([(*signed(e, key), e.scope)])


class BrainNode:
    def __init__(self, node_id: str, *, encoder: str, nli: str,
                 epoch: Optional[EpochRecord] = None, cache_dir: Optional[str] = None,
                 pol: Optional[dict] = None, key: Optional[bytes] = None,
                 admit: Optional[Callable] = None, provider=None,
                 clock: Optional[Callable[[], int]] = None):
        self.node_id = node_id
        self._mem_kw = dict(encoder=encoder, nli=nli, epoch=epoch, cache_dir=cache_dir,
                            pol=pol or DEFAULT_POL, key=key or group_key({}), origin=node_id,
                            provider=provider, clock=clock)
        self.admit = admit or (lambda kind, obj: self.mem._admit(obj) if kind == "D" else None)
        # ThreadingHTTPServer handles requests concurrently: insertion + admission and the
        # outbox are one critical section, so a redelivery race admits exactly once.
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        """Drop all state -> a fresh replica + empty published-set, so a campaign cell starts
        isolated (seeds are independent replicates; each RTT regime is actually re-measured
        rather than masked by publish-once leftover state from the previous cell)."""
        with self._lock:
            self.mem = SMPMemory(**self._mem_kw)
            self.pol, self.key = self.mem.pol, self.mem.key
            self.objects = self.mem.store.objects
            self.scope_of = self.objects.scope_of
            self._published: Set[bytes] = set()   # object ids published OR received

    # -- replication layer ---------------------------------------------------
    def publish_local(self, obj) -> bytes:
        """Sign a locally authored object, insert it (admitting it like a received one) and
        return its id; PUBLISH ships it on the next ``outbox``. Raises ValueError if the
        object is outside this node's scopes or (a Delta) under another encoder."""
        received, mac = signed(obj, self.key)
        if self._insert(received, mac, None) is None:
            raise ValueError(f"{type(obj).__name__} refused by this node's sets")
        return object_id(received)

    def outbox(self) -> List[bytes]:
        """Ids of own objects not yet published, sorted; each is returned once."""
        with self._lock:
            ids = sorted(i for k in KINDS for s in self.objects.scopes
                         for i in self.objects.ids(k, s) if i not in self._published)
            self._published.update(ids)
        return ids

    def receive_object(self, received: bytes, mac: str, scope: Optional[str]) -> bool:
        """The receive gate for PUBLISH and PULL: True iff the object verified and is held.
        A received id is marked published, so it is never re-published (echo guard)."""
        return self._insert(received, mac, scope, echo_guard=True) is not None

    def _insert(self, received: bytes, mac: str, scope: Optional[str], echo_guard=False):
        with self._lock:
            new = isinstance(received, bytes) and object_id(received) not in self.objects
            obj = self.mem.store.apply(received, mac, scope)
            if obj is None:
                return None
            oid = object_id(received)
            if echo_guard:
                self._published.add(oid)
            if new:
                self.admit(self.objects.kind_of(oid), obj)
            return obj

    # -- local observation and authorship ------------------------------------
    def ingest(self, text: str) -> None:
        """A local observation: a signed Delta in the node's cell enters the own sets and is
        admitted."""
        text = (text or "").strip()
        if text:
            self.publish_local(self.mem.delta(text))

    def apply_status(self, claim_text: str, *, link_id: str, effect: str) -> bytes:
        """Author a status effect on the claim (default cell, version 0) as a signed
        AuthoredEffect; ``link_id`` names the author, so re-applying one link is idempotent
        and two links are two effects. DEPRECATED is SUPERSEDE to the claim's next version."""
        return self.publish_local(status_effect(claim_text, link_id=link_id, effect=effect,
                                                scope=self.scope_of[DEFAULT_CELL]))

    # -- read side -----------------------------------------------------------
    def live_claim_texts(self) -> Set[str]:
        """The live claim texts (the eval-c convergence observable)."""
        nodes = self.mem.store.nodes()
        return {nodes[k][1].text for k in self.mem.store.live_nodes()}

    def status_of(self, claim_text: str) -> str:
        """vis of the claim's key in the default cell: authored status joined with the
        consolidation view (ACTIVE if nothing applies)."""
        return self.mem.store.vis(claim_key(claim_text, DEFAULT_CELL))


__all__ = ["BrainNode", "DEFAULT_POL", "status_effect", "status_envelope"]
