"""BrainNode: the integration kernel deployed per host in the multi-host sync runs.

Wraps an SMPMemory (a Brain + the merge-on-receive decision procedure) with the
network-facing operations the orchestration drives:

  * ``ingest(text)``        -- a LOCAL observation; merged into the own brain.
  * ``extract_delta()``     -- the claim texts to PUBLISH (new since last publish),
                               i.e. the payload PatchTransport carries to peers.
  * ``receive(claims)``     -- a PEER's delta; merged into the own brain via the same
                               merge-on-receive path (status-CRDT reconciles).
  * ``live_claim_texts()``  -- the status-live claims, the observable used by the
                               convergence/agreement metric (eval-c).

A claim that was only RECEIVED is never re-published (gossip-loop guard): both
locally-ingested and received claims are tracked, so extract_delta() returns each
claim at most once across the node's lifetime. The transport/merge are decoupled --
this class holds no socket; ``extract_delta()`` output is bytes-serialized by the
caller and handed to PatchTransport, keeping the node unit-testable in-process.
"""
from __future__ import annotations

from typing import List, Optional, Set

from smp.agent.memory import SMPMemory
from smp.config import Thresholds


class BrainNode:
    def __init__(self, node_id: str, *, encoder: str, nli: str,
                 cfg: Optional[Thresholds] = None, cache_dir: Optional[str] = None):
        self.node_id = node_id
        self._mem_kw = dict(encoder=encoder, nli=nli, cfg=cfg, cache_dir=cache_dir)
        self.mem = SMPMemory(**self._mem_kw)
        self._published: Set[str] = set()   # claim texts already published OR received

    def reset(self) -> None:
        """Drop all state -> a fresh brain + empty published-set, so a campaign cell starts
        isolated (seeds are independent replicates; each RTT regime is actually re-measured
        rather than masked by publish-once leftover state from the previous cell)."""
        self.mem = SMPMemory(**self._mem_kw)
        self._published.clear()

    # -- local observation ---------------------------------------------------
    def ingest(self, text: str) -> None:
        """Merge a local observation into the own brain (merge-on-receive)."""
        self.mem.write(text)

    # -- publish side --------------------------------------------------------
    def _live_claim_texts(self) -> Set[str]:
        b = self.mem.brain
        return {c.text
                for u in b._units.values() if b._unit_is_live(u)
                for c in u.claims}

    def live_claim_texts(self) -> Set[str]:
        """The status-live claim texts (the eval-c convergence observable)."""
        return self._live_claim_texts()

    def full_state(self) -> List[str]:
        """The entire live set -- the anti-entropy payload. Unlike extract_delta()
        this ignores the publish-once guard, so a peer that missed claims (e.g. during
        a partition) is brought fully up to date on the next re-sync after heal."""
        return sorted(self._live_claim_texts())

    # -- status-CRDT (eval-c: concurrent overrules + partition heal) ----------
    def _claim_key(self, claim_text: str) -> bytes:
        from smp.agent.memory import _cid
        return _cid(claim_text + "::claim")

    def apply_status(self, claim_text: str, *, link_id: str, effect: str) -> None:
        """Apply one overrule/supersede status link to a claim (the status-CRDT add).
        Order-independent + idempotent: re-applying the same link id is a no-op, and the
        same set of links yields the same effective status regardless of arrival order."""
        b = self.mem.brain
        ck = self._claim_key(claim_text)
        b._status[ck] = b.claim_status(ck).add(link_id, effect)

    def status_of(self, claim_text: str) -> str:
        """Effective per-claim status (lattice-join of all its links; ACTIVE if none)."""
        from smp.merge.status_crdt import effective_status
        return effective_status(self.mem.brain.claim_status(self._claim_key(claim_text)))

    def extract_delta(self) -> List[str]:
        """Claim texts to publish: live claims not yet published or received.
        Each claim is emitted at most once over the node's lifetime."""
        delta = sorted(self._live_claim_texts() - self._published)
        self._published.update(delta)
        return delta

    # -- receive side --------------------------------------------------------
    def receive(self, claim_texts: List[str]) -> None:
        """Merge a peer's published delta into the own brain. Received claims are
        marked published so they are never echoed back (gossip-loop guard).

        In-process API: peers hand over claim TEXT. The wire path uses
        ``receive_wire`` (a signed, typed delta) and merges identically."""
        for t in claim_texts:
            self.mem.write(t)
            self._published.add(t)

    # -- typed, signed wire seam (the real §7-s path) ------------------------
    def publish_wire_payloads(self, key: bytes) -> List[bytes]:
        """The publish payloads for this node's delta: one signed, serialized
        TYPED delta (a MemoryUnit) per new claim, ready for the transport. Each
        carries the originating node id and is HMAC-signed over the group ``key``."""
        from smp.sync.wire_delta import serialize_signed_delta
        payloads: List[bytes] = []
        for text in self.extract_delta():
            unit = self.mem.build_unit(text)
            payloads.append(serialize_signed_delta(self.node_id, unit, key))
        return payloads

    def receive_wire(self, payload: bytes, key: bytes) -> bool:
        """Verify a peer's signed, typed delta and -- only if it verifies -- merge
        the reconstructed MemoryUnit through the same path as a local write.
        Returns True if admitted, False if rejected before admission (forged /
        tampered / wrong-key / malformed). Verification precedes any merge signal."""
        from smp.sync.wire_delta import deserialize_verified_delta
        out = deserialize_verified_delta(payload, key)
        if out is None:
            return False                              # dropped before admission
        _node_id, unit = out
        self.mem.write_unit(unit)
        for c in unit.claims:
            self._published.add(c.text)               # gossip-loop guard
        return True


__all__ = ["BrainNode"]
