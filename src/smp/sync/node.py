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
        self.mem = SMPMemory(encoder=encoder, nli=nli, cfg=cfg, cache_dir=cache_dir)
        self._published: Set[str] = set()   # claim texts already published OR received

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

    def extract_delta(self) -> List[str]:
        """Claim texts to publish: live claims not yet published or received.
        Each claim is emitted at most once over the node's lifetime."""
        delta = sorted(self._live_claim_texts() - self._published)
        self._published.update(delta)
        return delta

    # -- receive side --------------------------------------------------------
    def receive(self, claim_texts: List[str]) -> None:
        """Merge a peer's published delta into the own brain. Received claims are
        marked published so they are never echoed back (gossip-loop guard)."""
        for t in claim_texts:
            self.mem.write(t)
            self._published.add(t)


__all__ = ["BrainNode"]
