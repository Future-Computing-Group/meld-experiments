"""Evaluation topology: N agent brains -> global-brain funnel sink.

Each agent consumes a disjoint slice of conversations into its own brain, then
PUBLISHes deltas that the global brain merges-on-receive (the N->F->1 funnel).
The centralized baseline is a single store ingesting all conversations with no
merge protocol -- the comparison point for H/M-1.

REUSE: the funnel/aggregation pattern (the transport's N->F->1 map) is the NPS
sink shape; here it is exercised in-process for the SIM accuracy claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from meld.brain.store import Brain
from meld.config import load_thresholds
from meld.merge.signals import EmbeddingSimilarity
from meld.protocol.merge_on_receive import merge_on_receive
from meld.protocol.messages import Patch
from meld.proto import MemoryUnit


@dataclass
class Topology:
    agent_brains: List[Brain]
    global_brain: Brain
    conversations: List[Dict]
    encoder: str
    patches: List[Patch] = field(default_factory=list)


def build_topology(conversations: List[Dict], n_agents: int, encoder: str) -> Topology:
    agents = [Brain(encoder=encoder) for _ in range(n_agents)]
    glob = Brain(encoder=encoder)
    # round-robin assign conversations to agents; each agent inserts its units
    for i, conv in enumerate(conversations):
        a = agents[i % n_agents]
        for u in conv["units"]:
            a.apply(Patch(decision="MERGE",
                          target=_target(u), emitted=[u],
                          gates_that_fired=["novel"], version=1, signature=b"s"))
    return Topology(agent_brains=agents, global_brain=glob,
                    conversations=conversations, encoder=encoder)


def _target(u: MemoryUnit):
    from meld.proto import Endpoint
    return Endpoint(node_id=u.id)


def run_funnel_merge(topo: Topology, nli: str = "lexical") -> Topology:
    """Each agent PUBLISHes its units; the global brain merges-on-receive,
    producing one Patch per incoming claim and applying it."""
    cfg = load_thresholds()
    es = EmbeddingSimilarity(encoder=topo.encoder)
    for agent in topo.agent_brains:
        for unit in agent.extract_delta():
            if not isinstance(unit, MemoryUnit):
                continue
            local = [u for u in topo.global_brain.extract_delta()
                     if isinstance(u, MemoryUnit)]
            patches = merge_on_receive(unit, local, cfg=cfg, es=es, nli=nli)
            for p in patches:
                topo.global_brain.apply(p)
                topo.patches.append(p)
    return topo


class CentralizedBaseline:
    """A single store ingesting all conversations with NO merge protocol."""

    def __init__(self, conversations: List[Dict], encoder: str):
        self.conversations = conversations
        self.brain = Brain(encoder=encoder)

    def ingest_all(self) -> None:
        for conv in self.conversations:
            for u in conv["units"]:
                self.brain.apply(Patch(decision="MERGE", target=_target(u),
                                       emitted=[u], gates_that_fired=["ingest"],
                                       version=1, signature=b"s"))


class NaiveUnionBaseline:
    """Ablated MELD: set-union with no merge decision (every incoming unit kept,
    double-counting paraphrases and keeping contradictions side-by-side)."""

    def __init__(self, conversations: List[Dict], encoder: str):
        self.conversations = conversations
        self.brain = Brain(encoder=encoder)

    def ingest_all(self) -> None:
        for conv in self.conversations:
            for u in conv["units"]:
                # naive: append every unit, no classify, no conflict detection
                self.brain.apply(Patch(decision="MERGE", target=_target(u),
                                       emitted=[u], gates_that_fired=["union"],
                                       version=1, signature=b"s"))


__all__ = ["Topology", "build_topology", "run_funnel_merge",
           "CentralizedBaseline", "NaiveUnionBaseline"]
