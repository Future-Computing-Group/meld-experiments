"""Task 1.7 — merge-on-receive orchestration.

An incoming Concept/Claim/Link produces exactly ONE Patch; the Patch (not the
raw delta) is the only state mutation. MERGE emits claim-status UPDATEs; RELATE
emits a Mapping link; CONFLICT emits a CONTRADICTS link with no silent winner;
REJECT is a recorded no-op.
"""
from __future__ import annotations

from smp.config import load_thresholds
from smp.merge.signals import EmbeddingSimilarity
from smp.proto import Claim, MemoryLink, MemoryUnit
from smp.protocol.merge_on_receive import merge_on_receive

CFG = load_thresholds()


def _incoming(text, key=b"ki"):
    return MemoryUnit(id=b"inc", claims=[Claim(claim_key=key, text=text)])


def _local(text, key=b"kl"):
    return MemoryUnit(id=b"loc", claims=[Claim(claim_key=key, text=text)])


def _es():
    return EmbeddingSimilarity(encoder="hashing")


def test_exact_key_match_emits_one_merge_patch():
    inc = _incoming("Paris is in France", key=b"same")
    loc = _local("Paris is in France", key=b"same")
    patches = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")
    assert len(patches) == 1
    assert patches[0].decision == "MERGE"
    assert "kappa" in patches[0].gates_that_fired


def test_contradiction_emits_conflict_with_contradicts_link_no_winner():
    inc = _incoming("Paris is in France", key=b"same")
    loc = _local("Paris is in Italy", key=b"same")
    patches = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")
    assert len(patches) == 1
    p = patches[0]
    assert p.decision == "CONFLICT"
    links = [x for x in p.emitted if isinstance(x, MemoryLink)]
    assert any(l.type == "CONTRADICTS" for l in links)


def test_unrelated_claim_is_novel_insert_not_a_conflict():
    # An incoming claim with NO real local candidate (best similarity below the
    # floor) is novel knowledge -> inserted as MERGE. REJECT is reserved for the
    # stale/overruled wire gate, not for "no match found".
    inc = _incoming("quantum chromodynamics is hard", key=b"a")
    loc = _local("the cat sat on the mat", key=b"b")
    patches = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")
    assert len(patches) == 1
    assert patches[0].decision == "MERGE"
    assert "novel" in patches[0].gates_that_fired


def test_stale_incoming_is_rejected_at_wire():
    # a stale incoming unit (retention below floor) IS rejected.
    inc = _incoming("Paris is in France", key=b"same")
    inc.relevance_magnitude = 0.001          # below r_min
    loc = _local("Paris is in France", key=b"same")
    patches = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")
    assert patches[0].decision == "REJECT"


def test_no_local_match_yields_a_patch_per_incoming_claim():
    inc = _incoming("a brand new fact about widgets", key=b"new")
    patches = merge_on_receive(inc, [], cfg=CFG, es=_es(), nli="lexical")
    # with no local claim, a novel concept is admitted as a MERGE (insert)
    assert len(patches) == 1
    assert patches[0].decision in {"MERGE", "RELATE"}
