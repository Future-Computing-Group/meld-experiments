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


def test_sigma_paraphrase_merge_deprecates_incoming_and_supersedes_local():
    # same-enough by SIMILARITY (kappa=0, sigma>=theta_merge, not contradicting):
    # the incoming is a redundant paraphrase of an existing local canonical claim.
    # The merge keeps it for provenance/audit but marks ITS claim status
    # DEPRECATED (superseded) and records a SUPERSEDES link to the canonical, so a
    # later status-aware recall surfaces ONE representative, not both. This is the
    # "status, not truth" consolidation that distinguishes merge from naive union.
    inc = _incoming("Paris is the capital city of France", key=b"ki")
    loc = _local("Paris is the capital of France", key=b"kl")
    patches = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")
    assert len(patches) == 1
    p = patches[0]
    assert p.decision == "MERGE"
    assert "sigma" in p.gates_that_fired          # paraphrase path, not kappa/novel
    status_updates = [x for x in p.emitted
                      if isinstance(x, tuple) and x and x[0] == "STATUS"]
    assert status_updates, "a MERGE must emit a claim-status update"
    _, ck, (_lid, eff) = status_updates[0]
    assert ck == inc.claims[0].claim_key
    assert eff == "DEPRECATED"                    # superseded, NOT left ACTIVE
    links = [x for x in p.emitted if isinstance(x, MemoryLink)]
    assert any(l.type == "SUPERSEDES" for l in links)


def test_relate_retains_the_incoming_claim_as_a_unit():
    # Section 4: a RELATE ("overlaps") keeps the two claims DISTINCT -- BOTH must
    # remain retrievable. The incoming claim must be stored as a unit AND linked
    # (ASSOCIATES), not dropped. The drop was collapsing the centralized store
    # (37 of 40 claims RELATEd away -> recall@5 0.056), caught on the minilm smoke.
    inc = _incoming("Mount Everest is located in the Himalayas", key=b"ki")
    loc = _local("Mount Everest is the tallest mountain", key=b"kl")
    p = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "RELATE"
    units = [x for x in p.emitted if isinstance(x, MemoryUnit)]
    links = [x for x in p.emitted if isinstance(x, MemoryLink)]
    assert any(u.id == inc.id for u in units), "RELATE must retain the incoming unit"
    assert any(l.type == "ASSOCIATES" for l in links)


def test_exact_key_and_novel_merges_stay_active():
    # the deprecation applies ONLY to the sigma-paraphrase path; an exact-key
    # merge and a novel insert remain ACTIVE (they are canonical, not redundant).
    exact = merge_on_receive(_incoming("Paris is in France", key=b"same"),
                             [_local("Paris is in France", key=b"same")],
                             cfg=CFG, es=_es(), nli="lexical")[0]
    novel = merge_on_receive(_incoming("quantum chromodynamics is hard", key=b"a"),
                             [_local("the cat sat on the mat", key=b"b")],
                             cfg=CFG, es=_es(), nli="lexical")[0]
    for p in (exact, novel):
        effs = [x[2][1] for x in p.emitted
                if isinstance(x, tuple) and x and x[0] == "STATUS"]
        assert all(e == "ACTIVE" for e in effs)
        assert not any(isinstance(x, MemoryLink) and x.type == "SUPERSEDES"
                       for x in p.emitted)
