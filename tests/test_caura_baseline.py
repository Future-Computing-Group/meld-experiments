"""Caura (caura-ai/caura, backend-v3.34.5) as a merge-on-receive baseline: prompt and excerpt
fidelity, the write gate and contradiction judge on hand-built cases, the event mapping onto the
pair protocol, and the plug-in to the runners (fake LLM and a fake embedder only)."""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pytest

from meld.agent.llm import FakeChatJSON
from meld.baselines import _caura_prompts as cp
from meld.baselines.caura_policy import CauraCrossAgentPolicy, CauraPolicy
from meld.baselines.memory import BaselineMemory, pair_rows, summarize_pairs

ENC, NLI = "hashing", "lexical"
A, B = "Paris is the capital of France", "Paris is the capital of Italy"
VENDOR = os.path.join(os.path.dirname(__file__), "..", "vendor", "caura", "src")
JSON = {"type": "json_object"}


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# -- prompt fidelity: hashes of the vendored prompts (byte identity checked below) -------

def test_caura_prompts_and_renderings_match_upstream():
    assert _h(cp.DEDUP_PROMPT) == "a19d469e1e4bebfe287c70e9572f7730233b1b04b7bfd94340ce2efbce79f853"
    assert _h(cp.CONTRADICTION_PROMPT) == "85d4492a1cd0561dfe23519a72a16c99828f91acf5e536430f1feb71ddafca1e"
    assert _h(cp.BATCH_CONTRADICTION_PROMPT) == (
        "ffff4ab3b7129dd9548170033492bb8c56f53e76e169a320ef1b3742689fb897")
    assert _h(cp.DEDUP_PROMPT.format(new_content=B, old_content=A)) == (
        "5024473da0175ad5f6c993ccb504cb220fd1e580b43a37ff3d55a48eb260c30f")
    assert _h(cp.CONTRADICTION_PROMPT.format(new_content=B, old_content=A)) == (
        "edcfb56bc669e5d80a3b92846db603f7f9d92f4d90748c0d56cb92b61f8f1317")
    assert _h(cp.BATCH_CONTRADICTION_PROMPT.format(new_content=B, candidates_block=f"[0] {A}\n[1] {B}")) == (
        "d2c1ace1e6f6287ffde83fb0a42fbb5bf37e1eaf9c4d76806e8f4e74beabb22b")


def test_caura_constants_are_the_upstream_defaults():
    assert (cp.SEMANTIC_DEDUP_AUTO_THRESHOLD, cp.SEMANTIC_DEDUP_JUDGE_THRESHOLD) == (0.97, 0.85)
    assert (cp.CONTRADICTION_SIMILARITY_THRESHOLD, cp.CONTRADICTION_CANDIDATE_WINDOW) == (0.45, 20)
    assert cp.DEDUP_JUDGE_CONFIDENCE_THRESHOLD == 0.60


# (upstream file, sha256, line ranges excerpted byte-identically into _caura_prompts.py)
EXCERPTS = [
    ("common/constants.py", "9fc38194803733214826c8347824e0813ba2a6f8ee0c4cee885c02f6adb746af",
     [(93, 98)]),
    ("core-api/src/core_api/services/dedup_judge.py",
     "6d7cdf92e1c913aa15cd12343de3d5661031d3f7bde6e3b335835ff836b2e20b",
     [(36, 38), (41, 61), (64, 92), (139, 142)]),
    ("core-api/src/core_api/services/dedup_identifier_filter.py",
     "8ccfaa57bd2ba2bb752f64c35459333ace536ca29a468051ecc3fea8954e3c59", [(27, 79)]),
    ("core-api/src/core_api/services/contradiction_detector.py",
     "0b5427e721e7bd1319f95cd80eb7892c5f148cdee4eaf77718cbad7f55918eb3",
     [(1251, 1360), (1510, 1594), (1600, 1611), (1636, 1666), (1743, 1839)]),
]


def test_vendored_module_records_every_source_and_sha256():
    doc = cp.__doc__ + open(cp.__file__, encoding="utf-8").read()
    assert "fd1da78b563b21f357fe5a7060a79954c9df94c0" in doc
    for path, sha, _ in EXCERPTS:
        assert path in doc and sha in doc


@pytest.mark.skipif(not os.path.isdir(VENDOR), reason="vendor/caura/src not fetched (vendor/caura/fetch.sh)")
def test_excerpts_are_byte_identical_to_the_pinned_tree():
    src = open(cp.__file__, encoding="utf-8").read()
    for path, sha, ranges in EXCERPTS:
        raw = open(os.path.join(VENDOR, path), "rb").read()
        assert hashlib.sha256(raw).hexdigest() == sha, path
        lines = raw.decode("utf-8").split("\n")
        for a, b in ranges:
            assert "\n".join(lines[a - 1:b]) + "\n" in src, (path, a, b)
    up = open(os.path.join(VENDOR, "common/constants.py"), encoding="utf-8").read()
    assert 'read_float_env(\n    "CONTRADICTION_SIMILARITY_THRESHOLD", 0.45\n)' in up
    assert 'read_int_env("CONTRADICTION_CANDIDATE_WINDOW", 20)' in up


# -- the write gate and the contradiction judge on hand-built cases -------------------

def _emb(sims: dict):
    """A fake embedder: the arrival is e0; held text t sits at cosine sims[t] to it."""
    def embed(texts):
        out = []
        for t in texts:
            if t in sims:
                c = sims[t]
                # each held text gets its own orthogonal direction so held-held cosines stay small
                k = 1 + sorted(sims).index(t)
                v = np.zeros(len(sims) + 1)
                v[0], v[k] = c, np.sqrt(max(0.0, 1 - c * c))
                out.append(v)
            else:
                v = np.zeros(len(sims) + 1)
                v[0] = 1.0
                out.append(v)
        return np.array(out)
    return embed


def _llm(*replies):
    it = iter(replies)
    return FakeChatJSON(lambda m, rf: next(it))


DUP = json.dumps({"is_duplicate": True, "reason": "same fact"})
NODUP = json.dumps({"is_duplicate": False, "reason": "different"})
YES = json.dumps({"subject_a": "Paris", "subject_b": "Paris", "same_subject": True,
                  "non_conflict_reason": "none", "contradicts": True, "reason": "x"})
NO = json.dumps({"subject_a": "Paris", "subject_b": "Paris", "same_subject": True,
                 "non_conflict_reason": "none", "contradicts": False, "reason": "x"})


def test_empty_store_and_exact_text_make_no_call():
    pol = CauraPolicy(_llm(), embedder=_emb({}))
    v = pol(B, [])
    assert (v.decision, v.event, v.add) == ("INSERT", "NEW", (B,))
    pol = CauraPolicy(_llm(), embedder=_emb({B: 1.0}))
    v = pol(B, [B])
    assert (v.decision, v.event, v.target, v.add, v.revoke) == ("MERGE", "EXACT", 0, (), ())
    assert pol.counters["calls"] == 0


def test_auto_band_rejects_without_a_call_and_names_the_nearest():
    pol = CauraPolicy(_llm(), embedder=_emb({A: 0.5, "Paris, capital of Italy": 0.98}))
    v = pol(B, [A, "Paris, capital of Italy"])
    assert (v.decision, v.event, v.target, v.add) == ("MERGE", "DUP_AUTO", 1, ())
    assert pol.counters["calls"] == 0


def test_judge_band_duplicate_is_a_merge_with_the_upstream_request():
    llm = _llm(DUP)
    pol = CauraPolicy(llm, embedder=_emb({A: 0.90}))
    v = pol(B, [A])
    assert (v.decision, v.event, v.target, v.add) == ("MERGE", "DUP_JUDGE", 0, ())
    assert llm.calls == [([{"role": "user", "content": cp.DEDUP_PROMPT.format(new_content=B, old_content=A)}],
                          JSON)]
    assert pol.counters == {**pol.counters, "calls": 1, "dedup_calls": 1, "contradiction_calls": 0}


def test_judge_band_not_duplicate_goes_on_to_the_contradiction_judge():
    llm = _llm(NODUP, YES)
    pol = CauraPolicy(llm, embedder=_emb({A: 0.90}))
    v = pol(B, [A])
    assert (v.decision, v.event, v.revoke, v.add, v.target) == ("CONFLICT", "CONFLICTED", (0,), (B,), 0)
    assert llm.calls[1][0][0]["content"] == cp.CONTRADICTION_PROMPT.format(new_content=B, old_content=A)


@pytest.mark.parametrize("reply", [json.dumps({"is_duplicate": True}),            # no reason: 0.50
                                   json.dumps({"is_duplicate": "yes", "reason": "r"}),
                                   json.dumps({"is_duplicate": True, "reason": "  "})])
def test_a_malformed_judge_reply_cannot_merge(reply):
    pol = CauraPolicy(_llm(reply, NO), embedder=_emb({A: 0.90}))
    v = pol(B, [A])
    assert (v.decision, v.event) == ("INSERT", "CLEAR") and pol.counters["parse_fail"] == 0


@pytest.mark.parametrize("bad", ["not json", "", "[1, 2]", "```json\n{\"is_duplicate\": true}\n```"])
def test_an_unparsable_reply_falls_back_like_upstream_and_is_counted(bad):
    # dedup: _fake_dedup_check at confidence 0.50 never 409s; contradiction: abstain
    pol = CauraPolicy(_llm(bad, bad), embedder=_emb({A: 0.90}))
    v = pol(B, [A])
    assert (v.decision, v.event, v.add) == ("INSERT", "CLEAR", (B,))
    assert pol.counters["parse_fail"] == 2


def test_identifier_bearing_content_skips_the_semantic_gate():
    new = "Build 1024 of PR#12 passed"
    pol = CauraPolicy(_llm(YES), embedder=_emb({"Build 1023 of PR#12 passed": 0.99}))
    v = pol(new, ["Build 1023 of PR#12 passed"])
    assert (v.decision, v.event) == ("CONFLICT", "CONFLICTED") and pol.counters["dedup_calls"] == 0


def test_below_the_candidate_floor_nothing_is_judged():
    pol = CauraPolicy(_llm(), embedder=_emb({A: 0.44}))
    v = pol(B, [A])
    assert (v.decision, v.event, v.add) == ("INSERT", "NO_CANDIDATE", (B,)) and pol.counters["calls"] == 0


def test_several_candidates_are_judged_in_one_batch_ordered_by_similarity():
    c1, c2, c3 = "Paris is in France", "Rome is the capital of Italy", "Bread is food"
    llm = _llm(json.dumps({"clean": [0], "hits": {"1": {"same_subject": True, "non_conflict_reason": "none",
                                                        "contradicts": True}}}))
    pol = CauraPolicy(llm, embedder=_emb({c1: 0.50, c2: 0.80, c3: 0.10}))
    v = pol(B, [c1, c2, c3])
    block = f"[0] {c2}\n[1] {c1}"                      # c3 below the floor, then by cosine
    assert llm.calls[0][0][0]["content"] == cp.BATCH_CONTRADICTION_PROMPT.format(
        new_content=B, candidates_block=block)
    assert (v.decision, v.revoke, v.target, v.add) == ("CONFLICT", (0,), 0, (B,))   # hit 1 = c1 = old[0]
    assert pol.counters["batch_calls"] == 1


def test_batch_parse_failure_abstains_for_every_candidate():
    pol = CauraPolicy(_llm("nope"), embedder=_emb({A: 0.6, "Paris is big": 0.5}))
    v = pol(B, [A, "Paris is big"])
    assert (v.decision, v.event) == ("INSERT", "CLEAR") and pol.counters["parse_fail"] == 1


@pytest.mark.parametrize("reply", [
    {"same_subject": False, "non_conflict_reason": "none", "contradicts": True},       # gate 1
    {"same_subject": True, "non_conflict_reason": "refinement", "contradicts": True},  # gate 2
    {"same_subject": True, "non_conflict_reason": "none", "contradicts": "true"}])     # not a bool
def test_the_upstream_safety_gates_veto_a_contradiction(reply):
    pol = CauraPolicy(_llm(json.dumps(reply)), embedder=_emb({A: 0.6}))
    assert pol(B, [A]).event == "CLEAR"


def test_temporal_supersession_does_not_veto():
    r = {"same_subject": True, "non_conflict_reason": "temporal_supersession", "contradicts": True}
    pol = CauraPolicy(_llm(json.dumps(r)), embedder=_emb({A: 0.6}))
    assert pol(B, [A]).decision == "CONFLICT"


def test_prompt_content_is_truncated_at_500_characters():
    long_old, long_new = "a" * 700, "b" * 700
    llm = _llm(NO)
    CauraPolicy(llm, embedder=_emb({long_old: 0.6}))(long_new, [long_old])
    assert llm.calls[0][0][0]["content"] == cp.CONTRADICTION_PROMPT.format(
        new_content="b" * 500, old_content="a" * 500)


def test_window_caps_the_judged_candidates():
    old = [f"Paris fact {i}" for i in range(25)]
    llm = _llm(json.dumps({"clean": list(range(20)), "hits": {}}))
    CauraPolicy(llm, k=25, embedder=_emb({t: 0.5 + i / 100 for i, t in enumerate(old)}))(B, old)
    assert len(re.findall(r"^\[\d+\] Paris fact", llm.calls[0][0][0]["content"], re.M)) == 20


def test_cross_agent_arm_has_no_write_gate():
    pol = CauraCrossAgentPolicy(_llm(YES), embedder=_emb({A: 0.99}))
    v = pol(B, [A])
    assert pol.name == "caura-xagent" and (v.decision, v.event) == ("CONFLICT", "CONFLICTED")
    assert CauraCrossAgentPolicy(_llm(NO), embedder=_emb({B: 1.0}))(B, [B]).event == "CLEAR"


def test_default_embedder_is_cauras_local_default():
    assert CauraPolicy(_llm()).embedder_name == "BAAI/bge-large-en-v1.5"
    assert CauraPolicy(_llm()).k == 20


# -- the pair protocol: events onto false merge / overwrite / missed / lost ------------

def test_pair_rows_map_caura_outcomes_onto_the_protocol():
    pairs = [("CONFLICT", A, B), ("CONFLICT", A, B), ("RELATE", A, B), ("MERGE", A, B), ("CONFLICT", A, B)]
    replies = iter([NODUP, YES,          # conflicted: held retired -> silent overwrite
                    NODUP, NO,           # judged clear -> missed contradiction
                    DUP,                 # duplicate on RELATE -> false merge
                    DUP,                 # duplicate on MERGE -> merge recall
                    DUP])                # duplicate on CONFLICT -> missed contradiction, no overwrite
    pol = CauraPolicy(FakeChatJSON(lambda m, rf: next(replies)), embedder=_emb({A: 0.90}))
    rows = pair_rows(pairs, lambda: BaselineMemory(pol, encoder=ENC, nli=NLI))
    assert [r["event"] for r in rows] == ["CONFLICTED", "CLEAR", "DUP_JUDGE", "DUP_JUDGE", "DUP_JUDGE"]
    assert [r["pred"] for r in rows] == ["CONFLICT", "NO_RELATION", "MERGE", "MERGE", "MERGE"]
    assert [r["silent_overwrite"] for r in rows] == [True, False, False, False, False]
    assert [r["missed_contradiction"] for r in rows] == [False, True, False, False, True]
    assert [r["false_merge"] for r in rows] == [False, False, True, False, True]
    s = summarize_pairs(rows)
    assert s["claim_lost_rate"] == 0.0 and s["events"] == {"CONFLICTED": 1, "CLEAR": 1, "DUP_JUDGE": 3}


def test_registered_as_runner_baselines():
    from meld.eval.runner_args import BASELINES
    assert BASELINES["caura"] is CauraPolicy and BASELINES["caura-xagent"] is CauraCrossAgentPolicy


def test_pilot_runs_caura_against_the_fake(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "pilot", Path(__file__).resolve().parents[1] / "scripts" / "run_baseline_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    res = pilot.main(["--policy", "caura", "--fake", "--n", "8", "--encoder", ENC, "--nli", NLI,
                      "--baseline-embedder", "hashing", "--out", str(tmp_path / "o.json")])
    assert res["summary"]["n"] == 8 and res["config"]["baseline_embedder"] == "hashing"
    assert res["policy_counters"]["embedder"] == "hashing"
    assert set(res["summary"]["events"]) <= {"NEW", "EXACT", "DUP_AUTO", "DUP_JUDGE", "CONFLICTED", "CLEAR",
                                             "NO_CANDIDATE"}
    with pytest.raises(SystemExit):
        pilot.main(["--policy", "graphiti", "--fake", "--encoder", ENC, "--nli", NLI,
                    "--baseline-embedder", "hashing"])


def test_runner_args_pass_the_embedder_and_stamp_it():
    import argparse
    from meld.eval.runner_args import add_baseline, baseline_policy, baseline_stamp
    ap = argparse.ArgumentParser()
    add_baseline(ap)
    a = ap.parse_args(["--baseline", "caura", "--baseline-embedder", "hashing"])
    pol = baseline_policy(a, _llm())
    assert pol.embedder_name == "hashing" and baseline_stamp(a)["baseline_embedder"] == "hashing"
    a = ap.parse_args(["--baseline", "caura"])
    assert baseline_stamp(a)["baseline_embedder"] == "BAAI/bge-large-en-v1.5"
    a = ap.parse_args(["--baseline", "graphiti"])
    assert "baseline_embedder" not in baseline_stamp(a)
    with pytest.raises(SystemExit):
        baseline_policy(ap.parse_args(["--baseline", "graphiti", "--baseline-embedder", "x"]), _llm())


def test_hashing_embedder_is_offline_and_normalised():
    pol = CauraPolicy(_llm(), embedder="hashing")
    v = pol._vectors([A, B])
    assert v.shape[0] == 2 and np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert 0.5 < float(v[0] @ v[1]) < 1.0
