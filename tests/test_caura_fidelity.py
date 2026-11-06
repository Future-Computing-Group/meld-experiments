"""Caura fidelity driver (scripts/run_caura_fidelity.py): Caura's outcome in event form, the
vendored side on a recorded Caura run (requests and outcome compared), the summary, and re-scoring
a saved step-1 file. Step 1 itself needs Caura's own venv and is exercised by the driver run."""
from __future__ import annotations

import json
import os
import sys

from meld.baselines import _caura_prompts as cp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_caura_fidelity as F  # noqa: E402

A, R, B = "Paris is the capital of France", "Paris has many museums", "Paris is the capital of Italy"
YES = {"same_subject": True, "non_conflict_reason": "none", "contradicts": True, "reason": "x"}


def _call(prompt, content):
    return {"request": {"model": "m", "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"}, "temperature": 0.0,
                        "max_completion_tokens": 8192}, "content": content}


def _vec(c):
    import math
    return [c, math.sqrt(1 - c * c), 0.0]


def _rec(sims, calls, **real):
    vecs = {B: [1.0, 0.0, 0.0], A: _vec(sims[0]), R: [sims[1], 0.0, (1 - sims[1] ** 2) ** 0.5]}
    base = {"sims": list(sims), "conflicted": [], "status_409": False, "calls": calls, "vectors": vecs}
    return {"id": "c", "new": B, "old": [A, R], "real": {**base, **real}}


def test_cases_cover_paraphrase_and_contradiction_per_seed():
    cs = F.cases(20)
    assert len(cs) == 20 and {c["id"][-1] for c in cs} == {"p", "x"} and all(len(c["old"]) == 2 for c in cs)


def test_real_event_maps_cauras_outcome():
    assert F.real_event({"status_409": True, "calls": []})["event"] == "DUP_AUTO"
    assert F.real_event({"status_409": True, "calls": [1]})["event"] == "DUP_JUDGE"
    assert F.real_event({"status_409": False, "conflicted": [1, 0], "supersedes": 1, "calls": [1]}) == {
        "event": "CONFLICTED", "revoke": [1, 0], "target": 1}
    assert F.real_event({"status_409": False, "conflicted": [], "calls": []})["event"] == "NO_CANDIDATE"
    assert F.real_event({"status_409": False, "conflicted": [], "calls": [1]})["event"] == "CLEAR"


def test_vendored_side_agrees_on_a_judged_duplicate():
    rec = _rec((0.9, 0.2), [_call(cp.DEDUP_PROMPT.format(new_content=B, old_content=A),
                                  json.dumps({"is_duplicate": True, "reason": "same"}))], status_409=True)
    out = F.vendored_side(rec)
    assert out["requests_equal"] and out["event_equal"] and out["vendored_event"] == "DUP_JUDGE"


def test_vendored_side_agrees_on_a_batch_contradiction_and_flags_a_mismatch():
    block = f"[0] {A}\n[1] {R}"
    batch = cp.BATCH_CONTRADICTION_PROMPT.format(new_content=B, candidates_block=block)
    rec = _rec((0.7, 0.5), [_call(batch, json.dumps({"clean": [1], "hits": {"0": YES}}))],
               conflicted=[0], supersedes=0)
    out = F.vendored_side(rec)
    assert out["requests_equal"] and out["event_equal"] and out["revoke_equal"] and out["target_equal"]
    rec["real"]["conflicted"] = [1]
    rec["real"]["supersedes"] = 1
    out = F.vendored_side(rec)
    assert out["event_equal"] and not out["revoke_equal"] and not out["target_equal"]


def test_vendored_side_flags_a_different_request():
    rec = _rec((0.6, 0.2), [_call("some other prompt", json.dumps(YES))], conflicted=[0], supersedes=0)
    out = F.vendored_side(rec)
    assert not out["requests_equal"] and not out["event_equal"] and out["vendored_event"] == "CLEAR"


def test_rescore_writes_a_summary(tmp_path):
    rows = [_rec((0.99, 0.2), [], status_409=True),
            _rec((0.6, 0.2), [_call(cp.CONTRADICTION_PROMPT.format(new_content=B, old_content=A), json.dumps(YES))],
                 conflicted=[0], supersedes=0)]
    src = tmp_path / "real.json"
    src.write_text(json.dumps({"config": {"model": "m", "embedder": "e"}, "rows": rows}))
    res = F.main(["--rescore", str(src), "--out", str(tmp_path / "check.json")])
    assert res["summary"] == {"cases": 2, "requests_compared": 1, "requests_equal": 2, "events_equal": 2,
                              "revoke_equal": 2, "target_equal": 2,
                              "caura_events": {"DUP_AUTO": 1, "CONFLICTED": 1}}
    assert json.loads((tmp_path / "check.json").read_text())["rows"][1]["check"]["caura_event"] == "CONFLICTED"


def test_audit_hook_refuses_non_loopback_connections():
    import pytest
    F._loopback_only("socket.connect", (None, ("127.0.0.1", 11434)))
    with pytest.raises(PermissionError):
        F._loopback_only("socket.connect", (None, ("104.18.0.1", 443)))
