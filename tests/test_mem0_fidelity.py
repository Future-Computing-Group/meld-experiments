"""Mem0 fidelity driver (scripts/run_mem0_fidelity.py): the vendored side (fact extraction and
update step) on a recorded Mem0 input with a fake LLM, the event mapping, re-scoring a saved run,
and (when mem0ai is importable) the whole driver against a fake OpenAI-compatible endpoint on
localhost."""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from meld.agent.llm import FakeChatJSON
from meld.baselines._mem0_prompts import USER_MEMORY_EXTRACTION_PROMPT, get_update_memory_messages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_mem0_fidelity as F  # noqa: E402

OLD = [{"uuid": "u-a", "id": "0", "text": "Alice lives in Paris"},
       {"uuid": "u-b", "id": "1", "text": "Alice likes tea"}]


NEW = "Alice lives in Lyon"
EXTRACT = [{"role": "system", "content": USER_MEMORY_EXTRACTION_PROMPT},
           {"role": "user", "content": f"Input:\nuser: {NEW}\n"}]


def _rec(results, facts=(NEW,), **kw):
    old = [{"id": o["id"], "text": o["text"]} for o in OLD]
    return {"new": NEW, "old": OLD if facts else [], "facts": list(facts),
            "prompt": get_update_memory_messages(old, list(facts)) if facts else None,
            "results": results, "extract_messages": EXTRACT, **kw}


def _reply(memory, facts=(NEW,)):
    return FakeChatJSON(lambda m, rf: json.dumps({"facts": list(facts)} if m[0]["role"] == "system"
                                                 else {"memory": memory}))


def test_cases_cover_paraphrase_and_contradiction_per_seed():
    cs = F.cases(20)
    assert len(cs) == 20 and len({c["id"] for c in cs}) == 20
    assert {c["id"][-1] for c in cs} == {"p", "x"} and all(len(c["old"]) == 2 for c in cs)


def test_mem0_events_maps_ids_to_indices():
    res = [{"id": "u-b", "memory": "x", "event": "DELETE"}, {"id": "n1", "memory": "Alice lives in Lyon", "event": "ADD"},
           {"id": "u-a", "memory": "Alice lives in Lyon now", "event": "UPDATE"}]
    assert F.mem0_events(res, OLD) == (("DELETE", 1, "x"), ("ADD", None, "Alice lives in Lyon"),
                                       ("UPDATE", 0, "Alice lives in Lyon now"))


def test_vendored_side_agrees_on_update_and_flags_a_mismatch():
    res = [{"id": "u-a", "memory": "Alice lives in Lyon", "event": "UPDATE"}]
    upd = [{"id": "0", "text": "Alice lives in Lyon", "event": "UPDATE", "old_memory": "Alice lives in Paris"}]
    out = F.vendored_side(_rec(res), _reply(upd))
    assert out["prompt_equal"] and out["events_equal"] and out["vendored"] == [["UPDATE", 0, NEW]]
    assert out["facts_equal"] and out["extract_prompt_equal"]
    out = F.vendored_side(_rec(res), _reply([{"id": "0", "text": "Alice lives in Paris", "event": "NONE"}]))
    assert out["events_equal"] is False


def test_vendored_side_agrees_on_an_unchanged_text_update():
    res = [{"id": "u-a", "memory": "Alice lives in Paris", "event": "UPDATE"}]
    upd = [{"id": "0", "text": "Alice lives in Paris", "event": "UPDATE"},
           {"id": "1", "text": "Alice likes tea", "event": "NONE"}]
    out = F.vendored_side(_rec(res), _reply(upd))
    assert out["events_equal"] and out["vendored_event"] == "UPDATE"


def test_vendored_side_compares_multi_fact_input_in_one_update_call():
    res = [{"id": "n", "memory": "b", "event": "ADD"}]
    out = F.vendored_side(_rec(res, facts=("a", "b")),
                          _reply([{"id": "2", "text": "b", "event": "ADD"}], facts=("a", "b")))
    assert out["prompt_equal"] and out["facts_equal"] and out["events_equal"] and out["n_facts"] == 2


def test_vendored_side_on_a_no_fact_case_compares_the_extraction_only():
    out = F.vendored_side(_rec([], facts=()), _reply([], facts=()))
    assert out["facts_equal"] and out["extract_prompt_equal"] and out["n_facts"] == 0
    assert out["prompt_equal"] is None and out["events_equal"] is None
    out = F.vendored_side(_rec([], facts=()), _reply([], facts=("Alice lives in Lyon",)))
    assert out["facts_equal"] is False


def test_vendored_side_without_recorded_extraction_messages():
    rec = _rec([], facts=())
    del rec["extract_messages"]
    assert F.vendored_side(rec, _reply([], facts=()))["extract_prompt_equal"] is None


def test_rescore_replays_the_saved_replies(tmp_path):
    upd = {"memory": [{"id": "0", "text": "Alice lives in Paris", "event": "UPDATE"}]}
    rows = [_rec([{"id": "u-a", "memory": "Alice lives in Paris", "event": "UPDATE"}],
                 replies=[json.dumps({"facts": [NEW]}), json.dumps(upd)]),
            _rec([], facts=(), replies=['{"facts": []}'])]
    for r in rows:
        del r["extract_messages"]
    src = tmp_path / "in.json"
    src.write_text(json.dumps({"summary": {}, "rows": rows}))
    s = F.main(["--rescore", str(src), "--out", str(tmp_path / "out.json"), "--embedder", "e"])
    assert {k: v for k, v in s.items() if k not in F.stamps("e", None, None)} == {"cases": 2, "extract_prompt_recorded": 0, "extract_prompt_equal": 0,
                 "facts_equal": 2, "no_fact_cases": 1, "prompt_equal": 1, "events_compared": 1,
                 "events_equal": 1, "http_calls_vendored": 0}
    assert json.loads((tmp_path / "out.json").read_text())["rows"][0]["check"]["events_equal"]


class _Fake(BaseHTTPRequestHandler):
    def do_POST(self):                     # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        if msgs[0]["role"] == "system":    # Mem0's fact extraction
            user = msgs[-1]["content"]
            content = json.dumps({"facts": [user.split("user: ")[-1].strip()]})
        else:                              # the update step: DELETE memory 0 when told to delete
            content = json.dumps({"memory": [{"id": "0", "text": "x", "event": "DELETE"}]})
        out = json.dumps({"id": "c", "object": "chat.completion", "created": 0, "model": body["model"],
                          "choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": content}}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_driver_end_to_end_against_a_fake_endpoint(tmp_path):
    pytest.importorskip("mem0")
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = F.main(["--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--model", "fake",
                    "--n", "2", "--out", str(tmp_path / "f.json"),
                    "--embedder", "sentence-transformers/all-MiniLM-L6-v2"])
    finally:
        srv.shutdown()
    assert s["cases"] == 2 and s["prompt_equal"] == 2 and s["events_compared"] == 2 and s["events_equal"] == 2
    assert s["facts_equal"] == 2 and s["extract_prompt_recorded"] == 2 and s["extract_prompt_equal"] == 2
