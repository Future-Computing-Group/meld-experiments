"""Mem0 and Graphiti merge-on-receive baselines: prompt fidelity, decision mapping, pair
scoring, and the plug-in to the memory-condition harnesses (fake LLM only)."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from meld.agent.hotpot_loader import HotpotTask
from meld.agent.llm import ChatJSON, FakeChatJSON
from meld.agent.mem_conditions import build_condition
from meld.baselines import _graphiti_prompts as gp
from meld.baselines import _mem0_prompts as mp
from meld.baselines.graphiti_policy import GraphitiPolicy
from meld.baselines.mem0_policy import Mem0FullPolicy, Mem0Policy
from meld.baselines.memory import BaselineMemory, Verdict, pair_rows, summarize_pairs
from meld.eval.recall_sweep import recall_rows
from meld.protocol.admission import Admission

ENC, NLI = "hashing", "lexical"
A, B = "Paris is the capital of France", "Paris is the capital of Italy"


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# -- prompt fidelity: hashes computed from the upstream files (module docstrings) --------

def test_mem0_prompt_and_rendering_match_upstream():
    assert _h(mp.DEFAULT_UPDATE_MEMORY_PROMPT) == (
        "d68a6ef706f5477c156e69b40f55c81d9fdc17704952d679e9658da7e15749dd")
    old = [{"id": "0", "text": A}]
    assert _h(mp.get_update_memory_messages(old, [B])) == (
        "f6b8701fec23a14437006a1a4d16cf054a47e454644b443bfd0f3a11ae308db8")
    assert _h(mp.get_update_memory_messages([], [B])) == (
        "4602a27dc2a8b95c0c60f710950154e0ca811996571a00bbbfbe27be7a2dc0c4")


def _undated(s: str) -> str:
    return re.sub(r"Today's date is \d{4}-\d{2}-\d{2}\.", "Today's date is YYYY-MM-DD.", s)


def test_mem0_extraction_prompts_and_helpers_match_upstream():
    assert _h(_undated(mp.USER_MEMORY_EXTRACTION_PROMPT)) == (
        "7c3dd5ed613f942ad9ab99af1574bd9b1769f516a1f0441f45b8fa43afd4eb6d")
    assert _h(_undated(mp.AGENT_MEMORY_EXTRACTION_PROMPT)) == (
        "61d12d2db724a3d5fa353252f3e283855d0ef5d66c682cb69d4134934c3f7c42")
    assert mp.parse_messages([{"role": "user", "content": B}]) == f"user: {B}\n"
    sys_p, user_p = mp.get_fact_retrieval_messages(f"user: {B}\n", False)
    assert sys_p is mp.USER_MEMORY_EXTRACTION_PROMPT and user_p == f"Input:\nuser: {B}\n"
    assert mp.ensure_json_instruction(sys_p, user_p) == (sys_p, user_p)
    assert mp.normalize_facts([{"fact": "x"}, "y", {"z": 1}, 3, ""]) == ["x", "y", "3"]


def test_graphiti_prompt_and_schema_match_upstream():
    ctx = {"existing_edges": [{"idx": 0, "fact": A}], "edge_invalidation_candidates": [],
           "new_edge": B}
    msgs = gp.resolve_edge(ctx)
    assert _h(json.dumps([[m.role, m.content] for m in msgs])) == (
        "162731c722850c5bb957eec5f3db39c0e015e0264bbe97e66442250279673e19")
    assert json.dumps(gp.EDGE_DUPLICATE_SCHEMA, sort_keys=True) == (
        '{"properties": {"contradicted_facts": {"description": "List of idx values of '
        'contradicted facts (from full idx range). Empty list if none.", "items": {"type": '
        '"integer"}, "title": "Contradicted Facts", "type": "array"}, "duplicate_facts": '
        '{"description": "List of idx values of duplicate facts (only from EXISTING FACTS '
        'range). Empty list if none.", "items": {"type": "integer"}, "title": "Duplicate '
        'Facts", "type": "array"}}, "required": ["duplicate_facts", "contradicted_facts"], '
        '"title": "EdgeDuplicate", "type": "object"}')


def test_vendored_modules_record_source_and_sha256():
    assert "ddb803bedc22bd514606d262407782e88df929f6991b59f6972fb8a25cc06001" in mp.__doc__
    assert "05be41d3d0ea7e877fbde2875d146a15624435e32ca04949bc4f6304a9bd7cad" in mp.__doc__
    assert "b8ac6999705d350ebb53d799505d49b76f5d74c20265a09df4094645b5d8ada9" in gp.__doc__
    assert "607cb830bb02a6020fdfea013e321345c0c0d5629f7895dc76622130cefc22c6" in gp.__doc__


# -- decision mapping on a two-claim store ---------------------------------------------

def _mem0(reply) -> Mem0Policy:
    return Mem0Policy(FakeChatJSON(lambda msgs, fmt: reply))


def _graphiti(reply) -> GraphitiPolicy:
    return GraphitiPolicy(FakeChatJSON(lambda msgs, fmt: json.dumps(reply)))


def _run(policy, a=A, b=B):
    """a held, b received: (decision, gate event, a live, b live, live texts)."""
    mem = BaselineMemory(policy, encoder=ENC, nli=NLI)
    da = mem.publish(a)
    dec = mem.write(b)
    live = set(mem.store.live_nodes())
    texts = {mem.store.nodes()[k][1].text for k in live}
    kb = mem.delta(b).claim_key
    return (dec[0].decision if dec else None, dec[0].gates[1] if dec else None,
            da.claim_key in live, kb in live, texts)


def _m(*events) -> str:
    return json.dumps({"memory": [dict(zip(("id", "text", "event"), e)) for e in events]})


@pytest.mark.parametrize("reply, want", [
    (_m(("0", A, "NONE")), ("MERGE", "NONE", True, False)),
    (_m(("0", A, "NONE"), ("1", B, "ADD")), ("INSERT", "ADD", True, True)),
    (_m(("0", B, "UPDATE")), ("MERGE", "UPDATE", False, True)),
    (_m(("0", A, "DELETE"), ("1", B, "ADD")), ("CONFLICT", "DELETE", False, True)),
    (_m(("0", A, "DELETE")), ("CONFLICT", "DELETE", False, False)),
    (_m(("0", A, "UPDATE")), ("MERGE", "UPDATE", True, False)),    # same text: store unchanged
    (_m((0, A, "DELETE"), ("1", B, "ADD")), ("INSERT", "ADD", True, True)),  # int id: skipped
    (_m(("7", A, "DELETE"), ("1", B, "ADD")), ("INSERT", "ADD", True, True)),  # unknown id
    ("```json\n" + _m(("1", B, "ADD")) + "\n```", ("INSERT", "ADD", True, True)),
    ("Sure! " + _m(("1", B, "ADD")) + " Done.", ("INSERT", "ADD", True, True)),
    ("not json at all", ("REJECT", "PARSE_FAIL", True, False)),
    ("", ("REJECT", "PARSE_FAIL", True, False)),            # upstream: empty -> nothing stored
])
def test_mem0_event_mapping(reply, want):
    assert _run(_mem0(reply))[:4] == want


def test_mem0_unchanged_text_update_is_an_update_event_with_the_store_unchanged():
    v = _mem0(_m(("0", A, "UPDATE"), ("1", "France is in Europe", "NONE")))(B, [A, "France is in Europe"])
    assert v == Verdict("MERGE", "UPDATE", (), (), 0, (("UPDATE", 0, A),))


def test_mem0_events_are_upstreams_returned_list_in_index_form():
    v = _mem0(_m(("0", A, "DELETE"), ("1", B, "ADD"), ("1", "x", "UPDATE"), ("9", "y", "DELETE"),
                 ("1", "", "ADD")))(B, [A, "z"])
    assert v.events == (("DELETE", 0, A), ("ADD", None, B), ("UPDATE", 1, "x"))


def test_mem0_update_with_rewritten_text_stores_the_rewrite_only():
    r = _run(_mem0(_m(("0", "Paris is the capital of France and Italy", "UPDATE"))))
    assert r[:4] == ("MERGE", "UPDATE", False, False)
    assert r[4] == {"Paris is the capital of France and Italy"}


def test_mem0_request_is_the_upstream_message_and_counts_parse_failures():
    fake = FakeChatJSON(lambda msgs, fmt: "nope")
    pol = Mem0Policy(fake)
    _run(pol)
    (msgs, fmt), = fake.calls
    assert fmt == {"type": "json_object"}
    assert msgs == [{"role": "user", "content": mp.get_update_memory_messages(
        [{"id": "0", "text": A}], [B])}]
    assert pol.counters == {"calls": 1, "parse_fail": 1, "extract_calls": 0, "no_facts": 0}
    assert pol.name == "mem0-update"


# -- the full upstream pipeline: fact extraction, then the update step -------------------

def _full(facts_reply, update_reply="{}"):
    fake = FakeChatJSON(lambda msgs, fmt: facts_reply if msgs[0]["role"] == "system" else update_reply)
    return Mem0FullPolicy(fake), fake


def test_mem0_full_extracts_then_runs_one_update_call_on_all_facts():
    pol, fake = _full(json.dumps({"facts": ["Paris is in Italy", "Italy has a capital"]}),
                      _m(("0", A, "DELETE"), ("1", "Paris is in Italy", "ADD")))
    r = _run(pol)
    assert r[:4] == ("CONFLICT", "DELETE", False, False) and r[4] == {"Paris is in Italy"}
    (m1, f1), (m2, f2) = fake.calls
    assert f1 == f2 == {"type": "json_object"}
    assert m1 == [{"role": "system", "content": mp.USER_MEMORY_EXTRACTION_PROMPT},
                  {"role": "user", "content": f"Input:\nuser: {B}\n"}]
    assert m2 == [{"role": "user", "content": mp.get_update_memory_messages(
        [{"id": "0", "text": A}], ["Paris is in Italy", "Italy has a capital"])}]
    assert pol.name == "mem0-full" and pol.counters == {
        "calls": 2, "parse_fail": 0, "extract_calls": 1, "no_facts": 0}


@pytest.mark.parametrize("facts_reply", ['{"facts": []}', "", "no json", '{"other": 1}',
                                         "```json\n{\"facts\": []}\n```"])
def test_mem0_full_no_facts_stores_nothing_and_skips_the_update_call(facts_reply):
    pol, fake = _full(facts_reply, _m(("1", B, "ADD")))
    assert _run(pol) == ("REJECT", "NO_FACTS", True, False, {A})
    assert len(fake.calls) == 1 and pol.counters["no_facts"] == 1


def test_mem0_full_facts_parse_like_upstream():
    pol, _ = _full("Here: ```json\n{\"facts\": [{\"fact\": \"a\"}, \"b\"]}\n```")
    assert pol.extract(B) == ["a", "b"]
    pol, _ = _full('Sure {"facts": ["c"]} ok')
    assert pol.extract(B) == ["c"]


@pytest.mark.parametrize("reply, want", [
    ({"duplicate_facts": [0], "contradicted_facts": []}, ("MERGE", "DUPLICATE", True, False)),
    ({"duplicate_facts": [], "contradicted_facts": [0]}, ("CONFLICT", "CONTRADICTED", False, True)),
    ({"duplicate_facts": [], "contradicted_facts": []}, ("INSERT", "NONE", True, True)),
    ({"duplicate_facts": [0], "contradicted_facts": [0]}, ("MERGE", "DUPLICATE", True, False)),
    ({"duplicate_facts": [3], "contradicted_facts": [-1]}, ("INSERT", "NONE", True, True)),
    ({"duplicate_facts": [0]}, ("REJECT", "PARSE_FAIL", True, False)),
])
def test_graphiti_mapping(reply, want):
    assert _run(_graphiti(reply))[:4] == want


def test_graphiti_no_call_on_empty_store_or_exact_match_and_request_shape():
    fake = FakeChatJSON(lambda msgs, fmt: json.dumps({"duplicate_facts": [],
                                                     "contradicted_facts": []}))
    mem = BaselineMemory(GraphitiPolicy(fake), encoder=ENC, nli=NLI)
    assert mem.write(A)[0][1:2] == ("INSERT",) and mem.decisions[0].gates == ("graphiti", "NEW")
    assert mem.write("PARIS  is the capital of france")[0].gates[1] == "EXACT"
    assert fake.calls == []
    mem.write(B)
    (msgs, fmt), = fake.calls
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["schema"] == gp.EDGE_DUPLICATE_SCHEMA
    assert msgs[0]["content"].endswith(gp.get_extraction_language_instruction())
    assert f"'fact': '{A}'" in msgs[1]["content"]


# -- pair-benchmark scoring ------------------------------------------------------------

def test_pair_scoring_flags_false_merge_silent_overwrite_and_missed_contradiction():
    pairs = [("CONFLICT", A, B), ("CONFLICT", A, B), ("RELATE", A, B), ("MERGE", A, B)]
    replies = iter([_m(("0", A, "DELETE"), ("1", B, "ADD")),   # silent overwrite
                    _m(("0", A, "NONE"), ("1", B, "ADD")),      # missed contradiction
                    _m(("0", B, "UPDATE")),                     # false merge (and overwrite)
                    _m(("0", A, "NONE"))])                      # correct merge
    pol = Mem0Policy(FakeChatJSON(lambda msgs, fmt: next(replies)))
    rows = pair_rows(pairs, lambda: BaselineMemory(pol, encoder=ENC, nli=NLI))
    assert [r["pred"] for r in rows] == ["CONFLICT", "NO_RELATION", "MERGE", "MERGE"]
    assert [r["silent_overwrite"] for r in rows] == [True, False, False, False]
    assert [r["missed_contradiction"] for r in rows] == [False, True, False, False]
    assert [r["false_merge"] for r in rows] == [False, False, True, False]
    s = summarize_pairs(rows)
    assert s["silent_overwrite_rate"] == 0.5 and s["missed_contradiction_rate"] == 0.5
    assert s["false_merge_rate"] == pytest.approx(1 / 3)
    assert s["events"] == {"DELETE": 1, "ADD": 1, "UPDATE": 1, "NONE": 1}
    assert s["metrics"]["confusion"]["MERGE"]["MERGE"] == 1
    assert not any(r["claim_lost"] for r in rows) and s["claim_lost_rate"] == 0.0


def test_pair_scoring_a_no_fact_claim_is_lost_and_missed():
    pol, _ = _full('{"facts": []}')
    rows = pair_rows([("MERGE", A, A + " indeed"), ("CONFLICT", A, B)],
                     lambda: BaselineMemory(pol, encoder=ENC, nli=NLI))
    assert [(r["pred"], r["event"], r["a_live"], r["claim_lost"]) for r in rows] == [
        ("NO_RELATION", "NO_FACTS", True, True), ("NO_RELATION", "NO_FACTS", True, True)]
    assert [r["missed_contradiction"] for r in rows] == [False, True]
    s = summarize_pairs(rows)
    assert s["claim_lost_rate"] == 1.0 and s["missed_contradiction_rate"] == 1.0
    assert s["metrics"]["confusion"]["MERGE"]["NO_RELATION"] == 1


# -- harness plug-in: MELD admission never runs on the baseline brain -----------------

def _task() -> HotpotTask:
    return HotpotTask(
        task_id="t1", question="In which city was Christopher Nolan born?", answer="London",
        level="hard", qtype="bridge",
        context=[["Christopher Nolan", ["Christopher Nolan was born in London.",
                                        "Christopher Nolan is a film director."]],
                 ["Inception", ["Inception is a 2010 film."]]],
        supporting_facts=[["Christopher Nolan", 0]])


def _addall(msgs, fmt):
    facts = ast.literal_eval(msgs[0]["content"].split("```")[-2].strip())
    return json.dumps({"memory": [{"id": "9", "text": t, "event": "ADD"} for t in facts]})


def test_build_condition_with_policy_bypasses_admission_on_the_global_brain(monkeypatch):
    calls = {"n": 0}
    real = Admission.admit

    def counting(self, d):
        calls["n"] += 1
        return real(self, d)
    monkeypatch.setattr(Admission, "admit", counting)
    pol = Mem0Policy(FakeChatJSON(_addall))
    b = build_condition(_task(), "distributed_merge", encoder=ENC, nli=NLI, policy=pol)
    sovereign_admissions = calls["n"]
    assert b.condition == "mem0-update" and isinstance(b.brain, BaselineMemory)
    assert pol.counters["calls"] == len(b.brain.decisions) == b.n_total_patches
    assert b.live_units == b.stored_units                 # every fact added, nothing revoked
    assert sovereign_admissions == sum(len(x) for x in b.split.per_brain)   # sovereigns only
    assert b.brain.recall("Nolan born", k=2)


def test_policy_needs_the_distributed_condition():
    with pytest.raises(ValueError):
        build_condition(_task(), "centralized", encoder=ENC, nli=NLI,
                        policy=Mem0Policy(FakeChatJSON(_addall)))


def _extract_all(msgs, fmt):
    if msgs[0]["role"] == "system":
        return json.dumps({"facts": [msgs[1]["content"].split("user: ", 1)[1].strip()]})
    return _addall(msgs, fmt)


def test_build_condition_runs_the_full_arm():
    pol = Mem0FullPolicy(FakeChatJSON(_extract_all))
    b = build_condition(_task(), "distributed_merge", encoder=ENC, nli=NLI, policy=pol)
    assert b.condition == "mem0-full" and pol.counters["extract_calls"] == b.n_total_patches
    assert pol.counters["calls"] == 2 * b.n_total_patches and b.brain.recall("Nolan born", k=2)


def test_recall_rows_accept_baseline_conditions():
    pols = {"mem0-update": Mem0Policy(FakeChatJSON(_addall)),
            "mem0-full": Mem0FullPolicy(FakeChatJSON(_extract_all))}
    rows = recall_rows([_task()], [0.5], [1, 3], ["distributed_merge", *pols], encoder=ENC,
                       nli=NLI, epoch=None, policies=pols)
    assert [(r["condition"], r["k"]) for r in rows] == [
        ("distributed_merge", 1), ("distributed_merge", 3), ("mem0-update", 1), ("mem0-update", 3),
        ("mem0-full", 1), ("mem0-full", 3)]
    assert rows[3]["recall"] == rows[5]["recall"] == 1.0


# -- OpenAI-compatible client: request body, response, cache --------------------------

def test_chat_json_posts_chat_completions_and_caches(tmp_path):
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            out = json.dumps({"choices": [{"message": {"content": '{"memory": []}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = ChatJSON(f"http://127.0.0.1:{srv.server_port}/v1", "m", cache_dir=str(tmp_path))
        msgs = [{"role": "user", "content": "x"}]
        assert c.chat_json(msgs, {"type": "json_object"}) == '{"memory": []}'
        assert c.chat_json(msgs, {"type": "json_object"}) == '{"memory": []}'
    finally:
        srv.shutdown()
    assert len(seen) == 1 and seen[0][0] == "/v1/chat/completions"
    assert seen[0][1] == {"model": "m", "messages": msgs, "response_format": {"type": "json_object"},
                          "temperature": 0.0, "max_tokens": 2000}
    assert c.counters == {"calls": 1, "cache_hits": 1}


# -- pilot script against the fake ----------------------------------------------------

@pytest.mark.parametrize("policy", ["mem0-update", "mem0-full", "graphiti"])
def test_pilot_runs_against_the_fake(tmp_path, policy):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "pilot", Path(__file__).resolve().parents[1] / "scripts" / "run_baseline_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    out = tmp_path / "o.json"
    res = pilot.main(["--policy", policy, "--fake", "--n", "8", "--encoder", ENC, "--nli", "lexical",
                      "--out", str(out)])
    assert res["summary"]["n"] == 8 and len(res["rows"]) == 8
    assert json.loads(out.read_text())["summary"]["n"] == 8


def test_pilot_reads_counters_of_a_client_without_calls(tmp_path, monkeypatch):
    """The real ChatJSON keeps counters and no ``calls`` list; the pilot must not touch ``calls``."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "pilot", Path(__file__).resolve().parents[1] / "scripts" / "run_baseline_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)

    class Client:
        def __init__(self, base_url, model, **kw):
            self.counters = {"calls": 0, "cache_hits": 0}

        def chat_json(self, messages, response_format):
            self.counters["calls"] += 1
            return pilot.toy_respond(messages, response_format)
    monkeypatch.setattr(pilot, "ChatJSON", Client)
    res = pilot.main(["--policy", "graphiti", "--base-url", "http://unused/v1", "--n", "3",
                      "--encoder", ENC, "--nli", "lexical"])
    assert res["llm_counters"]["calls"] >= 1
