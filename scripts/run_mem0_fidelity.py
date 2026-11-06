"""Mem0 fidelity check: the vendored Mem0 write path
(baselines/mem0_policy.py) against real ``mem0ai==1.0.11`` on the same OpenAI-compatible endpoint.

Per case (20 by default: gold seeds 0-9, old memories = canonical + related, new = the paraphrase
or the contradiction): a fresh real ``Memory`` (llm vllm at temperature 0, embedder huggingface
all-MiniLM-L6-v2, vector store faiss) stores the old texts with ``infer=False`` and then adds the
new text with ``infer=True``. The driver records the fact-extraction request and the extracted
facts, what Mem0 fed its update step (the retrieved old memories in Mem0's order), the update
prompt, every raw reply and Mem0's returned events. The vendored side then gets exactly that
input: its extraction request must equal Mem0's, its extraction on the same endpoint must give
the same facts, its update prompt must equal Mem0's byte for byte, and its update step on the
same endpoint must give the same event list: (event, index, text) in order, Mem0's UPDATE/DELETE
ids mapped to indices. A case with no facts makes no update call on either side.
Retrieval is out of scope here (MELD retrieves by sigma, Mem0 by its vector store).

Needs mem0ai==1.0.11 importable (a throwaway venv) and meld's src on PYTHONPATH. The summary and
every row carry topology.stamps with Mem0's --embedder as the encoder (no MELD NLI or policy runs).
Run:  PYTHONPATH=src python scripts/run_mem0_fidelity.py --base-url http://HOST:PORT/v1 \
          --model Qwen/Qwen2.5-7B-Instruct --embedder sentence-transformers/all-MiniLM-L6-v2 \
          --out outputs/mem0-fidelity.json
Re-score a saved run offline (the recorded raw replies stand in for the endpoint; mem0ai not
needed; the extraction request is compared only where the run recorded it):
      PYTHONPATH=src python scripts/run_mem0_fidelity.py --rescore saved.json --embedder E --out rescored.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from typing import Dict, List, Tuple

from meld.agent.llm import ChatJSON, FakeChatJSON
from meld.baselines._mem0_prompts import get_update_memory_messages
from meld.baselines.mem0_policy import Mem0FullPolicy, _parse, extraction_messages
from meld.eval.topology import stamps
from meld.merge.m2_gold import _SEEDS


def cases(n: int = 20) -> List[Dict]:
    out = []
    for sid, _diff, canon, para, rel, conf in _SEEDS[: (n + 1) // 2]:
        out += [{"id": f"{sid}-p", "old": [canon, rel], "new": para},
                {"id": f"{sid}-x", "old": [canon, rel], "new": conf}]
    return out[:n]


def mem0_events(results: List[Dict], old: List[Dict]) -> Tuple[Tuple, ...]:
    """Mem0's returned events as (event, old index or None for ADD, text), the Verdict's events.
    ``old`` is the retrieved list Mem0 numbered, each with its real id under "uuid"."""
    idx = {o["uuid"]: i for i, o in enumerate(old)}
    return tuple((r["event"], None if r["event"] == "ADD" else idx[r["id"]], r["memory"])
                 for r in results)


def vendored_side(rec: Dict, llm) -> Dict:
    """The vendored extraction on Mem0's message, then the vendored update prompt and policy on
    Mem0's own update-step input (its facts and retrieved memories)."""
    pol = Mem0FullPolicy(llm)
    facts = pol.extract(rec["new"])
    sent = rec.get("extract_messages")
    out = {"extract_prompt_equal": None if sent is None else extraction_messages(rec["new"]) == sent,
           "facts_equal": facts == rec["facts"], "vendored_facts": facts, "n_facts": len(rec["facts"]),
           "prompt_equal": None, "events_equal": None}
    if not rec["facts"]:
        return out                              # no facts: no update call on Mem0's side
    old = [{"id": str(i), "text": o["text"]} for i, o in enumerate(rec["old"])]
    out["prompt_equal"] = get_update_memory_messages(old, rec["facts"]) == rec["prompt"]
    v = pol.update(rec["facts"], [o["text"] for o in rec["old"]])
    theirs = mem0_events(rec["results"], rec["old"])
    out.update(events_equal=v.events == theirs, vendored=[list(e) for e in v.events],
               mem0=[list(e) for e in theirs], vendored_event=v.event,
               parse_fail=pol.counters["parse_fail"])
    return out


def replay(rec: Dict) -> FakeChatJSON:
    """The recorded raw replies as the endpoint: the extraction reply, then the update reply."""
    replies = rec["replies"]
    return FakeChatJSON(lambda m, rf: replies[0] if m[0]["role"] == "system"
                        else (replies[1] if len(replies) > 1 else ""))


def real_side(case: Dict, *, base_url: str, model: str, embedder: str, workdir: str) -> Dict:
    """One case through real mem0ai 1.0.11; records the update-step input, prompt, reply, results."""
    import mem0.memory.main as M
    from mem0 import Memory
    rec: Dict = {"id": case["id"], "new": case["new"], "extract_messages": None}
    orig = M.get_update_memory_messages

    def spy(old, facts, custom=None):
        rec["old"] = [{"uuid": None, **o} for o in copy.deepcopy(old)]
        rec["facts"] = list(facts)
        rec["prompt"] = orig(old, facts, custom)
        return rec["prompt"]
    cfg = {"llm": {"provider": "vllm", "config": {"model": model, "vllm_base_url": base_url,
                                                   "temperature": 0.0, "max_tokens": 2000}},
           "embedder": {"provider": "huggingface", "config": {"model": embedder}},
           "vector_store": {"provider": "faiss", "config": {"path": os.path.join(workdir, case["id"]),
                                                            "embedding_model_dims": 384}},
           "history_db_path": os.path.join(workdir, case["id"] + ".db")}
    m = Memory.from_config(cfg)
    gen = m.llm.generate_response
    replies = []

    def spy_llm(*a, **k):
        msgs = k.get("messages", a[0] if a else None)
        if rec["extract_messages"] is None and msgs and msgs[0]["role"] == "system":
            rec["extract_messages"] = copy.deepcopy(msgs)
        r = gen(*a, **k)
        replies.append(r)
        return r
    m.llm.generate_response = spy_llm
    uuids = {}
    for t in case["old"]:
        for r in m.add(t, user_id=case["id"], infer=False)["results"]:
            uuids[r["memory"]] = r["id"]
    M.get_update_memory_messages = spy
    try:
        res = m.add(case["new"], user_id=case["id"], infer=True)["results"]
    finally:
        M.get_update_memory_messages = orig
    rec["results"] = [{k: r.get(k) for k in ("id", "memory", "event")} for r in res]
    rec.setdefault("old", [])
    rec.setdefault("facts", [])
    rec.setdefault("prompt", None)
    # Mem0 renumbered the retrieved memories "0".."n-1" in place; recover their real ids by text
    for o in rec["old"]:
        o["uuid"] = uuids.get(o["text"])
    rec["replies"] = replies
    rec["update_reply_parsed"] = _parse(replies[-1]) if len(replies) > 1 else None
    return rec


def summarize(rows: List[Dict], http_calls: int) -> Dict:
    c = [r["check"] for r in rows]
    return {"cases": len(rows),
            "extract_prompt_recorded": sum(x["extract_prompt_equal"] is not None for x in c),
            "extract_prompt_equal": sum(bool(x["extract_prompt_equal"]) for x in c),
            "facts_equal": sum(x["facts_equal"] for x in c),
            "no_fact_cases": sum(x["n_facts"] == 0 for x in c),
            "prompt_equal": sum(bool(x["prompt_equal"]) for x in c),
            "events_compared": sum(x["events_equal"] is not None for x in c),
            "events_equal": sum(bool(x["events_equal"]) for x in c),
            "http_calls_vendored": http_calls}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--embedder", required=True, help="Mem0's embedder, e.g. sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rescore", help="a saved run: re-score it offline from its recorded replies")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.rescore:
        rows = json.load(open(a.rescore))["rows"]
        for rec in rows:
            rec.setdefault("extract_messages", None)
            rec["check"] = vendored_side(rec, replay(rec))
        calls = 0
    else:
        if not (a.base_url and a.model):
            ap.error("--base-url and --model (or --rescore)")
        llm = ChatJSON(a.base_url, a.model, temperature=0.0, max_tokens=2000)
        rows = []
        with tempfile.TemporaryDirectory() as wd:
            for c in cases(a.n):
                rec = real_side(c, base_url=a.base_url, model=a.model, embedder=a.embedder, workdir=wd)
                rec["check"] = vendored_side(rec, llm)
                rows.append(rec)
                print(c["id"], rec["check"])
        calls = llm.counters["calls"]
    st = stamps(a.embedder, None, None, pol=None)
    for rec in rows:
        rec.update(st)
    summary = {**summarize(rows, calls), **st}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(summary)
    return summary


if __name__ == "__main__":
    main()
