"""Pilot: run a Mem0 arm (mem0-update: the update step on the claim; mem0-full: fact extraction,
then the update step), the Graphiti receive-path policy or the Caura write path (caura: write gate
then contradiction detector; caura-xagent: the arrival from another agent, contradiction detector
only) on N labelled claim pairs.

Each pair holds text_a and receives text_b through the policy (meld.baselines.memory); the
output has one row per pair and the pair-benchmark summary (four-way metrics, false-merge,
silent-overwrite and missed-contradiction rates, event counts) plus the LLM counters.

  # against an OpenAI-compatible server (vLLM): every response cached under --cache-dir
  PYTHONPATH=src python scripts/run_baseline_pilot.py --policy mem0-full --n 50 --encoder minilm \
      --nli mnli-cross-encoder --base-url http://HOST:PORT/v1 --model Qwen/Qwen2.5-7B-Instruct \
      --pairs pairs.jsonl
  # offline wiring check: a toy token-overlap responder stands in for the model
  PYTHONPATH=src python scripts/run_baseline_pilot.py --policy graphiti --fake --encoder hashing --nli lexical

The encoder ranks the held memories the policy sees; the NLI is the replica's (the baseline policy
decides, so it is not consulted), stamped with the rest of topology.stamps on every row, with the
row's epoch bin (None: a baseline memory holds no epoch).

``--pairs`` is a JSONL of benchmark rows (``text_a``, ``text_b``, ``label``; unlabelled rows
are skipped); without it the built-in gold set is used. ``--k`` defaults to the policy's own (5 for
Mem0 and Graphiti, Caura's candidate window 20); ``--baseline-embedder`` sets Caura's embedder
(default BAAI/bge-large-en-v1.5; ``hashing`` for an offline check).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from meld.agent.llm import ChatJSON, FakeChatJSON                          # noqa: E402
from meld.baselines.memory import BaselineMemory, pair_rows, summarize_pairs  # noqa: E402
from meld.eval.runner_args import BASELINES as POLICIES, embedder_kw      # noqa: E402
from meld.eval.topology import stamps                                      # noqa: E402
from meld.merge.m2_eval import M2_GOLD                                     # noqa: E402


def _toks(t: str) -> set:
    return set(re.findall(r"\w+", t.lower()))


def _overlap(a: str, b: str) -> float:
    x, y = _toks(a), _toks(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def toy_respond(messages, response_format) -> str:
    """Offline stand-in: overlap >= 0.8 duplicate, >= 0.5 contradiction, else new; Mem0's fact
    extraction returns the user line as the one fact."""
    text = messages[-1]["content"]
    if text.startswith("You are a deduplication judge"):                  # Caura write gate
        new, old = (text.split(f"Statement {s}: ", 1)[1].split("\n", 1)[0] for s in ("A (NEW)", "B (EXISTING)"))
        return json.dumps({"is_duplicate": _overlap(new, old) >= 0.8, "reason": "toy"})
    if text.startswith("You are a contradiction detector"):               # Caura contradiction judge
        if "NEW statement:\n" in text:                                    # batch
            new = text.split("NEW statement:\n", 1)[1].split("\n", 1)[0]
            cands = re.findall(r"^\[(\d+)\] (.*)$", text.split("judge each, by index):\n", 1)[1], re.M)
            hits = {i: {"same_subject": True, "non_conflict_reason": "none", "contradicts": True}
                    for i, c in cands if 0.5 <= _overlap(new, c) < 0.8}
            return json.dumps({"clean": [int(i) for i, _ in cands if i not in hits], "hits": hits})
        new, old = (text.split(f"Statement {s}: ", 1)[1].split("\n", 1)[0] for s in ("A (NEW)", "B (EXISTING)"))
        c = 0.5 <= _overlap(new, old) < 0.8
        return json.dumps({"same_subject": True, "non_conflict_reason": "none", "contradicts": c, "reason": "toy"})
    if messages[0]["role"] == "system" and response_format["type"] == "json_object":
        return json.dumps({"facts": [text.split("user: ", 1)[1].strip()]})
    if response_format["type"] == "json_schema":          # Graphiti resolve_edge
        new = text.split("<NEW FACT>")[1].split("</NEW FACT>")[0].strip()
        old = ast.literal_eval(text.split("<EXISTING FACTS>")[1].split("</EXISTING FACTS>")[0].strip())
        ov = [(e["idx"], _overlap(new, e["fact"])) for e in old]
        return json.dumps({"duplicate_facts": [i for i, o in ov if o >= 0.8],
                           "contradicted_facts": [i for i, o in ov if 0.5 <= o < 0.8]})
    blocks = text.split("```")                              # Mem0 update step
    new = ast.literal_eval(blocks[-2].strip())[0]
    old = ast.literal_eval(blocks[1].strip()) if len(blocks) > 3 else []
    events = [{"id": m["id"], "text": m["text"],
               "event": "NONE" if _overlap(new, m["text"]) >= 0.8 or _overlap(new, m["text"]) < 0.5
               else "DELETE"} for m in old]
    if not any(_overlap(new, m["text"]) >= 0.8 for m in old):
        events.append({"id": str(len(old)), "text": new, "event": "ADD"})
    return json.dumps({"memory": events})


def load_pairs(path):
    if not path:
        return list(M2_GOLD)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [(r["label"], r["text_a"], r["text_b"]) for r in rows if r.get("label")]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", choices=sorted(POLICIES), required=True)
    ap.add_argument("--n", type=int, default=0, help="first N pairs (0 = all)")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--fake", action="store_true", help="toy offline responder, no endpoint")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, e.g. http://h:p/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--k", type=int, default=None, help="claims the policy sees (default: the policy's)")
    ap.add_argument("--baseline-embedder", default=None, help="embedder of a policy with its own (caura)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.fake == bool(a.base_url):
        ap.error("exactly one of --fake and --base-url")
    llm = (FakeChatJSON(toy_respond) if a.fake else
           ChatJSON(a.base_url, a.model, temperature=a.temperature, max_tokens=a.max_tokens,
                    cache_dir=a.cache_dir))
    policy = POLICIES[a.policy](llm, **({"k": a.k} if a.k else {}),
                                **embedder_kw(a.policy, a.baseline_embedder))
    if hasattr(policy, "embedder_name"):
        a.baseline_embedder = policy.embedder_name
    pairs = load_pairs(a.pairs)
    pairs = pairs[:a.n] if a.n else pairs
    rows = pair_rows(pairs, lambda: BaselineMemory(policy, encoder=a.encoder, nli=a.nli))
    st = stamps(a.encoder, a.nli, None)
    for r, (_, ta, tb) in zip(rows, pairs):
        r.update(text_a=ta, text_b=tb, **st, bin=None)     # no epoch: no bin
    res = {"config": {k: v for k, v in vars(a).items() if k != "out"},
           "summary": {**summarize_pairs(rows), **st}, "policy_counters": policy.counters,
           "llm_counters": llm.counters if hasattr(llm, "counters") else {"calls": len(llm.calls)}, "rows": rows}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    s = res["summary"]
    print(f"{a.policy}: n={s['n']} false_merge={s['false_merge_rate']:.3f} "
          f"silent_overwrite={s['silent_overwrite_rate']:.3f} "
          f"missed_contradiction={s['missed_contradiction_rate']:.3f} "
          f"parse_fail={policy.counters['parse_fail']} events={s['events']}")
    return res


if __name__ == "__main__":
    main()
