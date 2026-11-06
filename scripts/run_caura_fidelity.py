"""Caura fidelity check: the vendored write path (baselines/caura_policy.py) against Caura's own
functions from the pinned tree (vendor/caura/src, commit fd1da78b) on the same OpenAI-compatible
endpoint and the same embedder.

Two steps, two interpreters (Caura needs Python >= 3.12 and its own dependencies; MELD runs on its
own venv):

1. ``--real`` (a throwaway venv with core-api's locked dependencies and sentence-transformers):
   per case (gold seeds 0-9: held = canonical + related, arriving = the paraphrase or the
   contradiction) Caura's ``LocalEmbedding`` embeds every text, ``CheckSemanticDuplicate.execute``
   runs on a ``PipelineContext`` (strong mode, tenant defaults) and, when it does not refuse the
   write, ``contradiction_detector._detect`` runs on the new row. Storage is an in-memory stand-in
   answering ``find_semantic_duplicate`` (nearest held row at cosine >= min_similarity) and
   ``find_similar_candidates`` (held rows at cosine >= 0.45, nearest first, at most 20) as the
   storage SQL does, and recording ``batch_update_status``; ``_record_conflicts`` is recorded,
   not run. The driver records every wire request the OpenAI SDK sends and its raw reply, the
   vectors, and Caura's outcome (409 or not, rows set to ``conflicted``).
   Caura has no setting for the chat base URL (its OpenAI provider is pinned to
   api.openai.com), so this driver points ``common.llm._credentials.OPENAI_CHAT_BASE_URL`` at
   ``--base-url`` in-process; the key is a dummy. An audit hook refuses any socket connection to
   a non-loopback address, telemetry is off (CAURA_TELEMETRY=off, DO_NOT_TRACK=1; the heartbeat
   only starts with the server app, which is never built), and the embedder loads offline.
2. ``--rescore`` (MELD's venv): the vendored policy gets each case with Caura's recorded vectors
   as its embedder and Caura's recorded replies as the endpoint; its requests must equal Caura's
   (prompt, response_format, temperature) in order, and its event, retired rows and target must
   equal Caura's outcome. With ``--base-url`` the vendored policy is also run live on the endpoint
   (its own embedder) and its event compared again.

  # step 1 (Caura venv), e.g. against a local Ollama
  CAURA_SRC=vendor/caura/src $CAURA_VENV/bin/python -I scripts/run_caura_fidelity.py --real \\
      --base-url http://127.0.0.1:11434/v1 --model qwen2.5:7b --out outputs/wp6f/fidelity/real.json
  # step 2 (MELD venv)
  PYTHONPATH=src .venv/bin/python scripts/run_caura_fidelity.py --rescore outputs/wp6f/fidelity/real.json \\
      --base-url http://127.0.0.1:11434/v1 --model qwen2.5:7b --out outputs/wp6f/fidelity/check.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EMBEDDER = "BAAI/bge-large-en-v1.5"
NEW_ID = str(uuid.UUID(int=255))            # held rows are UUID(int=i + 1)


def cases(n: int = 20) -> List[Dict]:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from meld.merge.m2_gold import _SEEDS
    out = []
    for sid, _diff, canon, para, rel, conf in _SEEDS[: (n + 1) // 2]:
        out += [{"id": f"{sid}-p", "old": [canon, rel], "new": para},
                {"id": f"{sid}-x", "old": [canon, rel], "new": conf}]
    return out[:n]


def cosine(a, b) -> float:
    """pgvector's 1 - cosine distance over stored float32 vectors."""
    import numpy as np
    x, y = np.asarray(a, dtype=np.float32).astype(np.float64), np.asarray(b, dtype=np.float32).astype(np.float64)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def _loopback_only(event, args):
    if event == "socket.connect":
        addr = args[1]
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if isinstance(addr, tuple) and host not in ("127.0.0.1", "::1", "localhost"):
            raise PermissionError(f"non-loopback connection refused: {host}")


# -- step 1: Caura's own functions ----------------------------------------------------

def real_side(cs: List[Dict], *, base_url: str, model: str, embedder: str, caura_src: str) -> List[Dict]:
    import asyncio
    from types import SimpleNamespace

    os.environ.update(CAURA_TELEMETRY="off", DO_NOT_TRACK="1", OPENAI_API_KEY="local-endpoint",
                      HF_HUB_OFFLINE="1", SENTRY_DSN="")
    sys.addaudithook(_loopback_only)
    sys.path[:0] = [caura_src, os.path.join(caura_src, "core-api", "src")]
    import common.llm._credentials as CR
    CR.OPENAI_CHAT_BASE_URL = base_url              # the one transport patch (module doc)
    import openai.resources.chat.completions as OC
    from fastapi import HTTPException
    import core_api.pipeline.steps.write.check_semantic_duplicate as CSD
    import core_api.services.contradiction_detector as CD
    from common.constants import CONTRADICTION_CANDIDATE_WINDOW, CONTRADICTION_SIMILARITY_THRESHOLD
    from common.embedding.providers.local import LocalEmbedding
    from core_api.pipeline.context import PipelineContext

    calls: List[Dict] = []
    create = OC.AsyncCompletions.create

    async def spy_create(self, **kw):
        r = await create(self, **kw)
        calls.append({"request": {k: kw[k] for k in ("model", "messages", "response_format", "temperature",
                                                       "max_completion_tokens") if k in kw},
                      "content": r.choices[0].message.content})
        return r
    OC.AsyncCompletions.create = spy_create
    emb = LocalEmbedding(embedder)
    tenant = SimpleNamespace(semantic_dedup_enabled=True, entity_extraction_provider="openai",
                             entity_extraction_model=model, resolve_fallback=lambda: (None, None))

    async def one(case):
        vec = {t: await emb.embed(t) for t in [case["new"], *case["old"]]}
        rows = [{"id": str(uuid.UUID(int=i + 1)), "content": t, "status": "active", "subject_entity_id": None,
                 "created_at": f"2000-01-0{i + 1}T00:00:00+00:00", "visibility": "scope_team"}
                for i, t in enumerate(case["old"])]
        sims = [cosine(vec[case["new"]], vec[t]) for t in case["old"]]
        near = sorted(range(len(rows)), key=lambda i: -sims[i])

        async def find_semantic_duplicate(*a, min_similarity=None, **k):
            j = near[0]
            return {**rows[j], "similarity": sims[j]} if sims[j] >= min_similarity else None
        updates, recorded = [], []

        class SC:
            async def find_similar_candidates(self, payload):
                return [rows[i] for i in near if sims[i] >= CONTRADICTION_SIMILARITY_THRESHOLD][
                    :CONTRADICTION_CANDIDATE_WINDOW]

            async def batch_update_status(self, payload, tenant_id=None):
                updates.extend(payload["updates"])
                return {}

            async def find_rdf_conflicts(self, *a, **k):
                return []

        async def record_conflicts(new_memory, pairs, unlinked, **k):
            recorded.extend(uuid.UUID(str(p[0]["id"])).int - 1 for p in unlinked)
        CSD._find_semantic_duplicate = find_semantic_duplicate
        CD.get_storage_client = lambda: SC()
        CD._record_conflicts = record_conflicts
        n0 = len(calls)
        data = SimpleNamespace(content=case["new"], tenant_id="t", fleet_id=None, visibility="scope_team",
                               agent_id="a", subject_entity_id=None)
        ctx = PipelineContext(data={"input": data, "embedding": vec[case["new"]],
                                    "memory_fields": {"metadata": {}}}, tenant_config=tenant)
        out = {"sims": sims, "conflicted": [], "status_409": False}
        try:
            await CSD.CheckSemanticDuplicate().execute(ctx)
        except HTTPException as e:
            out.update(status_409=True, detail=e.detail if isinstance(e.detail, (str, dict)) else str(e.detail))
        if not out["status_409"]:
            new_mem = {"id": NEW_ID, "tenant_id": "t", "fleet_id": None, "content": case["new"],
                       "status": "active", "visibility": "scope_team", "agent_id": "a",
                       "created_at": "2000-02-01T00:00:00+00:00", "supersedes_id": None}
            await CD._detect(new_mem, vec[case["new"]], tenant)
            out["conflicted"] = [uuid.UUID(u["memory_id"]).int - 1 for u in updates
                                 if u["memory_id"] != NEW_ID and u.get("status") == "conflicted"]
            out["supersedes"] = next((uuid.UUID(u["supersedes_id"]).int - 1 for u in updates
                                      if u["memory_id"] == NEW_ID and u.get("supersedes_id")), None)
            out["recorded_unlinked"] = recorded
        out["calls"] = calls[n0:]
        out["vectors"] = {t: list(map(float, v)) for t, v in vec.items()}
        return {**case, "real": out}
    async def run_all():                     # one loop: Caura caches its provider's http client
        return [await one(c) for c in cs]
    return asyncio.run(run_all())


def real_event(r: Dict) -> Dict:
    """Caura's outcome in the vendored policy's terms."""
    if r["status_409"]:
        return {"event": "DUP_JUDGE" if r["calls"] else "DUP_AUTO", "revoke": [], "target": None}
    if r["conflicted"]:
        return {"event": "CONFLICTED", "revoke": r["conflicted"], "target": r["supersedes"]}
    return {"event": "CLEAR" if r["calls"] else "NO_CANDIDATE", "revoke": [], "target": None}


# -- step 2: the vendored policy on the recorded input ---------------------------------

def vendored_side(rec: Dict, live=None) -> Dict:
    """``live``: a CauraPolicy on the real endpoint with its own embedder, or None."""
    import numpy as np
    from meld.agent.llm import FakeChatJSON
    from meld.baselines.caura_policy import CauraPolicy
    real = rec["real"]
    replies = {json.dumps(c["request"]["messages"]): c["content"] for c in real["calls"]}
    fake = FakeChatJSON(lambda m, rf: replies.get(json.dumps(m), ""))
    vecs = real["vectors"]
    pol = CauraPolicy(fake, embedder=lambda texts: np.array([vecs[t] for t in texts], dtype=np.float32))
    v = pol(rec["new"], rec["old"])
    want = real_event(real)
    sent = [{"messages": m, "response_format": rf, "temperature": 0.0} for m, rf in fake.calls]
    theirs = [{k: c["request"][k] for k in ("messages", "response_format", "temperature")} for c in real["calls"]]
    out = {"requests_equal": sent == theirs, "n_requests": len(theirs),
           "event_equal": v.event == want["event"], "vendored_event": v.event, "caura_event": want["event"],
           "revoke_equal": sorted(v.revoke) == sorted(want["revoke"]),
           "target_equal": want["event"] != "CONFLICTED" or v.target == want["target"]}
    if live is not None:
        lv = live(rec["new"], rec["old"])
        x = live._vectors([rec["new"], *rec["old"]])
        out.update(live_event=lv.event, live_event_equal=lv.event == want["event"],
                   live_sims=[float(y) for y in x[1:] @ x[0]])
    return out


def summarize(rows: List[Dict]) -> Dict:
    c = [r["check"] for r in rows]
    s = {"cases": len(rows), "requests_compared": sum(x["n_requests"] for x in c),
         "requests_equal": sum(x["requests_equal"] for x in c), "events_equal": sum(x["event_equal"] for x in c),
         "revoke_equal": sum(x["revoke_equal"] for x in c), "target_equal": sum(x["target_equal"] for x in c),
         "caura_events": {}}
    for x in c:
        s["caura_events"][x["caura_event"]] = s["caura_events"].get(x["caura_event"], 0) + 1
    if c and "live_event_equal" in c[0]:
        s["live_events_equal"] = sum(x["live_event_equal"] for x in c)
        s["max_abs_sim_diff"] = max(abs(a - b) for r, x in zip(rows, c)
                                    for a, b in zip(x["live_sims"], r["real"]["sims"]))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real", action="store_true", help="step 1: run Caura's functions (Caura venv)")
    ap.add_argument("--rescore", help="step 2: a step-1 output, checked against the vendored policy")
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--embedder", default=EMBEDDER)
    ap.add_argument("--caura-src", default=os.environ.get("CAURA_SRC", os.path.join(ROOT, "vendor", "caura", "src")))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.real == bool(a.rescore):
        ap.error("exactly one of --real and --rescore")
    if a.real:
        if not a.base_url:
            ap.error("--real needs --base-url")
        rows = real_side(cases(a.n), base_url=a.base_url, model=a.model, embedder=a.embedder,
                         caura_src=os.path.abspath(a.caura_src))
        res = {"config": {k: v for k, v in vars(a).items() if k != "out"}, "rows": rows}
    else:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from meld.agent.llm import ChatJSON
        from meld.baselines.caura_policy import CauraPolicy
        src = json.load(open(a.rescore))
        rows = src["rows"]
        live = (CauraPolicy(ChatJSON(a.base_url, src["config"]["model"], temperature=0.0),
                            embedder=src["config"]["embedder"]) if a.base_url else None)
        for r in rows:
            r["check"] = vendored_side(r, live)
            print(r["id"], r["check"]["caura_event"], r["check"]["vendored_event"], r["check"]["requests_equal"])
        res = {"config": {**src["config"], "rescore": a.rescore}, "summary": summarize(rows), "rows": rows}
        print(res["summary"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
