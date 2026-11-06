"""Arguments shared by the runners: the group key file and the receive-path baseline arms.

``--key-file PATH``: the group HMAC key as the file's bytes, stripped (a binary key cannot travel
through ``SMP_GROUP_KEY``, which is UTF-8 text). It excludes ``SMP_GROUP_KEY``; without either
the runner keeps ``wire_delta.group_key``'s default. The key is never printed.

``--baseline NAME`` (mem0-update, mem0-full, graphiti, caura, caura-xagent): a baseline policy
(meld.baselines) on an OpenAI-compatible endpoint (``--llm-base-url``, ``--llm-model``; every
response cached under ``--llm-cache-dir``), run as one more condition next to the runner's own.
``--baseline-embedder`` sets the embedder of a policy that has its own (Caura; default its local
default, BAAI/bge-large-en-v1.5); rows of such a policy stamp it as ``baseline_embedder``.
"""
from __future__ import annotations

import os

from meld.agent.llm import ChatJSON
from meld.baselines.caura_policy import DEFAULT_EMBEDDER, CauraCrossAgentPolicy, CauraPolicy
from meld.baselines.graphiti_policy import GraphitiPolicy
from meld.baselines.mem0_policy import Mem0FullPolicy, Mem0Policy
from meld.sync.wire_delta import group_key

BASELINES = {p.name: p for p in (Mem0Policy, Mem0FullPolicy, GraphitiPolicy, CauraPolicy,
                                 CauraCrossAgentPolicy)}


def add_key_file(ap) -> None:
    ap.add_argument("--key-file", help="group key file (raw bytes, stripped); excludes SMP_GROUP_KEY")


def read_key_file(path: str) -> bytes:
    try:
        with open(path, "rb") as fh:
            key = fh.read().strip()
    except OSError as e:
        raise SystemExit(f"key file not readable: {path} ({e.strerror})")
    if not key:
        raise SystemExit(f"empty key file {path}")
    return key


def runner_key(key_file, env=os.environ) -> bytes:
    """The key file's bytes; without one, ``group_key(env)``."""
    if key_file is None:
        return group_key(env)
    if env.get("SMP_GROUP_KEY"):
        raise SystemExit("--key-file and SMP_GROUP_KEY are mutually exclusive")
    return read_key_file(key_file)


def add_baseline(ap) -> None:
    ap.add_argument("--baseline", choices=sorted(BASELINES))
    ap.add_argument("--llm-base-url", help="OpenAI-compatible base URL, e.g. http://h:p/v1")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--llm-cache-dir")
    ap.add_argument("--baseline-embedder", help="embedder of a baseline that has its own (caura)")


def baseline_policy(a, llm=None):
    """The ``--baseline`` policy (None without one) on ``llm``, else on a ChatJSON client."""
    if not a.baseline:
        return None
    if llm is None:
        if not a.llm_base_url:
            raise SystemExit("--baseline needs --llm-base-url")
        llm = ChatJSON(a.llm_base_url, a.llm_model, cache_dir=a.llm_cache_dir)
    return BASELINES[a.baseline](llm, **embedder_kw(a.baseline, a.baseline_embedder))


def embedder_kw(name: str, embedder) -> dict:
    """``{"embedder": embedder}`` for a policy with its own embedder; refuses one for any other."""
    if not issubclass(BASELINES[name], CauraPolicy):
        if embedder:
            raise SystemExit(f"--baseline-embedder: {name} has no embedder of its own")
        return {}
    return {"embedder": embedder} if embedder else {}


def add_read_tau(ap) -> None:
    """--read-tau: the read's near-duplicate threshold (recall_metrics.read_top), stamped on every
    row as ``read_tau``; default the fixed 0.95, ``none`` the reported wp6c read."""
    from meld.agent.recall_metrics import READ_TAU
    ap.add_argument("--read-tau", default=READ_TAU,
                    type=lambda s: None if s.lower() == "none" else float(s))


def read_stamp(a) -> dict:
    """What every recall row, cell and report carries about its read: ``read_tau`` and
    ``read_sim`` (what the near-duplicate test compares; r1.13 = the replicated int8 q)."""
    from meld.agent.recall_metrics import READ_SIM
    return {"read_tau": a.read_tau, "read_sim": READ_SIM}


def baseline_stamp(a) -> dict:
    """What a baseline row carries besides the provenance stamps."""
    out = {"baseline": a.baseline, "llm_model": a.llm_model}
    if a.baseline and issubclass(BASELINES[a.baseline], CauraPolicy):
        out["baseline_embedder"] = getattr(a, "baseline_embedder", None) or DEFAULT_EMBEDDER
    return out


__all__ = ["BASELINES", "embedder_kw", "add_key_file", "read_key_file", "runner_key", "add_baseline",
           "baseline_policy", "baseline_stamp", "add_read_tau", "read_stamp"]
