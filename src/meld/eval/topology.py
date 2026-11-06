"""N-brains driver: one world (a SplitPlan of per-agent claim lists) built under a condition and
a topology over the current replica objects.

Conditions:
  * ``centralized``  -- one memory writes every claim (pass the one-brain plan of the world);
  * ``meld``         -- N agent memories each write their own claims (admission under the
                        epoch), then exchange signed deltas by the topology;
  * ``exact_dedup``  -- the same without an epoch: no similarity merge, exact copies are one node;
  * ``naive_union``  -- every claim enters one store's D with no admission (the topology does
                        not change a union);
  * any name in ``baselines``: a callable (plan, kw) -> SMPMemory, kw the build's encoder, nli,
    epoch and cache_dir (the slot for merge-on-receive baseline policies); a callable may
    instead return a whole build (a dict, e.g. ``build(plan, "meld", "star", sink=...)``),
    which is returned as it is.

``sink`` (star only) replaces the sink memory, e.g. by a ``baselines.memory.BaselineMemory``
whose policy decides on every received delta in the same canonical order.

Topologies (``meld`` and ``exact_dedup``):
  * ``star``        -- every agent's deltas go to a sink memory in canonical order
                       (created_at, id), so no admission refolds; the sink is the readout;
  * ``all_to_all``  -- agents are BrainNodes that broadcast their outboxes to every peer
                       (sync.experiment.run_sync), then ``settle`` runs the pass; replica 0 is
                       the readout and the convergence report (post-pass divergence, L,
                       certificate fingerprints) is returned.

Every replica gets its own NLI provider and model instance (``nli_provider(own=True)``), and
the all-to-all path goes through ``settle``'s own-model guard. A plan's ``when`` is the world
time of each text; writes use it as created_at, so canonical order is world-time order (without
it a shared counter in ingestion order is used). ``stamps`` gives the provenance every output
row carries.
"""
from __future__ import annotations

import functools
import itertools
import os
import subprocess
import time
from collections import Counter
from typing import Callable, Dict, Optional

from meld.agent.mem_conditions import SplitPlan
from meld.agent.memory import DEFAULT_POL, DEFAULT_SCOPE, SMPMemory
from meld.merge import signals
from meld.merge.conformal import bin_of
from meld.proto import CANON_VERSION
from meld.protocol.fold import nli_provider
from meld.protocol.objects import canonical_bytes, m_nli, m_pol_of, object_id
from meld.sync.experiment import run_sync
from meld.sync.node import BrainNode

CONDITIONS = ("centralized", "meld", "exact_dedup", "naive_union")
TOPOLOGIES = ("star", "all_to_all")


def grid(ns, ds):
    """(N, d) cells with d <= N, so the duplication structure changes with N."""
    return [(n, d) for n in ns for d in ds if d <= n]


@functools.lru_cache(maxsize=None)
def _git_sha() -> Optional[str]:
    """HEAD of this checkout, suffixed "-dirty" when tracked files differ (read once per process);
    in an export without .git, MELD_GIT_SHA (set by whoever shipped the export)."""
    here = os.path.dirname(os.path.abspath(__file__))
    git = lambda *a: subprocess.run(["git", *a], cwd=here, capture_output=True,  # noqa: E731
                                    text=True).stdout.strip()
    sha = git("rev-parse", "HEAD") or None
    if sha is None:
        return os.environ.get("MELD_GIT_SHA") or None
    if git("status", "--porcelain", "--untracked-files=no"):
        sha += "-dirty"
    return sha


def bin_for(n_prior: int, epoch) -> Optional[int]:
    """The epoch's calibration bin of a decision taken at store size ``n_prior`` (None without an epoch)."""
    return None if epoch is None else bin_of(n_prior, epoch.bins[0])


def stamps(encoder: Optional[str], nli: Optional[str], epoch, pol: Optional[dict] = DEFAULT_POL) -> dict:
    """The provenance every result row carries: git sha (with "-dirty"), M_nli, M_pol, epoch id and
    seq (None without an epoch), and the NLI and encoder model ids. ``nli`` or ``encoder`` None (a
    run that uses no such model) gives None for it and for M_nli; ``pol`` None gives no M_pol.
    Per-decision and per-bin rows add ``bin`` (``bin_for``)."""
    return {"git_sha": _git_sha(),
            "m_nli": m_nli(encoder, nli, "int8", CANON_VERSION).hex() if encoder and nli else None,
            "m_pol": m_pol_of(pol).hex() if pol is not None else None,
            "epoch_id": object_id(canonical_bytes(epoch)).hex() if epoch is not None else None,
            "seq": epoch.seq if epoch is not None else None,
            "nli_id": signals._NLI_MODEL_IDS.get(nli, nli),
            "encoder": signals._SentenceTransformerEncoder._MODEL_IDS.get(encoder, encoder)}


def _own(nli: str, calls: list):
    """An own-instance provider that counts its calls into ``calls[0]``."""
    p = nli_provider(nli, own=True)

    def f(a, b):
        calls[0] += 1
        return p(a, b)
    f.model = p.model
    return f


def build(plan: SplitPlan, condition: str, topology: str = "star", *, encoder: str, nli: str,
          epoch=None, cache_dir: Optional[str] = None,
          baselines: Optional[Dict[str, Callable]] = None, sink=None) -> dict:
    """The world under one condition and topology: {memory (the readout), stored, live,
    decisions (count per kind), decisions_by_bin (epoch bin -> count per kind; bin None without
    an epoch), nli_calls, wall_s, convergence (all_to_all only)}."""
    t0 = time.perf_counter()
    kw = dict(encoder=encoder, nli=nli, epoch=epoch, cache_dir=cache_dir)
    calls, cur, tick = [0], [0], itertools.count(1)
    clock = lambda: cur[0]                                  # noqa: E731

    def feed(write, texts):
        for t in texts:
            cur[0] = plan.when[t] + 1 if plan.when else next(tick)
            write(t)

    def mem(origin, ep, provider):
        return SMPMemory(encoder=encoder, nli=nli, epoch=ep, cache_dir=cache_dir, origin=origin,
                         clock=clock, provider=provider)

    conv = None
    if condition in (baselines or {}):
        out = baselines[condition](plan, kw)
        if isinstance(out, dict):
            return out
    elif condition == "naive_union":
        out = mem("union", epoch, None)
        for texts in plan.per_brain:
            feed(out.publish, texts)
    elif condition == "centralized":
        if len(plan.per_brain) != 1:
            raise ValueError("centralized takes the one-brain plan of the world")
        out = mem("central", epoch, _own(nli, calls))
        feed(out.write, plan.per_brain[0])
    elif condition in ("meld", "exact_dedup") and topology in TOPOLOGIES:
        ep = epoch if condition == "meld" else None
        if topology == "star":
            held = []
            for i, texts in enumerate(plan.per_brain):
                # ponytail: an agent (and its model instance) is dropped once its deltas are
                # collected; a star sink never reads an agent again, so peak memory is two models
                agent = mem(f"agent{i}", ep, _own(nli, calls))
                feed(agent.write, texts)
                objs = agent.store.objects
                held += [(objs.get("D", d).created_at, d, objs.raw(d)[:2])
                         for d in objs.ids("D", DEFAULT_SCOPE)]
            out = sink if sink is not None else mem("sink", ep, _own(nli, calls))
            for *_, raw in sorted(held):
                out.receive(*raw)
        else:
            # ponytail: N live replicas each hold a model instance (about 0.6 GB each for the
            # deployed cross-encoder), and every replica admits every delta: N times the star cost
            nodes = {f"agent{i}": BrainNode(f"agent{i}", pol=DEFAULT_POL, provider=_own(nli, calls),
                                            clock=clock, **{**kw, "epoch": ep})
                     for i in range(len(plan.per_brain))}
            for i, texts in enumerate(plan.per_brain):
                feed(nodes[f"agent{i}"].ingest, texts)
            names = list(nodes)
            res = run_sync(nodes, {n: [p for p in names if p != n] for n in names},
                           routing_mode="broadcast", encoder=encoder)
            c = res.convergence[DEFAULT_SCOPE]
            conv = {"post_pass": c["post_pass"], "L": c["L"], "fingerprints": c["fingerprints"]}
            out = nodes["agent0"].mem
            decided = list(zip(c["decisions"][0], c["n_prior"][0]))
    else:
        raise ValueError(f"unknown condition {condition!r} or topology {topology!r}")
    if conv is None:
        decided = [(d.decision, d.n_prior) for d in out.admission.decisions(DEFAULT_SCOPE)]
        ep = epoch if condition != "exact_dedup" else None
    by_bin: Dict[Optional[int], Counter] = {}
    for kind, n_prior in decided:
        by_bin.setdefault(bin_for(n_prior, ep), Counter())[kind] += 1
    f = out.footprint()
    return {"memory": out, "stored": f["stored"], "live": f["live"],
            "decisions": dict(Counter(k for k, _ in decided)),
            "decisions_by_bin": {b: dict(c) for b, c in by_bin.items()},
            "nli_calls": calls[0], "wall_s": time.perf_counter() - t0, "convergence": conv}


__all__ = ["CONDITIONS", "TOPOLOGIES", "grid", "stamps", "bin_for", "build"]
