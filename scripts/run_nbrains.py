"""N-brains sweep (eval/topology.py): worlds built under each (N, d) cell with d <= N, each
condition and topology, recall@k per question, live and stored units, decision counts, NLI
calls and wall time per build, appended one row per build (resumable; the cell id carries the
git sha and a config hash) with the provenance stamps (topology.stamps: git sha, M_nli, M_pol, epoch id and seq, NLI and
encoder ids) and the decision counts per epoch bin.

Workloads:
  * ``lme``     one world per LongMemEval item (eval/lme_supersession.session_split; user turns
                are claims); KU items also report the stale-fact rate;
  * ``hotpot``  pooled HotpotQA validation stores (--data fixture: the four-task fixture) (mem_conditions.pool_tasks, split_across_brains with
                dup_degree); recall is scored per member question.

Only evaluation-pool tasks are used (eval/substrates.pool: the calibration pool is never evaluated).

Without --epoch or --dev-theta every theta is +inf: no similarity merge (meld equals
exact_dedup). --dev-theta builds an uncalibrated development epoch (every bin's theta the given
value) for smoke runs only.

Run:  PYTHONPATH=src python scripts/run_nbrains.py --workload lme --encoder minilm \
          --nli mnli-cross-encoder --ns 3 8 --ds 1 2 4 --n-items 5 --out outputs/nbrains/smoke.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import statistics

from meld.agent.mem_conditions import pool_tasks, split_across_brains
from meld.agent.memory import DEFAULT_POL
from meld.agent.recall_metrics import recall_at_k_and_mrr
from meld.baselines.memory import BaselineMemory
from meld.config import read_epoch
from meld.eval import topology
from meld.eval.campaign_checkpoint import CampaignCheckpoint
from meld.eval.lme_supersession import lme_recall, session_split, stale_fact_rate
from meld.eval.progress import status_emitter
from meld.eval.runner_args import (add_baseline, add_key_file, add_read_tau, read_stamp, baseline_policy,
                                   baseline_stamp, runner_key)
from meld.eval.substrates import eval_pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dev_epoch(encoder: str, nli: str, theta: float):
    from meld.proto import CANON_VERSION
    from meld.protocol.objects import FIXED_ONE, INF16, EpochRecord, m_nli, m_pol_of
    sel = DEFAULT_POL["bootstrap_selector"]
    return EpochRecord(scope="agent", seq=1, m_nli=m_nli(encoder, nli, "int8", CANON_VERSION),
                       m_pol=m_pol_of(DEFAULT_POL), policy="SEED", alpha=20_000, alpha_c=20_000,
                       mondrian_c=False, cal_root=b"\x00" * 16, cal_hw="cpu", bins=(256, "doubling"),
                       table=((0, 100, 0, round(theta * FIXED_ONE), INF16),), sigma_lo_q=sel[0],
                       sigma_c_q=sel[1], m=sel[2], author="calibration", created_at=0)


def worlds(a):
    """(world id, split(n, d) -> plan, score(memory, plan, k) -> {recall, stale})."""
    if a.workload == "lme":
        items = json.load(open(a.data, encoding="utf-8"))
        items = eval_pool([x for x in items if x["question_type"] in a.types], key=lambda x: x["question_id"])
        random.Random(a.seed).shuffle(items)
        for it in items[:a.n_items]:
            yield (it["question_id"],
                   lambda n, d, it=it: session_split(it, n_brains=n, dup_degree=d, seed=a.seed),
                   lambda m, p, k, it=it: {"recall": lme_recall(m, it, k, p.source, tau=a.read_tau),
                                           "stale": stale_fact_rate(m, it, k, p.source, tau=a.read_tau)})
    else:
        from meld.agent.hotpot_loader import load_hotpot_fixture, load_hotpot_real
        tasks = (load_hotpot_fixture() if a.data == "fixture"
                 else eval_pool(load_hotpot_real()))[:a.n_items]
        for pooled, members in pool_tasks(tasks, a.per_store):
            yield (pooled.task_id,
                   lambda n, d, t=pooled: split_across_brains(t, n_brains=n, dup_degree=d,
                                                              overlap_fraction=a.overlap),
                   lambda m, p, k, ms=members: {
                       "recall": statistics.mean(recall_at_k_and_mrr(m, t, k=k, source=p.source,
                                                                     tau=a.read_tau)["recall_at_k"]
                                                 for t in ms), "stale": None})


def main(argv=None, llm=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=("lme", "hotpot"), default="lme")
    ap.add_argument("--data")
    ap.add_argument("--types", nargs="+",
                    default=["knowledge-update", "multi-session", "temporal-reasoning"])
    ap.add_argument("--n-items", type=int, default=5)
    ap.add_argument("--per-store", type=int, default=20)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--ns", type=int, nargs="+", default=[3, 8, 16, 32])
    ap.add_argument("--ds", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--conditions", nargs="+", default=list(topology.CONDITIONS))
    ap.add_argument("--topologies", nargs="+", default=["star"])
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--epoch")
    add_key_file(ap)
    ap.add_argument("--dev-theta", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "outputs", "emb-cache"))
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("I", "K"),
                    help="run only worlds whose index is I mod K (one array task per shard)")
    ap.add_argument("--out", required=True)
    add_baseline(ap)
    add_read_tau(ap)
    a = ap.parse_args(argv)
    if a.workload == "lme":
        a.data = a.data or os.path.join(REPO, "data", "bench", "lme-cleaned",
                                        "longmemeval_s_cleaned.json")
    epoch = (read_epoch(a.epoch, runner_key(a.key_file)) if a.epoch
             else dev_epoch(a.encoder, a.nli, a.dev_theta) if a.dev_theta is not None else None)
    stamp = {**topology.stamps(a.encoder, a.nli, epoch), "dev_epoch": a.dev_theta, **read_stamp(a)}
    policy = baseline_policy(a, llm)
    baselines = {a.baseline: lambda plan, kw: topology.build(
        plan, "meld", "star", **kw, sink=BaselineMemory(policy, **kw, clock=itertools.count(1).__next__))
    } if policy else {}
    config = {k: v for k, v in vars(a).items()
              if k not in ("out", "cache_dir", "shard", "key_file", "llm_base_url", "llm_cache_dir")}
    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ckpt = CampaignCheckpoint(a.out)
    emit = status_emitter(a.out + ".status", "nbrains")
    kw = dict(encoder=a.encoder, nli=a.nli, epoch=epoch, cache_dir=a.cache_dir)
    conditions = a.conditions + [a.baseline] * bool(policy)
    for w, (wid, split, score) in enumerate(worlds(a)):
        if w % a.shard[1] != a.shard[0]:
            continue
        cells = [(1, 1, "centralized", "star")] if "centralized" in conditions else []
        cells += [(n, d, c, t) for n, d in topology.grid(a.ns, a.ds) for c in conditions
                  if c != "centralized"
                  for t in (a.topologies if c in ("meld", "exact_dedup") else ["star"])]
        for n, d, cond, top in cells:
            cid = f"{stamp['git_sha']}|{cfg_hash}|{wid}|{n}|{d}|{cond}|{top}"
            if ckpt.done(cid):
                continue
            plan = split(n, d)
            b = topology.build(plan, cond, top, **kw, baselines=baselines)
            row = {"world": wid, "n_brains": n, "dup": d, "condition": cond, "topology": top,
                   "n_claims": sum(map(len, plan.per_brain)), "stored": b["stored"],
                   "live": b["live"], "decisions": b["decisions"],
                   "decisions_by_bin": {str(g): v for g, v in b["decisions_by_bin"].items()},
                   "nli_calls": b["nli_calls"],
                   "wall_s": round(b["wall_s"], 3), "convergence": b["convergence"],
                   **{f"{m}@{k}": v for k in a.ks for m, v in score(b["memory"], plan, k).items()},
                   **stamp, **(baseline_stamp(a) if cond == a.baseline else {})}
            ckpt.record(cid, row)
            emit(f"{wid} N={n} d={d} {cond}/{top}: {row['wall_s']}s, {b['nli_calls']} NLI calls, "
                 f"live {b['live']}/{b['stored']}")
    return ckpt.results()


if __name__ == "__main__":
    main()
