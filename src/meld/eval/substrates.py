"""Calibration substrates of the recall workloads: the HotpotQA overlap split and the LongMemEval
session split, each folded as the evaluation folds it.

Task pools. ``pool`` splits every task id into a calibration pool (about one task in ten, by a
salted hash, so the split does not depend on how many tasks a run loads) and an evaluation pool;
a LongMemEval abstention variant (id suffix ``_abs``) shares the pool of its base question. The
recall runners keep ``eval_pool`` tasks only, the calibration streams read ``cal_pool`` tasks
only, so no calibration task is ever evaluated.

Worlds. A world is one store of the evaluation: one HotpotQA task, or ``per_store`` pooled tasks
(``mem_conditions.pool_tasks``, ``split_across_brains`` with the overlap fraction and duplication
degree of the run), or one LongMemEval question (``lme_supersession.session_split``). Each
placed text has a source text (the corpus sentence or turn it was placed from: itself, or the
original of a ``light_paraphrase`` copy) and a source item (the Wikipedia paragraph title, or the
haystack session id), so one item recurring in two worlds is one unit of the per-item draw.

``world_store`` builds the store the evaluation's sink holds: every agent publishes its texts as
in ``topology.build`` and ``mem_conditions._build_distributed_merge`` (origin ``agent<i>``,
created_at the plan's world time, else a counter over the agents' writes in order), and the sink
applies every agent delta. Admission adds no delta, so the sink's D, hence the fold's tested
pairs, equal the evaluation's; no NLI model is loaded.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

from meld.agent.mem_conditions import SplitPlan, pool_tasks, split_across_brains
from meld.agent.memory import DEFAULT_POL, DEFAULT_SCOPE, SMPMemory
from meld.brain.store import Store
from meld.eval.lme_supersession import session_split

RECALL_DOMAINS = ("hotpot", "lme")
CAL_PER_MILLE = 100
_SALT = b"meld-cal-pool\x00"

World = Tuple[str, SplitPlan, Dict[str, str], Dict[str, str]]   # (id, plan, item, source)


def pool(task_id: str) -> str:
    """'cal' or 'eval' for a task id (LongMemEval ``_abs`` variants with their base question)."""
    base = task_id[:-4] if task_id.endswith("_abs") else task_id
    h = int.from_bytes(hashlib.sha256(_SALT + base.encode()).digest()[:4], "big")
    return "cal" if h % 1000 < CAL_PER_MILLE else "eval"


def eval_pool(xs: Iterable, key: Callable = lambda t: t.task_id) -> list:
    return [x for x in xs if pool(key(x)) == "eval"]


def cal_pool(xs: Iterable, key: Callable = lambda t: t.task_id) -> list:
    return [x for x in xs if pool(key(x)) == "cal"]


def hotpot_worlds(tasks: Sequence, *, n_brains: int = 3, dup_degree: int = 2, overlap: float = 0.5,
                  per_store: int = 1) -> Iterator[World]:
    """One world per ``per_store`` consecutive tasks (1: the per-task stores of the overlap x k
    runner; 20: the pooled stores of the N-brains runner)."""
    for pooled, _ in pool_tasks(list(tasks), per_store):
        plan = split_across_brains(pooled, n_brains=n_brains, overlap_fraction=overlap,
                                   dup_degree=dup_degree)
        title = {f"{t}: {s}".strip(): t for t, ss in pooled.context for s in ss}
        src = {x: plan.source.get(x, x) for b in plan.per_brain for x in b}
        yield pooled.task_id, plan, {x: title[s] for x, s in src.items()}, src


def lme_worlds(items: Sequence[dict], *, n_brains: int = 3, dup_degree: int = 2,
               seed: int = 0) -> Iterator[World]:
    """One world per LongMemEval question, as ``run_nbrains`` splits it."""
    for it in items:
        plan = session_split(it, n_brains=n_brains, dup_degree=dup_degree, seed=seed)
        sess: Dict[str, str] = {}
        for sid, s in zip(it["haystack_session_ids"], it["haystack_sessions"]):
            for t in s:
                sess.setdefault((t.get("content") or "").strip(), sid)
        yield it["question_id"], plan, {x: sess[s] for x, s in plan.source.items()}, dict(plan.source)


def world_store(plan: SplitPlan, *, encoder: str, cache_dir: str = None) -> Store:
    """The sink store of the world (module doc), under DEFAULT_POL."""
    cur, tick = [0], itertools.count(1)
    held: List[tuple] = []
    key = None
    for i, texts in enumerate(plan.per_brain):
        agent = SMPMemory(encoder=encoder, nli="lexical", cache_dir=cache_dir, origin=f"agent{i}",
                          clock=lambda: cur[0])
        key = agent.key
        for t in texts:
            cur[0] = plan.when[t] + 1 if plan.when else next(tick)
            agent.publish(t)
        objs = agent.store.objects
        held += [(objs.get("D", d).created_at, d, objs.raw(d)[:2]) for d in objs.ids("D", DEFAULT_SCOPE)]
    store = Store(DEFAULT_POL, key, encoder=encoder)
    for *_, raw in sorted(held):
        store.apply(*raw)
    return store


def keyed(store: Store, item: Dict[str, str], src: Dict[str, str]):
    """(item, source) of the store's nodes by claim key."""
    text = {d.claim_key: d.text for _, d in store.nodes().values()}
    return {k: item[t] for k, t in text.items()}, {k: src[t] for k, t in text.items()}


LME_TYPES = ("knowledge-update", "multi-session", "temporal-reasoning")     # run_nbrains' default


def cal_worlds(domain: str, *, data: str = None, n: int = None, n_brains: int = 3, dup_degree: int = 2,
               overlap: float = 0.5, per_store: int = 1, types: Sequence[str] = LME_TYPES,
               seed: int = 0, pick: int = None, ids: Sequence[str] = None) -> List[World]:
    """The calibration-pool worlds of ``domain``, the first ``n`` cal tasks (HotpotQA validation,
    dataset order; LongMemEval-cleaned ``data``, file order, question types ``types``), or with
    ``pick`` a seeded draw of ``n`` cal tasks, kept in that order, or with ``ids`` those cal tasks in
    the given order (ValueError for an id outside the cal pool)."""
    def take(xs):
        if ids is not None:
            by = {getattr(x, "task_id", None) or x["question_id"]: x for x in xs}
            missing = [i for i in ids if i not in by]
            if missing:
                raise ValueError(f"not in the {domain} cal pool: {missing[:5]}")
            return [by[i] for i in ids]
        if n is None or pick is None:
            return xs[:n]
        keep = set(random.Random(pick).sample(range(len(xs)), min(n, len(xs))))
        return [x for i, x in enumerate(xs) if i in keep]
    if domain == "hotpot":
        from meld.agent.hotpot_loader import load_hotpot_real
        return list(hotpot_worlds(take(cal_pool(load_hotpot_real())), n_brains=n_brains,
                                  dup_degree=dup_degree, overlap=overlap, per_store=per_store))
    if domain == "lme":
        import json
        with open(data, encoding="utf-8") as fh:
            items = [x for x in json.load(fh) if x["question_type"] in types]
        return list(lme_worlds(take(cal_pool(items, key=lambda x: x["question_id"])), n_brains=n_brains,
                               dup_degree=dup_degree, seed=seed))
    raise ValueError(f"unknown recall domain {domain!r}")


def spec_worlds(spec: Dict[str, List[dict]], *, data: str = None) -> Dict[str, List[World]]:
    """The worlds of a world spec, per domain: a list of shapes, each ``ids`` (cal tasks, in order)
    and the split of the evaluation (``n_brains``, ``dup``, optional ``per_store``, ``overlap``,
    ``seed``), so one calibration set holds worlds of several evaluation cells."""
    return {d: [w for sh in shapes for w in cal_worlds(
        d, data=data, ids=sh["ids"], n_brains=sh["n_brains"], dup_degree=sh["dup"],
        per_store=sh.get("per_store", 1), overlap=sh.get("overlap", 0.5), seed=sh.get("seed", 0))]
        for d, shapes in spec.items()}


def load_spec(path: str = None):
    """(world spec, {path, sha256}) of a world-spec JSON file, or (None, None) without one."""
    if not path:
        return None, None
    import json
    with open(path, "rb") as fh:
        raw = fh.read()
    return json.loads(raw), {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}


def repeated_items(worlds: Iterable[World]) -> Dict[str, List[str]]:
    """Source items placed in more than one world, with those worlds (one Cal record per item)."""
    at: Dict[str, set] = {}
    for wid, _, item, _ in worlds:
        for it in set(item.values()):
            at.setdefault(it, set()).add(wid)
    return {it: sorted(ws) for it, ws in sorted(at.items()) if len(ws) > 1}


__all__ = ["RECALL_DOMAINS", "CAL_PER_MILLE", "pool", "eval_pool", "cal_pool", "hotpot_worlds",
           "lme_worlds", "world_store", "keyed", "LME_TYPES", "cal_worlds", "spec_worlds",
           "load_spec", "repeated_items"]
