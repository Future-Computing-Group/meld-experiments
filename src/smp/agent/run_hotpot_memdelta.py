"""M-1 memory-delta run driver: the SAME held-out questions answered under every
MEMORY_CONDITION x answer-mode x model, scored as DELTAS (not absolute F1).

This REPLACES the centralized-only framing of ``run_hotpot_gate.py`` /
``run_hotpot_gate_csc.py``: that driver fed ALL of a question's context into ONE
brain (= the ``centralized`` condition) and reported an absolute agent F1. SMP's
contribution is the merge ACROSS sovereign brains, so the headline is the
condition delta (see ``mem_delta_metrics.HEADLINE_DELTAS``).

Two answering modes:
  - ``fixed_recall`` (headline): deterministic top-k graph-walk read assembled
    into ONE LLM answer call -- removes agent-strategy variance.
  - ``agentic`` (realism): the registry agent recalls + reasons over the SAME
    brain / inline context.

Offline-safe: the LLM is injected (MockLLM locally / OpenWeightLLM on the gated
host). NO billable run here -- this module only orchestrates.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

from smp.agent.hotpot_loader import HotpotTask
from smp.agent.hotpot_scorer import em_score, f1_score
from smp.agent.mem_conditions import (ANSWER_MODES, MEMORY_CONDITIONS,
                                      ConditionBuild, assemble_recall,
                                      build_condition, fixed_recall_prompt)
from smp.agent.mem_delta_metrics import condition_means, headline_table
from smp.agent.memory import SMPMemory
from smp.agent.registry import make_agent
from smp.agent.run_hotpot_gate import _calibrated_cfg, extract_prediction
from smp.agent.recall_metrics import recall_at_k_and_mrr
from smp.agent.tools import FinalAnswerTool, RecallTool
from smp.config import Thresholds

# the question's id namespace for the paired delta (same question across conditions)
QID = "task_id"


def _answer_fixed_recall(llm, build: ConditionBuild, question: str, *,
                         encoder: str, k: int = 5):
    """One deterministic LLM call over the top-k recalled evidence -> answer.

    Returns ``(prediction, evidence, prompt)``: the assembled recall context and
    the exact answer prompt are returned (not discarded) so ``run_one`` can
    PERSIST them in the durable row -- making a future model-add a true zero-build
    replay and the context auditable."""
    evidence = assemble_recall(build, question, encoder=encoder, k=k)
    prompt = fixed_recall_prompt(question, evidence)
    raw = llm.complete(prompt)
    return extract_prediction(raw), evidence, prompt


def _answer_agentic(llm, build: ConditionBuild, question: str, *, baseline: str,
                    encoder: str, nli: str, cfg: Thresholds,
                    max_steps: int = 6) -> str:
    """Run a registry agent over the condition's memory.

    For brain-backed conditions the agent recalls from the SAME global/centralized
    brain. For ``none``/``full_context`` (no brain) the agent gets a fresh empty
    brain seeded with the inline context (so it still has the RecallTool surface);
    ``full_context`` seeds all paragraphs, ``none`` seeds nothing.
    """
    if build.brain is not None:
        mem = build.brain
    else:
        mem = SMPMemory(encoder=encoder, nli=nli, cfg=cfg)
        for text in build.inline_context:   # empty for 'none'
            mem.write(text)
    agent = make_agent(baseline, llm=llm, memory=mem,
                       tools=[RecallTool(mem), FinalAnswerTool()],
                       max_steps=max_steps)
    res = agent.run(question)
    return extract_prediction(res.answer)


def run_one(task: HotpotTask, condition: str, mode: str, llm, *, encoder: str,
            nli: str, cfg: Thresholds, baseline: str = "react",
            n_brains: int = 3, overlap_fraction: float = 0.5,
            inject_conflict: bool = False, k: int = 5,
            max_steps: int = 6) -> Dict:
    """Answer ONE (task, condition, mode); return a scored row + the merge audit."""
    if condition not in MEMORY_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if mode not in ANSWER_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    build = build_condition(task, condition, encoder=encoder, nli=nli, cfg=cfg,
                            n_brains=n_brains, overlap_fraction=overlap_fraction,
                            inject_conflict=inject_conflict)
    if mode == "fixed_recall":
        pred, evidence, prompt = _answer_fixed_recall(
            llm, build, task.question, encoder=encoder, k=k)
    else:
        pred = _answer_agentic(llm, build, task.question, baseline=baseline,
                               encoder=encoder, nli=nli, cfg=cfg,
                               max_steps=max_steps)
        # agentic mode has no single deterministic answer prompt; persist the
        # assembled recall context (same model-independent build) + the prompt the
        # fixed-recall path WOULD use, so the durable keys are always present and a
        # replay/audit has the exact context the agent was given.
        evidence = assemble_recall(build, task.question, encoder=encoder, k=k)
        prompt = fixed_recall_prompt(task.question, evidence)
    row = {
        "question_id": task.task_id, "task_id": task.task_id,
        "condition": condition, "mode": mode, "baseline": baseline,
        "level": task.level, "prediction": pred, "gold": task.answer,
        "em": em_score(pred, task.answer), "f1": f1_score(pred, task.answer),
        # persisted recall context + answer prompt (zero-build replay + audit).
        "evidence": evidence, "prompt": prompt,
        # store footprint (the overhead/compactness claim).
        "stored_units": build.stored_units, "live_units": build.live_units,
        # the merge audit (load-bearing: distributed_merge must show overlap merges)
        "n_merge": build.n_merge, "n_relate": build.n_relate,
        "n_conflict": build.n_conflict, "n_overlap_merges": build.n_overlap_merges,
        "n_total_patches": build.n_total_patches,
    }
    # retrieval-quality observable (PRIMARY): does the gold supporting fact reach
    # the status-live top-k? Defined ONLY for store-backed conditions -- none and
    # full_context have no store (full_context delivers gold by inlining), so
    # recall@k is N/A there and the keys are OMITTED (aggregation skips them; the
    # floor stays an F1 question, not a spurious recall@k 0.000).
    if build.brain is not None:
        rm = recall_at_k_and_mrr(build.brain.brain, task, encoder=encoder, k=k)
        row.update(recall_at_k=rm["recall_at_k"], mrr=rm["mrr"], n_gold=rm["n_gold"])
    return row


def run_grid(tasks: List[HotpotTask], llm_factory: Callable[[], object], *,
             conditions: Optional[List[str]] = None,
             modes: Optional[List[str]] = None, encoder: str = "hashing",
             nli: str = "lexical", cfg: Optional[Thresholds] = None,
             baseline: str = "react", n_brains: int = 3,
             overlap_fraction: float = 0.5, inject_conflict: bool = False,
             k: int = 5, max_steps: int = 6,
             on_row: Optional[Callable[[Dict], None]] = None) -> List[Dict]:
    """Run every (task x condition x mode) cell. ``llm_factory`` returns a FRESH
    LLM per cell (so a scripted MockLLM is not consumed across cells; the real
    OpenWeightLLM factory just returns the shared client). ``on_row`` is an
    optional durable-append callback (CSC checkpointing)."""
    conditions = conditions or list(MEMORY_CONDITIONS)
    modes = modes or ["fixed_recall"]
    cfg = cfg or _calibrated_cfg(encoder)
    rows: List[Dict] = []
    for mode in modes:
        for cond in conditions:
            for task in tasks:
                row = run_one(task, cond, mode, llm_factory(), encoder=encoder,
                              nli=nli, cfg=cfg, baseline=baseline,
                              n_brains=n_brains, overlap_fraction=overlap_fraction,
                              inject_conflict=inject_conflict, k=k,
                              max_steps=max_steps)
                rows.append(row)
                if on_row is not None:
                    on_row(row)
    return rows


def summarize(rows: List[Dict], *, modes: Optional[List[str]] = None,
              conditions: Optional[List[str]] = None,
              n_boot: int = 2000, seed: int = 0) -> Dict:
    """Per-mode floor/ceiling means + the three headline delta CIs (F1)."""
    modes = modes or sorted({r["mode"] for r in rows})
    conditions = conditions or list(MEMORY_CONDITIONS)
    out: Dict = {"per_mode": {}}
    for mode in modes:
        mrows = [r for r in rows if r["mode"] == mode]
        out["per_mode"][mode] = {
            "means_f1": condition_means(mrows, conditions, metric="f1"),
            "means_em": condition_means(mrows, conditions, metric="em"),
            # recall@k is the PRIMARY retrieval-quality observable; F1 is secondary.
            "means_recall": condition_means(mrows, conditions, metric="recall_at_k"),
            "means_mrr": condition_means(mrows, conditions, metric="mrr"),
            "headline_f1": headline_table(mrows, metric="f1", n_boot=n_boot,
                                          seed=seed),
            "headline_recall": headline_table(mrows, metric="recall_at_k",
                                              n_boot=n_boot, seed=seed),
            "merge_audit": _merge_audit(mrows),
            "footprint": _footprint_agg(mrows),
        }
    return out


def _footprint_agg(rows: List[Dict]) -> Dict:
    """Mean stored vs status-live unit count per condition -- the storage/
    compactness signal (merge: live < stored < naive-union)."""
    agg: Dict[str, Dict[str, float]] = {}
    tally: Dict[str, Dict[str, float]] = {}
    for r in rows:
        t = tally.setdefault(r["condition"], {"stored": 0.0, "live": 0.0, "n": 0.0})
        t["stored"] += r.get("stored_units", 0)
        t["live"] += r.get("live_units", 0)
        t["n"] += 1
    for c, t in tally.items():
        n = t["n"] or 1
        agg[c] = {"stored": t["stored"] / n, "live": t["live"] / n}
    return agg


def _merge_audit(rows: List[Dict]) -> Dict:
    """Aggregate the merge audit per condition (proves the merge fired)."""
    agg: Dict[str, Dict[str, int]] = {}
    for r in rows:
        a = agg.setdefault(r["condition"],
                           {"merge": 0, "relate": 0, "conflict": 0,
                            "overlap_merges": 0, "total_patches": 0, "n": 0})
        a["merge"] += r.get("n_merge", 0)
        a["relate"] += r.get("n_relate", 0)
        a["conflict"] += r.get("n_conflict", 0)
        a["overlap_merges"] += r.get("n_overlap_merges", 0)
        a["total_patches"] += r.get("n_total_patches", 0)
        a["n"] += 1
    return agg


def format_delta_table(summary: Dict) -> str:
    """Pretty-print the floor/ceiling means + headline deltas (for the smoke)."""
    lines: List[str] = []
    for mode, m in summary["per_mode"].items():
        lines.append(f"== mode={mode} ==")
        lines.append("  condition means (F1 | EM):")
        means_f1, means_em = m["means_f1"], m["means_em"]
        for c in MEMORY_CONDITIONS:
            if c in means_f1:
                lines.append(f"    {c:18s} F1={means_f1[c]['mean']:.3f} "
                             f"EM={means_em[c]['mean']:.3f} (n={means_f1[c]['n']})")
        lines.append("  HEADLINE deltas (F1, paired bootstrap 95% CI):")
        for d in m["headline_f1"]:
            lines.append(f"    {d['a']} - {d['b']:18s} = {d['delta']:+.3f} "
                         f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]  ({d['label']})")
        lines.append("  merge audit (per condition, summed over questions):")
        for c, a in m["merge_audit"].items():
            lines.append(f"    {c:18s} MERGE={a['merge']} (overlap={a['overlap_merges']}) "
                         f"RELATE={a['relate']} CONFLICT={a['conflict']} "
                         f"patches={a['total_patches']}")
    return "\n".join(lines)


def write_summary(summary: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


__all__ = ["run_one", "run_grid", "summarize", "format_delta_table",
           "write_summary"]
