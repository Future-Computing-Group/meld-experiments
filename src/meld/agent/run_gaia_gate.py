"""G3 competence-gate driver: agent <-> MELD-brain <-> GAIA scorer.

Two modes:
  - LOCAL WIRING SMOKE (default): a deterministic ORACLE MockLLM drives the
    ReAct agent over a tiny GAIA-shaped fixture. Each task's clues are written
    through the MELD merge path (PUBLISH -> merge-on-receive -> Patch); the agent
    recalls them via the brain's graph-walk read and answers. This proves the
    harness + MELD-brain integration RUNS and the scorer + gate work. It is NOT a
    GAIA result -- the oracle just exercises the wiring end-to-end.
  - GATED CSC RUN: swap in ``OpenWeightLLM`` (Qwen2.5-7B-Instruct) and
    ``load_gaia_real`` (the gated GAIA validation slice). That is the only run
    that yields a real GAIA L2 number; it is PI-gated (G5) and not launched here.

The oracle is NOT the model under test: it is a fixed policy (remember each
clue, recall, answer the gold) used to verify that a *correct* agent's actions
traverse the MELD brain and that the scorer/gate classify the result correctly.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from meld.agent.gaia_loader import GaiaTask, load_gaia_fixture
from meld.agent.gaia_scorer import competence_gate, extract_final_answer, score_predictions
from meld.agent.llm import MockLLM
from meld.agent.memory import SMPMemory
from meld.agent.react_agent import ReActAgent
from meld.agent.registry import make_agent
from meld.agent.tools import FinalAnswerTool, RecallTool
# Single source of truth for the encoder-matched, calibrated thresholds (the
# duplicate inverted-0.5/0.55 copy that used to live here is gone -- it tripped the
# Thresholds ordering invariant and decoupled the GAIA gate from the calibration).
from meld.agent.run_hotpot_gate import _calibrated_cfg


def _oracle_script(task: GaiaTask) -> List[str]:
    """A fixed, correct ReAct policy for the wiring smoke: remember each clue,
    recall, then answer the gold in the GAIA FINAL ANSWER format."""
    turns = [f"Thought: note clue.\nAction: remember[{c}]" for c in task.clues]
    turns.append(f"Thought: recall.\nAction: recall[{task.question}]")
    turns.append(f"Thought: I can answer.\nAction: final_answer[{task.gold}]")
    return turns


def _run_one_baseline(baseline: str, tasks: List[GaiaTask], *, encoder: str,
                      nli: str, bar: float) -> Dict:
    """Run ONE registry baseline over the fixture with the ORACLE LLM. Each
    baseline drives the SAME GAIA gate through the SAME MELD brain backend."""
    preds: List[Dict] = []
    integration: List[Dict] = []
    for t in tasks:
        # Fresh brain per task (a GAIA task is independent; the integration we
        # exercise is agent<->brain<->tools, not cross-task memory).
        mem = SMPMemory(encoder=encoder, nli=nli, cfg=_calibrated_cfg(encoder))
        agent = make_agent(baseline, llm=MockLLM(_oracle_script(t)), memory=mem,
                           tools=[RecallTool(mem), FinalAnswerTool()],
                           max_steps=len(t.clues) + 4)
        result = agent.run(t.question)
        pred = result.answer
        if pred and pred.upper().startswith("FINAL ANSWER:"):
            pred = extract_final_answer(pred)
        preds.append({"task_id": t.task_id, "level": t.level,
                      "prediction": pred or "", "gold": t.gold})
        integration.append({
            "task_id": t.task_id,
            "n_steps": result.n_steps,
            "n_patches": len(mem.brain.audit_log),
            "decisions": [p.decision for p in mem.brain.audit_log],
            "brain_units": sum(1 for u in mem.brain.extract_delta()
                               if getattr(u, "claims", None)),
        })
    return {"scored": score_predictions(preds),
            "gate": competence_gate(preds, bar=bar),
            "integration": integration, "n_tasks": len(tasks),
            "mode": "LOCAL-WIRING-SMOKE (oracle LLM; NOT a GAIA result)"}


def run_gate_smoke(n: int = 6, encoder: str = "hashing",
                   nli: str = "lexical", bar: float = 0.60,
                   baseline: str = "react") -> Dict:
    """Run a wired baseline (default ReAct) over a tiny GAIA-shaped fixture with
    the ORACLE LLM, score with quasi-exact-match, and evaluate the gate."""
    tasks = load_gaia_fixture(n=n)
    return _run_one_baseline(baseline, tasks, encoder=encoder, nli=nli, bar=bar)


def run_gate_smoke_multibaseline(n: int = 6, encoder: str = "hashing",
                                 nli: str = "lexical", bar: float = 0.60,
                                 baselines: Optional[List[str]] = None) -> Dict:
    """Run SEVERAL registry baselines over the SAME GAIA-shaped fixture, each
    through the SAME MELD brain backend. Returns per-baseline gate + integration.
    This shows baselines are SWAPPABLE and each writes through the MELD merge path
    (wiring + robustness, NOT a GAIA competence claim)."""
    baselines = baselines or ["react", "reflexion"]
    tasks = load_gaia_fixture(n=n)
    per = {b: _run_one_baseline(b, tasks, encoder=encoder, nli=nli, bar=bar)
           for b in baselines}
    return {"per_baseline": per, "n_tasks": len(tasks),
            "baselines": baselines,
            "mode": "LOCAL-WIRING-SMOKE multibaseline (oracle LLM; "
                    "NOT a GAIA result)"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--encoder", default="hashing")
    ap.add_argument("--nli", default="lexical")
    ap.add_argument("--bar", type=float, default=0.60)
    a = ap.parse_args(argv)
    res = run_gate_smoke(n=a.n, encoder=a.encoder, nli=a.nli, bar=a.bar)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_gate_smoke", "run_gate_smoke_multibaseline"]
