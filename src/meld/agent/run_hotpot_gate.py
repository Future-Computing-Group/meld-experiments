"""G3 competence-gate driver (HotpotQA pivot): agent <-> MELD-brain <-> HotpotQA
EM/F1 scorer + gate.

The competence benchmark is PIVOTED from GAIA (gated) to HotpotQA distractor
(open). Each item's ``context`` paragraphs are fed THROUGH the MELD brain as merged
observations (``feed_context_through_brain``), the ``question`` is posed to a
registry agent, the gold is the ``answer``, and the prediction is scored with the
official HotpotQA EM + token-F1; the gate is token-F1 >= 0.60 on medium+hard.

Two modes:
  - LOCAL WIRING SMOKE (default): a deterministic ORACLE ``MockLLM`` drives a
    registry baseline over the HotpotQA-shaped fixture. The context is fed through
    the MELD merge path; the oracle recalls + answers the gold. Proves wiring; NOT
    a competence claim.
  - GATED / NETWORKED REAL RUN: swap in ``OpenWeightLLM`` + ``load_hotpot_real``.
    The local real-model ext-smoke (``scripts/run_hotpot_realmodel_extsmoke.py``)
    is the 0-cost rung that yields a GENUINE EM/F1.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from meld.agent.hotpot_loader import (HotpotTask, feed_context_through_brain,
                                     load_hotpot_fixture)
from meld.agent.hotpot_scorer import competence_gate, score_predictions
from meld.agent.gaia_scorer import extract_final_answer, tighten_gaia_answer
from meld.agent.llm import MockLLM
from meld.agent.memory import SMPMemory
from meld.agent.registry import make_agent
from meld.agent.tools import FinalAnswerTool, RecallTool
from meld.config import DEFAULT_PATH, Thresholds, load_thresholds

# Thresholds are calibrated PER ENCODER (the similarity scale differs by encoder);
# the calibrated pairs live next to the schema and are reproduced by
# tests/test_threshold_calibration.py (provenance meld.merge.calibration).
_CFG_BY_ENCODER = {
    "minilm": "thresholds-minilm.yaml",    # headline encoder; classes separate, acc 1.0
    "hashing": "thresholds-hashing.yaml",  # fast unit-test path; compressed scale, acc ~0.89
}


def _calibrated_cfg(encoder: Optional[str] = None) -> Thresholds:
    """Encoder-matched, CALIBRATED merge thresholds (Section 4 / M-2).

    Thresholds are calibrated PER ENCODER because the similarity scale differs:
    minilm -> theta_merge=0.90/sigma_lo=0.12 (acc 1.0); hashing ->
    theta_merge=0.49/sigma_lo=0.29 (acc ~0.89). An unknown encoder falls back to the
    documented default (0.85/0.55). The earlier hardcoded theta_merge=0.5 /
    sigma_lo=0.55 was INVERTED (empty RELATE band) and is gone; every branch now
    satisfies the Thresholds ordering invariant.
    """
    fname = _CFG_BY_ENCODER.get(encoder)
    if fname:
        return load_thresholds(os.path.join(os.path.dirname(DEFAULT_PATH), fname))
    return load_thresholds()


def extract_prediction(raw: Optional[str]) -> str:
    """Get the agent's answer string: honor a 'FINAL ANSWER:' template if present
    (the agents emit it), else take the tightened raw text. HotpotQA scoring then
    applies its own normalization, so we only need the short answer text."""
    if not raw:
        return ""
    # honor BOTH the spaced "FINAL ANSWER:" and the underscore "FINAL_ANSWER:"
    # template (open models emit both; the underscore form was leaking the whole
    # "FINAL_ANSWER: British" string through tighten and tanking F1 1.0 -> 0.67).
    up = raw.upper()
    if "FINAL ANSWER" in up or "FINAL_ANSWER" in up:
        fa = extract_final_answer(raw)
        if fa:
            return fa
    return tighten_gaia_answer(raw)


def _oracle_script(task: HotpotTask) -> List[str]:
    """A fixed, correct ReAct policy for the wiring smoke: recall against the
    brain-fed context, then answer the gold in the FINAL ANSWER template."""
    return [
        f"Thought: recall the evidence.\nAction: recall[{task.question}]",
        f"Thought: I can answer.\nAction: final_answer[{task.answer}]",
    ]


def _run_one_baseline(baseline: str, tasks: List[HotpotTask], *, encoder: str,
                      nli: str, bar: float) -> Dict:
    """Run ONE registry baseline over the fixture with the ORACLE LLM. The
    context is fed THROUGH the MELD brain first (the load-bearing MELD-on-message
    step), so the agent recalls brain-resident evidence."""
    preds: List[Dict] = []
    integration: List[Dict] = []
    for t in tasks:
        mem = SMPMemory(encoder=encoder, nli=nli, cfg=_calibrated_cfg(encoder))
        ctx_patches = feed_context_through_brain(t, mem)
        agent = make_agent(baseline, llm=MockLLM(_oracle_script(t)), memory=mem,
                           tools=[RecallTool(mem), FinalAnswerTool()],
                           max_steps=6)
        result = agent.run(t.question)
        pred = extract_prediction(result.answer)
        preds.append({"task_id": t.task_id, "level": t.level,
                      "prediction": pred, "gold": t.answer})
        integration.append({
            "task_id": t.task_id,
            "n_steps": result.n_steps,
            "n_patches": len(mem.brain.audit_log),
            "n_context_patches": len(ctx_patches),
            "decisions": sorted({p.decision for p in mem.brain.audit_log}),
        })
    return {"scored": score_predictions(preds),
            "gate": competence_gate(preds, bar=bar),
            "integration": integration, "n_tasks": len(tasks),
            "mode": "LOCAL-WIRING-SMOKE (oracle LLM; NOT a HotpotQA result)"}


def run_gate_smoke(n: int = 6, encoder: str = "hashing", nli: str = "lexical",
                   bar: float = 0.60, baseline: str = "react") -> Dict:
    tasks = load_hotpot_fixture(n=n)
    return _run_one_baseline(baseline, tasks, encoder=encoder, nli=nli, bar=bar)


def run_gate_smoke_multibaseline(n: int = 6, encoder: str = "hashing",
                                 nli: str = "lexical", bar: float = 0.60,
                                 baselines: Optional[List[str]] = None) -> Dict:
    baselines = baselines or ["react", "reflexion"]
    tasks = load_hotpot_fixture(n=n)
    per = {b: _run_one_baseline(b, tasks, encoder=encoder, nli=nli, bar=bar)
           for b in baselines}
    return {"per_baseline": per, "n_tasks": len(tasks), "baselines": baselines,
            "mode": "LOCAL-WIRING-SMOKE multibaseline (oracle; NOT a result)"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--encoder", default="hashing")
    ap.add_argument("--nli", default="lexical")
    ap.add_argument("--bar", type=float, default=0.60)
    a = ap.parse_args(argv)
    print(json.dumps(run_gate_smoke(n=a.n, encoder=a.encoder, nli=a.nli,
                                    bar=a.bar), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_gate_smoke", "run_gate_smoke_multibaseline",
           "extract_prediction", "_calibrated_cfg"]
