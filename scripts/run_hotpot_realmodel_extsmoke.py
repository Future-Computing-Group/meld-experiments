"""G3 LOCAL REAL-MODEL ext-smoke (HotpotQA pivot; 0-cost; small open model via a
local OpenAI-compatible endpoint) — the rung between the oracle wiring smoke and
the gated/networked CSC run.

Unlike the GAIA fixture (whose oracle/clue setup made accuracy incidental), this
yields a GENUINE EM/F1: each item's HotpotQA-shaped ``context`` paragraphs are fed
THROUGH the SMP brain (PUBLISH -> merge_on_receive -> Patch), then the question is
posed and the agent must RECALL the brain-resident evidence and MULTI-HOP REASON
to answer. So the score is a real competence signal (still on a tiny fixture +
7B model, so read it as a signal, not the gate verdict — that is the CSC run).

Reports per baseline: PATH-clean (contract_fail=0; parseable, tight answers) AND
HotpotQA EM / token-F1 (+ the medium+hard gate F1).

DRIVES a real model with ZERO new python deps (urllib over an OpenAI-compatible
endpoint), so the protobuf-6.x ``.venv`` pin is untouched. Default endpoint is a
local ollama server (``ollama serve``; ``ollama pull qwen2.5:7b``).

Env: SMP_REALMODEL (default qwen2.5:7b), SMP_ENDPOINT
(default http://localhost:11434/v1/completions), SMP_N (default 6).
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from smp.agent.hotpot_loader import (feed_context_through_brain,            # noqa: E402
                                     load_hotpot_fixture)
from smp.agent.hotpot_scorer import (competence_gate, em_score,             # noqa: E402
                                     f1_score, score_predictions)
from smp.agent.llm import OpenWeightLLM                                     # noqa: E402
from smp.agent.memory import SMPMemory                                      # noqa: E402
from smp.agent.registry import make_agent                                   # noqa: E402
from smp.agent.run_hotpot_gate import _calibrated_cfg, extract_prediction   # noqa: E402
from smp.agent.tools import FinalAnswerTool, RecallTool                     # noqa: E402


def _is_clean_final(pred: str) -> tuple:
    """A PATH-clean answer: non-empty, NOT a leaked tool-call blob, single line."""
    p = (pred or "").strip()
    if not p:
        return False, "NO-FINAL-ANSWER"
    low = p.lower()
    if ('"name"' in low and ("recall" in low or "arguments" in low)) \
            or low.startswith("action:") or (p.startswith("{") and "name" in low):
        return False, "TOOL-CALL-LEAK"
    if "\n" in p:
        return False, "MULTI-LINE-NOT-TIGHT"
    return True, "OK"


def main() -> int:
    model = os.environ.get("SMP_REALMODEL", "qwen2.5:7b")
    endpoint = os.environ.get("SMP_ENDPOINT",
                              "http://localhost:11434/v1/completions")
    n = int(os.environ.get("SMP_N", "6"))

    llm = OpenWeightLLM(model=model, transport="vllm-openai", endpoint=endpoint,
                        max_new_tokens=256, temperature=0.0)

    baselines = ["react", "reflexion"]
    if importlib.util.find_spec("smolagents") is not None:
        baselines.append("smolagents-toolcalling")

    tasks = load_hotpot_fixture(n=n)
    print(f"== G3 HotpotQA LOCAL REAL-MODEL ext-smoke ==  model={model}")
    print(f"   endpoint={endpoint}  baselines={baselines}  tasks={len(tasks)}")
    print("   context fed THROUGH the SMP brain -> recall+multi-hop reason "
          "=> a GENUINE EM/F1 signal (tiny fixture + 7B: a signal, not the "
          "gate verdict)\n")

    overall_ok = True
    for b in baselines:
        preds, contract_fail = [], 0
        steps_lo = steps_hi = patches_lo = patches_hi = None
        for t in tasks:
            mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
            feed_context_through_brain(t, mem)        # SMP-on-message: context -> brain
            agent = make_agent(b, llm=llm, memory=mem,
                               tools=[RecallTool(mem), FinalAnswerTool()],
                               max_steps=6)
            try:
                res = agent.run(t.question)
                pred = extract_prediction(res.answer)
                ns, npatch = res.n_steps, len(mem.brain.audit_log)
                steps_lo = ns if steps_lo is None else min(steps_lo, ns)
                steps_hi = ns if steps_hi is None else max(steps_hi, ns)
                patches_lo = npatch if patches_lo is None else min(patches_lo, npatch)
                patches_hi = npatch if patches_hi is None else max(patches_hi, npatch)
                ok, why = _is_clean_final(pred)
                contract_fail += 0 if ok else 1
                em, f1 = em_score(pred, t.answer), f1_score(pred, t.answer)
                print(f"   [{b}] {t.task_id} {t.level} steps={ns} patches={npatch} "
                      f"pred={pred!r} gold={t.answer!r} EM={em:.0f} F1={f1:.2f} "
                      f"{'OK' if ok else why}")
            except Exception as e:
                contract_fail += 1
                print(f"   [{b}] {t.task_id} CONTRACT-FAIL: "
                      f"{type(e).__name__}: {str(e)[:140]}")
                preds.append({"task_id": t.task_id, "level": t.level,
                              "prediction": "", "gold": t.answer})
                continue
            preds.append({"task_id": t.task_id, "level": t.level,
                          "prediction": pred, "gold": t.answer})
        sc = score_predictions(preds)
        g = competence_gate(preds, bar=0.60)
        path_ok = contract_fail == 0 and (steps_hi or 0) >= 1
        overall_ok = overall_ok and path_ok
        print(f"   == {b}: PATH {'OK' if path_ok else 'BROKEN'} "
              f"(contract_fail={contract_fail}, steps={steps_lo}-{steps_hi}, "
              f"patches={patches_lo}-{patches_hi}); "
              f"EM={sc['overall']['em']:.2f} F1={sc['overall']['f1']:.2f} "
              f"(n={sc['overall']['n']}); gate(med+hard)F1={g['gate_f1']:.2f} "
              f"n={g['gate_n']} PASS={g['passed']}  "
              f"[REAL competence signal on a tiny fixture]\n")
    print("REAL-MODEL PATH:", "OK across all baselines" if overall_ok
          else "BROKEN on >=1 baseline (see CONTRACT-FAIL above)")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
