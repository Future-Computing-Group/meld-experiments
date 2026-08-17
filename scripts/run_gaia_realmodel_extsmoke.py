"""G3 LOCAL REAL-MODEL ext-smoke (0-cost; small open model via a local OpenAI
endpoint) — the rung between the oracle wiring smoke and the gated CSC run.

PURPOSE: exercise the REAL-LLM path the MockLLM cannot — prompt construction,
the ReAct text contract, output parsing, tool dispatch, the open-weight-quirk
surface — across the multi-baseline registry, each through the MELD Brain. It is
NOT a competence claim: a small local model + a tiny fixture means correctness is
incidental; the signal is "does a real model drive the harness end-to-end without
contract/parse/dispatch failures?"

DRIVES a real model with ZERO new python deps (urllib over an OpenAI-compatible
endpoint), so the protobuf-6.x `.venv` pin is untouched. Default endpoint is a
local ollama server (`ollama serve`; `ollama pull qwen2.5:7b`).

Env: SMP_REALMODEL (default qwen2.5:7b), SMP_ENDPOINT
(default http://localhost:11434/v1/completions), SMP_N (default 3).
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from meld.agent.gaia_loader import load_gaia_fixture            # noqa: E402
from meld.agent.gaia_scorer import (competence_gate,            # noqa: E402
                                    extract_final_answer, score_predictions)
from meld.agent.llm import OpenWeightLLM                        # noqa: E402
from meld.agent.memory import SMPMemory                         # noqa: E402
from meld.agent.registry import make_agent                      # noqa: E402
from meld.agent.run_gaia_gate import _calibrated_cfg            # noqa: E402
from meld.agent.tools import (CalculatorTool, FinalAnswerTool,  # noqa: E402
                             RecallTool)


def _is_clean_final(pred: str) -> tuple:
    """A PATH-clean final answer: non-empty, NOT a leaked tool-call blob (the
    smolagents recall-as-final_answer confusion), and GAIA-format-tight (a short
    single-line string/number/list, not a JSON dump or a paragraph)."""
    p = (pred or "").strip()
    if not p:
        return False, "NO-FINAL-ANSWER"
    low = p.lower()
    # a leaked tool-call (JSON {"name": ...} or an unconsumed Action: tool[...])
    if ('"name"' in low and ("recall" in low or "arguments" in low)) \
            or low.startswith("action:") or (p.startswith("{") and "name" in low):
        return False, "TOOL-CALL-LEAK"
    if "\n" in p:                       # GAIA answers are a single short line
        return False, "MULTI-LINE-NOT-TIGHT"
    return True, "OK"


def main() -> int:
    model = os.environ.get("SMP_REALMODEL", "qwen2.5:7b")
    endpoint = os.environ.get("SMP_ENDPOINT",
                              "http://localhost:11434/v1/completions")
    n = int(os.environ.get("SMP_N", "3"))

    llm = OpenWeightLLM(model=model, transport="vllm-openai", endpoint=endpoint,
                        max_new_tokens=256, temperature=0.0)

    baselines = ["react", "reflexion"]
    if importlib.util.find_spec("smolagents") is not None:
        baselines.append("smolagents-toolcalling")

    tasks = load_gaia_fixture(n=n)
    print(f"== G3 LOCAL REAL-MODEL ext-smoke ==  model={model} endpoint={endpoint}")
    print(f"   baselines={baselines}  tasks={len(tasks)}")
    print("   (real-LLM PATH validation; small model => correctness incidental)\n")

    overall_ok = True
    for b in baselines:
        preds, contract_fail = [], 0
        steps_lo = steps_hi = patches_lo = patches_hi = None
        for t in tasks:
            mem = SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())
            for clue in t.clues:                 # pre-load via the MELD merge path
                mem.write(clue)
            agent = make_agent(b, llm=llm, memory=mem,
                               tools=[RecallTool(mem), FinalAnswerTool(),
                                      CalculatorTool()],
                               max_steps=6)
            try:
                res = agent.run(t.question)
                pred = res.answer or ""
                if pred.upper().startswith("FINAL ANSWER:"):
                    pred = extract_final_answer(pred)
                ns, npatch = res.n_steps, len(mem.brain.audit_log)
                steps_lo = ns if steps_lo is None else min(steps_lo, ns)
                steps_hi = ns if steps_hi is None else max(steps_hi, ns)
                patches_lo = npatch if patches_lo is None else min(patches_lo, npatch)
                patches_hi = npatch if patches_hi is None else max(patches_hi, npatch)
                # PATH-clean = a parseable final answer that is NOT a leaked
                # tool-call blob (the smolagents recall-as-final_answer bug) and
                # is GAIA-format-tight (not a multi-line / JSON dump).
                ok, why = _is_clean_final(pred)
                contract_fail += 0 if ok else 1
                print(f"   [{b}] {t.task_id} L{t.level} steps={ns} patches={npatch} "
                      f"pred={pred!r} gold={t.gold!r} {'OK' if ok else why}")
            except Exception as e:           # contract/parse/dispatch failure = the bug we hunt
                contract_fail += 1
                print(f"   [{b}] {t.task_id} CONTRACT-FAIL: {type(e).__name__}: {str(e)[:140]}")
                preds.append({"task_id": t.task_id, "level": t.level,
                              "prediction": "", "gold": t.gold})
                continue
            preds.append({"task_id": t.task_id, "level": t.level,
                          "prediction": pred, "gold": t.gold})
        sc = score_predictions(preds)
        g = competence_gate(preds, bar=0.60)
        path_ok = contract_fail == 0 and (steps_hi or 0) >= 1
        overall_ok = overall_ok and path_ok
        print(f"   == {b}: PATH {'OK' if path_ok else 'BROKEN'} "
              f"(contract_fail={contract_fail}, steps={steps_lo}-{steps_hi}, "
              f"patches={patches_lo}-{patches_hi}); incidental acc "
              f"{sc['overall']['correct']}/{sc['overall']['n']}, "
              f"gateL2={g['l2_rate']:.2f} (NOT a competence claim)\n")
    print("REAL-MODEL PATH:", "OK across all baselines" if overall_ok
          else "BROKEN on >=1 baseline (see CONTRACT-FAIL above)")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
