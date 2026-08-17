"""M-1 memory-delta LOCAL validation (0-cost) — the conditions yield DISTINCT F1,
distributed_merge exercises same-enough/overlaps (audit log), and the DELTA table
prints with paired bootstrap CIs.

This is the local rung for the redesigned HotpotQA experiment (the M-1
memory-condition DELTA, not absolute agent F1). It runs the SAME held-out
questions under every MEMORY_CONDITION x answer-mode, scores EM/token-F1, and
prints the floor/ceiling means + the three headline deltas:
  distributed_merge - naive_union   (does the merge beat dumping)
  distributed_merge - centralized   (is the merge lossless vs a single store)
  full_context      - none          (the measurable headroom / competence FLOOR)

Two LLM backends (no new python deps; urllib over an OpenAI-compatible endpoint,
so the protobuf-6.x .venv pin is untouched):
  - SMP_LLM=ollama (default): a real small open model (qwen2.5:7b) at the local
    ollama endpoint — a GENUINE EM/F1 delta signal.
  - SMP_LLM=oracle: a deterministic gold-emitting MockLLM — a 0-dependency CI/wiring
    check (NOT a competence claim; every condition that can recall the gold scores
    1.0, so it validates plumbing, not the delta).

Env: SMP_LLM (ollama|oracle), SMP_REALMODEL (qwen2.5:7b), SMP_ENDPOINT
(http://localhost:11434/v1/completions), SMP_N (10), SMP_OVERLAP (0.5),
SMP_MODES (fixed_recall or "fixed_recall,agentic"), SMP_AGENTIC_N (2).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from meld.agent.hotpot_loader import load_hotpot_fixture                    # noqa: E402
from meld.agent.llm import (MockLLM, OPEN_MODEL_AXIS, OpenWeightLLM,        # noqa: E402
                           open_model)
from meld.agent.mem_conditions import MEMORY_CONDITIONS                     # noqa: E402
from meld.agent.run_hotpot_gate import _calibrated_cfg                      # noqa: E402
from meld.agent.run_hotpot_memdelta import (format_delta_table, run_one,    # noqa: E402
                                           summarize)


def _model_axis_smoke() -> int:
    """SMP_MODELS axis-loop smoke (0-BU, offline). For each named OPEN_MODEL_AXIS
    key: resolve it via open_model() (axis lookup -> exact HF id, import-light),
    then run the full per-cell pipeline (build -> recall -> row schema incl. the
    new evidence/prompt -> summarize -> headline_f1) with a deterministic MockLLM.

    MockLLM ignores the weights (compute is offline), but this exercises EXACTLY
    the new wiring an added size-axis model touches: axis resolution + the
    model-independent CPU build + the persisted evidence/prompt + the per-model
    summary. The gated CSC run swaps MockLLM for the served OpenWeightLLM.
    """
    keys = [k for k in os.environ["SMP_MODELS"].split() if k]
    n = int(os.environ.get("SMP_N", "4"))
    overlap = float(os.environ.get("SMP_OVERLAP", "0.6"))
    enc, nli, cfg = "hashing", "lexical", _calibrated_cfg()
    tasks = load_hotpot_fixture(n=n)

    print("== M-1 model-SIZE axis LOCAL smoke (offline, 0 BU) ==")
    print(f"   SMP_MODELS={keys} n={n} overlap={overlap} "
          f"(MockLLM answers; axis lookup + build + row schema exercised)\n")

    ok = True
    for key in keys:
        # axis lookup: resolve the key to its OpenWeightLLM (exact HF id), proving
        # the new size-axis entries round-trip exactly as the 7B does.
        if key not in OPEN_MODEL_AXIS:
            print(f"   [{key}] FAIL: not on OPEN_MODEL_AXIS ({list(OPEN_MODEL_AXIS)})")
            ok = False
            continue
        client = open_model(key)
        assert isinstance(client, OpenWeightLLM)
        assert client.model == OPEN_MODEL_AXIS[key]["model"]
        print(f"   [{key}] resolves -> {client.model} "
              f"(role={OPEN_MODEL_AXIS[key]['role']})")

        rows = []
        for cond in MEMORY_CONDITIONS:
            for t in tasks:
                row = run_one(t, cond, "fixed_recall",
                              MockLLM([f"FINAL ANSWER: {t.answer}"]),
                              encoder=enc, nli=nli, cfg=cfg, baseline="react",
                              n_brains=3, overlap_fraction=overlap, k=5)
                row["model_key"] = key       # mirrors the CSC driver's row stamp
                rows.append(row)

        # row-schema invariants: model_key + the persisted evidence/prompt present.
        missing = [kk for kk in ("model_key", "evidence", "prompt")
                   if any(kk not in r for r in rows)]
        if missing:
            print(f"   [{key}] FAIL: rows missing keys {missing}")
            ok = False
            continue

        summary = summarize(rows, modes=["fixed_recall"], n_boot=200, seed=0)
        fr = summary["per_mode"]["fixed_recall"]
        hf1 = fr["headline_f1"]
        f1_populated = all(isinstance(d.get("delta"), float) for d in hf1)
        means = {c: round(fr["means_f1"][c]["mean"], 3) for c in MEMORY_CONDITIONS
                 if c in fr["means_f1"]}
        print(f"      rows={len(rows)} model_key+evidence+prompt OK | "
              f"headline_f1 deltas={len(hf1)} populated={f1_populated} | "
              f"means_f1={means}")
        if not (len(hf1) == 3 and f1_populated):
            ok = False

    print("\nLOCAL M-1 MODEL-AXIS SMOKE:", "OK" if ok else "CHECK ABOVE")
    return 0 if ok else 1


def main() -> int:
    # SMP_MODELS set -> the offline model-SIZE axis smoke (0 BU); else the
    # ollama/oracle delta-signal validation.
    if os.environ.get("SMP_MODELS"):
        return _model_axis_smoke()
    backend = os.environ.get("SMP_LLM", "ollama")
    model = os.environ.get("SMP_REALMODEL", "qwen2.5:7b")
    endpoint = os.environ.get("SMP_ENDPOINT",
                              "http://localhost:11434/v1/completions")
    n = int(os.environ.get("SMP_N", "10"))
    overlap = float(os.environ.get("SMP_OVERLAP", "0.5"))
    modes = os.environ.get("SMP_MODES", "fixed_recall").split(",")
    agentic_n = int(os.environ.get("SMP_AGENTIC_N", "2"))
    enc, nli, cfg = "hashing", "lexical", _calibrated_cfg()

    tasks = load_hotpot_fixture(n=n)

    def llm_factory(task):
        if backend == "oracle":
            return MockLLM([f"FINAL ANSWER: {task.answer}"])
        return OpenWeightLLM(model=model, transport="vllm-openai",
                             endpoint=endpoint, max_new_tokens=256,
                             temperature=0.0)

    print("== M-1 memory-delta LOCAL validation ==")
    print(f"   backend={backend} model={model if backend=='ollama' else 'oracle'} "
          f"n={n} overlap={overlap} modes={modes}")
    print("   the SAME questions under every condition; headline = the DELTA, "
          "not absolute F1\n")

    rows = []
    # fixed_recall over ALL conditions on all n questions; agentic on a subset of
    # conditions+questions (realism; agentic is slower and noisier).
    for mode in modes:
        cond_tasks = tasks if mode == "fixed_recall" else tasks[:agentic_n]
        for cond in MEMORY_CONDITIONS:
            for t in cond_tasks:
                row = run_one(t, cond, mode, llm_factory(t), encoder=enc,
                              nli=nli, cfg=cfg, baseline="react",
                              n_brains=3, overlap_fraction=overlap, k=5,
                              max_steps=6)
                rows.append(row)
                tag = "OK" if row["em"] == 1.0 else f"F1={row['f1']:.2f}"
                print(f"   [{mode}/{cond:18s}] {t.task_id} {t.level} "
                      f"pred={row['prediction']!r} gold={row['gold']!r} {tag} "
                      f"merge={row['n_overlap_merges']} conf={row['n_conflict']}")
        print()

    summary = summarize(rows, modes=modes, n_boot=2000, seed=0)
    print(format_delta_table(summary))

    # the validation assertions: (a) conditions yield DISTINCT F1 (not all equal);
    # (b) distributed_merge fired >= 1 overlap merge; (c) naive_union fired none.
    fr = summary["per_mode"].get("fixed_recall")
    ok = True
    if fr is not None:
        means = {c: fr["means_f1"][c]["mean"] for c in MEMORY_CONDITIONS
                 if c in fr["means_f1"]}
        distinct = len(set(round(v, 3) for v in means.values())) > 1
        dm_merge = fr["merge_audit"]["distributed_merge"]["overlap_merges"]
        nu_merge = fr["merge_audit"]["naive_union"]["overlap_merges"]
        print(f"\n[validate] distinct condition F1? {distinct} "
              f"({ {c: round(v,3) for c,v in means.items()} })")
        print(f"[validate] distributed_merge overlap merges = {dm_merge} (>0 wanted); "
              f"naive_union overlap merges = {nu_merge} (==0 wanted)")
        ok = distinct and dm_merge >= 1 and nu_merge == 0
    print("\nLOCAL M-1 VALIDATION:", "OK" if ok else "CHECK ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
