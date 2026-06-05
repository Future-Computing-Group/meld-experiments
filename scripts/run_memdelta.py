"""CSC M-1 memory-delta run (compute node; offline) — the REDESIGNED HotpotQA
experiment: measure the merge-protocol DELTA, not absolute agent F1.

REPLACES run_hotpot_gate_csc.py's centralized-only framing. That driver fed ALL of
a question's context into ONE brain (= the ``centralized`` condition) and reported
an absolute agent F1; it never exercised the merge ACROSS sovereign brains, which
is SMP's whole contribution. This driver runs the SAME held-out questions under
EVERY MEMORY_CONDITION x answer-mode x model and the headline is the condition
DELTA (distributed_merge - naive_union; distributed_merge - centralized; the
floor/ceiling none/full_context). See src/smp/agent/mem_delta_metrics.py.

DURABLE CHECKPOINT + RESUME (G4 acceptance criterion): each completed
(condition,mode,model,question) prediction is appended atomically (write+fsync) to
a JSONL on persistent scratch. On (re)start, already-scored keys are SKIPPED, so an
induced abort resumes without redoing or losing work. Final scoring (the paired
bootstrap deltas) reads the JSONL, so a resumed run scores ALL keys.

LLM TRANSPORT: vLLM OpenAI-compatible endpoint (transport=vllm-openai), urllib-only
so the protobuf-6.x .venv pin is untouched. One model is served per pass (swap the
served checkpoint between model passes). transformers is the fallback only if vLLM
bring-up is fiddly on the node.

Env (set by the sbatch): SMP_ENDPOINT, SMP_TRANSPORT, SMP_MODEL_KEY, SMP_OUT,
SMP_PER_LEVEL (e.g. "easy:20,medium:60,hard:60"), SMP_CONDITIONS, SMP_MODES,
SMP_BASELINE (agentic mode only), SMP_N_BRAINS, SMP_OVERLAP, SMP_INJECT_CONFLICT,
SMP_MAX_STEPS, SMP_MAX_NEW_TOKENS, SMP_K.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from smp.agent.hotpot_loader import load_hotpot_real, stratified_slice    # noqa: E402
from smp.agent.llm import OPEN_MODEL_AXIS, open_model                     # noqa: E402
from smp.agent.mem_conditions import ANSWER_MODES, MEMORY_CONDITIONS      # noqa: E402
from smp.agent.run_hotpot_gate import _calibrated_cfg                     # noqa: E402
from smp.agent.run_hotpot_memdelta import run_one, summarize             # noqa: E402


def _parse_per_level(spec: str) -> dict:
    out = {}
    for part in (spec or "").split(","):
        if ":" in part:
            lvl, n = part.split(":")
            out[lvl.strip()] = int(n)
    return out or {"easy": 20, "medium": 60, "hard": 60}


def _durable_append(path: str, rec: dict) -> None:
    """Atomic append + fsync so an abort never loses a completed prediction."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _done_keys(path: str) -> set:
    """Resume set keyed by (model_key, condition, mode, question)."""
    done = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                    done.add((r["model_key"], r["condition"], r["mode"],
                              r["question_id"]))
                except Exception:  # noqa: BLE001
                    continue
    return done


def main() -> int:
    endpoint = os.environ.get("SMP_ENDPOINT",
                              "http://localhost:8000/v1/completions")
    transport = os.environ.get("SMP_TRANSPORT", "vllm-openai")
    model_key = os.environ.get("SMP_MODEL_KEY", "qwen2.5-7b")
    out = os.environ["SMP_OUT"]  # required; durable JSONL (no silent default)
    per_level = _parse_per_level(os.environ.get("SMP_PER_LEVEL",
                                                "easy:20,medium:60,hard:60"))
    conditions = [c for c in os.environ.get(
        "SMP_CONDITIONS", ",".join(MEMORY_CONDITIONS)).split(",") if c]
    modes = [m for m in os.environ.get("SMP_MODES", "fixed_recall").split(",")
             if m]
    baseline = os.environ.get("SMP_BASELINE", "react")
    n_brains = int(os.environ.get("SMP_N_BRAINS", "3"))
    overlap = float(os.environ.get("SMP_OVERLAP", "0.5"))
    inject_conflict = os.environ.get("SMP_INJECT_CONFLICT", "0") == "1"
    max_steps = int(os.environ.get("SMP_MAX_STEPS", "8"))
    max_new = int(os.environ.get("SMP_MAX_NEW_TOKENS", "512"))
    k = int(os.environ.get("SMP_K", "5"))
    # minilm is the headline encoder (its similarity separates the merge classes;
    # the hashing encoder cannot -- see smp.merge.calibration). Requires the
    # all-MiniLM-L6-v2 model cached in HF_HOME for the offline (HF_HUB_OFFLINE) run.
    encoder = os.environ.get("SMP_ENCODER", "minilm")

    if model_key not in OPEN_MODEL_AXIS:
        print(f"ERROR: unknown model key {model_key!r}; axis={list(OPEN_MODEL_AXIS)}")
        return 2
    bad_c = [c for c in conditions if c not in MEMORY_CONDITIONS]
    bad_m = [m for m in modes if m not in ANSWER_MODES]
    if bad_c or bad_m:
        print(f"ERROR: unknown conditions={bad_c} modes={bad_m}")
        return 2

    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = _done_keys(out)
    print(f"[run] model={model_key} transport={transport} endpoint={endpoint}")
    print(f"[run] conditions={conditions} modes={modes} baseline={baseline} "
          f"N={n_brains} overlap={overlap} conflict={inject_conflict}")
    print(f"[run] per_level={per_level} resume: {len(done)} scored keys in {out}")

    # OFFLINE load of the real HotpotQA distractor validation split, then a
    # deterministic level-stratified slice (reads the prefetched HF cache under
    # HF_HOME with HF_HUB_OFFLINE=1). The SAME tasks across all conditions/modes.
    all_levels = list(per_level)
    pool = load_hotpot_real(split="validation", config="distractor",
                            levels=all_levels)
    tasks = stratified_slice(pool, per_level)
    counts = {lvl: sum(1 for t in tasks if t.level == lvl) for lvl in all_levels}
    print(f"[run] sliced {len(tasks)} tasks ({counts})")

    cfg = _calibrated_cfg(encoder)        # encoder-matched calibrated thresholds
    llm = open_model(model_key, transport=transport, endpoint=endpoint,
                     max_new_tokens=max_new, temperature=0.0)

    for mode in modes:
        for cond in conditions:
            for t in tasks:
                key = (model_key, cond, mode, t.task_id)
                if key in done:
                    continue
                try:
                    row = run_one(t, cond, mode, llm, encoder=encoder,
                                  nli="lexical", cfg=cfg, baseline=baseline,
                                  n_brains=n_brains, overlap_fraction=overlap,
                                  inject_conflict=inject_conflict, k=k,
                                  max_steps=max_steps)
                except Exception as e:  # noqa: BLE001
                    # carry ALL metric keys (incl. recall_at_k/mrr/footprint) so
                    # summarize() can't KeyError on an error row.
                    row = {"question_id": t.task_id, "task_id": t.task_id,
                           "condition": cond, "mode": mode, "baseline": baseline,
                           "level": t.level, "prediction": "", "gold": t.answer,
                           "em": 0.0, "f1": 0.0,
                           "evidence": [], "prompt": "",
                           "recall_at_k": 0.0, "mrr": 0.0, "n_gold": 0,
                           "stored_units": 0, "live_units": 0,
                           "n_merge": 0, "n_relate": 0,
                           "n_conflict": 0, "n_overlap_merges": 0,
                           "n_total_patches": 0,
                           "error": f"{type(e).__name__}: {str(e)[:200]}"}
                row["model_key"] = model_key
                _durable_append(out, row)

    # Final scoring reads the durable JSONL (so a resumed run scores ALL keys).
    rows = [json.loads(ln) for ln in open(out, encoding="utf-8") if ln.strip()]
    rows = [r for r in rows if r.get("model_key") == model_key]

    # INTEGRITY GATE: run_one writes an all-zero row with an `error` string on any
    # failure (e.g. a dead inference endpoint), and summarize() averages those zeros
    # as if they were data -- so a fully-failed run is statistically indistinguishable
    # from a clean null result. Make that impossible: count error rows and FAIL loudly
    # if too many, rather than emitting a deceptively tidy all-zero summary.
    n_err = sum(1 for r in rows if r.get("error"))
    err_frac = (n_err / len(rows)) if rows else 1.0
    print(f"[integrity] {model_key}: {n_err}/{len(rows)} rows carry an error "
          f"(frac={err_frac:.3f})")
    if err_frac > 0.10:
        sample = next((r.get("error") for r in rows if r.get("error")), "")
        print(f"[FATAL] error-row fraction {err_frac:.1%} exceeds 10% for {model_key}: "
              f"the inference path is failing; the all-zero summary is INVALID, not a "
              f"null result. First error: {str(sample)[:160]}")
        return 3

    summary = summarize(rows, modes=modes, conditions=conditions,
                        n_boot=2000, seed=0)
    for mode, m in summary["per_mode"].items():
        # PRIMARY observable: retrieval quality (recall@k) + the storage footprint.
        print(f"\n[summary {model_key}/{mode}] condition means "
              f"(recall@{k} | F1 | live/stored units):")
        for c in conditions:
            mr = m["means_recall"].get(c)
            mf = m["means_f1"].get(c)
            fp = m["footprint"].get(c, {})
            if mr:
                print(f"   {c:18s} recall@{k}={mr['mean']:.3f}  F1={mf['mean']:.3f}  "
                      f"live/stored={fp.get('live', 0):.1f}/{fp.get('stored', 0):.1f} "
                      f"(n={mr['n']})")
        print(f"[summary {model_key}/{mode}] HEADLINE deltas (recall@{k}, PRIMARY, 95% CI):")
        for d in m["headline_recall"]:
            if d["n"] == 0:
                print(f"   {d['a']} - {d['b']:18s} =    N/A   "
                      f"[condition not run; n=0]  ({d['label']})")
            else:
                print(f"   {d['a']} - {d['b']:18s} = {d['delta']:+.3f} "
                      f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]  (n={d['n']}; {d['label']})")
        print(f"[summary {model_key}/{mode}] secondary deltas (answer F1, 95% CI):")
        for d in m["headline_f1"]:
            if d["n"] == 0:
                # No paired questions (a referenced condition was not run / failed
                # to populate). A missing delta is N/A, NOT a confident +0.000 --
                # never let an absent condition masquerade as a clean zero.
                print(f"   {d['a']} - {d['b']:18s} =    N/A   "
                      f"[condition not run; n=0]  ({d['label']})")
            else:
                print(f"   {d['a']} - {d['b']:18s} = {d['delta']:+.3f} "
                      f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]  (n={d['n']}; {d['label']})")
        ma = m["merge_audit"]
        dm = ma.get("distributed_merge", {})
        print(f"[summary {model_key}/{mode}] merge audit distributed_merge: "
              f"overlap_merges={dm.get('overlap_merges')} "
              f"conflict={dm.get('conflict')} "
              f"naive_union overlap_merges="
              f"{ma.get('naive_union', {}).get('overlap_merges')}")
    with open(out + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
