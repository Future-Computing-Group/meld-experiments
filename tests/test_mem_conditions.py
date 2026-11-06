"""M-1 memory-condition harness tests (TDD).

Cover: (1) condition wiring (all 5 conditions build a recall source); (2) the
load-bearing OVERLAP injection actually FIRES the same-enough/overlaps MERGE in
distributed_merge (and NEVER in naive_union); (3) the conflict variant forms a
first-class CONTRADICTS edge; (4) fixed_recall determinism; (5) the paired
bootstrap delta CI; (6) the scorer integration through run_one.
"""
from __future__ import annotations

from meld.agent.hotpot_loader import HotpotTask, load_hotpot_fixture
from meld.agent.llm import MockLLM
from meld.agent.mem_conditions import (DEFAULT_N_BRAINS, MEMORY_CONDITIONS,
                                      assemble_recall, build_condition,
                                      fixed_recall_prompt, light_paraphrase,
                                      split_across_brains)
from meld.agent.mem_delta_metrics import bootstrap_delta, headline_table
from meld.agent.run_hotpot_memdelta import run_one, summarize
from test_agent_meld_memory import agent_epoch

ENC, NLI = "hashing", "lexical"
EPOCH = agent_epoch(ENC, NLI)      # a test epoch on the hashing + lexical substrate


def _task() -> HotpotTask:
    # the medium bridge fixture item (Inception director nationality).
    return load_hotpot_fixture(n=1)[0]


# -- (1) condition wiring ---------------------------------------------------


def test_all_conditions_build():
    t = _task()
    for cond in MEMORY_CONDITIONS:
        b = build_condition(t, cond, encoder=ENC, nli=NLI, epoch=EPOCH)
        assert b.condition == cond
        if cond == "none":
            assert b.brain is None and b.inline_context == []
        elif cond == "full_context":
            assert b.brain is None and len(b.inline_context) > 0
        else:
            assert b.brain is not None and b.stored_units > 0
            assert (b.n_total_patches > 0) == (cond != "naive_union")   # the union admits nothing


def test_unknown_condition_raises():
    import pytest
    with pytest.raises(ValueError):
        build_condition(_task(), "bogus", encoder=ENC, nli=NLI, epoch=EPOCH)


# -- (2) overlap injection FIRES the merge (load-bearing) -------------------


def test_split_injects_non_disjoint_overlap():
    """The split must NOT be a clean disjoint partition: a controlled fraction of
    gold sentences is duplicated (paraphrased) across brains."""
    t = _task()
    plan = split_across_brains(t, n_brains=DEFAULT_N_BRAINS,
                               overlap_fraction=0.5)
    assert plan.n_gold >= 1
    assert plan.n_duplicated >= 1, "no gold sentence was duplicated across brains"
    assert plan.n_paraphrased >= 1, "duplicates were not lightly paraphrased"
    # the duplicated sentence lives on >= 2 brains (overlap, not a partition)
    nonempty = [b for b in plan.per_brain if b]
    assert len(nonempty) >= 2


def test_light_paraphrase_changes_text_but_keeps_tokens():
    s = "Inception was directed by Christopher Nolan."
    p = light_paraphrase(s)
    assert p != s
    assert "Christopher Nolan" in p  # entity preserved -> stays dense-similar


def test_distributed_merge_fires_overlap_merge_in_audit_log():
    """distributed_merge: the global brain MUST emit >= 1 MERGE-via-sigma
    (same-enough/overlap) Patch -- the merge actually does something. This is the
    assertion the brief requires (verify via the audit log)."""
    t = _task()
    b = build_condition(t, "distributed_merge", encoder=ENC, nli=NLI, epoch=EPOCH,
                        n_brains=DEFAULT_N_BRAINS, overlap_fraction=0.6)
    assert b.n_overlap_merges >= 1, (
        "distributed_merge fired no overlap MERGE -- the merge is trivial; "
        f"audit decisions={[p.decision for p in b.brain.decisions]}")
    # without an epoch the same build makes no similarity merge
    bare = build_condition(t, "distributed_merge", encoder=ENC, nli=NLI,
                           n_brains=DEFAULT_N_BRAINS, overlap_fraction=0.6)
    assert bare.n_merge == 0 and bare.live_units == bare.stored_units


def test_naive_union_never_merges():
    """The ablation: naive_union is a raw union -- NO MERGE-via-overlap, RELATE,
    or CONFLICT ever fires (every write is a plain novel insert)."""
    t = _task()
    b = build_condition(t, "naive_union", encoder=ENC, nli=NLI, epoch=EPOCH,
                        n_brains=DEFAULT_N_BRAINS, overlap_fraction=0.6)
    assert b.n_overlap_merges == 0
    assert b.n_relate == 0
    assert b.n_conflict == 0
    # nothing is admitted (the union): no decision, no link
    assert b.brain.decisions == [] and b.brain.store.derived.links == []


def test_distributed_merge_smaller_than_naive_union_store():
    """A direct consequence of merging overlaps: the merged global store holds
    FEWER distinct units than the naive union of the same split."""
    t = _task()
    dm = build_condition(t, "distributed_merge", encoder=ENC, nli=NLI, epoch=EPOCH,
                         overlap_fraction=0.6)
    nu = build_condition(t, "naive_union", encoder=ENC, nli=NLI, epoch=EPOCH,
                         overlap_fraction=0.6)
    assert dm.stored_units <= nu.stored_units


def test_distributed_merge_has_fewer_live_recall_units_than_naive_union():
    # the real dedup signal (fix #1): the merge deprecates paraphrased duplicates,
    # so FEWER units are status-live (recallable) than the naive union, even though
    # both keep every unit in the store (status, not truth). This live-unit count
    # is the storage/compactness metric for the overhead claim -- NOT the audit
    # patch count (n_total_patches), which is identical across the two conditions.
    t = _task()
    dm = build_condition(t, "distributed_merge", encoder=ENC, nli=NLI, epoch=EPOCH,
                         overlap_fraction=0.6)
    nu = build_condition(t, "naive_union", encoder=ENC, nli=NLI, epoch=EPOCH,
                         overlap_fraction=0.6)
    assert dm.live_units < dm.stored_units      # at least one duplicate deprecated
    assert dm.live_units < nu.live_units        # merge recalls fewer redundant units
    assert nu.live_units == nu.stored_units     # naive union deprecates nothing


# -- (2b) conflict variant forms a CONTRADICTS edge -------------------------


def test_conflict_variant_forms_contradicts_edge():
    """The conflict variant injects a contradictory sentence; the global merge
    must emit >= 1 CONFLICT Patch (a first-class CONTRADICTS edge)."""
    # a single-supporting-fact item makes the contradiction the best candidate.
    t = HotpotTask(
        task_id="conf1",
        question="In which city was Christopher Nolan born?",
        answer="London", level="hard", qtype="bridge",
        context=[["Christopher Nolan",
                  ["Christopher Nolan was born in London.",
                   "Christopher Nolan is a director."]]],
        supporting_facts=[["Christopher Nolan", 0]],
    )
    b = build_condition(t, "distributed_merge", encoder=ENC, nli=NLI, epoch=EPOCH,
                        n_brains=2, overlap_fraction=0.0, inject_conflict=True)
    n_contradicts = sum(1 for ln in b.brain.store.derived.links
                        if ln[0] == "CONTRADICTS")
    assert b.n_conflict >= 1, "conflict variant fired no CONFLICT Patch"
    assert n_contradicts >= 1, "no first-class CONTRADICTS edge formed"


# -- (3) fixed_recall determinism -------------------------------------------


def test_fixed_recall_prompt_is_deterministic():
    t = _task()
    b = build_condition(t, "centralized", encoder=ENC, nli=NLI, epoch=EPOCH)
    ev1 = assemble_recall(b, t.question, encoder=ENC, k=5)
    ev2 = assemble_recall(b, t.question, encoder=ENC, k=5)
    assert ev1 == ev2  # graph-walk read is deterministic
    assert fixed_recall_prompt(t.question, ev1) == fixed_recall_prompt(t.question, ev2)


def test_fixed_recall_run_is_deterministic_with_fixed_llm():
    """Same (task, condition, fixed LLM) -> identical prediction + scores."""
    t = _task()
    script = [f"FINAL ANSWER: {t.answer}"]
    r1 = run_one(t, "distributed_merge", "fixed_recall", MockLLM(script),
                 encoder=ENC, nli=NLI, epoch=EPOCH)
    r2 = run_one(t, "distributed_merge", "fixed_recall", MockLLM(script),
                 encoder=ENC, nli=NLI, epoch=EPOCH)
    assert r1["prediction"] == r2["prediction"]
    assert r1["f1"] == r2["f1"] and r1["em"] == r2["em"]
    assert r1["n_overlap_merges"] == r2["n_overlap_merges"]


# -- (3b) deterministic-rebuild faithfulness (cross-size F1 validity) -------


def test_rebuild_is_byte_identical_across_two_builds_distributed_merge():
    """The cross-SIZE F1 comparison is only valid if a newly-added model sees the
    EXACT context the 7B arm saw. The per-cell build (split + merge + recall +
    prompt) is deterministic (CPU, model-independent), so building twice with the
    same task/seed/params must yield a BYTE-IDENTICAL assembled evidence + prompt.
    """
    t = _task()
    kw = dict(encoder=ENC, nli=NLI, epoch=EPOCH, n_brains=DEFAULT_N_BRAINS,
              overlap_fraction=0.6, inject_conflict=False)
    b1 = build_condition(t, "distributed_merge", **kw)
    b2 = build_condition(t, "distributed_merge", **kw)
    ev1 = assemble_recall(b1, t.question, encoder=ENC, k=5)
    ev2 = assemble_recall(b2, t.question, encoder=ENC, k=5)
    assert ev1 == ev2, "distributed_merge recall is NOT reproducible across builds"
    p1 = fixed_recall_prompt(t.question, ev1)
    p2 = fixed_recall_prompt(t.question, ev2)
    assert p1.encode("utf-8") == p2.encode("utf-8"), "prompt not byte-identical"
    # the store footprint / merge audit must also be identical (model-independent)
    assert (b1.stored_units, b1.live_units, b1.n_overlap_merges) == \
           (b2.stored_units, b2.live_units, b2.n_overlap_merges)


def test_rebuild_byte_identical_across_all_store_conditions():
    """Faithfulness across EVERY store-backed condition (centralized,
    distributed_merge, naive_union) -- each is a model-independent CPU build, so
    re-running it must reproduce the exact evidence/prompt a prior model saw."""
    t = _task()
    for cond in ("centralized", "distributed_merge", "naive_union"):
        kw = dict(encoder=ENC, nli=NLI, epoch=EPOCH, overlap_fraction=0.6)
        b1 = build_condition(t, cond, **kw)
        b2 = build_condition(t, cond, **kw)
        ev1 = assemble_recall(b1, t.question, encoder=ENC, k=5)
        ev2 = assemble_recall(b2, t.question, encoder=ENC, k=5)
        assert ev1 == ev2, f"{cond}: recall not reproducible across builds"
        assert fixed_recall_prompt(t.question, ev1).encode("utf-8") == \
               fixed_recall_prompt(t.question, ev2).encode("utf-8"), \
               f"{cond}: prompt not byte-identical"


# -- (4) scorer integration through run_one ---------------------------------


def test_run_one_scores_correct_answer():
    t = _task()
    r = run_one(t, "full_context", "fixed_recall",
                MockLLM([f"FINAL ANSWER: {t.answer}"]),
                encoder=ENC, nli=NLI, epoch=EPOCH)
    assert r["em"] == 1.0 and r["f1"] == 1.0
    assert r["condition"] == "full_context" and r["question_id"] == t.task_id


def test_run_one_scores_wrong_answer_zero():
    t = _task()
    r = run_one(t, "none", "fixed_recall", MockLLM(["FINAL ANSWER: wrongplace"]),
                encoder=ENC, nli=NLI, epoch=EPOCH)
    assert r["em"] == 0.0


def test_run_one_records_recall_at_k_and_footprint():
    # the row must carry the retrieval-quality + footprint observables (the
    # systems-relevant primary metrics), not just answer F1, so the re-run records
    # them per (condition, task).
    t = _task()
    r = run_one(t, "distributed_merge", "fixed_recall",
                MockLLM([f"FINAL ANSWER: {t.answer}"]),
                encoder=ENC, nli=NLI, epoch=EPOCH, overlap_fraction=0.6, k=5)
    for key in ("recall_at_k", "mrr", "n_gold", "stored_units", "live_units"):
        assert key in r, key
    assert 0.0 <= r["recall_at_k"] <= 1.0
    assert 0.0 <= r["mrr"] <= 1.0
    assert r["live_units"] <= r["stored_units"]


def test_run_one_persists_evidence_and_prompt():
    # the durable row must carry the assembled recall evidence + the full answer
    # prompt, so a future model-add is a true zero-build REPLAY and the context is
    # auditable. evidence is the list of recalled context strings; prompt is the
    # exact string fed to llm.complete().
    t = _task()
    r = run_one(t, "distributed_merge", "fixed_recall",
                MockLLM([f"FINAL ANSWER: {t.answer}"]),
                encoder=ENC, nli=NLI, epoch=EPOCH, overlap_fraction=0.6, k=5)
    assert "evidence" in r and "prompt" in r
    assert isinstance(r["evidence"], list)
    assert all(isinstance(e, str) for e in r["evidence"])
    assert isinstance(r["prompt"], str) and r["prompt"]
    # the persisted evidence is exactly what assemble_recall produced for this build
    b = build_condition(t, "distributed_merge", encoder=ENC, nli=NLI, epoch=EPOCH,
                        overlap_fraction=0.6)
    assert r["evidence"] == assemble_recall(b, t.question, encoder=ENC, k=5)
    # and the persisted prompt is exactly the fixed_recall answer prompt over it
    assert r["prompt"] == fixed_recall_prompt(t.question, r["evidence"])


def test_run_one_persists_prompt_for_agentic_mode():
    # agentic mode has no single deterministic answer prompt; the persisted prompt
    # is the assembled-evidence context prompt (same assemble_recall), so the keys
    # are always present and summarize() never KeyErrors on mode.
    t = _task()
    r = run_one(t, "centralized", "agentic",
                MockLLM([f"FINAL ANSWER: {t.answer}"]),
                encoder=ENC, nli=NLI, epoch=EPOCH)
    assert "evidence" in r and "prompt" in r
    assert isinstance(r["evidence"], list) and isinstance(r["prompt"], str)


def test_summarize_does_not_break_on_evidence_prompt_keys():
    # the new durable keys must not perturb aggregation (no KeyError, deltas intact).
    t = _task()
    rows = [run_one(t, c, "fixed_recall", MockLLM([f"FINAL ANSWER: {t.answer}"]),
                    encoder=ENC, nli=NLI, epoch=EPOCH, overlap_fraction=0.6)
            for c in MEMORY_CONDITIONS]
    for r in rows:
        assert "evidence" in r and "prompt" in r
    summary = summarize(rows, modes=["fixed_recall"], n_boot=100)
    fr = summary["per_mode"]["fixed_recall"]
    assert len(fr["headline_f1"]) == 3
    assert "means_recall" in fr


def test_run_one_omits_recall_for_no_store_conditions():
    # recall@k is a STORE-retrieval metric; none/full_context have no store
    # (full_context inlines the gold), so the keys are omitted (N/A) rather than a
    # misleading 0.0 -- the recall@k floor would otherwise read as no-headroom.
    t = _task()
    none_row = run_one(t, "none", "fixed_recall", MockLLM(["FINAL ANSWER: x"]),
                       encoder=ENC, nli=NLI, epoch=EPOCH)
    full_row = run_one(t, "full_context", "fixed_recall",
                       MockLLM([f"FINAL ANSWER: {t.answer}"]), encoder=ENC, nli=NLI, epoch=EPOCH)
    cen_row = run_one(t, "centralized", "fixed_recall",
                      MockLLM([f"FINAL ANSWER: {t.answer}"]), encoder=ENC, nli=NLI, epoch=EPOCH)
    assert "recall_at_k" not in none_row and "recall_at_k" not in full_row
    assert "recall_at_k" in cen_row


def test_summarize_includes_recall_deltas_and_footprint():
    # the summary must surface recall@k (the PRIMARY observable) condition means +
    # headline deltas, and the per-condition live/stored footprint.
    t = _task()
    rows = [run_one(t, c, "fixed_recall", MockLLM([f"FINAL ANSWER: {t.answer}"]),
                    encoder=ENC, nli=NLI, epoch=EPOCH, overlap_fraction=0.6)
            for c in ("none", "full_context", "centralized",
                      "distributed_merge", "naive_union")]
    m = summarize(rows)["per_mode"]["fixed_recall"]
    assert "means_recall" in m and "headline_recall" in m and "footprint" in m
    assert len(m["headline_recall"]) == 3        # the same three contrasts, on recall@k
    fp = m["footprint"]
    assert fp["distributed_merge"]["live"] <= fp["naive_union"]["live"]


# -- (5) paired bootstrap delta CI ------------------------------------------


def test_bootstrap_delta_paired_sign_and_ci():
    # construct paired rows: A beats B on every question -> delta strictly > 0,
    # and the CI must not straddle a negative when the effect is unanimous.
    rows = []
    for i in range(8):
        rows.append({"question_id": f"q{i}", "condition": "A", "f1": 1.0,
                     "gold": "x"})
        rows.append({"question_id": f"q{i}", "condition": "B", "f1": 0.0,
                     "gold": "x"})
    ci = bootstrap_delta(rows, "A", "B", metric="f1", n_boot=500, seed=1)
    assert ci.point == 1.0
    assert ci.lo == 1.0 and ci.hi == 1.0  # unanimous -> degenerate tight CI
    assert ci.n == 8


def test_bootstrap_delta_is_paired_not_two_sample():
    # rows present in only ONE condition are excluded from the paired set.
    rows = [
        {"question_id": "q0", "condition": "A", "f1": 1.0, "gold": "x"},
        {"question_id": "q0", "condition": "B", "f1": 0.0, "gold": "x"},
        {"question_id": "q1", "condition": "A", "f1": 1.0, "gold": "x"},  # no B
    ]
    ci = bootstrap_delta(rows, "A", "B", metric="f1", n_boot=100, seed=0)
    assert ci.n == 1  # only q0 is paired


def test_bootstrap_delta_absent_condition_is_na_not_zero():
    # a delta whose B condition was never run yields n==0 -> the runner prints
    # N/A, NOT a confident +0.000 (an absent condition must not masquerade
    # as a clean zero with a zero-width CI).
    rows = [{"question_id": "q0", "condition": "A", "f1": 1.0, "gold": "x"}]
    ci = bootstrap_delta(rows, "A", "naive_union", metric="f1", n_boot=100, seed=0)
    assert ci.n == 0
    assert ci.point == 0.0 and ci.lo == 0.0 and ci.hi == 0.0  # the n==0 sentinel


def test_tost_declares_equivalence_when_within_margin():
    # identical paired scores -> delta 0 -> equivalent at any positive margin
    from meld.agent.mem_delta_metrics import tost_equivalent
    rows = []
    for i in range(20):
        rows.append({"question_id": f"q{i}", "condition": "A", "f1": 0.5, "gold": "x"})
        rows.append({"question_id": f"q{i}", "condition": "B", "f1": 0.5, "gold": "x"})
    res = tost_equivalent(rows, "A", "B", margin=0.05, metric="f1", n_boot=500)
    assert res["equivalent"] is True


def test_tost_rejects_equivalence_when_gap_exceeds_margin():
    from meld.agent.mem_delta_metrics import tost_equivalent
    rows = []
    for i in range(20):
        rows.append({"question_id": f"q{i}", "condition": "A", "f1": 1.0, "gold": "x"})
        rows.append({"question_id": f"q{i}", "condition": "B", "f1": 0.0, "gold": "x"})
    res = tost_equivalent(rows, "A", "B", margin=0.05, metric="f1", n_boot=500)
    assert res["equivalent"] is False


def test_aggregation_skips_rows_missing_a_nonscore_metric():
    # rows from an older harness lack recall_at_k; aggregating recall_at_k must SKIP
    # them, not KeyError via score_one (which only recomputes em/f1). Caught on the
    # CSC validation smoke: resume mixed pre-fix rows lacking the new metric.
    from meld.agent.mem_delta_metrics import bootstrap_delta, condition_means
    rows = [
        {"question_id": "q1", "condition": "A", "recall_at_k": 0.8,
         "prediction": "x", "gold": "x"},
        {"question_id": "q2", "condition": "A",                       # OLD: no recall_at_k
         "prediction": "x", "gold": "x"},
        {"question_id": "q1", "condition": "B", "recall_at_k": 0.4,
         "prediction": "y", "gold": "x"},
    ]
    cm = condition_means(rows, ["A", "B"], metric="recall_at_k")
    assert cm["A"]["n"] == 1 and abs(cm["A"]["mean"] - 0.8) < 1e-9   # q2 skipped
    ci = bootstrap_delta(rows, "A", "B", metric="recall_at_k", n_boot=50)
    assert ci.n == 1                                                 # only q1 is paired
    # em/f1 still recompute from (prediction, gold) for rows lacking the key
    cmf = condition_means(rows, ["A"], metric="f1")
    assert cmf["A"]["n"] == 2


def test_holm_correction_thresholds_family():
    from meld.agent.mem_delta_metrics import holm_correct
    res = holm_correct({"a": 0.001, "b": 0.4, "c": 0.8}, alpha=0.05)
    assert res["a"] is True and res["b"] is False and res["c"] is False


def test_headline_table_has_three_deltas():
    rows = []
    for i in range(4):
        for c, f in [("distributed_merge", 0.8), ("naive_union", 0.4),
                     ("centralized", 0.8), ("full_context", 0.9), ("none", 0.1)]:
            rows.append({"question_id": f"q{i}", "condition": c, "f1": f,
                         "gold": "x"})
    table = headline_table(rows, metric="f1", n_boot=200, seed=0)
    assert len(table) == 3
    labels = {d["b"] for d in table}
    assert {"naive_union", "centralized", "none"} == labels


# -- (6) summarize end-to-end (oracle LLM) ----------------------------------


def test_summarize_produces_floor_ceiling_and_merge_audit():
    tasks = load_hotpot_fixture(n=4)
    rows = []
    for mode in ["fixed_recall"]:
        for cond in MEMORY_CONDITIONS:
            for t in tasks:
                rows.append(run_one(t, cond, mode,
                                    MockLLM([f"FINAL ANSWER: {t.answer}"]),
                                    encoder=ENC, nli=NLI, epoch=EPOCH,
                                    overlap_fraction=0.6))
    summary = summarize(rows, modes=["fixed_recall"], n_boot=200)
    fr = summary["per_mode"]["fixed_recall"]
    # the merge audit must show distributed_merge fired overlap merges, union 0.
    assert fr["merge_audit"]["distributed_merge"]["overlap_merges"] >= 1
    assert fr["merge_audit"]["naive_union"]["overlap_merges"] == 0
    # the three headline deltas exist
    assert len(fr["headline_f1"]) == 3


def test_distributed_merge_delivers_in_canonical_order_so_the_global_build_never_refolds():
    # the sovereign deltas reach the global brain in canonical order (created_at, delta id):
    # every admission is canonical-order last, so the build is one setup plus the latency path
    b = build_condition(_task(), "distributed_merge", encoder=ENC, nli=NLI, epoch=EPOCH)
    assert b.brain.admission.counters == {"not_last": 0, "refolds": 1}
    assert b.n_total_patches == b.stored_units > 3
