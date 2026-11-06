"""N-brains plumbing: duplication degree in the HotpotQA split, pooled stores, the LongMemEval
session split with its knowledge-update ground truth, and the star / all-to-all driver."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from meld.agent.hotpot_loader import load_hotpot_fixture
from meld.agent.mem_conditions import (_gold_sentences, light_paraphrase, pool_tasks,
                                      split_across_brains)
from meld.eval import topology
from meld.eval.lme_supersession import ku_truth, lme_recall, session_split, stale_fact_rate
from test_agent_meld_memory import agent_epoch

ENC, NLI = "hashing", "lexical"
EPOCH = agent_epoch(ENC, NLI)


# -- HotpotQA split: duplication degree ---------------------------------------------------


def test_default_dup_degree_reproduces_legacy_split():
    """dup_degree=2 (the default) is byte-identical to the split before the parameter existed
    (sha256 over the fixture, N in {2, 3, 5}, both overlaps, conflict on and off). Re-pinned when
    the copy generator became surface-only (no lexical rules): the earlier pin 52b9e27b... is
    reproduced exactly by this split with the earlier ``light_paraphrase`` substituted, so the
    split itself is unchanged and only the copies' text moved. Re-pinned again when each brain's
    list was put in world-time order: the second pin, over each brain's sorted list, is the same
    before and after, so only the order within a brain moved."""
    h, members = hashlib.sha256(), hashlib.sha256()
    for t in load_hotpot_fixture():
        for n in (2, 3, 5):
            for ov in (0.5, 1.0):
                for c in (False, True):
                    plan = split_across_brains(t, n_brains=n, overlap_fraction=ov, inject_conflict=c)
                    h.update(json.dumps(plan.per_brain).encode())
                    members.update(json.dumps([sorted(b) for b in plan.per_brain]).encode())
    assert h.hexdigest() == "c9e4188e3a12131517317f42da5077b87f632b4456058edc727010306f598ab1"
    assert members.hexdigest() == "fbd5de1bd8f315b2ba0bb6648628d924b05a1be8aac02dd7e9f2873965b331ea"


@pytest.mark.parametrize("n,d", [(3, 1), (3, 3), (8, 4), (8, 8)])
def test_each_duplicated_gold_on_exactly_d_brains(n, d):
    t = load_hotpot_fixture(n=1)[0]
    plan = split_across_brains(t, n_brains=n, overlap_fraction=1.0, dup_degree=d)
    for title, sent in _gold_sentences(t):
        holders = [b for b, obs in enumerate(plan.per_brain)
                   if any(o.startswith(f"{title}: ") and (o == f"{title}: {sent}" or
                          o in {f"{title}: {light_paraphrase(sent, variant=v)}" for v in range(d)})
                          for o in obs)]
        assert len(holders) == d
    assert plan.n_duplicated == (plan.n_gold if d > 1 else 0)


def test_dup_degree_above_n_brains_raises():
    with pytest.raises(ValueError):
        split_across_brains(load_hotpot_fixture(n=1)[0], n_brains=3, dup_degree=4)


# adversarial inputs for the copy generator: a contraction, a country and a year after "born in",
# "is a", a question, an exclamation, several sentences, an abbreviation, a closing quote, no
# terminal punctuation, a multi-line turn
ADVERSARIAL = ["It's a film by Christopher Nolan.", "He was born in France in 1990.",
               "Paris is the capital of France.", "Inception was directed by Christopher Nolan.",
               "Who directed Inception?", "What a great film!", "It rained. He left. It is a river.",
               "She moved to the U.S.", "They met Smith et al.", 'He said "hello."',
               "no terminal punctuation here", "I like:\n- tea\n- coffee", "Please send me the list."]


def _source_intact(src: str, out: str) -> bool:
    """The rewrite only appends a connective to the last declarative sentence or prepends a label:
    the source text is recoverable verbatim (up to its final period)."""
    return out.endswith(src) or (src.endswith(".") and out.startswith(src[:-1]) and out.endswith("."))


@pytest.mark.parametrize("s", ADVERSARIAL)
def test_paraphrase_preserves_the_source_text(s):
    for v in range(9):
        p = light_paraphrase(s, variant=v)
        assert _source_intact(s, p), (v, p)
        for bad in ("This work", "This person", "the city of", "was a ", "made by director"):
            assert bad not in p or bad in s, (v, p)


@pytest.mark.parametrize("s", ["Who directed Inception?", "What a great film!", 'He said "hello."',
                               "She moved to the U.S.", "They met Smith et al.", "no terminal punctuation here"])
def test_no_connective_suffix_where_the_text_does_not_end_a_plain_declarative(s):
    for v in range(9):
        p = light_paraphrase(s, variant=v)
        assert p.endswith(s) and p != s, (v, p)      # a leading label, the text untouched


def test_declarative_gets_a_connective_on_its_last_sentence_only():
    p = light_paraphrase("It rained. He left.", variant=2)
    assert p.startswith("It rained. He left, ") and p.endswith(".")


@pytest.mark.parametrize("s", ADVERSARIAL)
def test_paraphrase_variants_are_pairwise_distinct_and_dense_similar(s):
    from meld.merge.signals import _make_encoder
    enc = _make_encoder("hashing")
    vs = [light_paraphrase(s, variant=v) for v in range(9)]       # duplication degree up to 10
    assert len(set(vs)) == 9 and s not in vs
    assert len({" ".join(v.lower().split()) for v in vs}) == 9     # distinct under the queue's folding
    e = enc.encode_one(s)
    for v in vs:
        assert float(e @ enc.encode_one(v)) > 0.6, v
    assert light_paraphrase(s) == light_paraphrase(s, variant=0)


def test_pooled_store_contains_all_tasks_sentences():
    ts = load_hotpot_fixture()
    pools = pool_tasks(ts, per_store=3)
    assert [len(p) for _, p in pools] == [3, 1]
    for pooled, members in pools:
        flat = {(ti, s) for ti, ss in pooled.context for s in ss}
        for t in members:
            assert {(ti, s) for ti, ss in t.context for s in ss} <= flat


# -- LongMemEval session split ------------------------------------------------------------


def _turn(c, ans=False, role="user"):
    return {"role": role, "content": c, **({"has_answer": True} if ans else {})}


def _item(qtype="knowledge-update"):
    """Five sessions listed out of date order; the old value (s_old) is listed AFTER the new."""
    return {
        "question_id": "q1", "question_type": qtype, "question": "What is my 5K personal best time?",
        "answer": "25 minutes",
        "haystack_session_ids": ["s_new", "s_a", "s_old", "s_b", "s_c"],
        "haystack_dates": ["2023/05/02 (Tue) 12:00", "2023/05/01 (Mon) 09:00",
                           "2023/05/02 (Tue) 09:00", "2023/05/03 (Wed) 09:00", "2023/05/10 (Wed) 09:00"],
        "haystack_sessions": [
            [_turn("My 5K personal best time is now 25 minutes.", ans=True), _turn("Nice!", role="assistant")],
            [_turn("I bought new running shoes today.")],
            [_turn("My 5K personal best time is 27 minutes.", ans=True)],
            [_turn("I want to bake bread this weekend."), _turn("I like sourdough most.")],
            [_turn("My sister moved to Oslo.")],
        ],
    }


def test_session_split_assigns_every_session_once():
    plan = session_split(_item(), n_brains=3)
    placed = [t for b in plan.per_brain for t in b]
    user = [t["content"] for s in _item()["haystack_sessions"] for t in s if t["role"] == "user"]
    assert sorted(placed) == sorted(user)          # d=1: every user turn on exactly one brain
    owner = {t: b for b, obs in enumerate(plan.per_brain) for t in obs}
    # sessions are round-robin over date order: s_a, s_old, s_new, s_b, s_c -> 0, 1, 2, 0, 1
    assert owner["I bought new running shoes today."] == 0
    assert owner["My 5K personal best time is 27 minutes."] == 1
    assert owner["My 5K personal best time is now 25 minutes."] == 2
    assert owner["I like sourdough most."] == owner["I want to bake bread this weekend."] == 0
    assert owner["My sister moved to Oslo."] == 1
    assert plan.n_gold == 2


def test_session_split_time_order_within_each_brain():
    plan = session_split(_item(), n_brains=2, dup_degree=2)
    for obs in plan.per_brain:
        assert [plan.when[t] for t in obs] == sorted(plan.when[t] for t in obs)
    old, new = ku_truth(_item())
    assert plan.when[old] < plan.when[new]


def test_split_places_old_and_new_value_on_different_brains_when_n_ge_2():
    old, new = ku_truth(_item())
    for n in (2, 3, 8):
        plan = session_split(_item(), n_brains=n)
        owner = {t: b for b, obs in enumerate(plan.per_brain) for t in obs}
        assert owner[old] != owner[new]


@pytest.mark.parametrize("n,d", [(3, 2), (8, 4), (8, 8)])
def test_session_split_dup_degree(n, d):
    plan = session_split(_item(), n_brains=n, dup_degree=d)
    copies = {}
    for b, obs in enumerate(plan.per_brain):
        for t in obs:
            copies.setdefault(plan.source[t], set()).add(b)
    assert all(len(bs) == d for bs in copies.values())
    placed = [t for obs in plan.per_brain for t in obs]
    assert len(placed) == len(set(placed))         # copies are paraphrase variants, never exact
    assert plan.n_duplicated == 2 and plan.n_paraphrased == 2


def test_session_split_is_deterministic_and_seeded():
    a = session_split(_item(), n_brains=3, assign="random", seed=7)
    b = session_split(_item(), n_brains=3, assign="random", seed=7)
    assert a.per_brain == b.per_brain
    assert session_split(_item(), n_brains=3).per_brain == session_split(_item(), n_brains=3).per_brain
    with pytest.raises(ValueError):
        session_split(_item(), n_brains=3, dup_degree=4)


def test_ku_truth_orders_by_date_and_is_none_otherwise():
    assert ku_truth(_item()) == ("My 5K personal best time is 27 minutes.",
                                 "My 5K personal best time is now 25 minutes.")
    assert ku_truth(_item("multi-session")) is None
    one = _item()
    one["haystack_sessions"][2][0].pop("has_answer")
    assert ku_truth(one) is None


def test_stale_fact_rate_on_toy_store():
    item = _item()
    old, new = ku_truth(item)
    # a stand-in brain has no Store (no replicated q), so it is read plainly (tau None): this tests
    # the source mapping and the stale logic, not the near-duplicate read (test_read_near_dup.py)
    brain = lambda texts: SimpleNamespace(recall=lambda q, k: texts[:k])      # noqa: E731
    assert stale_fact_rate(brain([old, "x"]), item, 5, tau=None) == 1.0
    assert stale_fact_rate(brain([old, new]), item, 5, tau=None) == 0.0
    assert stale_fact_rate(brain([new]), item, 5, tau=None) == 0.0
    assert stale_fact_rate(brain(["x", new, old]), item, 1, tau=None) == 0.0
    assert stale_fact_rate(brain([]), item, 5, tau=None) == 0.0
    assert stale_fact_rate(brain([old]), _item("multi-session"), 5, tau=None) is None
    para = light_paraphrase(old, variant=1)          # a copy of the old turn counts as the old turn
    assert stale_fact_rate(brain([para]), item, 5, source={para: old}, tau=None) == 1.0
    assert lme_recall(brain([para, "x"]), item, 5, source={para: old}, tau=None) == 0.5


# -- driver -------------------------------------------------------------------------------


def _world(n, d, cond, top, **kw):
    plan = session_split(_item(), n_brains=1 if cond == "centralized" else n,
                         dup_degree=1 if cond == "centralized" else d)
    return plan, topology.build(plan, cond, top, encoder=ENC, nli=NLI, epoch=EPOCH, **kw)


@pytest.mark.parametrize("n", [3, 8])
def test_driver_smoke_all_conditions_and_topologies(n):
    for cond in topology.CONDITIONS:
        for top in topology.TOPOLOGIES:
            plan, b = _world(n, 2, cond, top)
            assert b["live"] >= 1 and b["stored"] >= b["live"]
            assert lme_recall(b["memory"], _item(), 5, source=plan.source) is not None


def test_conditions_differ_where_they_should():
    _, union = _world(3, 3, "naive_union", "star")
    _, dedup = _world(3, 3, "exact_dedup", "star")
    _, meld = _world(3, 3, "meld", "star")
    _, central = _world(3, 3, "centralized", "star")
    assert union["decisions"] == {} and union["live"] == union["stored"]
    assert dedup["decisions"].get("MERGE", 0) == 0 and dedup["live"] == dedup["stored"]
    assert meld["decisions"].get("MERGE", 0) >= 1 and meld["live"] < union["live"]
    assert union["stored"] == dedup["stored"] == meld["stored"] > central["stored"]


def test_star_and_all_to_all_converge_to_the_same_live_set():
    _, star = _world(3, 2, "meld", "star")
    _, a2a = _world(3, 2, "meld", "all_to_all")
    live = lambda m: sorted(m.store.nodes()[k][1].text for k in m.store.live_nodes())  # noqa: E731
    assert live(star["memory"]) == live(a2a["memory"])
    assert len(set(a2a["convergence"]["fingerprints"])) == 1


def test_every_replica_gets_its_own_provider(monkeypatch):
    made = []
    real = topology.nli_provider

    def spy(model, *, own=False):
        made.append(own)
        return real(model, own=own)
    monkeypatch.setattr(topology, "nli_provider", spy)
    _world(3, 2, "meld", "star")
    assert made == [True] * 4                       # three agents and the sink
    made.clear()
    _world(3, 2, "meld", "all_to_all")
    assert made == [True] * 3


def test_baseline_slot_and_unknown_condition():
    plan = session_split(_item(), n_brains=3)
    seen = []

    def policy(p, kw):
        seen.append(p)
        return topology.build(p, "naive_union", "star", **kw)["memory"]
    b = topology.build(plan, "toy", "star", encoder=ENC, nli=NLI, epoch=EPOCH, baselines={"toy": policy})
    assert seen == [plan] and b["live"] >= 1
    with pytest.raises(ValueError):
        topology.build(plan, "bogus", "star", encoder=ENC, nli=NLI)


def test_grid_and_stamps():
    assert topology.grid((3, 8, 16, 32), (1, 2, 4, 8)) == [
        (3, 1), (3, 2), (8, 1), (8, 2), (8, 4), (8, 8), (16, 1), (16, 2), (16, 4), (16, 8),
        (32, 1), (32, 2), (32, 4), (32, 8)]
    s = topology.stamps(ENC, NLI, EPOCH)
    assert set(s) == {"git_sha", "m_nli", "m_pol", "epoch_id", "seq", "nli_id", "encoder"}
    assert s["seq"] == EPOCH.seq and topology.stamps(ENC, NLI, None)["seq"] is None
    assert topology.stamps(ENC, None, None, pol=None) | {"git_sha": 0} == {
        "git_sha": 0, "m_nli": None, "m_pol": None, "epoch_id": None, "seq": None, "nli_id": None,
        "encoder": ENC}
    assert topology.bin_for(0, None) is None and topology.bin_for(EPOCH.bins[0], EPOCH) == 1
    assert len(s["m_nli"]) == 32 and len(s["m_pol"]) == 32 and len(s["epoch_id"]) == 32
    assert topology.stamps(ENC, NLI, None)["epoch_id"] is None
    assert topology.stamps(ENC, "mnli-cross-encoder", None)["nli_id"] == "cross-encoder/nli-deberta-v3-small"


def test_runner_smoke_rows_stamps_and_resume(tmp_path):
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "run_nbrains", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_nbrains.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    data = tmp_path / "lme.json"
    data.write_text(json.dumps([_item(), _item("multi-session") | {"question_id": "q2"}]))
    argv = ["--data", str(data), "--encoder", ENC, "--nli", NLI, "--ns", "3", "--ds", "1", "2",
            "--topologies", "star", "all_to_all", "--ks", "5", "--dev-theta", "0.5",
            "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "o.jsonl")]
    rows = list(run.main(argv).values())
    # per world: centralized once; per (N, d): union on star, meld and dedup on both topologies
    assert len(rows) == 2 * (1 + 2 * (1 + 2 * 2))
    assert all(r["git_sha"] and r["m_nli"] and r["epoch_id"] and r["nli_id"] == NLI and r["seq"] == 1
               for r in rows)
    # decisions per epoch bin add up to the decision counts (the dev epoch has one bin row)
    for r in rows:
        assert sum(n for by in r["decisions_by_bin"].values() for n in by.values()) == sum(r["decisions"].values())
    assert any(r["decisions_by_bin"] for r in rows if r["condition"] == "meld")
    ku = [r for r in rows if r["world"] == "q1"]
    assert all(r["stale@5"] in (0.0, 1.0) for r in ku)
    assert all(r["stale@5"] is None for r in rows if r["world"] == "q2")
    n_lines = len((tmp_path / "o.jsonl").read_text().splitlines())
    run.main(argv)                                   # resume: every cell is done, nothing appended
    assert len((tmp_path / "o.jsonl").read_text().splitlines()) == n_lines


# -- HotpotQA world time: every condition folds the same canonical order -------------------


def _hotpot(n, d, cond, epoch):
    pooled, members = pool_tasks(load_hotpot_fixture(), 4)[0]
    plan = split_across_brains(pooled, n_brains=n, dup_degree=d, overlap_fraction=0.5)
    return plan, members, topology.build(plan, cond, "star", encoder=ENC, nli=NLI, epoch=epoch)


def _live(m):
    return sorted(m.store.nodes()[u][1].text for u in m.store.live_nodes())


def _recalls(m, members, source):
    from meld.agent.recall_metrics import recall_at_k_and_mrr
    return [recall_at_k_and_mrr(m, t, k=k, source=source) for t in members for k in (1, 5, 10)]


def test_hotpot_world_time_is_the_position_in_the_pooled_context():
    pooled, _ = pool_tasks(load_hotpot_fixture(), 4)[0]
    flat = [f"{title}: {s}".strip() for title, sents in pooled.context for s in sents]
    plan = split_across_brains(pooled, n_brains=3, dup_degree=3, overlap_fraction=1.0)
    for text in (x for b in plan.per_brain for x in b):
        assert plan.when[text] == flat.index(plan.source.get(text, text))
    assert all(b == sorted(b, key=plan.when.get) for b in plan.per_brain)
    assert split_across_brains(pooled, n_brains=1, dup_degree=1).per_brain[0] == sorted(flat, key=flat.index)


@pytest.mark.parametrize("cond,epoch", [("exact_dedup", None), ("meld", EPOCH)])
def test_hotpot_order_term_is_zero(cond, epoch):
    """At d = 1 the sink holds the centralized store's claims; with one canonical order the two
    give the same store and the same recall (exact_dedup at d = 1 minus centralized is 0)."""
    _, members, sink = _hotpot(3, 1, cond, epoch)
    _, _, cen = _hotpot(1, 1, "centralized", epoch)
    assert _live(sink["memory"]) == _live(cen["memory"])
    assert _recalls(sink["memory"], members, {}) == _recalls(cen["memory"], members, {})


def test_centralized_fed_the_sink_stream_equals_the_sink():
    from meld.agent.mem_conditions import SplitPlan
    from meld.agent.memory import DEFAULT_SCOPE
    plan, members, sink = _hotpot(3, 2, "meld", EPOCH)
    objs = sink["memory"].store.objects
    stream = [objs.get("D", d).text for _, d in
              sorted((objs.get("D", d).created_at, d) for d in objs.ids("D", DEFAULT_SCOPE))]
    one = SplitPlan(per_brain=[stream], source=plan.source, when=plan.when)
    cen = topology.build(one, "centralized", "star", encoder=ENC, nli=NLI, epoch=EPOCH)
    assert _live(sink["memory"]) == _live(cen["memory"])
    assert _recalls(sink["memory"], members, plan.source) == _recalls(cen["memory"], members, plan.source)


@pytest.mark.parametrize("epoch", [None, EPOCH])
def test_condition_harness_centralized_equals_the_merge_without_copies(epoch):
    from meld.agent.mem_conditions import build_condition
    pooled, members = pool_tasks(load_hotpot_fixture(), 4)[0]
    kw = dict(encoder=ENC, nli=NLI, epoch=epoch, overlap_fraction=0.0)
    cen = build_condition(pooled, "centralized", **kw)
    dm = build_condition(pooled, "distributed_merge", **kw)
    assert _live(dm.brain) == _live(cen.brain)
    assert _recalls(dm.brain, members, {}) == _recalls(cen.brain, members, {})
