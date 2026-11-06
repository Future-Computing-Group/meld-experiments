"""The multi-replica harnesses over the fold (design §7.1 harnesses, §7.2 H1-H3, §7.4 freeze gate;
Props 1 and 3): replicas are BrainNodes, each with its own signal provider (model instance); a
replica's state is compared on five components (authored sets, Sig, derived representatives and
vis, links, decisions); ``settle`` measures the pre-pass divergence (the window), then runs a Sig
exchange and a pass (and one more of each if a pass still missed, L <= 1) and measures the
post-pass divergence and the PASS certificate fingerprints. Hashing encoder, lexical or scripted
NLI: offline."""
from __future__ import annotations

import pytest

from meld.agent.memory import DEFAULT_CELL, DEFAULT_POL
from meld.eval import graph_fingerprint as gf
from meld.protocol.objects import FIXED_ONE, AuthoredEffect
from meld.sync.node import BrainNode

from test_agent_meld_memory import agent_epoch

F = FIXED_ONE
GATED = ("authored", "reps", "links", "decisions")     # Sig ids differ by record batching
ENTAIL = (0, F, 0)
NEUTRAL = (0, 0, F)
TEXTS = ["the valve on line two is open", "the valve on line two is open now",
         "alpha bravo charlie", "alpha bravo charlie delta", "golf hotel india"]


def _deltas(texts=TEXTS):
    from meld.eval.order_shuffle import claims
    return claims("hashing", DEFAULT_POL, [(t, DEFAULT_CELL) for t in texts])


def _node(i, provider, epoch=True, **kw):
    return BrainNode(f"r{i}", encoder="hashing", nli="lexical",
                     epoch=agent_epoch() if epoch else None, provider=provider, **kw)


def _scripted(payload):
    return lambda a, b: payload


def _deliver(node, objs):
    from meld.protocol.objects import signed
    for o in objs:
        assert node.receive_object(*signed(o, node.key), None)


# --------------------------------------------------------------------------- components, counters
def test_divergence_counts_replicas_off_the_modal_value_per_component():
    views = [dict.fromkeys(gf.COMPONENTS, b"a") for _ in range(5)]
    views[3]["links"] = views[4]["links"] = b"b"            # 3 agree, 2 differ: 2 off the mode
    views[0]["sig"] = b"x"
    views[1]["sig"] = b"y"                                  # 3 distinct, mode "a" held by 3
    assert gf.divergence(views) == {"authored": 0, "sig": 2, "reps": 0, "links": 2, "decisions": 0}
    assert gf.divergence(views[:1]) == dict.fromkeys(gf.COMPONENTS, 0)


def test_components_see_links_reps_and_decisions_separately():
    n = _node(0, _scripted(NEUTRAL))
    _deliver(n, _deltas())
    store, decs = n.mem.store, n.mem.admission.decisions("agent")
    assert store.derived.links                                  # RELATEs left ASSOCIATES links
    base = gf.components(store, "agent", decs)
    t, s, d, w = store.derived.links[0]
    store.derived.links.append((t, s, d, w / 2))                # one more link, nothing else
    assert {c for c in gf.COMPONENTS if gf.components(store, "agent", decs)[c] != base[c]} == {"links"}
    store.derived.links.pop()
    ends = {x for _, a, b, _ in store.derived.links for x in (a, b)}
    k = next(x for x in store.nodes() if x not in ends)      # rep change, links unaffected
    store.derived.rep[k] = next(x for x in store.nodes() if x != k)
    assert {c for c in gf.COMPONENTS if gf.components(store, "agent", decs)[c] != base[c]} == {"reps"}
    del store.derived.rep[k]
    assert {c for c in gf.COMPONENTS
            if gf.components(store, "agent", decs[:-1])[c] != base[c]} == {"decisions"}
    assert gf.components(store, "agent", decs) == base
    # a held Cal or Ep record flips the authored component only
    from dataclasses import replace
    from meld.protocol.objects import CalRecord, signed
    k1, k2 = sorted(list(store.nodes())[:2])
    cal = CalRecord(scope="agent", c_x=k1, c_y=k2, bin=0, s_q=100, hw="cpu", obs=1, label="NON_MERGEABLE",
                    subtype="OTHER", source="SEED", item="i1", author="calibration", created_at=0,
                    sigma_lo_q=0, m=3, m_pol=b"\0" * 16, n_prior=0)
    for obj in (cal, replace(agent_epoch(theta=0.9), seq=2)):
        before = gf.components(store, "agent", decs)
        assert n.receive_object(*signed(obj, n.key), None)
        after = gf.components(store, "agent", decs)
        assert {c for c in gf.COMPONENTS if after[c] != before[c]} == {"authored"}


def test_components_are_restricted_to_the_scope():
    # a replica lacking content of another scope agrees on this scope's components and pass
    from meld.eval.order_shuffle import EU, US, claims
    from meld.protocol.fold import nli_provider
    from meld.sync.experiment import settle
    pol = {**DEFAULT_POL, "scopes": {"eu": [EU], "us": [US]}}
    ds = claims("hashing", pol, [("alpha bravo", EU), ("alpha bravo charlie", EU), ("delta echo", US),
                                 ("delta echo foxtrot", US)])
    nodes = [BrainNode(f"r{i}", encoder="hashing", nli="lexical", pol=pol, provider=nli_provider("lexical"))
             for i in range(2)]
    _deliver(nodes[0], ds)
    _deliver(nodes[1], ds[:3])                                  # lacks one US delta
    views = [gf.components(n.mem.store, "eu", n.mem.admission.decisions("eu")) for n in nodes]
    assert {c for c in gf.COMPONENTS if views[0][c] != views[1][c]} <= {"sig"}   # issuers differ
    assert gf.components(nodes[0].mem.store, "us", []) != gf.components(nodes[1].mem.store, "us", [])
    assert settle(nodes, "eu")["post_pass"] == dict.fromkeys(gf.COMPONENTS, 0)


# --------------------------------------------------------------------------- settle
def _hetero(n=3):
    """Three replicas on two 'hardware classes' (one reads every pair as entailment, one as
    neutral), each receiving the workload in another order (r1 in reverse canonical order)."""
    ds = _deltas()
    nodes = [_node(i, _scripted(ENTAIL if i % 2 == 0 else NEUTRAL)) for i in range(n)]
    for i, node in enumerate(nodes):
        _deliver(node, ds[::-1] if i == 1 else ds[i:] + ds[:i])
    return nodes


def test_settle_measures_the_window_then_agrees_after_sig_exchange_and_pass():
    from meld.sync.experiment import settle
    nodes = _hetero()
    rep = settle(nodes, "agent")
    # before any pass the replicas disagree: r0 merged the valve paraphrase, r1 related it
    assert rep["pre_pass"]["decisions"] >= 1 and rep["pre_pass"]["reps"] >= 1
    assert rep["pre_pass"]["sig"] >= 1                      # each holds only its own records
    assert {"decisions", "reps"} <= set(rep["window"][0])
    assert all(set(w) <= {"reps", "links", "decisions"} for w in rep["window"])   # Sig apart
    assert rep["not_last"][1] == len(TEXTS) - 1            # every arrival but the first was late
    # after the Sig exchange and the pass: zero divergence on all five counters
    assert rep["post_pass"] == dict.fromkeys(gf.COMPONENTS, 0)
    assert len(set(rep["fingerprints"])) == 1
    assert rep["pass_misses"][-1] == [0, 0, 0] and rep["L"] == 0
    # the joined evidence (entailment met neutral) merges nothing: Lemma 7b's direction
    assert all("MERGE" not in d for d in rep["decisions"])


def test_settle_on_empty_replicas_without_epoch_is_trivially_converged():
    from meld.sync.experiment import settle
    nodes = [_node(i, _scripted(NEUTRAL), epoch=False) for i in range(2)]
    rep = settle(nodes, "agent")
    assert rep["pre_pass"] == rep["post_pass"] == dict.fromkeys(gf.COMPONENTS, 0)


def test_settle_refuses_replicas_that_share_a_model_instance(monkeypatch):
    from meld.merge import signals
    from meld.protocol.fold import nli_provider
    from meld.sync.experiment import settle
    shared = _scripted(NEUTRAL)
    with pytest.raises(ValueError, match="model instance"):
        settle([_node(0, shared), _node(1, shared)], "agent")
    # a cross-encoder replica on the module-level model cache has no model of its own
    ce = [BrainNode(f"c{i}", encoder="hashing", nli="mnli-cross-encoder") for i in range(2)]
    with pytest.raises(ValueError, match="model instance"):
        settle(ce, "agent")
    made = []

    class Stub:
        def __init__(self, model_id):
            made.append(self)

        def probs(self, a, b):
            return (0.0, 0.0, 1.0)
    monkeypatch.setattr(signals, "_CrossEncoderNLI", Stub)
    with pytest.raises(ValueError, match="unknown NLI model"):
        nli_provider("no-such-model", own=True)
    own = [nli_provider("mnli-cross-encoder", own=True) for _ in range(2)]
    assert len(made) == 2 and own[0].model is not own[1].model
    assert own[0]("a", "b") == (0, 0, F)
    ok = [BrainNode(f"c{i}", encoder="hashing", nli="mnli-cross-encoder", provider=p)
          for i, p in enumerate(own)]
    assert settle(ok, "agent")["post_pass"] == dict.fromkeys(gf.COMPONENTS, 0)


def test_node_factory_gives_each_replica_its_own_model(monkeypatch):
    from meld.merge import signals
    from meld.sync.experiment import node_factory

    class Stub:
        def __init__(self, model_id):
            pass
    monkeypatch.setattr(signals, "_CrossEncoderNLI", Stub)
    make = node_factory(encoder="hashing", nli="mnli-cross-encoder")
    a, b = make(0), make(1)
    assert a.node_id == "r0" and b.node_id == "r1"
    assert a.mem.admission.provider.model is not b.mem.admission.provider.model


def test_a_late_authored_effect_is_repaired_by_a_second_exchange_and_pass():
    # A DISTINGUISHES on a merged pair, held by every replica but read by no admission, changes
    # a_e at the pass: the first pass misses on the new tested pair, evaluates it on each
    # replica's own provider; the second exchange and pass agree (Prop 3, L = 1).
    from meld.protocol.objects import signed
    from meld.sync.experiment import settle
    ds = _deltas(["alpha bravo charlie", "alpha bravo charlie delta", "alpha bravo charlie delta echo"])
    nodes = [_node(i, _scripted(ENTAIL if i == 0 else (0, F - 7, 7))) for i in range(2)]
    for node in nodes:
        _deliver(node, ds)
    c1, c2 = sorted((ds[1].claim_key, ds[2].claim_key))
    dist = AuthoredEffect(scope="agent", kind="DISTINGUISHES", c1=c1, c2=c2, target_cell=DEFAULT_CELL,
                          author="auditor", author_cell=DEFAULT_CELL, author_auth=0, created_at=0)
    for node in nodes:
        node.receive_object(*signed(dist, node.key), None)
    rep = settle(nodes, "agent")
    assert rep["L"] == 1 and rep["pass_misses"][0] != [0, 0] and rep["pass_misses"][-1] == [0, 0]
    assert rep["post_pass"] == dict.fromkeys(gf.COMPONENTS, 0)


def test_settle_l_counts_passes_that_missed_so_a_prop3_violation_is_visible(monkeypatch):
    from meld.sync import experiment
    real = experiment.readjudicate

    def always_missing(*a, **k):
        res = real(*a, **k)
        res.counters["pass_time_misses"] = 1
        return res
    monkeypatch.setattr(experiment, "readjudicate", always_missing)
    rep = experiment.settle(_hetero(2), "agent")
    assert rep["L"] == 3 and len(rep["pass_misses"]) == 3


# --------------------------------------------------------------------------- order shuffle
def _rx_provider():
    from meld.eval.order_shuffle import C2_CONTRADICTION
    from meld.proto import canon
    from meld.protocol.fold import nli_provider
    lex = nli_provider("lexical")

    def p(a, b):     # the lexical rung cannot see 'retained seven years' vs 'deleted after two'
        return (F, 0, 0) if canon(C2_CONTRADICTION) in (a, b) else lex(a, b)
    return p


def test_order_shuffle_running_example_200_orders_one_post_pass_fingerprint():
    from meld.eval.order_shuffle import (
        C1_PARAPHRASE, C2_CONTRADICTION, EU, US, run_order_shuffle, running_example)
    from meld.proto import claim_key
    pol, seed_objs, msgs = running_example("hashing")
    sel = [round(0.3 * F), round(0.3 * F), 3]
    epoch = agent_epoch(theta=0.7, pol=pol, sel=sel, scope="jur")
    made = []

    def make_node(i):
        made.append(BrainNode("sink", encoder="hashing", nli="lexical", pol=pol, epoch=epoch,
                              provider=_rx_provider()))
        return made[-1]
    rep = run_order_shuffle(seed_objs, msgs, k=200, seed=2026, make_node=make_node, scope="jur",
                            probe_queries=["How long must customer data be retained?"])
    assert rep["k"] == 200 and len(made) == 200 and len(set(map(tuple, rep["orders"]))) > 20
    assert rep["distinct_fingerprints"] == 1 and rep["fingerprints"].count(rep["fingerprints"][0]) == 200
    assert all(rep["post_pass"][c] == 0 for c in GATED) and rep["final_misses"] == [0] * 200
    assert [v for v, _ in rep["decision_variants"]] == [["INSERT", "MERGE", "RELATE", "CONFLICT"]]
    assert "decisions" not in rep
    assert rep["topk_distinct"] == {"How long must customer data be retained?": 1}
    assert any(rep["not_last"]) and len(rep["window"]) == 200
    assert rep["orders_with_window"] == sum(bool(w) for w in rep["window"])
    s = made[0].mem.store
    assert s.vis(claim_key(C2_CONTRADICTION, EU)) == "OVERRULED"
    assert s.vis(claim_key(C1_PARAPHRASE, EU)) == "DEPRECATED"
    assert s.vis(claim_key(C1_PARAPHRASE, US)) == "ACTIVE"


def _jitter(seed):
    """A nondeterministic provider (run-to-run jitter around theta = 0.5)."""
    import random
    rng = random.Random(seed)

    def p(a, b):
        e = rng.choice((int(0.45 * F), int(0.55 * F)))
        return (0, e, F - e)
    return p


def test_order_shuffle_runs_orders_one_at_a_time_and_exposes_provider_jitter():
    # each order is folded alone (its own Sig, no exchange across orders that would join the
    # jitter away), and the harness drops each replica before building the next: one model live
    import gc
    import weakref
    from meld.eval.order_shuffle import gold_workload, run_order_shuffle
    deltas, roles = gold_workload("hashing")
    epoch, live = agent_epoch(theta=0.5), []

    def make_node(i):
        gc.collect()
        assert not [r for r in live if r() is not None]          # the previous order was dropped
        n = BrainNode("sink", encoder="hashing", nli="lexical", epoch=epoch, provider=_jitter(i))
        live.append(weakref.ref(n))
        return n
    rep = run_order_shuffle([], deltas, k=10, seed=7, scope="agent", make_node=make_node)
    assert rep["distinct_fingerprints"] == 10 and rep["post_pass"]["reps"] > 0


def test_gold_workload_has_26_seeds_four_roles_and_ground_truth():
    from meld.eval.order_shuffle import gold_workload, truth
    deltas, roles = gold_workload("hashing")
    assert len(deltas) == 104 == len(roles) and len({s for s, _ in roles.values()}) == 26
    assert {d.origin for d in deltas} == {"src0", "src1", "src2"}            # three source brains
    by = {v: k for k, v in roles.items()}
    t = truth(roles)
    assert t(by["geo-paris", "c"], by["geo-paris", "p"]) == "MERGEABLE"
    assert t(by["geo-paris", "x"], by["geo-paris", "p"]) == "CONTRADICTION"
    assert t(by["geo-paris", "c"], by["geo-paris", "r"]) == "NON_MERGEABLE"
    assert t(by["geo-paris", "r"], by["geo-paris", "x"]) == "NON_MERGEABLE"
    assert t(by["geo-paris", "c"], by["geo-everest", "c"]) == "NON_MERGEABLE"


def test_edge_recovery_counts_typed_edges_per_seed():
    from meld.brain.store import Derived
    from meld.eval.order_shuffle import edge_recovery, gold_workload
    deltas, roles = gold_workload("hashing")
    by = {v: k for k, v in roles.items()}

    class S:
        derived = Derived(rep={by["geo-paris", "p"]: by["geo-paris", "c"],
                               by["geo-everest", "c"]: by["geo-paris", "c"]},
                          links=[("CONTRADICTS", by["geo-paris", "x"], by["geo-paris", "c"], 1.0),
                                 ("ASSOCIATES", by["geo-paris", "r"], by["geo-paris", "p"], 0.5),
                                 ("CONTRADICTS", by["lit-1984", "x"], by["geo-paris", "r"], 1.0)])
    e = edge_recovery(S, roles)
    assert e["MERGE"] == {"recovered": 1, "group_recovered": 1, "total": 26}
    assert e["CONFLICT"] == {"recovered": 1, "group_recovered": 2, "total": 52}   # p-x via its rep c
    assert e["RELATE"] == {"recovered": 1, "group_recovered": 2, "total": 52}
    # the cross-seed CONTRADICTS link, and paris-c/p with everest-c in one group (2 pairs)
    assert e["spurious_links"] == 1 and e["spurious_merges"] == 2
    assert e["per_seed"]["geo-paris"] == {"MERGE": 1, "CONFLICT": 1, "RELATE": 1}
    assert e["per_seed"]["lit-1984"] == {"MERGE": 0, "CONFLICT": 0, "RELATE": 0} and len(e["per_seed"]) == 26
    c, p, x, r = (by["geo-paris", t] for t in "cpxr")

    class T:      # the representative is p, a third member x wrongly in the group (same seed)
        derived = Derived(rep={c: p, x: p}, links=[("CONTRADICTS", x, c, 1.0)])
    e = edge_recovery(T, roles)
    assert e["MERGE"]["recovered"] == 1 and e["spurious_merges"] == 2      # c-x and p-x
    # the raw link exists, but x sits in c's group: not a group-level CONFLICT recovery
    assert e["CONFLICT"]["recovered"] == 1 and e["CONFLICT"]["group_recovered"] == 0


def test_e2t1_reports_link_misses_per_m_with_one_fold_per_epoch():
    from dataclasses import replace
    from meld.eval.order_shuffle import e2t1, gold_workload
    from meld.protocol.fold import nli_provider
    deltas, roles = gold_workload("hashing")
    lo = round(0.3 * F)
    eps = [replace(agent_epoch(theta=0.5, sel=[lo, lo, m]), cal_root=bytes([m]) * 16) for m in (1, 3, 5)]
    mk = lambda ep: BrainNode("sink", encoder="hashing", nli="lexical", epoch=ep,    # noqa: E731
                              provider=nli_provider("lexical"))
    out = e2t1(deltas, roles, eps, make_node=mk, scope="agent")
    assert sorted(out) == [1, 3, 5]
    for m, row in out.items():
        assert set(row["miss_rate"]) == set(row["group_miss_rate"]) == {"MERGE", "CONFLICT", "RELATE"}
        assert len(row["per_seed"]) == 26 and row["L"] <= 1
    # a larger neighbourhood collects at least the raw CONTRADICTS links a smaller one does
    assert out[5]["miss_rate"]["CONFLICT"] <= out[1]["miss_rate"]["CONFLICT"]
    with pytest.raises(ValueError, match="cal_root"):
        e2t1(deltas, roles, [eps[0], replace(eps[1], cal_root=eps[0].cal_root)], make_node=mk, scope="agent")


def test_order_shuffle_gold_workload_reports_purity_and_edges_before_and_after():
    from meld.eval.order_shuffle import gold_workload, run_order_shuffle
    from meld.protocol.fold import nli_provider
    deltas, roles = gold_workload("hashing")
    epoch = agent_epoch(theta=0.5)
    rep = run_order_shuffle([], deltas, k=4, seed=1, roles=roles, scope="agent",
                            make_node=lambda i: BrainNode("sink", encoder="hashing", nli="lexical",
                                                          epoch=epoch, provider=nli_provider("lexical")))
    assert rep["distinct_fingerprints"] == 1 and all(rep["post_pass"][c] == 0 for c in GATED)
    assert rep["pre_pass"]["sig"] > 0 and rep["orders_with_window"] == 0    # Sig is not a window
    pur = rep["purity"]
    assert pur["group_pairs"] >= pur["impure_pairs"] >= 0 and 0.0 <= pur["purity"] <= 1.0
    for when in ("edges_pre", "edges_post"):
        assert {"MERGE", "CONFLICT", "RELATE"} <= set(rep[when]["miss_rate"])
        assert all(0.0 <= v <= 1.0 for v in rep[when]["miss_rate"].values())
    assert rep["edges_post"]["total"] == {"MERGE": 26, "CONFLICT": 52, "RELATE": 52}


def test_order_shuffle_edges_before_and_after_differ_when_an_effect_arrives_last():
    # a DISTINGUISHES on the merged c-p pair arriving after every delta is read by no admission:
    # the admission state still has the merge, the pass splits it
    from meld.eval.anti_entropy_eval import with_effects
    from meld.eval.order_shuffle import gold_workload, run_order_shuffle
    from meld.protocol.fold import nli_provider
    deltas, roles = gold_workload("hashing", n_seeds=1)
    work = with_effects(deltas, roles, DEFAULT_POL)[:len(deltas) + 1]      # + DISTINGUISHES(c, p)
    epoch = agent_epoch(theta=0.5)
    rep = run_order_shuffle([], work, k=30, seed=3, roles=roles, scope="agent",
                            make_node=lambda i: BrainNode("sink", encoder="hashing", nli="lexical",
                                                          epoch=epoch, provider=nli_provider("lexical")))
    assert rep["edges_pre"]["miss_rate"]["MERGE"] < rep["edges_post"]["miss_rate"]["MERGE"] == 1.0
    assert rep["orders_with_window"] > 0 and rep["distinct_fingerprints"] == 1
    # the effect arrives last in every order: every admission state is stale, so the window is
    # measured against the post-pass state, not against the (stale) pre-pass majority
    rep = run_order_shuffle(deltas, work[len(deltas):], k=3, seed=3, roles=roles, scope="agent",
                            make_node=lambda i: BrainNode("sink", encoder="hashing", nli="lexical",
                                                          epoch=epoch, provider=nli_provider("lexical")))
    assert rep["orders_with_window"] == 3 and rep["pre_pass"]["reps"] == 0


# --------------------------------------------------------------------------- run_sync over objects
def test_run_sync_publishes_objects_and_reports_post_pass_convergence():
    from meld.sync.experiment import run_sync
    nodes = {n: _node(i, _scripted(ENTAIL if i else NEUTRAL)) for i, n in enumerate("ABC")}
    for node, t in zip(nodes.values(), TEXTS[2:]):
        node.ingest(t)
    nodes["B"].ingest(TEXTS[3])                  # a paraphrase: A relates it, B and C merge it
    peers = {n: [p for p in "ABC" if p != n] for n in "ABC"}
    res = run_sync(nodes, peers, routing_mode="broadcast", encoder="hashing", clock=lambda: 0.0)
    assert res.presence_agreement == 1.0                  # measured after the pass
    assert all(len(d) == 2 for d in res.delivered.values())
    conv = res.convergence["agent"]
    assert conv["post_pass"] == dict.fromkeys(gf.COMPONENTS, 0) and len(set(conv["fingerprints"])) == 1


# --------------------------------------------------------------------------- runner scripts
def _load(name):
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem + "_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_scripts_smoke(monkeypatch, tmp_path, capsys):
    rep = _load("run_order_shuffle.py").main(["--encoder", "hashing", "--nli", "lexical",
                                              "--workload", "running", "--k", "3",
                                              "--out", str(tmp_path / "os.json")])
    assert rep["distinct_fingerprints"] == 1 and (tmp_path / "os.json").exists()
    assert "no similarity merge" in capsys.readouterr().out
    from dataclasses import replace
    from meld.config import save_epoch
    from meld.sync.wire_delta import group_key
    lo, files = round(0.3 * F), []
    for m in (1, 3):
        files.append(str(tmp_path / f"ep{m}.json"))
        save_epoch(replace(agent_epoch(sel=[lo, lo, m]), cal_root=bytes([m]) * 16), group_key({}), files[-1])
    rep = _load("run_order_shuffle.py").main(["--encoder", "hashing", "--nli", "lexical", "--workload",
                                              "gold", "--k", "1", "--e2t1-epochs", *files])
    assert sorted(rep["e2t1"]) == [1, 3]
    ae = _load("run_anti_entropy_loss.py")
    monkeypatch.setattr(ae, "LOSSES", [0.05])
    monkeypatch.setattr(ae, "SEEDS", [0])
    rows = ae.main(["--encoder", "hashing", "--nli", "lexical", "--n", "3"])["rows"]
    assert rows[0]["with_ae_post_pass_max"] == 0
