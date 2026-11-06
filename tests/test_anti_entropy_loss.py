"""Loss-injection convergence over the five replicated sets (design §2.2, §7.2 H1-H2, §7.4).

N replicas, each with its own provider, receive the gold workload with authored effects, each in
its own seeded order, each delivery dropped independently at the loss rate. WITHOUT anti-entropy
the lost objects stay lost and the authored sets diverge after the pass. WITH ring anti-entropy
(DIGEST / PULL / PUBLISH over the five sets), a Sig exchange and a pass, every replica agrees on
all five components (authored sets, Sig, representatives, links, decisions) and on the PASS
certificate fingerprint; the pre-pass divergence (the window) is measured. Deterministic in the
seed; hashing encoder and the lexical rung with per-replica hardware offsets.
"""
from __future__ import annotations

from meld.eval.graph_fingerprint import COMPONENTS
from meld.protocol.objects import FIXED_ONE

from test_harnesses import _node

ZERO = dict.fromkeys(COMPONENTS, 0)


def _work(n_seeds=8):
    from meld.agent.memory import DEFAULT_POL
    from meld.eval.anti_entropy_eval import with_effects
    from meld.eval.order_shuffle import gold_workload
    deltas, roles = gold_workload("hashing", n_seeds=n_seeds)
    return with_effects(deltas, roles, DEFAULT_POL), roles


def _hw(i):
    """Replica i's provider: the lexical rung with a hardware-class offset of (i % 3) - 1 units."""
    from meld.protocol.fold import nli_provider
    lex = nli_provider("lexical")
    return lambda a, b: tuple(min(FIXED_ONE, max(0, x + (i % 3) - 1)) for x in lex(a, b))


def _make(i):
    return _node(i, _hw(i))


def _cell(**kw):
    from meld.eval.anti_entropy_eval import lossy_cell
    work, roles = _work()
    return lossy_cell(work, make_node=_make, scope="agent", roles=roles, **kw)


def test_with_effects_adds_distinguishes_and_overrule_per_fourth_seed():
    from meld.agent.memory import DEFAULT_POL
    from meld.eval.anti_entropy_eval import with_effects
    from meld.eval.order_shuffle import gold_workload
    deltas, roles = gold_workload("hashing", n_seeds=8)
    extra = with_effects(deltas, roles, DEFAULT_POL)[len(deltas):]
    assert [e.kind for e in extra] == ["DISTINGUISHES", "OVERRULE"] * 2
    assert all(e.c1 < e.c2 for e in extra if e.kind == "DISTINGUISHES")


def test_zero_loss_agrees_without_anti_entropy():
    cell = _cell(loss=0.0, n=5, seed=0, anti_entropy=False)
    assert cell["dropped"] == 0 and cell["delivered"]["authored"] == 0
    assert cell["post_pass"] == ZERO and len(set(cell["fingerprints"])) == 1


def test_loss_without_anti_entropy_diverges():
    cell = _cell(loss=0.2, n=4, seed=0, anti_entropy=False)
    assert cell["dropped"] > 0 and cell["rounds"] == 0
    assert cell["post_pass"]["authored"] > 0 and len(set(cell["fingerprints"])) > 1


def test_lossy_reordered_replicas_agree_after_anti_entropy_exchange_and_pass():
    windows = []
    for seed in range(3):
        cell = _cell(loss=0.2, n=4, seed=seed)
        assert cell["dropped"] > 0 and cell["rounds"] >= 1 and cell["moved"] > 0
        assert all(cell["ae_bytes"][k] > 0 for k in ("/digest", "/pull", "/patch"))
        assert cell["delivered"]["authored"] > 0                  # loss left holes before repair
        assert cell["pre_pass"]["authored"] == cell["pre_pass"]["sig"] == 0   # repaired sets
        assert cell["post_pass"] == ZERO, seed
        assert len(set(cell["fingerprints"])) == 1 and cell["pass_misses"][-1] == [0] * 4
        assert 0.0 <= cell["purity"]["purity"] <= 1.0
        windows.append(sum(cell["pre_pass"].values()))
    assert any(windows)          # admission state still differed after repair, before the pass


def test_effects_last_exercises_a_second_pass_and_converges():
    # all deltas (canonical order), then the effects, then anti-entropy: the A_c effects change
    # tested pairs no admission read, so the first pass misses and the second agrees (L = 1)
    from meld.eval.anti_entropy_eval import lossy_cell
    work, _ = _work(n_seeds=12)              # from 12 seeds on, an effect moves a tested pair
    cell = lossy_cell(work, make_node=_make, scope="agent", loss=0.0, n=4, seed=0, effects_last=True)
    assert cell["L"] == 1 and cell["pass_misses"][0] != [0] * 4 and cell["pass_misses"][-1] == [0] * 4
    assert cell["post_pass"] == ZERO


def test_anti_entropy_converges_at_the_low_loss_rate():
    cell = _cell(loss=0.003, n=12, seed=2)
    assert cell["post_pass"] == ZERO and len(set(cell["fingerprints"])) == 1


def test_sweep_is_deterministic_in_seed():
    from meld.eval.anti_entropy_eval import loss_sweep
    work, roles = _work(n_seeds=3)
    a = loss_sweep(work, losses=[0.0, 0.1], n=3, seeds=[0, 1], scope="agent", make_node=_make)
    b = loss_sweep(work, losses=[0.0, 0.1], n=3, seeds=[0, 1], scope="agent", make_node=_make)
    assert a == b


def test_sweep_with_anti_entropy_is_zero_everywhere_and_without_it_loses():
    from meld.eval.anti_entropy_eval import loss_sweep
    work, roles = _work(n_seeds=12)
    rows = loss_sweep(work, losses=[0.0, 0.05, 0.2], n=4, seeds=[0, 1, 2], scope="agent",
                      make_node=_make, roles=roles)["rows"]
    assert [r["loss"] for r in rows] == [0.0, 0.05, 0.2]
    for r in rows:
        assert r["with_ae_post_pass_max"] == 0 and r["final_misses_max"] == 0
        assert r["effects_last_post_pass_max"] == 0 and r["effects_last_L_max"] <= 1
        assert (r["without_ae_authored_mean"] > 0) == (r["loss"] > 0)
    assert rows[0]["effects_last_L_max"] == 1                    # no loss: every replica misses once
