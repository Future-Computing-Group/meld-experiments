"""WP6g multi-target admission counterfactual (meld.eval.wp6g), evaluation only: the variant step
(single / first-pass / best-pass over the eligible candidates of K(b)), its fold and its Cal draw."""
from __future__ import annotations

import random

import numpy as np

from meld.merge import classifier
from meld.merge.classifier import FoldState, Node, StepContext
from meld.protocol.objects import FIXED_ONE, INF16
from meld.eval import wp6g as G

N = lambda p_c, p_e: (round(p_c * FIXED_ONE), round(p_e * FIXED_ONE), round((1 - p_c - p_e) * FIXED_ONE))  # noqa: E731


class Sig:
    """Scripted payloads: (premise key, hyp key) -> triple; default entails 0.5 (no contradiction)."""

    def __init__(self, pay=None):
        self.pay, self.calls = dict(pay or {}), 0

    def dir(self, p, h):
        self.calls += 1
        return self.pay.get((p.key, h.key), N(0.0, 0.5))

    def pair(self, x, y):
        return self.dir(x, y), self.dir(y, x)


def _node(k, t, q):
    return Node(k, "c", (t, k), np.array(q, np.int8), k.decode(), "ACTIVE", FIXED_ONE)


def _ctx(sig, theta=0, m=3, dist=frozenset(), sigma_c=0):
    return StepContext(sigma_lo=0, sigma_c=sigma_c, m=m, theta=lambda n: theta, rank={}, distinguishes=dist,
                       cal_pairs=set(), sig=sig, rep=min, r_min=1)


def _rank(ctx, nodes):
    return ctx._replace(rank={n.key: 1 for n in nodes})


# three earlier nodes r1, r2, r3 most to least similar to b
R1, R2, R3 = _node(b"r1", 1, [100, 10, 0]), _node(b"r2", 2, [100, 30, 0]), _node(b"r3", 3, [100, 60, 0])
B = _node(b"b", 4, [100, 0, 0])


def _state(*ns):
    st = FoldState()
    for n in ns:
        st.add(n)
        st.group_of[n.key] = n.key
        st.members[n.key] = [n.key]
    return st


def _s(p_e):
    return N(0.0, p_e)


def test_first_pass_skips_a_failing_nearest_and_merges_with_the_next_passing():
    sig = Sig({(b"b", b"r1"): _s(0.1), (b"r1", b"b"): _s(0.1),
               (b"b", b"r2"): _s(0.9), (b"r2", b"b"): _s(0.9),
               (b"b", b"r3"): _s(0.95), (b"r3", b"b"): _s(0.95)})
    th = round(0.5 * FIXED_ONE)
    ns = [R1, R2, R3, B]
    for mode, target, n_tested in (("single", None, 1), ("first", b"r2", 2), ("best", b"r3", 3)):
        st = _state(R1, R2, R3)
        d = G.vstep(st, B, _rank(_ctx(sig, th), ns), mode)
        assert d.target == target and len(d.tested) == n_tested, mode
        assert d.decision == ("MERGE" if target else "RELATE")
        if target:
            assert st.group_of[b"b"] == target and b"b" in st.members[target]


def test_contradiction_gates_block_a_candidate_and_its_group_only():
    th = round(0.5 * FIXED_ONE)
    ns = [R1, R2, R3, B]
    # r1 passes s but its pair contradicts (pair mask): skipped; r2 passes cleanly
    sig = Sig({(b"b", b"r1"): N(0.1, 0.9), (b"r1", b"b"): N(0.9, 0.05),
               (b"b", b"r2"): _s(0.9), (b"r2", b"b"): _s(0.9)})
    d = G.vstep(_state(R1, R2, R3), B, _rank(_ctx(sig, th), ns), "first")
    assert d.target == b"r2" and "pair_mask" in d.gates
    # r1 and r2 in one group, r1's pair contradicts: r2 is blocked by group contradiction, r3 merges
    sig.pay.update({(b"b", b"r3"): _s(0.9), (b"r3", b"b"): _s(0.9)})
    st = _state(R1, R2, R3)
    st.group_of[b"r2"] = b"r1"
    st.members[b"r1"].append(b"r2")
    d = G.vstep(st, B, _rank(_ctx(sig, th), ns), "best")
    assert d.target == b"r3" and "group_contradiction" in d.gates


def test_mask_on_a_star_is_conflict_and_the_tested_set_does_not_depend_on_it():
    sig = Sig({(b"b", b"r1"): N(0.9, 0.05)})
    for mode, n in (("single", 1), ("first", 3), ("best", 3)):
        d = G.vstep(_state(R1, R2, R3), B, _rank(_ctx(sig, 0), [R1, R2, R3, B]), mode)
        assert d.decision == "CONFLICT" and d.target is None and [r for r, _ in d.tested] == [b"r1", b"r2", b"r3"][:n]
        assert d.eligible == (b"r1", b"r2", b"r3")


def test_single_mode_equals_the_released_step_on_random_streams():
    rng = random.Random(7)
    seen = set()
    for trial in range(30):
        ns = [_node(bytes([65 + i]), i + 1, [rng.randint(-50, 100) for _ in range(4)]) for i in range(12)]
        pay = {(x.key, y.key): N(*((0.6, 0.2) if rng.random() < 0.1 else (0.0, rng.random() * 0.9)))
               for x in ns for y in ns if x != y}
        th = rng.randint(0, FIXED_ONE // 2)
        dist = frozenset({frozenset((ns[3].key, ns[1].key))}) if trial % 2 else frozenset()
        ctx = _ctx(Sig(pay), th, m=rng.choice([1, 3, 5]), dist=dist, sigma_c=round(0.3 * FIXED_ONE))
        ctx = _rank(ctx, ns)
        s1, s2 = FoldState(), FoldState()
        for b in ns:
            rel = classifier.step(s1, b, ctx)
            var = G.vstep(s2, b, ctx, "single")
            assert var.decision == rel.decision, (trial, b.key)
            assert var.target == (rel.tested[0] if rel.decision == "MERGE" else None)
            assert [r for r, _ in var.tested] == ([rel.tested[0]] if rel.tested else [])
            seen.add(rel.decision)
        assert s1.group_of == s2.group_of
    assert {"MERGE", "RELATE", "CONFLICT", "INSERT"} <= seen


def test_cal_draw_is_one_non_mergeable_tested_pair_per_item_of_the_arrival():
    D = lambda b, tested, n=0: G.VDecision(b, "RELATE", tested, None, (), n, ())  # noqa: E731
    ds = [D(b"x", ()), D(b"y", ((b"x", 5),), 1), D(b"z", ((b"x", 7), (b"y", 3)), 2)]
    item = {b"x": "i1", b"y": "i1", b"z": "i2"}
    lab = {frozenset((b"x", b"y")): True, frozenset((b"x", b"z")): False, frozenset((b"y", b"z")): False}
    cand = G.cal_candidates(ds, item, lambda a, b: lab[frozenset((a, b))])
    assert {it: sorted(s for _, s, _ in v) for it, v in cand.items()} == {"i2": [3, 7]}
    assert G.cal_candidates([D(b"y", ((b"x", 5),), 1)], item, lambda a, b: None) == {}


def test_theta_inf_never_merges():
    sig = Sig({(b"b", b"r1"): _s(1.0), (b"r1", b"b"): _s(1.0)})
    d = G.vstep(_state(R1, R2, R3), B, _rank(_ctx(sig, INF16), [R1, R2, R3, B]), "first")
    assert d.decision == "RELATE" and len(d.tested) == 3
