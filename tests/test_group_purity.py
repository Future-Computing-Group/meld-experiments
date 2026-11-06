"""E2-B5 / PH-14: group purity and hidden-node share on the folded benchmark test streams."""
from __future__ import annotations

import itertools

from meld.eval import benchmark_test as B

from test_benchmark_test import Scripted, _epoch, _eval, _manifest


def test_group_purity_counts_every_member_pair_by_hand():
    m = _manifest()
    ep = _epoch(m)["epoch"]
    ev = _eval(m, ep, provider=Scripted())
    g = B.group_purity(ev, source="paws")
    n_pairs = n_hidden = n_untested = n_nodes = 0
    for c in ev["chunks"]:
        rep = c["store"].derived.rep
        groups = {}
        for k in c["item"]:
            groups.setdefault(rep.get(k, k), []).append(k)
        merged = {frozenset(d.tested) for d in c["decisions"] if d.decision == "MERGE"}
        n_nodes += len(c["item"])
        for r, ms in groups.items():
            n_pairs += len(list(itertools.combinations(ms, 2)))
            hid = [x for x in ms if x != r]
            n_hidden += len(hid)
            n_untested += sum(frozenset((x, r)) not in merged for x in hid)
    assert g["group_pairs"] == n_pairs and g["hidden_nodes"] == n_hidden and g["nodes"] == n_nodes
    assert g["hidden_untested_vs_rep"] == n_untested
    assert g["group_pairs"] == g["mergeable"] + g["non_mergeable"] + g["unlabelled"]
    assert g["purity_labelled"] is None or 0.0 <= g["purity_labelled"] <= 1.0
    assert g["hidden_share"] == (n_hidden / n_nodes)
    assert set(g["unlabelled_by_class"]) <= {"cross_item", "same_item", "manifest_pending"}


def test_chunks_carry_their_decisions():
    m = _manifest()
    ev = _eval(m, _epoch(m)["epoch"], provider=Scripted())
    assert all(len(c["decisions"]) == c["texts"] for c in ev["chunks"])
