"""Item re-split spread of the E2-B3 per-bin false-merge rate (meld.eval.resplit)."""
from __future__ import annotations

import os
import sys

from meld.eval import benchmark_test as B
from meld.eval import calibration as C
from meld.eval import resplit as R

sys.path.insert(0, os.path.dirname(__file__))
from test_benchmark_test import KEY, Scripted, _epoch, _eval, _manifest  # noqa: E402


def _p(side, item, s, bin=0, blocked=False, decision="RELATE"):
    return {"side": side, "item": (side, 0, item), "bin": bin, "s_q": s, "blocked": blocked, "decision": decision}


# alpha 0.34 with n = 2 Cal scores: k = ceil(3 * 0.66) = 2, theta = the larger Cal score
A34 = 340_000
FOUR = [_p("cal", "a", 1), _p("cal", "b", 2), _p("test", "c", 3), _p("test", "d", 4)]


def test_observed_counts_exceedances_and_skips_blocked_pairs():
    pairs = FOUR + [_p("test", "e", 9, blocked=True), _p("test", "f", 9, bin=1)]
    th = {g: t for g, (_, t) in R.seed_table(pairs, A34).items()}
    assert th == {0: 2}                                   # bin 1 has no Cal record: disabled
    assert R.exceed(pairs, th) == {0: (2, 3), 1: (0, 1)}  # blocked pair in the denominator, never above


def test_resplit_matches_the_exact_split_distribution():
    # the 6 equally likely 2-of-4 Cal sets give test rates 1 (once), 1/2 (twice), 0 (three times);
    # exchangeability: mean = (n + 1 - k) / (n + 1) = 1/3
    r = R.resplit(FOUR, A34, splits=4000, seed=0)[0]
    assert len(r) == 4000 and set(r) <= {0.0, 0.5, 1.0}
    assert abs((r == 1).mean() - 1 / 6) < 0.03 and abs((r == 0.5).mean() - 1 / 3) < 0.03
    assert abs(r.mean() - 1 / 3) < 0.02
    s = R.spread(r, observed=1.0, alpha=0.34)
    assert abs(s["p_ge_observed"] - 1 / 6) < 0.03 and abs(s["share_above_alpha"] - 1 / 2) < 0.03
    assert (R.resplit(FOUR, A34, splits=4000, seed=0)[0] == r).all()      # fixed seed


def test_resplit_draws_one_cal_pair_per_item():
    # one cal item with two pairs (1 and 5) and one with a single pair (2): theta = max of the drawn two;
    # test pair 4 exceeds iff the draw takes 1 from item a, when the items keep their sides
    pairs = [_p("cal", "a", 1), _p("cal", "a", 5), _p("cal", "b", 2), _p("test", "c", 4)]
    assert R.seed_table(pairs, A34, seed=0)[0] in ((2, 2), (2, 5))
    r = R.resplit(pairs, A34, splits=4000, seed=1)[0]
    # Cal {a, b} (1/3): theta 2 or 5 by a's draw, test c = 4 gives 1 or 0; Cal {a, c} (1/3): theta >= 4,
    # test b = 2 gives 0; Cal {b, c} (1/3): theta 4, test a = {1, 5} gives 1/2. So P(1) = 1/6, P(1/2) = 1/3
    assert set(r) <= {0.0, 0.5, 1.0}
    assert abs((r == 1).mean() - 1 / 6) < 0.03 and abs((r == 0.5).mean() - 1 / 3) < 0.03


def test_pools_from_the_folds_rebuild_the_epoch_and_reproduce_the_realised_false_merges():
    m = _manifest()
    res = _epoch(m)
    ep = res["epoch"]
    cal = R.cal_pool(C.cal_rows(m, "paws"), ep, key=KEY, encoder="hashing", nli="lexical", provider=Scripted())
    ev = _eval(m, ep)
    test = R.test_pool(ev, ep)
    assert {p["side"] for p in cal} == {"cal"} and {p["side"] for p in test} == {"test"}
    assert len({p["item"] for p in cal}) == len(res["cal"])
    assert R.seed_table(cal, ep.alpha) == {g: (n, t) for g, n, _, t, _ in ep.table}
    th = {g: t for g, n, _, t, _ in ep.table}
    realised = {int(g): (b["false_merge"]["k"], b["false_merge"]["n"]) for g, b in ev["summary"]["bin"].items()
                if b["false_merge"]["n"]}
    assert R.merges(test) == realised
    ub = R.exceed(test, th)
    assert all(ub[g][0] >= k and ub[g][1] == n for g, (k, n) in realised.items())
