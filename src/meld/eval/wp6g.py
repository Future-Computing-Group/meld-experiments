"""WP6g: multi-target admission counterfactual, evaluation only (the protocol modules are read, never changed).

The released step (merge/classifier.step) tests one pair, (a_e, b), a_e the first eligible r in K(b). ``vstep``
replays the step with the tested set widened to the eligible candidates of K(b) (same cell, rank(b) <= rank(r),
no DISTINGUISHES; the a_e rule applied to every r) in K's order (sigma_q descending, ties canonical):

  single  only the first eligible candidate (a_e): the released step, decision for decision (tests check it).
  first   test the candidates in K's order and MERGE with the first that passes; stop there.
  best    test every eligible candidate and MERGE with the passing one of largest s (ties: K's order).

A candidate r passes when s(r, b) > theta_{g(b)} and none of the released gates fires on it: pair mask on (r, b),
group contradiction (a CONTRADICTS target of b, from the collected set or a pair mask read so far, lies in
group(r)), group cannot-link, Cal veto. The a* mask still sends b to CONFLICT before any test; first and best
then still read every eligible pair, so their tested set is the same under any Sig (single: a_e, as released).
Under theta = INF16 (calibration) first and best test every eligible candidate, a theta-independent population
(``eligible`` on each decision); first-pass at a finite theta tests a theta-dependent prefix of it. Without a MERGE the decision is CONFLICT when a contradiction gate fired on a tested
candidate, else RELATE. Read set added to the released one: both directions of every tested candidate.
Links are not kept (no metric here reads them).

``cal_candidates``: the released Cal draw's candidate pool (calibration.fold_calibrate) over the variant's tested
pairs: per source item of the arriving node, its non-mergeable tested pairs (one per pair).
"""
from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from meld.merge import classifier
from meld.merge.signals import sigma_q_cmp
from meld.protocol.objects import contradicts, score_s

MODES = ("single", "first", "best")


class VDecision(NamedTuple):
    b: bytes
    decision: str
    tested: Tuple[Tuple[bytes, Optional[int]], ...]   # (r, s(r, b)) in test order
    target: Optional[bytes]                            # the merge partner, None unless MERGE
    gates: Tuple[str, ...]
    n_prior: int
    K: Tuple[bytes, ...]
    eligible: Tuple[bytes, ...] = ()                   # the eligible candidates of K, in K's order


def vstep(state: classifier.FoldState, b: classifier.Node, ctx: classifier.StepContext, mode: str) -> VDecision:
    """One variant step; appends b to ``state`` (module doc). REJECT and INSERT as released."""
    if mode not in MODES:
        raise ValueError(mode)
    n_prior = len(state.keys)
    if b.status != "ACTIVE" or b.retention < ctx.r_min:
        return VDecision(b.key, "REJECT", (), None, (), n_prior, ())
    K, a_e = classifier.select_tested_pair(state, b, ctx.sigma_lo, ctx.m, ctx.rank, ctx.distinguishes)
    nodes, gof = state.nodes, state.group_of
    tested: List[Tuple[bytes, Optional[int]]] = []
    gates: List[str] = []
    target = None
    if not K:
        decision = "INSERT"
    else:
        elig = [r for r in K if nodes[r].cell == b.cell and ctx.rank[b.key] <= ctx.rank[r]
                and frozenset((b.key, r)) not in ctx.distinguishes]
        theta = ctx.theta(n_prior)
        # the released read order: the tested pair first, then the a* mask, then the collected directions
        first = ctx.sig.pair(b, nodes[a_e]) if a_e is not None else None
        mask = contradicts(ctx.sig.dir(b, nodes[K[0]]))
        collected = [r for r in K[1:] if sigma_q_cmp(b.q, nodes[r].q, ctx.sigma_c) >= 0
                     and contradicts(ctx.sig.dir(b, nodes[r]))]
        contra = set(collected)
        if mask:
            decision, gates = "CONFLICT", ["mask"]
            for r in (elig if mode != "single" else elig[:1]):     # the tested set does not depend on Sig
                tested.append((r, score_s(*(first if r == a_e else ctx.sig.pair(b, nodes[r])))))
        else:
            passing = []
            for r in (elig if mode != "single" else elig[:1]):
                fwd, back = first if r == a_e else ctx.sig.pair(b, nodes[r])
                s = score_s(fwd, back)
                tested.append((r, s))
                if contradicts(fwd) or contradicts(back):
                    contra.add(r)
                    gates.append("pair_mask")
                    continue
                if mode == "best":
                    passing.append((r, s))      # gated below, once every pair mask is known
                elif _passes(r, s, theta, contra, state, b, ctx, gates):
                    target = r
                    break
            if mode == "best":
                ok = [(r, s) for r, s in passing if _passes(r, s, theta, contra, state, b, ctx, gates)]
                target = max(ok, key=lambda x: x[1])[0] if ok else None   # max keeps the first of ties
            decision = "MERGE" if target else "CONFLICT" if {"pair_mask", "group_contradiction"} & set(gates) \
                else "RELATE"
    state.add(b)
    if target:
        gof[b.key] = gof[target]
        state.members[gof[target]].append(b.key)
    else:
        gof[b.key] = b.key
        state.members[b.key] = [b.key]
    return VDecision(b.key, decision, tuple(tested), target, tuple(dict.fromkeys(gates)), n_prior, tuple(K),
                     tuple(elig) if K else ())


def _passes(r, s, theta, contra, state, b, ctx, gates) -> bool:
    gof = state.group_of
    if any(gof[c] == gof[r] for c in contra):
        gates.append("group_contradiction")
        return False
    if s is None or not s > theta:
        return False
    hit = False
    if any(frozenset((b.key, g)) in ctx.distinguishes for g in state.members[gof[r]]):
        gates.append("group_cannot_link")
        hit = True
    if tuple(sorted((r, b.key))) in ctx.cal_pairs:
        gates.append("cal_veto")
        hit = True
    return not hit


def vfold(nodes, ctx: classifier.StepContext, mode: str) -> List[VDecision]:
    state = classifier.FoldState()
    return [vstep(state, b, ctx, mode) for b in nodes]


def cal_candidates(decisions, item: Dict[bytes, str], label: Callable[[bytes, bytes], Optional[bool]]) -> Dict[str, list]:
    """item -> [(pair (c_x, c_y), s, decision)] of the non-mergeable tested pairs whose arrival is of that item."""
    out: Dict[str, list] = {}
    seen = set()
    for d in decisions:
        for r, s in d.tested:
            p = tuple(sorted((r, d.b)))
            if p in seen or label(r, d.b) is not False:
                continue
            seen.add(p)
            out.setdefault(item[d.b], []).append((p, s, d))
    return out


__all__ = ["MODES", "VDecision", "vstep", "vfold", "cal_candidates"]
