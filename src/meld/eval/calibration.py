"""SEED calibration of one domain scope from the benchmark's cal split (design r3 §4.1-4.3, §4.5,
§4.6, §7.4).

The domain's distinct cal texts are shuffled (seeded) into one stream, published as deltas of
one cell and folded from the log start by ``fold.readjudicate`` without an epoch: the selector is
the policy's bootstrap selector, so every tested pair (a_e, b) is drawn by the fold's own
``select_tested_pair`` and read through the same Sig path (one record, both directions, issuer
hw). Labels follow the fold: every tested pair takes its mergeability from the label store
``label_queue.resolve``, (1) the manifest, else (2) the raw-source label, else (3) the annotator
label; a pair of two items is never labelled by construction (the annotators' labels on such pairs
also estimate eps_x). Tested pairs nobody labels are counted per class; more than
``max_unresolved`` of them (default 0, None for a development run) refuse the epoch. One
non-mergeable tested pair per source item (seeded draw) becomes a CalRecord; a tested pair belongs
to the item of its arriving node b, whose arrival drew it, so the per-item draw is over the test
unit (an arriving claim and the pair the selector gives it) and items stay disjoint. The epoch is
``conformal.build_epoch`` (policy SEED, mondrian_c off) and must pass ``verify_epoch`` against the
pass's own Sig records.

Only the cal split is read: ``cal_rows`` filters it and ``calibrate`` refuses any other row.
A domain whose cal rows lack a final mergeability after the label store (``mergeable`` None, or a
filtered ``source_filtered`` pre-label, with no source or annotator label) is refused. A tested pair
the store marks excluded (an AMBIGUOUS item under ``--ambiguous exclude``) is decided, counted per
class as ``excluded``, and never a Cal record; ``ambiguous`` counts AMBIGUOUS items per class.

The recall workloads (``calibrate_substrate``, domains of ``substrates``) fold their cal-pool
worlds, one store each as the evaluation's sink holds it, through the same ``fold_calibrate``.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence

from meld.agent.memory import DEFAULT_POL
from meld.brain.store import Store
from meld.merge.conformal import bin_of, build_epoch, n_min, verify_epoch
from meld.proto import CANON_VERSION
from meld.protocol.fold import readjudicate
from meld.protocol.objects import CalRecord, m_nli, m_pol_of, signed

# Domain strata (§4.5): each domain is its own scope and stream with its own SEED epoch.
DOMAINS = {"paws": ("S1_paws_para", "S2_paws_nonpara"),
           "vitaminc": ("S3_vitc_nonfactual", "S4_vitc_factual"),
           "sick": ("S8_sick_para", "S8_sick_oneway", "S8_sick_contra")}
AUTHOR = "calibration"


def split_rows(manifest: dict, domain: str, split: str) -> List[dict]:
    """The ``split`` rows of ``domain``, labelled or not."""
    return [r for r in manifest["pairs"] if r["split"] == split and r["stratum"] in DOMAINS[domain]]


def cal_rows(manifest: dict, domain: str, source_label: Optional[Callable] = None,
             annotations: Optional[dict] = None) -> List[dict]:
    """The cal-split rows of ``domain``; ValueError if any still lacks a final mergeability after the
    label store (``label_queue.resolve``: manifest, else ``source_label``, else ``annotations``)."""
    from meld.eval.label_queue import resolve
    rows = split_rows(manifest, domain, "cal")
    bad = Counter((r["stratum"], r["label_provenance"]) for r in rows
                  if resolve(r, r["text_a"], r["text_b"], r["source_item_id"], r["source_item_id"],
                             source=domain, source_label=source_label, annotations=annotations)[2] is None)
    if bad:
        raise ValueError(f"domain {domain!r}: cal rows without a final mergeability label "
                         f"{sorted(bad.items())}; they come from annotation, calibrate after it")
    return rows


def policy(scope: str, selector: Sequence[int]) -> dict:
    """The scope's M_pol: the default ranks and authorization, one cell named as the scope, the
    calibration authority, and ``selector`` as bootstrap selector (the epoch's selector)."""
    return {**DEFAULT_POL, "scopes": {scope: [scope]}, "bootstrap_selector": list(selector),
            "auth_policy": {**DEFAULT_POL["auth_policy"], "calibration_authority": {scope: AUTHOR}}}


def stream(rows: Sequence[dict], *, scope: str, encoder: str, selector: Sequence[int], key: bytes,
           seed: int = 0):
    """The scope's stream over ``rows`` of one split: the distinct texts shuffled (seeded),
    published as deltas of one cell and applied to a fresh Store under ``policy(scope, selector)``.
    Returns (store, pol, item: claim key -> source item, pair: frozenset of claim keys -> row)."""
    from meld.eval.order_shuffle import claims
    if len({r["split"] for r in rows}) != 1:
        raise ValueError("a stream holds the rows of exactly one split")
    pol = policy(scope, selector)
    texts = sorted({t for r in rows for t in (r["text_a"], r["text_b"])})
    random.Random(seed).shuffle(texts)
    deltas = claims(encoder, pol, [(t, scope) for t in texts])
    kof = {d.text: d.claim_key for d in deltas}
    item: Dict[bytes, str] = {}
    pair: Dict[frozenset, dict] = {}
    for r in rows:
        ka, kb = kof[r["text_a"]], kof[r["text_b"]]
        for k in (ka, kb):
            if item.setdefault(k, r["source_item_id"]) != r["source_item_id"]:
                raise ValueError("a text in two source items: items must be text-disjoint")
        pair[frozenset((ka, kb))] = r
    store = Store(pol, key, encoder=encoder)
    for d in deltas:
        store.apply(*signed(d, key))
    return store, pol, item, pair


def calibrate(rows: Sequence[dict], *, scope: str, encoder: str, nli: str,
              provider: Callable, key: bytes, alpha: int, B_0: int, selector: Sequence[int],
              hw: str, seed: int = 0, seq: int = 1, created_at: int = 0,
              source_label: Optional[Callable] = None, annotations: Optional[dict] = None,
              max_unresolved: Optional[int] = 0) -> dict:
    """Fold ``rows`` (cal split only) and build the verified SEED epoch; see the module doc.
    ``selector`` is (sigma_lo_q, sigma_c_q, m); ``alpha`` in ppm (alpha_c = alpha);
    ``source_label`` and ``annotations`` as in ``label_queue.resolve``."""
    if not rows or any(r["split"] != "cal" for r in rows):
        raise ValueError("calibrate reads the cal split only")
    store, pol, item, pair = stream(rows, scope=scope, encoder=encoder, selector=selector, key=key,
                                    seed=seed)
    res = fold_calibrate([(store, item, pair, {})], scope=scope, pol=pol, source=scope,
                         encoder=encoder, nli=nli, provider=provider, alpha=alpha, B_0=B_0, hw=hw,
                         seed=seed, seq=seq, created_at=created_at, source_label=source_label,
                         annotations=annotations, max_unresolved=max_unresolved)
    return {**res, "store": store, "unlabelled_subtype": any(r["label"] is None for r in rows)}


def calibrate_substrate(domain: str, worlds, *, encoder: str, nli: str, provider: Callable,
                        alpha: int, B_0: int, hw: str, seed: int = 0, seq: int = 1,
                        created_at: int = 0, annotations: Optional[dict] = None,
                        max_unresolved: Optional[int] = 0, cache_dir: Optional[str] = None,
                        copies_mergeable: bool = False) -> dict:
    """The SEED epoch of a recall workload (``substrates``: domain hotpot or lme) from its cal-pool
    ``worlds``, each folded as the evaluation's sink holds it, in the evaluation memory's scope and
    M_pol (``DEFAULT_SCOPE``, ``DEFAULT_POL``), so the epoch is the one its memories accept. No pair
    has a raw-source label: every tested pair is labelled by the annotators (``annotations``), the
    ``light_paraphrase`` copies too unless ``copies_mergeable`` (``label_queue.copies_validated``)."""
    from meld.agent.memory import DEFAULT_SCOPE
    from meld.eval import substrates as S
    built = []
    for _, plan, item, src in worlds:
        store = S.world_store(plan, encoder=encoder, cache_dir=cache_dir)
        k_item, k_src = S.keyed(store, item, src)
        built.append((store, k_item, {}, k_src))
    return fold_calibrate(built, scope=DEFAULT_SCOPE, pol=DEFAULT_POL, source=domain, encoder=encoder,
                          nli=nli, provider=provider, alpha=alpha, B_0=B_0, hw=hw, seed=seed, seq=seq,
                          created_at=created_at, annotations=annotations,
                          max_unresolved=max_unresolved, copies_mergeable=copies_mergeable)


def fold_calibrate(worlds, *, scope: str, pol: dict, source: str, encoder: str, nli: str,
                   provider: Callable, alpha: int, B_0: int, hw: str, seed: int = 0, seq: int = 1,
                   created_at: int = 0, source_label: Optional[Callable] = None,
                   annotations: Optional[dict] = None, max_unresolved: Optional[int] = 0,
                   copies_mergeable: bool = False) -> dict:
    """Fold every world (store, item: claim key -> source item, pair: frozenset of claim keys ->
    manifest row, src: claim key -> source text) under ``pol`` and build the verified SEED epoch
    over all their tested pairs, one Cal record per source item; ``source`` names the label
    queue. Without a non-mergeable tested pair there is no Cal and ``epoch`` is None."""
    from meld.eval.label_queue import AMBIGUOUS, resolve
    selector = pol["bootstrap_selector"]
    m_n = m_nli(encoder, nli, "int8", CANON_VERSION)
    labels, cand, truth, sig, decisions = defaultdict(Counter), {}, {}, [], []
    items_by_bin, calls, worlds = defaultdict(set), 0, list(worlds)
    for store, item, pair, src in worlds:
        res = readjudicate(store, scope, m_nli=m_n, provider=provider, issuer=AUTHOR, hw=hw)
        sig += res.new_sig
        calls += res.counters["nli_calls"]
        decisions += res.decisions
        text = {d.claim_key: d.text for _, d in store.nodes().values()}
        for d in res.decisions:
            if d.tested is None:
                continue
            a, b = d.tested
            items_by_bin[bin_of(d.n_prior, B_0)].add(item[b])
            r = pair.get(frozenset(d.tested))
            cls, mergeable, prov = resolve(r, text[a], text[b], item[a], item[b], source=source,
                                           source_label=source_label, annotations=annotations,
                                           copy=a in src and src[a] == src.get(b),
                                           copies_mergeable=copies_mergeable)
            labels[cls]["excluded" if mergeable is None and prov else "unresolved" if mergeable is None
                        else "mergeable" if mergeable else "non_mergeable"] += 1
            if prov == AMBIGUOUS:
                labels[cls]["ambiguous"] += 1
            if mergeable is not False:
                continue
            c_x, c_y = sorted(d.tested)
            if (c_x, c_y) in truth:                 # one pair recurring in two worlds is one unit
                continue
            cand.setdefault(item[b], []).append(CalRecord(
                scope=scope, c_x=c_x, c_y=c_y, bin=bin_of(d.n_prior, B_0), s_q=d.s_q, hw=hw, obs=1,
                label="NON_MERGEABLE", subtype="CONTRADICTION" if prov == "manifest" and r["label"] == "CONFLICT"
                else "OTHER", source="SEED", item=item[b], author=AUTHOR, created_at=created_at,
                sigma_lo_q=selector[0], m=selector[2], m_pol=m_pol_of(pol), n_prior=d.n_prior))
            truth[c_x, c_y] = {"class": cls, "provenance": prov, "pair_id": r["pair_id"] if r else None}
    labels = {c: dict(n) for c, n in sorted(labels.items())}
    unresolved = sum(n.get("unresolved", 0) for n in labels.values())
    if max_unresolved is not None and unresolved > max_unresolved:
        raise ValueError(f"scope {scope!r} ({source}): {unresolved} tested pairs unresolved (no manifest, "
                         f"source or annotator label; {labels}) exceed the tolerance {max_unresolved}: "
                         "annotate them (scripts/build_label_queue.py) before building an epoch")
    rng = random.Random(seed)
    cal = [rng.choice(cand[it]) for it in sorted(cand)]     # one tested pair per source item
    truth = {(k.c_x, k.c_y): truth[k.c_x, k.c_y] for k in cal}

    epoch = build_epoch(cal, scope=scope, seq=seq, alpha=alpha, alpha_c=alpha, policy="SEED",
                        bins=(B_0, "doubling"), selector_params=(selector[0], selector[2],
                                                                  m_pol_of(pol)),
                        m_nli=m_n, sigma_c_q=selector[1], author=AUTHOR, created_at=created_at,
                        mondrian_c=False) if cal else None
    if epoch is not None and not verify_epoch(epoch, cal, sig):
        raise ValueError("epoch does not verify by recomputation; not written")
    rows_g = {g: n for g, n, *_ in epoch.table} if epoch else {}
    populated = sorted({bin_of(d.n_prior, B_0) for d in decisions if d.tested})
    stats = dict(worlds=len(worlds), nodes=len(decisions), tested=sum(d.tested is not None for d in decisions),
                 nli_calls=calls, cal=len(cal), unresolved=unresolved)
    return {"epoch": epoch, "cal": cal, "sig": sig, "pol": pol, "truth": truth, "stats": stats,
            "labels": labels, "populated_bins": populated,
            "items_by_bin": {g: len(items_by_bin[g]) for g in populated},
            "disabled_bins": [g for g in populated if rows_g.get(g, 0) < n_min(alpha)]}


__all__ = ["DOMAINS", "split_rows", "cal_rows", "policy", "stream", "calibrate", "calibrate_substrate",
           "fold_calibrate"]
