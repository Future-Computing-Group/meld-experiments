"""MELD through the baselines' pair protocol (PH-7) and the predicate diagnostic (PH-12), evaluation only.

Pair protocol (the row semantics of ``meld.baselines.memory.pair_rows``, the protocol Mem0 and Graphiti are scored on): each
labelled pair is a fresh two-claim store; text_a is published (held, no admission), text_b is
written and admitted. Here the store is an ``SMPMemory`` holding the deployed epoch of the pair's
domain (the source item prefix: paws, sick, vitaminc, lme), its claims in that epoch's scope and
cell (``calibration.policy`` for the benchmark domains, the default policy for the LongMemEval
substrate epoch, scope "agent"); a domain without an epoch (gold, scifact) is admitted without
one, so no similarity merge happens there (C8: no default operating point). MELD never retires a
held claim, so a silent overwrite cannot occur; the row flags are those of ``pair_rows``.

Every row keeps the signals the decision read, so other thresholds and the submitted predicate are
derived from the stored rows without a model (``derive``, ``submitted``): the admission decision,
its gates, s_q (min of the two entailment probabilities, uint16) and n_prior (1: the arrival's bin
is the epoch's bin 0), sigma (float cosine of the deployed encoder, as the submitted code computed
it) and chi_ba (the three-way NLI label with the arrival as premise and the held claim as
hypothesis, raw texts, as the submitted ``merge_on_receive`` called ``chi(incoming, local)``).

``submitted`` is the v1.0.6 ``merge.classifier.classify`` at the submitted operating point
(theta_merge 0.90, sigma_lo 0.12, eps_hi 0, ctx_ok and authority_ok true): exact key MERGE unless
contradicting; CONFLICT on a contradiction at sigma >= sigma_lo; MERGE at sigma >= theta_merge
without contradiction; RELATE in [sigma_lo, theta_merge); else NO_RELATION.

``derive(row, theta_q)`` is the two-claim step at another threshold: the contradiction gates and
an empty candidate set decide before the test (classifier.step), so CONFLICT, INSERT and REJECT
stand; otherwise MERGE iff s_q > theta_q and no other gate fired, else RELATE. An exact copy
(no admission decision) stays MERGE.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

from meld.agent.memory import DEFAULT_CELL, DEFAULT_POL, SMPMemory, signed
from meld.baselines.memory import PRED
from meld.eval import calibration as C
from meld.eval.benchmark_test import cp
from meld.eval.topology import bin_for, stamps
from meld.merge.signals import EmbeddingSimilarity, chi
from meld.protocol.fold import nli_provider
from meld.protocol.objects import m_pol_of

THETA_MERGE, SIGMA_LO = 0.90, 0.12          # the submitted fixed operating point
CONTRA_GATES = {"mask", "pair_mask", "group_contradiction"}


def domain_of(row: dict) -> str:
    return row["source_item_id"].split(":", 1)[0]


class PairMemory(SMPMemory):
    """An SMPMemory whose local claims are published into one given cell."""

    def __init__(self, *, cell: str, **kw):
        self.cell = cell
        super().__init__(**kw)

    def publish(self, text: str):
        text = (text or "").strip()
        return self.store.apply(*signed(self.delta(text, self.cell), self.key)) if text else None


def _policy(epoch) -> tuple:
    """(pol, cell) of the store that holds ``epoch``; refuses an epoch of another policy."""
    if epoch is None:
        return DEFAULT_POL, DEFAULT_CELL
    if epoch.scope == "agent":
        pol, cell = DEFAULT_POL, DEFAULT_CELL
    else:
        pol, cell = C.policy(epoch.scope, (epoch.sigma_lo_q, epoch.sigma_c_q, epoch.m)), epoch.scope
    if m_pol_of(pol) != epoch.m_pol:
        raise ValueError(f"epoch of scope {epoch.scope!r}: its M_pol is not the pair store's policy")
    return pol, cell


def submitted(*, sigma: float, chi: str, identical: bool) -> str:
    """The v1.0.6 classify at theta_merge 0.90, sigma_lo 0.12 (module doc), as a four-way label."""
    if identical or sigma >= SIGMA_LO:
        if chi == "contradicts":
            return "CONFLICT"
    if identical:
        return "MERGE"
    if sigma >= THETA_MERGE:
        return "MERGE"
    return "RELATE" if sigma >= SIGMA_LO else "NO_RELATION"


def derive(row: dict, theta_q: int) -> str:
    """The row's admission decision at threshold ``theta_q`` (module doc)."""
    d = row["decision"]
    if d not in ("MERGE", "RELATE") or row["s_q"] is None or row["event"] == "EXACT_COPY":
        return d
    if set(row["gates"]) - {"cal_veto", "group_cannot_link"}:
        return d
    if row["s_q"] <= theta_q:
        return "RELATE"
    return "RELATE" if row["gates"] else "MERGE"


def meld_rows(pairs: Iterable[dict], *, epochs: Dict[str, object], encoder: str, nli: str,
              key: bytes) -> List[dict]:
    """One pair-protocol row per pair (module doc); ``epochs``: domain -> EpochRecord."""
    pairs = list(pairs)
    setup = {d: _policy(e) for d, e in epochs.items()}
    provider = nli_provider(nli)
    es = EmbeddingSimilarity(encoder=encoder)
    out = []
    for p in pairs:
        dom = domain_of(p)
        epoch = epochs.get(dom)
        pol, cell = setup[dom] if dom in setup else _policy(None)
        mem = PairMemory(cell=cell, encoder=encoder, nli=nli, epoch=epoch, pol=pol, key=key, provider=provider)
        ka = mem.publish(p["text_a"]).claim_key
        decs = mem.write(p["text_b"])
        dec = decs[0] if decs else None
        decision, event = (dec.decision, "ADMISSION") if dec else ("MERGE", "EXACT_COPY")
        a_live = ka in set(mem.store.live_nodes())
        row = {"gold": p["label"], "decision": decision, "event": event, "pred": PRED[decision],
               "a_live": a_live, **flags(p["label"], decision, a_live)}
        a, b = p["text_a"], p["text_b"]
        row.update(pair_id=p.get("pair_id"), stratum=p["stratum"], domain=dom, label=p["label"],
                   label_source=p.get("label_source"), text_a=a, text_b=b,
                   s_q=dec.s_q if dec else None, theta_q=dec.theta_q if dec else None,
                   gates=list(dec.gates) if dec else [], n_prior=dec.n_prior if dec else None,
                   tested=bool(dec and dec.tested), bin=bin_for(dec.n_prior, epoch) if dec else None,
                   sigma=es.sigma(b.strip(), a.strip()), chi_ba=chi(b.strip(), a.strip(), model=nli),
                   identical=row["event"] == "EXACT_COPY", **stamps(encoder, nli, epoch, pol))
        out.append(row)
    return out


def flags(gold: str, decision: str, a_live: bool = True) -> dict:
    """pair_rows' four flags for a (possibly derived) decision."""
    return {"false_merge": gold != "MERGE" and decision == "MERGE",
            "silent_overwrite": gold == "CONFLICT" and not a_live,
            "missed_contradiction": gold == "CONFLICT" and a_live and decision != "CONFLICT",
            "claim_lost": decision == "REJECT"}


def _rate(k: int, n: int) -> dict:
    lo, hi = cp(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def _metrics(rows: List[dict], key: str) -> dict:
    f = [(r, flags(r["label"], r[key], r["a_live"])) for r in rows]
    nm = [x for r, x in f if r["label"] != "MERGE"]
    cf = [x for r, x in f if r["label"] == "CONFLICT"]
    mg = [r for r in rows if r["label"] == "MERGE"]
    return {"pairs": len(rows),
            "false_merge": _rate(sum(x["false_merge"] for x in nm), len(nm)),
            "silent_overwrite": _rate(sum(x["silent_overwrite"] for x in cf), len(cf)),
            "missed_contradiction": _rate(sum(x["missed_contradiction"] for x in cf), len(cf)),
            "claim_lost": _rate(sum(x["claim_lost"] for _, x in f), len(f)),
            "merge_recall": _rate(sum(PRED.get(r[key], r[key]) == "MERGE" for r in mg), len(mg)),
            "decisions": dict(Counter(r[key] for r in rows))}


def score(rows: List[dict], key: str) -> dict:
    """Overall and per-stratum metrics of decision field ``key``."""
    return {"overall": _metrics(rows, key),
            "stratum": {s: _metrics([r for r in rows if r["stratum"] == s], key)
                        for s in sorted({r["stratum"] for r in rows})}}


__all__ = ["PairMemory", "domain_of", "submitted", "derive", "meld_rows", "flags", "score",
           "THETA_MERGE", "SIGMA_LO"]
