"""B2 risk-specified admission (protocol design r3 §4.1-4.6, Prop 5): per-bin split-conformal
thresholds over single-observation NON_MERGEABLE Cal scores, in exact rationals.

alpha is an integer in parts per million (ALPHA_ONE). theta_g is the ceil((n+1)(1-alpha))-th
smallest s_q of bin g, or INF16 when that index exceeds n (n < n_min). The epoch table always
records theta_c (same quantile at alpha_c over the CONTRADICTION records); mondrian_c only
decides whether the test reads it (``threshold``). MERGE iff s > theta, strictly: the fold's
merge site must not merge at s == theta (T6 owns that test).

Binding rules for callers:
- The selector is (sigma_lo_q, m, M_pol), carried by every Cal record and checked against the
  epoch's. M_pol hashes the rank map and the authorization policy, both of which fix the tested
  pairs (§4.3), so ANY M_pol change requires a new Cal set (a new cal_root).
- Every Cal record's bin is recomputed from its n_prior and the epoch's B_0; an epoch cannot
  rebin a reused Cal set. One record per item and per tested pair.
- ``verify_epoch`` needs the full Sig set of the scope: every s_q must equal the score of a
  single Sig observation on cal_hw and be at least the joined score.
- ``active_epoch`` does not check authorization: filter epochs by
  ``status_crdt.epoch_authorized`` first (protocol/fold.py does).
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Iterable, Optional, Sequence

from meld.protocol.objects import (
    ALPHA_ONE, INF16, EpochRecord, cal_pair, canonical_bytes, object_id, score_s,
)
from meld.sync.merkle import merkle_root


def _alpha(a: int) -> Fraction:
    if type(a) is not int or not 0 < a < ALPHA_ONE:
        raise ValueError(f"alpha must be an int in (0, {ALPHA_ONE}) ppm, got {a!r}")
    return Fraction(a, ALPHA_ONE)


def n_min(alpha: int) -> int:
    """Fewest Cal records giving a finite threshold: ceil(1/alpha - 1)."""
    return math.ceil(1 / _alpha(alpha) - 1)


def quantile_threshold(scores: Sequence[int], alpha: int) -> int:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - _alpha(alpha)))
    return INF16 if k > n else sorted(scores)[k - 1]


def bin_of(n_prior: int, B_0: int) -> int:
    """0 below B_0, else 1 + floor(log2(n_prior / B_0))."""
    if B_0 < 1 or n_prior < 0:
        raise ValueError(f"bin_of needs n_prior >= 0 and B_0 >= 1, got {n_prior}, {B_0}")
    return (n_prior // B_0).bit_length()


def derive_B0(alpha: int, q, r, t) -> int:
    """Smallest power of two >= n_min(alpha) / ((1 - q) r t); q, r, t exact via str()."""
    q, r, t = (Fraction(str(v)) for v in (q, r, t))
    if not (0 <= q < 1 and 0 < r <= 1 and 0 < t <= 1):
        raise ValueError("need 0 <= q < 1 and 0 < r, t <= 1")
    return 1 << (math.ceil(n_min(alpha) / ((1 - q) * r * t)) - 1).bit_length()


def cal_root(cal_records: Iterable) -> bytes:
    """The Merkle root over the Cal ids (sync/merkle.py), the same root DIGEST ships for Cal."""
    return merkle_root(object_id(canonical_bytes(k)) for k in cal_records)


def build_epoch(cal_records: Sequence, *, scope: str, seq: int, alpha: int, alpha_c: int,
                policy: str, bins, selector_params, m_nli: bytes,
                sigma_c_q: int, author: str, created_at: int,
                mondrian_c: bool = False) -> EpochRecord:
    """The epoch record over ``cal_records``; ValueError on mixed sources, Cal from another scope,
    more than one hardware class, obs != 1, two records for one item or one tested pair, a
    carried bin other than bin_of(n_prior, B_0), or a record drawn under a selector
    (sigma_lo_q, m, M_pol) other than ``selector_params``."""
    def refuse(ok, what):
        if not ok:
            raise ValueError(f"build_epoch refused: {what}")

    sel = tuple(selector_params)
    refuse(all((k.sigma_lo_q, k.m, k.m_pol) == sel for k in cal_records),
           "Cal record drawn under another selector")
    refuse(all(k.bin == bin_of(k.n_prior, bins[0]) for k in cal_records),
           "Cal bin is not bin_of(n_prior, B_0)")
    refuse(len({cal_pair(k) for k in cal_records}) == len(cal_records),
           "two Cal records for one tested pair")
    refuse(all(k.scope == scope for k in cal_records), "Cal record from another scope")
    refuse(all(k.source == policy for k in cal_records), f"source other than policy {policy}")
    refuse(len({k.hw for k in cal_records}) == 1, "Cal scores not on exactly one hw class")
    refuse(all(k.obs == 1 for k in cal_records), "Cal score with obs != 1")
    refuse(len({k.item for k in cal_records}) == len(cal_records), "two Cal records for one item")
    table = []
    for g in sorted({k.bin for k in cal_records}):
        s = [k.s_q for k in cal_records if k.bin == g]
        s_c = [k.s_q for k in cal_records if k.bin == g and k.subtype == "CONTRADICTION"]
        table.append((g, len(s), len(s_c), quantile_threshold(s, alpha),
                      quantile_threshold(s_c, alpha_c)))
    sigma_lo_q, m, m_pol = sel
    return EpochRecord(scope=scope, seq=seq, m_nli=m_nli, m_pol=m_pol, policy=policy,
                       alpha=alpha, alpha_c=alpha_c, mondrian_c=mondrian_c,
                       cal_root=cal_root(cal_records), cal_hw=cal_records[0].hw,
                       bins=tuple(bins), table=tuple(table), sigma_lo_q=sigma_lo_q,
                       sigma_c_q=sigma_c_q, m=m, author=author, created_at=created_at)


def threshold(epoch: Optional[EpochRecord], g: int) -> int:
    """The test threshold of bin g: MERGE iff s > threshold (strict). theta_g, or
    max(theta_g, theta_c_g) with mondrian_c on; INF16 without an epoch or a row for g."""
    row = next((r for r in epoch.table if r[0] == g), None) if epoch else None
    if row is None:
        return INF16
    return max(row[3], row[4]) if epoch.mondrian_c else row[3]


def verify_epoch(epoch: EpochRecord, cal_records: Sequence, sig_records: Iterable) -> bool:
    """Recompute the epoch from the Cal set (root, refusals, every row) and check every s_q is
    the score of a single Sig observation of its pair on cal_hw under the epoch's M_nli and
    scope (hence at least the joined score: Sig only lowers s)."""
    try:
        sel = (epoch.sigma_lo_q, epoch.m, epoch.m_pol)
        rebuilt = build_epoch(cal_records, scope=epoch.scope, seq=epoch.seq, alpha=epoch.alpha,
                              alpha_c=epoch.alpha_c, policy=epoch.policy, bins=epoch.bins,
                              selector_params=sel, m_nli=epoch.m_nli,
                              sigma_c_q=epoch.sigma_c_q, author=epoch.author,
                              created_at=epoch.created_at, mondrian_c=epoch.mondrian_c)
    except ValueError:
        return False
    if canonical_bytes(rebuilt) != canonical_bytes(epoch):
        return False
    # s_q must be one observation's own two-direction score; s_q >= the joined score follows,
    # since the join takes the per-direction minimum of p_e over observations (Lemma 7)
    observed = {((r.c_x, r.c_y), score_s(r.pay_xy, r.pay_yx)) for r in sig_records
                if r.m_nli == epoch.m_nli and r.scope == epoch.scope and r.hw == epoch.cal_hw}
    return all((cal_pair(k), k.s_q) in observed for k in cal_records)


def active_epoch(epochs: Iterable[EpochRecord], scope: str) -> Optional[EpochRecord]:
    """The scope's epoch with the largest seq (ties by id). Pass authorized epochs only."""
    return max((e for e in epochs if e.scope == scope), default=None,
               key=lambda e: (e.seq, object_id(canonical_bytes(e))))


__all__ = ["n_min", "quantile_threshold", "bin_of", "derive_B0", "cal_root", "build_epoch",
           "threshold", "verify_epoch", "active_epoch"]
