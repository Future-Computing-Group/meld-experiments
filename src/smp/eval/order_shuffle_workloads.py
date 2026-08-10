"""S3: the two order-shuffle workloads.

(i) The running-example message set -- the four messages walked in
    sections/04-merge-semantics.tex + figures/fig-running-example.tex + the
    S1 memo's table (EU paraphrase merge, cross-boundary US relate, intra-EU
    conflict, status overrule) -- against a receiver that already holds c1
    ACTIVE, exactly the figure's starting state.
(ii) A gold-derived workload: ~4 texts (canonical / paraphrase / related /
     conflict) per seed from smp.merge.m2_gold, spanning all four
     pair-adjudication classes (MERGE/RELATE/CONFLICT from within a seed;
     NO_RELATION/INSERT arises for free across seeds on different topics),
     ingested into N synthetic source brains and concatenated into one flat
     delta workload -- the funnel-merge shape (smp.eval.topology).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from smp.agent.memory import _unit_from_text, claim_key
from smp.brain.store import Brain
from smp.merge.m2_gold import _SEEDS
from smp.proto import MemoryUnit

# ---------------------------------------------------------------------------
# (i) running example
# ---------------------------------------------------------------------------

EU_CELL = "jurisdiction.eu"
US_CELL = "jurisdiction.us"

C1_CANONICAL = "Customer data must be retained for seven years"
C1_PARAPHRASE = "We are required to retain customer data for seven years"   # message 1
C2_CONTRADICTION = "Customer data must be deleted after two years"          # message 3

RUNNING_EXAMPLE_PROBE_QUERIES = (
    "How long must customer data be retained?",
    "When must customer data be deleted?",
)


def build_running_example_workload() -> Tuple[List, List[MemoryUnit]]:
    """Returns ``(workload, seed_units)``. ``seed_units`` is B's fixed
    starting state (holds c1 ACTIVE, per Figure~\\ref{fig:running});
    ``workload`` is the three claim deltas (messages 1-3) plus message 4 (the
    EU authority's overrule on c2) as a bare status event -- all delivered in
    whatever order the driver shuffles."""
    seed_units = [_unit_from_text(C1_CANONICAL, EU_CELL)]
    msg1 = _unit_from_text(C1_PARAPHRASE, EU_CELL)
    msg2 = _unit_from_text(C1_PARAPHRASE, US_CELL)   # "the very same sentence, scoped US"
    msg3 = _unit_from_text(C2_CONTRADICTION, EU_CELL)
    c2_key = claim_key(C2_CONTRADICTION, EU_CELL)
    msg4 = (c2_key, "C.overrule.c2", "OVERRULED")    # authority C overrules c2
    return [msg1, msg2, msg3, msg4], seed_units


# ---------------------------------------------------------------------------
# (ii) gold-derived workload
# ---------------------------------------------------------------------------

GOLD_SEED_IDS: Tuple[str, ...] = ("geo-paris", "geo-everest", "lit-1984", "chem-water")

GOLD_PROBE_QUERIES = tuple(
    canon for sid, _diff, canon, _para, _rel, _conf in _SEEDS if sid in GOLD_SEED_IDS
)


def _seed_row(sid: str):
    for row in _SEEDS:
        if row[0] == sid:
            return row
    raise KeyError(sid)


def build_gold_derived_workload(seed_ids: Sequence[str] = GOLD_SEED_IDS, *,
                                encoder: str, n_sources: int = 3,
                                cache_dir: str | None = None) -> List[MemoryUnit]:
    """~4 texts per seed (canonical/paraphrase/related/conflict; single cell,
    so distinct claim keys follow from distinct content -- S1 memo 2b),
    ingested as novel inserts into ``n_sources`` synthetic source brains
    (round-robin), then every source's extracted delta concatenated into ONE
    flat workload."""
    from smp.protocol.merge_on_receive import _make_patch

    texts: List[str] = []
    for sid in seed_ids:
        _sid, _diff, canon, para, rel, conf = _seed_row(sid)
        texts.extend([canon, para, rel, conf])

    sources = [Brain(encoder=encoder, cache_dir=cache_dir) for _ in range(n_sources)]
    for i, text in enumerate(texts):
        src = sources[i % n_sources]
        unit = _unit_from_text(text)
        for claim in unit.claims:
            src.apply(_make_patch("INSERT", unit, claim, gates=["novel"]))

    workload: List[MemoryUnit] = []
    for src in sources:
        workload.extend(u for u in src.extract_delta() if isinstance(u, MemoryUnit))
    return workload


__all__ = [
    "build_running_example_workload", "RUNNING_EXAMPLE_PROBE_QUERIES",
    "build_gold_derived_workload", "GOLD_SEED_IDS", "GOLD_PROBE_QUERIES",
]
