"""The one record every benchmark source emits.

A Pair is two short claim texts plus provenance. ``label`` is the MELD verdict
when the source fixes it (``label_provenance`` source / construction), a
pre-label that the annotation panel must confirm (``source_filtered``: a source
label that survived rule filters but was measured unreliable), or ``None`` when
it must come from annotation (``to_annotate``). S7_organic is the hook for
pairs MELD itself adjudicated (contradiction flags and others), added later. ``mergeable``
carries the source's MERGE versus non-MERGE decision separately, because for
several strata the source fixes mergeability but not the four-way verdict.
``pair_id`` is an opaque content hash, so it can be shown to blinded annotators.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

LABELS = ("MERGE", "RELATE", "CONFLICT", "NO_RELATION")
SUBTYPES = ("SUPERSESSION", "CORRECTION", "ELABORATION", "SCOPE", "ADVERSARIAL_NONPARA")
PROVENANCE = ("source", "source_filtered", "derived", "construction", "to_annotate")
STRATA = ("S0_gold", "S1_paws_para", "S2_paws_nonpara", "S3_vitc_nonfactual",
          "S4_vitc_factual", "S5_lme_ku", "S6_scifact_conflict", "S7_organic",
          "S8_sick_para", "S8_sick_oneway", "S8_sick_contra")
BADGE_KEYS = ("scope", "time")


def pair_id_of(source: str, source_item_id: str, text_a: str, text_b: str) -> str:
    blob = "\x00".join((source, source_item_id, text_a, text_b)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class Pair:
    pair_id: str
    text_a: str
    text_b: str
    source: str
    source_item_id: str
    stratum: str
    label: Optional[str]
    label_provenance: str
    mergeable: Optional[bool]
    subtype: Optional[str] = None
    origin_split: Optional[str] = None          # the source's official split: dev, test or None
    badges_a: Dict[str, str] = field(default_factory=dict)
    badges_b: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.text_a.strip() or not self.text_b.strip():
            raise ValueError("empty claim text")
        if self.stratum not in STRATA:
            raise ValueError(f"unknown stratum {self.stratum!r}")
        if self.label_provenance not in PROVENANCE:
            raise ValueError(f"unknown label_provenance {self.label_provenance!r}")
        if (self.label is None) != (self.label_provenance == "to_annotate"):
            raise ValueError("label must be None exactly when label_provenance is to_annotate")
        if self.label is not None and self.label not in LABELS:
            raise ValueError(f"unknown label {self.label!r}")
        if self.subtype is not None and self.subtype not in SUBTYPES:
            raise ValueError(f"unknown subtype {self.subtype!r}")
        if self.origin_split not in (None, "dev", "test"):
            raise ValueError(f"unknown origin_split {self.origin_split!r}")
        for b in (self.badges_a, self.badges_b):
            if set(b) - set(BADGE_KEYS):
                raise ValueError(f"badges outside {BADGE_KEYS}: {sorted(b)}")
        if self.pair_id != pair_id_of(self.source, self.source_item_id, self.text_a, self.text_b):
            raise ValueError("pair_id does not match content")

    def to_dict(self) -> dict:
        return asdict(self)


def make_pair(*, text_a: str, text_b: str, source: str, source_item_id: str, **kw) -> Pair:
    return Pair(pair_id=pair_id_of(source, source_item_id, text_a, text_b), text_a=text_a,
                text_b=text_b, source=source, source_item_id=source_item_id, **kw)
