"""Claim-pair benchmark: the common Pair schema (validation and opaque ids)."""
import re

import pytest

from meld.bench.schema import Pair, make_pair


def _p(**kw):
    base = dict(text_a="Paris is the capital of France.", text_b="France's capital is Paris.",
                source="paws", source_item_id="paws:dev:1", stratum="S1_paws_para",
                label="MERGE", label_provenance="source", mergeable=True)
    base.update(kw)
    return make_pair(**base)


def test_valid_pair_round_trips_through_dict():
    p = _p(badges_a={"time": "2023/05/23"}, badges_b={"time": "2023/05/30"})
    assert Pair(**p.to_dict()) == p


def test_pair_id_is_opaque_and_content_derived():
    p = _p()
    assert re.fullmatch(r"[0-9a-f]{16}", p.pair_id)
    assert "paws" not in p.pair_id and "S1" not in p.pair_id
    assert _p().pair_id == p.pair_id
    assert _p(text_b="Another sentence entirely.").pair_id != p.pair_id


@pytest.mark.parametrize("bad", [
    dict(label="SUPERSEDE"),                                  # not a MELD verdict
    dict(label=None),                                         # unlabelled but claims a source label
    dict(subtype="TYPO"),                                     # unknown subtype
    dict(label_provenance="guess"),                           # unknown provenance
    dict(text_a="   "),                                       # empty text
    dict(badges_a={"colour": "red"}),                         # badge outside scope/time
    dict(stratum="S9_unknown"),                               # unknown stratum
    dict(origin_split="validation"),                          # official split must be dev/test/None
])
def test_invalid_pair_is_refused(bad):
    with pytest.raises(ValueError):
        _p(**bad)


def test_to_annotate_pair_has_no_label():
    p = _p(stratum="S4_vitc_factual", source="vitaminc", label=None,
           label_provenance="to_annotate", mergeable=False)
    assert p.label is None and p.mergeable is False
    with pytest.raises(ValueError):
        _p(stratum="S4_vitc_factual", label="CONFLICT", label_provenance="to_annotate")


def test_tampered_pair_id_is_detected():
    d = _p().to_dict()
    d["pair_id"] = "0" * 16
    with pytest.raises(ValueError, match="pair_id"):
        Pair(**d)


def test_source_filtered_provenance_and_organic_hook_stratum_accepted():
    p = _p(stratum="S3_vitc_nonfactual", source="vitaminc", label_provenance="source_filtered")
    assert p.label == "MERGE"
    assert _p(stratum="S7_organic", source="meld_organic", label=None, label_provenance="to_annotate",
              mergeable=None).stratum == "S7_organic"
