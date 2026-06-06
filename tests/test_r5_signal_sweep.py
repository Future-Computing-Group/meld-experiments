"""R5: encoder/NLI alias extensions for the merge-signal sensitivity sweep (TDD).

Resolution is tested WITHOUT downloading weights (lazy load); the actual cross-model
inference is a smoke-ladder step, not a unit test. Run:
  PYTHONPATH=src python3 -m pytest tests/test_r5_signal_sweep.py -q
"""
import pytest

from smp.merge import signals
from smp.merge.signals import _make_encoder, _SentenceTransformerEncoder


def test_make_encoder_resolves_added_encoder_aliases():
    for alias in ("mpnet", "minilm-l12"):
        enc = _make_encoder(alias)
        assert isinstance(enc, _SentenceTransformerEncoder)
        assert enc.rung == alias
        assert _SentenceTransformerEncoder._MODEL_IDS[alias].startswith("sentence-transformers/")
    # distinct aliases map to distinct model ids (so a sweep cell is a real swap)
    ids = {_SentenceTransformerEncoder._MODEL_IDS[a] for a in ("minilm", "mpnet", "minilm-l12")}
    assert len(ids) == 3
    with pytest.raises(ValueError):
        _make_encoder("no-such-encoder")


def test_chi_resolves_added_nli_aliases_via_model_map():
    # an alias->model-id map drives chi, so adding an NLI is data, not a new branch
    assert "mnli-cross-encoder" in signals._NLI_MODEL_IDS
    for alias in ("roberta-nli", "deberta-base-nli"):
        assert alias in signals._NLI_MODEL_IDS
        assert signals._NLI_MODEL_IDS[alias] != signals._NLI_MODEL_IDS["mnli-cross-encoder"]
    with pytest.raises(ValueError):
        signals.chi("a", "b", model="no-such-nli")
