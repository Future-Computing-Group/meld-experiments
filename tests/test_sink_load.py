"""S-4: sink state-load vs cluster size (TDD).

The single global merge sink is a serialization point; the manuscript claims its load grows
with the rate of NOVEL claims, not the number of brains N. This pins the state side of that
claim: when N brains re-publish a fixed novel corpus (replication = N), the merge sink
consolidates the redundant copies so its live store stays ~corpus-sized (flat in N), whereas a
naive union grows as N x corpus. Uses the deterministic hashing encoder so exact re-publications
merge by construction and the test is model-free; the real-encoder run is the smoke step.

Run:  PYTHONPATH=src python3 -m pytest tests/test_sink_load.py -q
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from meld.config import load_thresholds
from meld.eval.sink_load import summarize_sink_load, sink_load_row

_MINILM = os.path.join(REPO, "src", "meld", "config", "thresholds-minilm.yaml")


def test_summarize_sink_load_ratios():
    r = summarize_sink_load(n=8, n_corpus=20, merge_live=21)
    assert r["n"] == 8 and r["n_corpus"] == 20
    assert r["union_live"] == 160          # n x corpus, no consolidation
    assert r["merge_live"] == 21
    assert r["union_over_corpus"] == 8.0   # union grows as N
    assert abs(r["merge_over_corpus"] - 1.05) < 1e-9  # merge stays ~1x corpus


def test_merge_sink_stays_corpus_sized_as_replication_grows():
    cfg = load_thresholds(_MINILM)
    corpus = ["paris is the capital of france",
              "the heart pumps blood through the body",
              "water boils at one hundred degrees celsius"]
    base = sink_load_row(1, corpus, encoder="hashing", nli="lexical", cfg=cfg)
    rep8 = sink_load_row(8, corpus, encoder="hashing", nli="lexical", cfg=cfg)
    # naive union would grow 8x; the merge sink consolidates the re-publications
    assert rep8["union_live"] == 8 * len(corpus)
    assert rep8["merge_live"] <= base["merge_live"] + 1      # flat in N (no growth)
    assert rep8["merge_live"] < rep8["union_live"]           # strictly compacter
