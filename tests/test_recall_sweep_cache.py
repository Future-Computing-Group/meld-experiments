"""R2 performance fix (TDD): a shared embedding cache_dir threaded through the sweep so the
same sentence embeds ONCE across builds, instead of being re-embedded per build_condition
(the redundant-load slowness). Run:
  PYTHONPATH=src python3 -m pytest tests/test_recall_sweep_cache.py -q
"""
import os

from smp.eval.recall_sweep import recall_rows
from smp.merge.signals import EmbeddingSimilarity
from smp.agent.mem_conditions import _gold_sentences

_CFG = os.path.join(os.path.dirname(__file__), "..", "src", "smp", "config",
                    "thresholds-minilm.yaml")


def test_recall_rows_threads_shared_cache_dir_so_embeddings_are_reused(tmp_path):
    from smp.config import load_thresholds
    from smp.agent.hotpot_loader import load_hotpot_fixture
    cfg = load_thresholds(_CFG)
    tasks = load_hotpot_fixture(n=2)
    cache = str(tmp_path / "emb")
    # build TWO conditions over the same tasks with a shared cache_dir: the second
    # condition's brains must reuse the first's cached sentence embeddings.
    recall_rows(tasks, overlaps=[0.5], ks=[5],
                conditions=["centralized", "distributed_merge"],
                encoder="minilm", nli="lexical", cfg=cfg, cache_dir=cache)
    files = os.listdir(cache)
    assert files, "no embeddings were cached -> cache_dir was not threaded through"
    # a fresh encoder over a sentence the sweep already embedded loads it from the shared
    # disk cache (a cache hit), proving cross-build reuse rather than re-embedding.
    sent = [s for _, s in _gold_sentences(tasks[0])][0]
    es = EmbeddingSimilarity(encoder="minilm", cache_dir=cache)
    es.sigma(sent, sent)
    assert es.cache_hits > 0, "the shared cache was not reused"
