"""R2 performance fix (TDD): a shared embedding cache_dir threaded through the sweep so the
same sentence embeds ONCE across builds, instead of being re-embedded per build_condition
(the redundant-load slowness). Run:
  PYTHONPATH=src python3 -m pytest tests/test_recall_sweep_cache.py -q
"""
import os

from meld.eval.recall_sweep import recall_rows
from meld.merge.signals import EmbeddingSimilarity
from meld.agent.mem_conditions import _gold_sentences

from test_agent_meld_memory import agent_epoch


def test_recall_rows_threads_shared_cache_dir_so_embeddings_are_reused(tmp_path):
    from meld.agent.hotpot_loader import load_hotpot_fixture
    epoch = agent_epoch("minilm", "lexical")
    tasks = load_hotpot_fixture(n=2)
    cache = str(tmp_path / "emb")
    # build TWO conditions over the same tasks with a shared cache_dir: the second
    # condition's brains must reuse the first's cached sentence embeddings.
    recall_rows(tasks, overlaps=[0.5], ks=[5],
                conditions=["centralized", "distributed_merge"],
                encoder="minilm", nli="lexical", epoch=epoch, cache_dir=cache)
    files = os.listdir(cache)
    assert files, "no embeddings were cached -> cache_dir was not threaded through"
    # a fresh encoder over a sentence the sweep already embedded loads it from the shared
    # disk cache (a cache hit), proving cross-build reuse rather than re-embedding.
    sent = [s for _, s in _gold_sentences(tasks[0])][0]
    es = EmbeddingSimilarity(encoder="minilm", cache_dir=cache)
    es.sigma(sent, sent)
    assert es.cache_hits > 0, "the shared cache was not reused"
