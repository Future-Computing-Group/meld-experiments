"""S-4: sink state-load vs cluster size --- does the global merge sink stay bounded as N grows?

The single global merge sink is a serialization point. Its per-delta compute cost is the M1
knee (one embedding + one inference call), independent of N. Its STATE load is what this harness
measures: when N brains redundantly re-publish a fixed novel corpus (replication factor N), the
merge-on-receive sink consolidates the duplicates, so its status-live store stays ~corpus-sized
regardless of N, whereas a naive union grows as N x corpus. This is the empirical state-side of
"sink load grows with the rate of novel claims, not the number of brains": novel content, not
cluster size, sets the sink's footprint.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from meld.agent.mem_conditions import _new_memory  # the shared SMPMemory builder
from meld.config import Thresholds


def summarize_sink_load(n: int, n_corpus: int, merge_live: int) -> Dict:
    """Pure: merge-sink live store vs the naive-union baseline (N x corpus)."""
    union_live = n * n_corpus
    return {
        "n": n,
        "n_corpus": n_corpus,
        "merge_live": merge_live,
        "union_live": union_live,
        "merge_over_corpus": round(merge_live / n_corpus, 2) if n_corpus else 0.0,
        "union_over_corpus": round(union_live / n_corpus, 2) if n_corpus else 0.0,
    }


def sink_load_row(n: int, corpus: List[str], *, encoder: str, nli: str,
                  cfg: Thresholds, cache_dir: Optional[str] = None) -> Dict:
    """Re-publish ``corpus`` ``n`` times into one merge sink (replication = N brains
    re-discovering the same facts); report the sink's live store vs the union baseline."""
    mem = _new_memory(encoder, nli, cfg, cache_dir)
    for _ in range(n):
        for text in corpus:
            mem.write(text)
    merge_live = mem.brain.live_unit_count()
    return summarize_sink_load(n, len(corpus), merge_live)


__all__ = ["summarize_sink_load", "sink_load_row"]
