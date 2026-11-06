"""Approximate candidate index over int8 claim embeddings (FAISS).

The protocol's candidate set is exact (signals.topm_exact): the fold, the calibration builder and
the admission path all rank by exact sigma_q, so replicas agree whatever their history (design
§3.1). This index is outside that path. It serves two uses: the scalability instrument
(eval/index_scaling.py), and an optional admission-path pre-filter (``Admission(index=...)``) whose
rows are then re-ranked exactly, so the admission decision equals the exact one whenever the exact
top-m lies inside the pre-filter, and the pass repairs any miss.

``CandidateIndex(dim, kind="flat"|"hnsw")``: ``faiss.IndexIDMap`` over ``IndexFlatIP`` or
``IndexHNSWFlat`` (inner product), fed the int8 rows cast to float32 and L2-normalised, so the inner
product is the cosine (sigma_q up to float rounding). ``add(Q, ids=None)`` appends int8 rows (ids
default to the running row count, i.e. canonical-order row positions); ``search(q, k)`` returns up
to k ids, best first. HNSW graph construction depends on insertion order and threads, which is why
the protocol never reads this index.

faiss is imported on first use, never at module import: the macOS faiss-cpu and torch wheels each
bundle their own OpenMP runtime, and a process that loads both aborts. Keep FAISS out of any process
that runs a torch model (the sentence encoder or the NLI cross-encoder) unless the two share one
OpenMP runtime.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def _faiss():
    import faiss
    return faiss


def _unit_rows(Q: np.ndarray) -> np.ndarray:
    if not isinstance(Q, np.ndarray) or Q.dtype != np.int8:
        raise TypeError("int8 rows required")
    X = np.ascontiguousarray(np.atleast_2d(Q), dtype=np.float32)
    _faiss().normalize_L2(X)                    # zero rows stay zero
    return X


class CandidateIndex:
    def __init__(self, dim: int, *, kind: str = "hnsw", M: int = 32, ef_construction: int = 200,
                 ef_search: int = 128):
        faiss = _faiss()
        if kind == "flat":
            base = faiss.IndexFlatIP(dim)
        elif kind == "hnsw":
            base = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efConstruction = ef_construction
            base.hnsw.efSearch = ef_search
        else:
            raise ValueError(f"kind {kind!r}: 'flat' or 'hnsw'")
        self.kind, self.base, self.index = kind, base, faiss.IndexIDMap(base)

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, Q: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        X = _unit_rows(Q)
        if ids is None:
            ids = np.arange(len(self), len(self) + len(X))
        self.index.add_with_ids(X, np.asarray(ids, dtype=np.int64))

    def search(self, q: np.ndarray, k: int) -> List[int]:
        if not len(self) or k <= 0:
            return []
        _, I = self.index.search(_unit_rows(q), min(k, len(self)))
        return [i for i in I[0].tolist() if i >= 0]

    def nbytes(self) -> int:
        """Serialized size of the index (vectors, graph and id map)."""
        return int(_faiss().serialize_index(self.index).nbytes)


__all__ = ["CandidateIndex"]
