"""Zero-friction experiment progress: a status emitter every experiment loop uses, unconditionally.

The recurring failure is a long experiment loop that runs SILENT, so a wedge or
silent failure only surfaces after minutes of blind waiting. The fix is to make
per-iteration progress one line of code, applied at WRITE time to every loop (not gated on
a runtime ">5 min" prediction): ``status_emitter`` returns an ``emit(msg)`` that overwrites
a status file (readable mid-run via ``cat``/``tail``) and prints flushed stderr, each line
stamped with elapsed seconds.

    emit = status_emitter("outputs/r2-status.log", label="r2")
    for i, cell in enumerate(cells, 1):
        ...
        emit(f"cell {i}/{n}")
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable


def status_emitter(status_path: str, label: str = "run") -> Callable[[str], None]:
    """Return ``emit(msg)``: writes ``[label] msg | elapsed Ns`` to ``status_path``
    (overwritten -> latest state, readable while the run is in progress) and to flushed
    stderr. The directory is created if needed."""
    d = os.path.dirname(status_path)
    if d:
        os.makedirs(d, exist_ok=True)
    t0 = time.perf_counter()

    def emit(msg: str) -> None:
        line = f"[{label}] {msg} | elapsed {time.perf_counter() - t0:.1f}s"
        print(line, file=sys.stderr, flush=True)
        with open(status_path, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return emit


__all__ = ["status_emitter"]
