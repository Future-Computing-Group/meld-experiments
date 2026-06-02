"""Resumable campaign checkpoint (rung-4, decision 6: flaky connectivity).

A durable JSONL log of completed cells, appended the instant each cell finishes -- so a
control-link drop or VM1 restart loses nothing: a fresh instance replays the file and the
orchestrator skips already-done cells (``pending``). Results live on the VM disk, never
only in the laptop session. Last write wins for a re-run cell.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List


class CampaignCheckpoint:
    def __init__(self, path: str):
        self.path = path
        self._done: Dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._done[rec["cell"]] = rec["result"]   # last line wins

    def done(self, cell_id: str) -> bool:
        return cell_id in self._done

    def record(self, cell_id: str, result: dict) -> None:
        self._done[cell_id] = result
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cell": cell_id, "result": result}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())          # durable before we proceed (survive a hard drop)

    def completed_cells(self) -> set:
        return set(self._done)

    def results(self) -> Dict[str, dict]:
        return dict(self._done)

    def pending(self, all_cells: List[str]) -> List[str]:
        return [c for c in all_cells if c not in self._done]


__all__ = ["CampaignCheckpoint"]
