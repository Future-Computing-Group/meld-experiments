"""Experiment progress helper (TDD): a one-line, zero-friction status emitter so every experiment
loop can write per-iteration progress to a status file readable mid-run. Run:
  PYTHONPATH=src python3 -m pytest tests/test_progress.py -q
"""
import os
import tempfile

from meld.eval.progress import status_emitter


def test_status_emitter_writes_readable_status_file_with_label_and_elapsed():
    p = os.path.join(tempfile.mkdtemp(), "status.log")
    emit = status_emitter(p, label="r2")
    emit("cell 1/16 task 3/12")
    content = open(p).read()
    assert "[r2]" in content and "cell 1/16 task 3/12" in content and "elapsed" in content


def test_status_emitter_overwrites_to_latest_state():
    p = os.path.join(tempfile.mkdtemp(), "status.log")
    emit = status_emitter(p, label="run")
    emit("cell 1/16")
    emit("cell 2/16")
    content = open(p).read()
    assert "cell 2/16" in content and "cell 1/16" not in content   # latest only, readable mid-run
