"""G3 — the PUBLIC tool surface for GAIA L2 achievability.

GAIA L2 tasks need real tools (web browse, file/document reading, calculation).
These are the TDD spec for ``smp.agent.tools``' additions: a web-search tool with
an OFFLINE cache/replay mode (CSC compute nodes have no internet), a file/document
reader (PDF/CSV/text), and a sandboxed calculator (no eval/exec of arbitrary code).

The load-bearing SMP decision (documented in the campaign plan): a tool that reads
EVIDENCE (web page / attached file) feeds its content THROUGH the SMP Brain as a
merged observation (PUBLISH -> merge-on-receive -> Patch), not only as a raw tool
string. So the brain accrues the evidence and a later recall can retrieve it. The
calculator is pure computation and does NOT write to the brain.

All offline, deterministic, no network. The web tool in this test runs in REPLAY
mode against a prefetched cache; LIVE mode is the gated/networked path.
"""
from __future__ import annotations

import csv
import os

import pytest

from smp.agent.memory import SMPMemory
from smp.agent.tools import (
    CalculatorTool,
    FileReaderTool,
    OfflineWebSearchTool,
)
from smp.config import Thresholds, load_thresholds


def _calibrated_cfg() -> Thresholds:
    # delegate to the canonical encoder-matched calibrated thresholds (was a
    # duplicated inverted-0.5/0.55 copy that tripped the ordering invariant)
    from smp.agent.run_hotpot_gate import _calibrated_cfg as _canonical
    return _canonical("hashing")


def _mem() -> SMPMemory:
    return SMPMemory(encoder="hashing", nli="lexical", cfg=_calibrated_cfg())


# --------------------------------------------------------------------------
# smolagents-shaped interface (name / description / __call__)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls,kw", [
    (CalculatorTool, {}),
])
def test_tools_have_smolagents_shape(cls, kw):
    t = cls(**kw)
    assert isinstance(t.name, str) and t.name
    assert isinstance(t.description, str) and t.description
    assert callable(t)


# --------------------------------------------------------------------------
# CalculatorTool: sandboxed arithmetic, NO eval of arbitrary code
# --------------------------------------------------------------------------


def test_calculator_evaluates_arithmetic():
    calc = CalculatorTool()
    assert calc("2 + 3 * 4") == "14"
    assert calc("(1000 - 1) / 3") == "333"
    assert calc("2 ** 10") == "1024"


def test_calculator_is_sandboxed_against_code_injection():
    """A sandboxed calculator must REFUSE names/calls/attribute access -- it must
    not be able to import, open files, or reach builtins."""
    calc = CalculatorTool()
    for hostile in ["__import__('os').system('echo x')",
                    "open('/etc/passwd').read()",
                    "().__class__.__bases__",
                    "exec('x=1')"]:
        out = calc(hostile)
        assert "error" in out.lower() or "refused" in out.lower(), \
            f"calculator did not refuse: {hostile!r} -> {out!r}"


# --------------------------------------------------------------------------
# FileReaderTool: text / CSV / PDF, content goes THROUGH the SMP brain
# --------------------------------------------------------------------------


def test_file_reader_reads_text_and_writes_through_the_brain(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("The launch venue is Helsinki. The budget rose this quarter.")
    mem = _mem()
    reader = FileReaderTool(memory=mem)
    out = reader(str(p))
    assert "launch venue" in out.lower()
    # content reached the brain as a merged observation (a Patch in the log)
    assert len(mem.brain.audit_log) >= 1, "file content did not reach the brain"
    assert any(p_.decision in ("INSERT", "MERGE", "RELATE", "CONFLICT")
               for p_ in mem.brain.audit_log)
    # and a later recall can retrieve it
    hits = mem.recall("where is the launch", k=3)
    assert any("helsinki" in h.lower() or "launch" in h.lower() for h in hits)


def test_file_reader_reads_csv(tmp_path):
    p = tmp_path / "data.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", "population"])
        w.writerow(["Paris", "2148000"])
        w.writerow(["Lyon", "513000"])
    mem = _mem()
    reader = FileReaderTool(memory=mem)
    out = reader(str(p))
    assert "Paris" in out and "2148000" in out
    assert len(mem.brain.audit_log) >= 1


def test_file_reader_missing_file_is_handled(tmp_path):
    mem = _mem()
    reader = FileReaderTool(memory=mem)
    out = reader(str(tmp_path / "nope.txt"))
    assert "error" in out.lower() or "not found" in out.lower()
    # a failed read must NOT write garbage into the brain
    assert len(mem.brain.audit_log) == 0


# --------------------------------------------------------------------------
# OfflineWebSearchTool: replay from a prefetched cache (CSC has no internet)
# --------------------------------------------------------------------------


def test_web_search_replays_from_cache_and_writes_through_the_brain():
    cache = {
        "capital of france":
            "Paris is the capital and most populous city of France.",
    }
    mem = _mem()
    web = OfflineWebSearchTool(memory=mem, cache=cache)
    out = web("capital of France")
    assert "Paris" in out
    # the retrieved page content was merged into the brain
    assert len(mem.brain.audit_log) >= 1
    hits = mem.recall("what is the capital of France", k=3)
    assert any("paris" in h.lower() for h in hits)


def test_web_search_cache_miss_is_explicit_not_a_fabrication():
    """A cache MISS in offline/replay mode must say so -- never fabricate a page
    (that would be a hallucinated source). And it must not write to the brain."""
    mem = _mem()
    web = OfflineWebSearchTool(memory=mem, cache={})
    out = web("a query that was never prefetched")
    assert "miss" in out.lower() or "no cached" in out.lower() \
        or "not prefetched" in out.lower()
    assert len(mem.brain.audit_log) == 0


def test_web_search_live_mode_is_refused_offline():
    """LIVE mode must be an explicit, gated path. Constructed with mode='live'
    but no fetcher, a call must refuse rather than attempt a network call (the
    offline/test path stays clean; CSC compute nodes have no internet)."""
    mem = _mem()
    web = OfflineWebSearchTool(memory=mem, cache={}, mode="live")
    out = web("anything")
    assert "live" in out.lower() and ("refus" in out.lower()
                                      or "no fetcher" in out.lower()
                                      or "unavailable" in out.lower())
