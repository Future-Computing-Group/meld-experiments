"""Self-enforcing public-safety hygiene guard.

Scans every git-tracked file and FAILS if any contains internal vocabulary
that must never reach the public repo or its permanent history (host aliases,
account/project paths, scheduler-process detail, dev-era process dates,
internal lesson/ticket numbers, decision-narrative defect names).

Needles are CONSTRUCTED AT RUNTIME from base64 fragments, so this guard file
itself contains none of the forbidden tokens as bare text -- it cannot trip on
itself. The set of scanned files is ``git ls-files`` (the single source of
truth for what becomes public).

Also asserts the public/internal boundary: ``data/`` and ``plan/`` trees and
``*-LOG.md`` files are never tracked.
"""
from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _d(b64: str) -> str:
    return base64.b64decode(b64.encode("ascii")).decode("ascii")


# (label, base64_fragment, word_boundaried, treat_as_regex)
_GENERIC_NEEDLES = [
    # dev-era ISO date  20(2[5-9]|[3-9][0-9])-\d{2}-\d{2}  (process timestamp,
    # NOT data: historical data years like 2017 are excluded by the year range)
    ("dev-era-date", "MjAoMls1LTldfFszLTldWzAtOV0pLVxkezJ9LVxkezJ9", False, True),
    # statistical theatre
    ("process-jargon", "c3RhdGlzdGljYWwgdGhlYXRyZQ==", False, False),
]

# PROJECT-SPECIFIC needles: shared-host aliases, scratch/project paths, the
# cluster env profile, scheduler-accounting tool, edge-testbed alias, the
# off-limits vGPU node-family alias, billing-unit token, and ops-log filenames.
_PROJECT_NEEDLES: list[tuple[str, str, bool, bool]] = [
    ("hpc-host-a", "bWFodGk=", True, False),
    ("hpc-host-b", "cHVodGk=", True, False),
    ("scratch-project-path", "L3NjcmF0Y2gvcHJvamVjdA==", False, False),
    ("cluster-env-profile", "enotY3NjLWVudg==", False, False),
    ("scheduler-accounting", "c2VmZiA=", False, False),
    ("edge-testbed", "NWd0bg==", True, False),
    ("offlimits-node-family", "bnJvdXRlci12bQ==", False, False),
    ("billing-unit", "QlU9", False, False),
    ("ops-log-1", "T1BFUkFUSU9OU19MT0c=", False, False),
    ("ops-log-2", "T1BTTE9H", True, False),
]


# Legit public domain tokens that collide with the lesson-number shape but are
# NOT internal references. GAIA's published difficulty levels are L1/L2/L3
# (verified at source). These are documented benchmark vocabulary, not lessons.
_LESSON_ALLOWLIST = {chr(0x4C) + d for d in ("1", "2", "3")}  # {"L1","L2","L3"}

# Files whose JOB is to name excluded artifacts: a .gitignore listing an excluded
# ops-log filename OR the edge-testbed dir/script glob is the leak-prevention
# mechanism, not a leak (the ignore patterns MUST name the artifact to exclude it).
_FILENAME_NEEDLE_LABELS = {"ops-log-1", "ops-log-2", "edge-testbed"}
_FILENAME_NEEDLE_EXEMPT_FILES = {".gitignore"}


def _build_patterns() -> list[tuple[str, "re.Pattern[str]"]]:
    patterns: list[tuple[str, "re.Pattern[str]"]] = []

    # Internal lesson/ticket reference: prefix letter + 1-3 digits, isolated
    # from identifier chars (so model/hardware ids like all-MiniLM-L6-v2 / L40S
    # are NOT flagged). Prefix letter via chr() so this file embeds no literal.
    prefix = chr(0x4C)  # 'L'
    lesson_re = r"(?<![\w-])" + re.escape(prefix) + r"[0-9]{1,3}(?![\w-])"
    patterns.append(("internal-lesson-reference", re.compile(lesson_re)))

    for label, b64, boundaried, is_regex in _GENERIC_NEEDLES + _PROJECT_NEEDLES:
        needle = _d(b64)
        body = needle if is_regex else re.escape(needle)
        if boundaried:
            body = r"\b" + body + r"\b"
        patterns.append((label, re.compile(body, re.IGNORECASE)))
    return patterns


FORBIDDEN = _build_patterns()


def test_no_internal_vocabulary_in_tracked_files() -> None:
    violations: list[str] = []
    self_rel = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    for rel in _tracked_files():
        if rel == self_rel:
            continue  # this guard legitimately documents the patterns
        p = REPO_ROOT / rel
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        # skip binary artifacts (figures, embeddings): null byte => not text.
        if b"\x00" in raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        basename = rel.rsplit("/", 1)[-1]
        for i, line in enumerate(text.splitlines(), 1):
            for label, rx in FORBIDDEN:
                m = rx.search(line)
                if not m:
                    continue
                # legit-domain allowlist for the lesson-shape collision
                if label == "internal-lesson-reference" and m.group(0) in _LESSON_ALLOWLIST:
                    # only skip if EVERY lesson-shaped match on this line is allowlisted
                    if all(t in _LESSON_ALLOWLIST for t in rx.findall(line)):
                        continue
                # a .gitignore naming an excluded ops-log file is prevention, not leak
                if label in _FILENAME_NEEDLE_LABELS and basename in _FILENAME_NEEDLE_EXEMPT_FILES:
                    continue
                violations.append(f"{rel}:{i}: [{label}] -> {line.strip()[:100]!r}")
    assert not violations, (
        "Public-safety hygiene guard tripped. Tracked files must not contain "
        "internal vocabulary before the repo goes public:\n  "
        + "\n  ".join(violations)
    )


def test_data_and_plan_not_tracked() -> None:
    tracked = set(_tracked_files())
    assert not any(p.startswith(("data/", "plan/")) for p in tracked), (
        "data/ and plan/ must be gitignored, never tracked"
    )
    assert not any(p.endswith("-LOG.md") for p in tracked), (
        "*-LOG.md files must be gitignored, never tracked"
    )


if __name__ == "__main__":
    test_no_internal_vocabulary_in_tracked_files()
    test_data_and_plan_not_tracked()
    print("hygiene guard OK")
