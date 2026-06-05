"""Smoke tests for the generic HPC runner scaffolding (hpc/).

The orchestrator ships as a generic runner + a committed, sanitized example config;
the real site config (hpc/site.env) is gitignored. These tests pin: the example config
defines every key the runner needs and only with placeholder values; the shell scripts
are syntactically valid; and the committed scaffolding carries no real site values.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

HPC = Path(__file__).resolve().parents[1] / "hpc"

REQUIRED_KEYS = {
    "SMP_ACCOUNT", "SMP_PARTITION", "SMP_GRES", "SMP_TIME", "SMP_MEM", "SMP_CPUS",
    "SMP_REPO", "SMP_SCRATCH", "SMP_OLLAMA_ROOT", "SMP_PY", "SMP_SITE_PROFILE",
    "SMP_MODELS", "SMP_PER_LEVEL", "SMP_ENCODER",
}


def _example_assignments() -> dict[str, str]:
    text = (HPC / "site.example.env").read_text()
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Z_]+)=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"')
    return out


def test_example_config_defines_every_required_key():
    keys = set(_example_assignments())
    missing = REQUIRED_KEYS - keys
    assert not missing, f"site.example.env missing keys: {missing}"


def test_example_config_uses_only_placeholders_no_real_site_values():
    a = _example_assignments()
    # placeholders, not a real account / partition / absolute site path
    assert a["SMP_ACCOUNT"].startswith("your_")
    assert a["SMP_PARTITION"].startswith("your_")
    assert a["SMP_REPO"].startswith("/path/")
    assert a["SMP_SCRATCH"].startswith("/path/")
    # (the broader internal-token scan over all tracked files, including hpc/, is the
    # job of the public-safety hygiene guard in test_repo_hygiene.py.)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("script", ["submit.sh", "run_memdelta.sbatch"])
def test_shell_scripts_are_syntactically_valid(script):
    r = subprocess.run(["bash", "-n", str(HPC / script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_runner_sbatch_has_no_hardcoded_slurm_account_or_partition():
    # account/partition come from site.env via submit.sh, never hardcoded in the runner.
    text = (HPC / "run_memdelta.sbatch").read_text()
    assert "#SBATCH --account" not in text
    assert "#SBATCH --partition" not in text
    assert 'source "$SITE_ENV"' in text or "source \"${SMP_SITE_ENV" in text
