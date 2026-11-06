"""Provenance in a git-archive export (no .git, as shipped to a remote host): the commit comes from
MELD_GIT_SHA, the convention scripts/run_h4_determinism.py already uses; a checkout's own HEAD wins."""
from types import SimpleNamespace


def test_git_sha_falls_back_to_meld_git_sha_without_a_checkout(monkeypatch):
    from meld.eval import topology
    monkeypatch.setattr(topology.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    monkeypatch.setenv("MELD_GIT_SHA", "70287aa")
    topology._git_sha.cache_clear()
    try:
        assert topology._git_sha() == "70287aa"
    finally:
        topology._git_sha.cache_clear()
