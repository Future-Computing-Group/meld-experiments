"""The brain images build from the tracked tree, and the campaign image boots offline.

Every file a Dockerfile under edge/docker COPYs exists in the tree (an image built from a clean
export needs nothing from gitignored directories). The campaign image bakes in exactly the
encoder and NLI model the service loads for `minilm` / `mnli-cross-encoder` (prefetched through
the service's own loaders, so the ids cannot drift), carries the tokenizer dependency of the
DeBERTa-v3 NLI model, and sets HF_HUB_OFFLINE=1 so a brain never downloads at boot.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCKER = REPO / "edge" / "docker"


@pytest.mark.parametrize("dockerfile", sorted(p.name for p in DOCKER.glob("Dockerfile.*")))
def test_every_copied_file_is_in_the_tree(dockerfile):
    for src in re.findall(r"^COPY\s+(\S+)\s+\S+", (DOCKER / dockerfile).read_text(), re.M):
        assert (REPO / src).exists(), f"{dockerfile} copies {src}, which is not in the tree"


def test_campaign_image_bakes_both_models_and_boots_offline():
    text = (DOCKER / "Dockerfile.meld-minilm").read_text()
    assert re.search(r"^ENV HF_HUB_OFFLINE=1", text, re.M)
    assert "new_cross_encoder('mnli-cross-encoder')" in text and "encoder='minilm'" in text
    reqs = (DOCKER / "requirements-minilm.txt").read_text()
    assert "sentencepiece" in reqs and "sentence-transformers" in reqs
