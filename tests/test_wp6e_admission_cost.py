"""PH-18 / PH-18b per-delta admission cost (scripts/run_wp6e_admission_cost.py): --device moves the encoder and
the NLI model, CUDA is synchronised before every timer read, the device is stamped, and a run's decisions are
compared with a reference run's rows (counted, never assumed equal). The real-model tests need the gitignored
dev key, the dev-signed HotpotQA r4 epoch, the committed laptop rows and the cached models and dataset."""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import run_wp6e_admission_cost as A  # noqa: E402

ROWS = os.path.join(REPO, "outputs", "wp6e", "ph18", "admission-cost-rows.jsonl")
HAVE = all(os.path.exists(os.path.join(REPO, p)) for p in (A.EPOCH[0], "outputs/calibration-labelled/key", ROWS))
OLD_KEYS = {"git_sha", "epoch_id", "cpu", "machine", "torch_threads", "omp", "torch", "device", "batch_size", "gpu",
            "world", "texts", "timed", "warmup", "refolds_in_timed_window", "m", "summary", "nli_calls_per_delta",
            "share_of_mean_total", "decisions", "max_n_prior"}


def test_compare_decisions_counts_differences_and_unmatched_rows():
    ref = [{"i": 20, "decision": "RELATE"}, {"i": 21, "decision": "MERGE"}, {"i": 22, "decision": "INSERT"}]
    rows = [{"i": 20, "decision": "RELATE"}, {"i": 21, "decision": "RELATE"}, {"i": 23, "decision": "INSERT"}]
    assert A.compare_decisions(rows, ref) == {"compared": 2, "differing": 1, "only_this_run": 1, "only_reference": 1,
                                              "changes": {"MERGE->RELATE": 1}}


def _run(tmp_path, device, n=60):
    out = A.main(["--device", device, "--max-texts", str(n), "--out-dir", str(tmp_path), "--reference-rows", ROWS])
    on_disk = json.load(open(tmp_path / "admission-cost.json"))
    assert on_disk["decisions_vs_reference"] == out["decisions_vs_reference"]
    return out


@pytest.mark.skipif(not HAVE, reason="dev key, dev-signed epoch or laptop PH-18 rows not present")
def test_cpu_run_keeps_the_schema_and_is_compared_with_the_laptop_rows(tmp_path):
    out = _run(tmp_path, "cpu")
    assert OLD_KEYS <= set(out)
    assert out["device"] == "cpu" and out["encoder_device"] == "cpu" and out["nli_device"] == "cpu"
    assert out["timed"] == 40 and out["max_texts"] == 60
    cmp = out["decisions_vs_reference"]
    assert cmp["compared"] == 40 and cmp["only_this_run"] == 0
    assert cmp["differing"] == 0        # CPU and the laptop rows agree on this prefix (measured, not assumed)


@pytest.mark.skipif(not HAVE, reason="dev key, dev-signed epoch or laptop PH-18 rows not present")
def test_cuda_run_is_synchronised_stamped_and_compared(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    out = _run(tmp_path, "cuda", n=30)
    assert out["device"] == "cuda" and out["cuda_synchronized"] is True
    assert out["gpu"] == torch.cuda.get_device_name(0) and out["cuda"] == torch.version.cuda
    assert out["encoder_device"].startswith("cuda") and out["nli_device"].startswith("cuda")
    assert out["decisions_vs_reference"]["compared"] == out["timed"]
