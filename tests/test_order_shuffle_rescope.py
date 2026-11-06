"""run_order_shuffle --rescope: a deployed epoch's table on the running example's policy (PH-11)."""
from __future__ import annotations

import pytest

from meld.config import save_epoch
from meld.protocol.objects import canonical_bytes, object_id
from meld.sync.wire_delta import group_key

from test_agent_meld_memory import agent_epoch
from test_harnesses import _load


def test_rescope_runs_the_running_example_under_a_deployed_table(tmp_path):
    ep = agent_epoch(theta=0.0)                                   # scope agent, DEFAULT_POL
    f = str(tmp_path / "ep.json")
    save_epoch(ep, group_key({}), f)
    run = _load("run_order_shuffle.py")
    with pytest.raises(ValueError):                               # another policy: the store refuses it
        run.main(["--encoder", "hashing", "--nli", "lexical", "--workload", "running", "--k", "2", "--epoch", f])
    rep = run.main(["--encoder", "hashing", "--nli", "lexical", "--workload", "running", "--k", "3",
                    "--epoch", f, "--rescope"])
    assert rep["epoch_rescoped_from"] == object_id(canonical_bytes(ep)).hex()
    assert rep["epoch_id"] != rep["epoch_rescoped_from"] and rep["distinct_fingerprints"] == 1
    assert any("MERGE" in v for v, _ in rep["decision_variants"])
    with pytest.raises(SystemExit):                               # --rescope needs an epoch
        run.main(["--encoder", "hashing", "--nli", "lexical", "--workload", "running", "--k", "1", "--rescope"])
