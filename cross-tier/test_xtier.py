#!/usr/bin/env python3
"""Local end-to-end smoke for the cross-tier run kit (broker + brains + driver).

Runs the WHOLE experiment on localhost (3 brains as subprocesses, no SSH) so the
run logic is proven before any tier is real. What the real run measures is exactly
what is asserted here: per-brain RTT (latency phase) and the running-example
decisions + OVERRULED convergence (correctness phase). Each brain is a replica that
admits real claim deltas with the protocol's step (hashing encoder + lexical NLI,
torch-free); the epoch travels on the claims topic like any other object.

Run:  PYTHONPATH=src python3 -m pytest cross-tier/test_xtier.py -q
   or: PYTHONPATH=src python3 cross-tier/test_xtier.py
"""
import os, socket, subprocess, sys, time, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BROKER = os.path.join(HERE, "probe_broker.py")
BRAIN = os.path.join(HERE, "brain.py")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def _wait(port, t=5.0):
    end = time.time() + t
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close(); return True
        except OSError:
            time.sleep(0.05)
    return False


class _Fleet:
    """A broker + N brains on localhost, all torch-free."""
    def __init__(self, brain_ids):
        self.port = _free_port()
        env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
        self.broker = subprocess.Popen([sys.executable, BROKER, str(self.port)])
        assert _wait(self.port), "broker did not come up"
        self.logs = {}
        self.brains = []
        for bid in brain_ids:
            log = tempfile.NamedTemporaryFile(prefix=f"brain-{bid}-", suffix=".jsonl", delete=False)
            log.close(); self.logs[bid] = log.name
            self.brains.append(subprocess.Popen(
                [sys.executable, BRAIN, "--broker-host", "127.0.0.1",
                 "--broker-port", str(self.port), "--id", bid, "--log", log.name,
                 "--encoder", "hashing", "--nli", "lexical"],
                env=env))
        time.sleep(1.0)  # let brains SUB before the driver publishes

    def stop(self):
        for p in self.brains + [self.broker]:
            p.terminate()
        for p in self.brains + [self.broker]:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()


def test_latency_phase_acks_every_brain():
    import driver
    f = _Fleet(["local", "hpc", "edge"])
    try:
        res = driver.run_latency("127.0.0.1", f.port, ["local", "hpc", "edge"], k=8, timeout=8)
        for bid in ("local", "hpc", "edge"):
            assert res["per_brain"][bid]["acked"] == 8, res
            assert res["per_brain"][bid]["p50_ms"] >= 0.0, res
    finally:
        f.stop()


def test_running_example_decisions_and_convergence():
    import driver
    f = _Fleet(["local", "hpc", "edge"])
    try:
        from meld.config import load_epoch
        from test_brain_core import epoch_file
        res = driver.run_running_example("127.0.0.1", f.port, ["local", "hpc", "edge"],
                                         epoch=load_epoch(epoch_file()), timeout=10)
        # decisions identical across tiers, matching the running example
        for bid in ("local", "hpc", "edge"):
            d = res["per_brain"][bid]
            assert d["seed"] == "INSERT", d
            assert d["merge"] == "MERGE", d
            assert d["relate"] == "RELATE", d
            assert d["conflict"] == "CONFLICT", d
            assert d["c2_status"] == "OVERRULED", d
        assert res["all_converged"] is True, res
    finally:
        f.stop()


def test_relay_brain_drops_an_unverified_delta_and_logs_it():
    """The relay brain runs the shared MAC-verified step: unsigned, forged and malformed
    messages get no ack and leave a drop event; only the signed one is received."""
    import driver, json as _json
    f = _Fleet(["solo"])
    try:
        signed = _json.loads(driver.effect_message("o", "OVERRULE"))
        for bad in (_json.dumps(dict(signed, nonce="unsigned", m="")),
                    _json.dumps(dict(signed, nonce="forged", m="00" * 32)), "[1]", "[" * 5000):
            driver._publish("127.0.0.1", f.port, "claims", bad)
        driver._publish("127.0.0.1", f.port, "claims", _json.dumps(signed))
        time.sleep(0.8)
        recs = [_json.loads(l) for l in open(f.logs["solo"]) if l.strip()]
        merged = [r["nonce"] for r in recs if r.get("ev") == "merge"]
        drops = [r for r in recs if r.get("ev") == "drop"]
        assert merged == ["o"], recs
        assert len(drops) == 4, recs
        assert f.brains[0].poll() is None, "brain died on hostile input"
    finally:
        f.stop()


def test_partition_heal_reconnect_retains_state():
    """A brain whose broker connection drops (a partition) must RETAIN its CRDT
    state, reconnect on heal, and converge when the missed link is re-delivered.
    This is the local analogue of the real-tier partition-heal (drop a tunnel)."""
    import driver, json as _json
    port = _free_port()
    log = tempfile.NamedTemporaryFile(prefix="brain-pn-", suffix=".jsonl", delete=False); log.close()
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    b1 = subprocess.Popen([sys.executable, BROKER, str(port)]); assert _wait(port)
    brain = subprocess.Popen([sys.executable, BRAIN, "--broker-host", "127.0.0.1",
                              "--broker-port", str(port), "--id", "pn", "--log", log.name,
                              "--encoder", "hashing", "--nli", "lexical"], env=env)
    try:
        time.sleep(1.0)
        driver._publish("127.0.0.1", port, "claims", driver.c2_message())
        driver._publish("127.0.0.1", port, "claims", driver.effect_message("o", "OVERRULE"))
        time.sleep(0.5)
        # --- partition: kill the broker; the brain keeps its OVERRULED state and retries ---
        b1.terminate(); b1.wait(timeout=3); time.sleep(1.2)
        # --- heal: bring the broker back on the same port; the brain reconnects ---
        b2 = subprocess.Popen([sys.executable, BROKER, str(port)]); assert _wait(port); time.sleep(1.8)
        # re-deliver a (late) deprecate; join(OVERRULED, DEPRECATED) must stay OVERRULED
        driver._publish("127.0.0.1", port, "claims", driver.effect_message("d", "SUPERSEDE"))
        time.sleep(0.6)
        recs = [_json.loads(l) for l in open(log.name) if l.strip()]
        readys = [r for r in recs if r.get("ev") == "ready"]
        c2 = [r for r in recs if r.get("ev") == "merge" and r.get("claim_id") == driver.C2_ID]
        assert len(readys) >= 2, "brain did not reconnect after the partition: %r" % readys
        assert c2 and c2[-1]["eff_status"] == "OVERRULED", "state lost / wrong join: %r" % c2
    finally:
        for p in (brain, b1, locals().get("b2")):
            if p:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, "->", repr(e))
    print("=== %d failure(s) ===" % fails)
    sys.exit(1 if fails else 0)
