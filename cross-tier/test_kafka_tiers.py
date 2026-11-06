#!/usr/bin/env python3
"""kafka_tiers.sh: the job-mode HPC tier, the per-tier partition signal and the image-mode edge.

No cluster and no broker: ssh, scp and the Slurm client are the local stand-ins in fakehpc/, which
log every call; ship() is replaced by a no-op so nothing is copied. Each test sources the helper
in a fresh bash and asserts on the logged calls.

Run:  python3 -m pytest cross-tier/test_kafka_tiers.py -q
"""
import os
import subprocess
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fakehpc")


def run(tmp_path, body, env=None, check=True):
    """Source kafka_tiers.sh with the variables the callers set, then run ``body``."""
    state = tmp_path / "state"
    pre = f"""
set -euo pipefail
REPO='{os.path.dirname(HERE)}'; HERE='{HERE}'; PY=python3; PORT=29092; OUT='{tmp_path}'; STAMP=t
CLAIMS=c; ACKS=a; ENCODER=hashing; NLI=lexical; EPOCH_FILE=/dev/null
EDGE_SSH=edgehost; EDGE_DIR='{tmp_path}/edge'; HPC_SSH=hpchost; HPC_DIR='{tmp_path}/hpc'
BRAIN_PIDS=(); TUN_PIDS=(); EDGE_TUN=""; EDGE_PID=""; CAFF_PID=""; LAUNCHED=()
source '{HERE}/kafka_tiers.sh'
ship() {{ echo "ship $*" >> "$FAKEHPC_LOG"; }}
"""
    e = dict(os.environ, PATH=FAKE + os.pathsep + os.environ["PATH"],
             FAKEHPC_STATE=str(state), FAKEHPC_LOG=str(tmp_path / "calls.log"))
    e.update(env or {})
    (tmp_path / "hpc").mkdir(exist_ok=True)
    r = subprocess.run(["bash", "-c", pre + body], cwd=tmp_path, env=e,
                       capture_output=True, text=True, timeout=60)
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
    log = (tmp_path / "calls.log").read_text() if (tmp_path / "calls.log").exists() else ""
    return r, log


@pytest.fixture
def job_script(tmp_path):
    """A stand-in job body: records its arguments, then idles like a brain."""
    p = tmp_path / "job.sh"
    p.write_text('echo "$@" > "$PWD/job.args"; exec sleep 30\n')
    return str(p)


def test_open_tun_forwards_every_argument(tmp_path):
    _, log = run(tmp_path, 'open_tun -J hpchost node7; sleep 0.3; kill "$TUN_PID"; trap - EXIT')
    line = [l for l in log.splitlines() if l.startswith("ssh -N")][0]
    assert "-R 29092:localhost:29092" in line
    assert line.endswith("-J hpchost node7")


def test_job_mode_submits_waits_and_tunnels_to_the_node(tmp_path, job_script):
    r, log = run(tmp_path, 'start_brain hpc; echo "JOB=$HPC_JOBID LAUNCHED=${LAUNCHED[*]:-}"',
                 env={"HPC_JOB": job_script})
    assert "ship hpchost" in log
    sub = [l for l in log.splitlines() if "sbatch --parsable" in l][0]
    assert sub.startswith("ssh hpchost") and job_script in sub and " 29092 " in sub
    args = (tmp_path / "hpc" / "job.args").read_text()
    assert "--id hpc" in args and "--bootstrap localhost:29092" in args
    assert "squeue" in log
    tun = [l for l in log.splitlines() if l.startswith("ssh -N")][0]
    assert tun.endswith("-J hpchost localhost")          # the node squeue reported
    assert "JOB=1001" in r.stdout and "LAUNCHED=\n" in r.stdout   # no tmux/docker teardown for a job
    assert "scancel 1001" in log                         # the EXIT trap cancels the job


def test_job_mode_gives_up_when_the_job_never_runs(tmp_path):
    dead = tmp_path / "dead.sh"; dead.write_text("exit 0\n")   # squeue never reports it RUNNING
    r, log = run(tmp_path, "start_brain hpc", check=False,
                 env={"HPC_JOB": str(dead), "HPC_QUEUE_WAIT": "2"})
    assert r.returncode != 0 and "not running" in r.stdout
    assert "scancel 1001" in log


def test_signal_tier_job_mode_goes_through_srun_on_the_allocation(tmp_path):
    _, log = run(tmp_path, "HPC_JOBID=4242; sig_tier hpc STOP || true; sig_tier hpc CONT || true; trap - EXIT")
    stop = [l for l in log.splitlines() if "srun" in l and "STOP" in l][0]
    assert stop.startswith("ssh hpchost") and "--overlap --jobid=4242" in stop
    assert "--id hpc " in stop
    assert any("srun" in l and "-CONT" in l for l in log.splitlines())


def test_signal_tier_local_freezes_and_thaws_that_brain(tmp_path):
    p = subprocess.Popen(["sleep", "30"])
    try:
        run(tmp_path, f"HPC_SSH=local; HPC_PID={p.pid}; sig_tier hpc STOP; trap - EXIT")
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(p.pid)], capture_output=True, text=True).stdout
        assert st.strip().startswith("T")
        run(tmp_path, f"HPC_SSH=local; HPC_PID={p.pid}; sig_tier hpc CONT; trap - EXIT")
        time.sleep(0.1)
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(p.pid)], capture_output=True, text=True).stdout
        assert not st.strip().startswith("T")
    finally:
        p.kill()


def test_signal_tier_refuses_a_tier_it_cannot_reach(tmp_path):
    r, _ = run(tmp_path, "sig_tier edge STOP", check=False)
    assert r.returncode != 0 and "FATAL" in r.stdout


def test_edge_image_mode_runs_the_brain_in_the_container(tmp_path):
    ssh_log_only = 'ssh() { echo "ssh $*" >> "$FAKEHPC_LOG"; }; '   # nothing runs locally
    _, log = run(tmp_path, ssh_log_only + "start_brain edge; trap - EXIT",
                 env={"EDGE_IMAGE": "meld-brain:minilm"})
    start = [l for l in log.splitlines() if "docker run" in l][0]
    assert start.startswith("ssh edgehost")
    for s in ("--network host", "PYTHONDONTWRITEBYTECODE=1", "meld-brain:minilm", "--id edge"):
        assert s in start


def test_remote_brain_logs_are_per_run(tmp_path):
    """Trial claim ids repeat across runs, and wait_ready greps for a ready line, so a remote log
    reused from an earlier run would satisfy both before the new brain has done anything."""
    r, _ = run(tmp_path, 'STAMP=run1; log_of hpc; log_of edge; STAMP=run2; log_of hpc; trap - EXIT')
    a, b, c = r.stdout.split()
    assert a.endswith("phwan-hpc-run1.jsonl") and b.endswith("phwan-edge-run1.jsonl") and a != c
