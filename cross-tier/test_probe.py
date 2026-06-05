#!/usr/bin/env python3
"""Loopback smoke test for the cross-tier probe kit.

Validates the probe broker + client on localhost ONLY (no SSH, no remote) so the
kit is proven working before it is ever pointed at remote hosts. What the real
cross-tier probe relies on is exactly what is exercised here:
  - PING/PONG reachability (the tunnel-reachability check), and
  - SUB/PUB relay through the hub (the cross-tier-delivery check).

Run:  PYTHONPATH=. python3 -m pytest cross-tier/test_probe.py -q
   or: python3 cross-tier/test_probe.py   (self-running, no pytest needed)
"""
import socket, subprocess, sys, threading, time, os

HERE = os.path.dirname(os.path.abspath(__file__))
BROKER = os.path.join(HERE, "probe_broker.py")
CLIENT = os.path.join(HERE, "probe_client.py")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_listening(port, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


class _Broker:
    def __init__(self):
        self.port = _free_port()
        self.proc = subprocess.Popen([sys.executable, BROKER, str(self.port)])
        assert _wait_listening(self.port), "broker did not come up"

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def test_ping_pong():
    """Phase-3 analogue: a client reaches the broker and gets a PONG (rtt printed)."""
    b = _Broker()
    try:
        out = subprocess.run(
            [sys.executable, CLIENT, "ping", "127.0.0.1", str(b.port)],
            capture_output=True, text=True, timeout=10)
        assert out.returncode == 0, out.stderr
        assert "PONG" in out.stdout and "rtt_ms=" in out.stdout, out.stdout
    finally:
        b.stop()


def test_pub_sub_relay():
    """Phase-4 analogue: a message PUBlished by one client reaches a SUBscriber
    via the hub. This is the exact mechanic the edge->hub->HPC delivery uses."""
    b = _Broker()
    received = []

    def subscriber():
        out = subprocess.run(
            [sys.executable, CLIENT, "sub", "127.0.0.1", str(b.port), "tier-x", "1"],
            capture_output=True, text=True, timeout=10)
        received.append(out.stdout)

    try:
        t = threading.Thread(target=subscriber)
        t.start()
        # Give the subscriber time to register before publishing.
        time.sleep(0.5)
        pub = subprocess.run(
            [sys.executable, CLIENT, "pub", "127.0.0.1", str(b.port), "tier-x", "hello-continuum"],
            capture_output=True, text=True, timeout=10)
        assert pub.returncode == 0, pub.stderr
        assert "OK PUB tier-x -> 1 sub(s)" in pub.stdout, pub.stdout
        t.join(timeout=10)
        assert received, "subscriber produced no output"
        assert "MSG tier-x hello-continuum" in received[0], received[0]
    finally:
        b.stop()


def test_unknown_topic_delivers_to_zero():
    """A publish to a topic with no subscribers acks 0 sub(s) (no crash, no hang)."""
    b = _Broker()
    try:
        pub = subprocess.run(
            [sys.executable, CLIENT, "pub", "127.0.0.1", str(b.port), "nobody", "x"],
            capture_output=True, text=True, timeout=10)
        assert pub.returncode == 0, pub.stderr
        assert "-> 0 sub(s)" in pub.stdout, pub.stdout
    finally:
        b.stop()


if __name__ == "__main__":
    # Self-running mode (no pytest dependency).
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failures += 1
                print("FAIL", name, "->", repr(e))
    print("=== %d failure(s) ===" % failures)
    sys.exit(1 if failures else 0)
