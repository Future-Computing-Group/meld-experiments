#!/usr/bin/env python3
"""Cross-tier probe client (stdlib only; runs on the laptop, an edge node, or an HPC node).

Connects to the probe broker (reached through a local SSH-reverse-tunnel port on
a remote tier, or directly on the laptop) and exercises one of three modes.

Usage:
  python3 probe_client.py ping HOST PORT
        -> prints "PONG <t> rtt_ms=<x>"; non-zero exit on failure.
  python3 probe_client.py sub  HOST PORT TOPIC [COUNT]
        -> subscribes; prints each "RECV MSG <topic> <payload> at=<epoch>";
           exits after COUNT messages (default: stream until killed).
  python3 probe_client.py pub  HOST PORT TOPIC PAYLOAD
        -> publishes one message; prints the broker's "OK PUB ..." ack.
"""
import socket, sys, time


def _conn(host, port):
    return socket.create_connection((host, int(port)), timeout=10)


def ping(host, port):
    s = _conn(host, port)
    f = s.makefile("rw")
    t0 = time.time()
    f.write("PING\n"); f.flush()
    reply = f.readline().strip()
    rtt = (time.time() - t0) * 1000.0
    if not reply.startswith("PONG"):
        print("FAIL: expected PONG, got %r" % reply); return 1
    print("%s rtt_ms=%.1f" % (reply, rtt))
    return 0


def sub(host, port, topic, count=None):
    s = _conn(host, port)
    f = s.makefile("rw")
    f.write("SUB %s\n" % topic); f.flush()
    ack = f.readline().strip()
    if not ack.startswith("OK SUB"):
        print("FAIL: expected OK SUB, got %r" % ack); return 1
    print(ack, flush=True)
    n = 0
    for line in f:
        print("RECV %s at=%.3f" % (line.strip(), time.time()), flush=True)
        n += 1
        if count is not None and n >= count:
            break
    return 0


def pub(host, port, topic, payload):
    s = _conn(host, port)
    f = s.makefile("rw")
    f.write("PUB %s %s\n" % (topic, payload)); f.flush()
    ack = f.readline().strip()
    print(ack)
    return 0 if ack.startswith("OK PUB") else 1


def main(argv):
    if len(argv) < 4:
        print(__doc__); return 2
    mode, host, port = argv[1], argv[2], argv[3]
    if mode == "ping":
        return ping(host, port)
    if mode == "sub":
        count = int(argv[5]) if len(argv) > 5 else None
        return sub(host, port, argv[4], count)
    if mode == "pub":
        return pub(host, port, argv[4], argv[5])
    print("unknown mode %r" % mode); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
