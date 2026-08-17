#!/usr/bin/env python3
"""Minimal line-relay "broker" for the MELD cross-tier reachability probe.

This is NOT the MELD transport. It is a ~40-line stand-in whose only job is to
prove the SSH-reverse-tunnel hub topology end-to-end: that a process on each
remote tier can reach a broker bound to the laptop's loopback through its local
tunnel port, and that a message PUBlished on one tier is delivered to a SUBscriber
on another via the laptop hub. No third-party dependencies (stdlib only), so it
runs identically on the laptop, an edge node, and an HPC login/interactive node.

Wire protocol (one line per request, newline-terminated):
  PING                      -> PONG <epoch>
  SUB  <topic>              -> OK SUB <topic>; then streams "MSG <topic> <payload>"
  PUB  <topic> <payload>    -> OK PUB <topic> -> <n> sub(s)

Usage:  python3 probe_broker.py [PORT]      (default 1883; binds 127.0.0.1 only)
"""
import socket, threading, sys, time

HOST = "127.0.0.1"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 1883
subs, lock = [], threading.Lock()


def handle(conn):
    f = conn.makefile("rw")
    try:
        line = f.readline()
        if not line:
            return
        parts = line.strip().split(" ", 2)
        verb = parts[0].upper()
        if verb == "PING":
            f.write("PONG %f\n" % time.time()); f.flush()
        elif verb == "SUB" and len(parts) >= 2:
            topic = parts[1]
            with lock:
                subs.append((conn, topic))
            f.write("OK SUB %s\n" % topic); f.flush()
            while f.readline():  # block until the subscriber disconnects
                pass
        elif verb == "PUB" and len(parts) >= 2:
            topic = parts[1]
            payload = parts[2] if len(parts) > 2 else ""
            with lock:
                targets = [c for (c, t) in subs if t == topic]
            for c in targets:
                try:
                    c.sendall(("MSG %s %s\n" % (topic, payload)).encode())
                except OSError:
                    pass
            f.write("OK PUB %s -> %d sub(s)\n" % (topic, len(targets))); f.flush()
        else:
            f.write("ERR bad-request\n"); f.flush()
    finally:
        conn.close()


def main():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(16)
    print("probe-broker listening on %s:%d" % (HOST, PORT), flush=True)
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
