"""The on-the-wire send seam for PatchTransport.

``make_http_send`` returns a ``send(peer_url, payload) -> requests.Response`` that POSTs
a Patch delta to a peer's ``/patch`` endpoint. PatchTransport owns the delivery contract
(2xx only) and exception handling (a connection error becomes non-delivery, never a
swallowed success), so this stays a thin adapter: build the URL, POST the bytes, return
the response. Identical seam local (live server in a thread) and on the testbed (peer VM URL).
"""
from __future__ import annotations

from typing import Callable

import requests


def make_http_send(*, timeout: float = 10.0) -> Callable[[str, bytes], "requests.Response"]:
    def send(peer_url: str, payload: bytes) -> "requests.Response":
        return requests.post(peer_url.rstrip("/") + "/patch", data=payload, timeout=timeout)
    return send


def make_http_call(*, timeout: float = 10.0) -> Callable[[str, str, bytes], tuple]:
    """Request/response seam for DIGEST and PULL: ``call(peer_url, path, body) ->
    (status, body)``; exceptions propagate to the caller (a lost call)."""
    def call(peer_url: str, path: str, body: bytes) -> tuple:
        r = requests.post(peer_url.rstrip("/") + path, data=body, timeout=timeout)
        return r.status_code, r.content
    return call


__all__ = ["make_http_send", "make_http_call"]
