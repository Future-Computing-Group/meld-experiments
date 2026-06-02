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


__all__ = ["make_http_send"]
