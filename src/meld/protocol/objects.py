"""The replicated objects of the merge protocol (design §2): D deltas, A authored effects, Sig
signal records, Cal calibration records, Ep epoch records; canonical bytes, ids, MACs, the Sig
join and the scope partition.

Canonical bytes are the sorted-key compact ASCII JSON of an object without its ``mac``, with a
type tag member ``"t"`` (the class name) so that no two object types share a body or a MAC.
Bytes fields are standard padded base64, tuples are JSON arrays, and every scalar has its
exact declared type (int never bool, no floats: timestamps are integer microseconds,
probabilities and thresholds fixed-point integers), checked at construction and at parse.
An object's id is ``h16`` over the bytes as received; ``from_bytes`` refuses any encoding
that is not the canonical one, so one object has one encoding and one id. Id fields of the
spec (delta_id, eid, kid) are computed by ``object_id``, never carried. The MAC is
HMAC-SHA256 over the canonical bytes under the non-empty group key (``trust.signing``).

Claim keys are 16 bytes. A Delta binds its key: claim_key == claim_key(text, cell, version).
DISTINGUISHES is unordered, so its canonical form has c1 < c2 bytewise (the constructor
refuses the reverse; callers sort) and target_cell is the cell of that c1. Its two keys may
lie in different cells; ``effect_valid`` checks the target scope and the held c1 only
(validity never depends on whether c2 is held).
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, Tuple

from meld.proto import h16
from meld.trust.signing import _canonical, sign_delta, verify_delta

FIXED_ONE = 65535      # uint16 fixed point: probability p is round(p * FIXED_ONE)
INF16 = FIXED_ONE + 1  # +inf threshold sentinel in an epoch table: no uint16 score exceeds it
ALPHA_ONE = 10**6      # alpha and alpha_c are integers in parts per million (0.02 -> 20000)

Payload = Optional[Tuple[int, int, int]]   # (p_c, p_e, p_n) or None for bottom

_SCALAR = {"int": int, "str": str, "bytes": bytes, "bool": bool}


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, (list, tuple)) else v


def _require(ok: bool, what: str):
    if not ok:
        raise ValueError(what)


def _u16(v) -> bool:
    return type(v) is int and 0 <= v <= FIXED_ONE


def _key(*ks: bytes):
    _require(all(len(k) == 16 for k in ks), "claim keys are 16 bytes")


def _check(obj):
    """Exact scalar types; sequences normalised to tuples (validated by the class)."""
    for f in fields(obj):
        v = getattr(obj, f.name)
        t = _SCALAR.get(f.type)
        if t is None:
            object.__setattr__(obj, f.name, _tuples(v))
        else:
            _require(type(v) is t, f"{type(obj).__name__}.{f.name} must be {f.type}")


@dataclass(frozen=True)
class Delta:
    unit_id: bytes
    claim_key: bytes        # claim_key(text, cell, version)
    text: str
    cell: str
    version: int
    auth: int               # asserted rank, smaller = more authoritative
    status: str
    retention: int          # fixed point (FIXED_ONE = 1.0); required
    created_at: int         # microseconds since the Unix epoch, set by the publisher
    origin: str
    encoder: str
    q: bytes                # int8 embedding under encoder
    mac: str = ""

    def __post_init__(self):
        from meld.proto import claim_key
        _check(self)
        _require(0 <= self.retention <= FIXED_ONE and self.created_at >= 0
                 and self.version >= 0, "Delta retention/created_at/version out of range")
        _require(self.auth >= 0, "Delta auth is a rank >= 0")
        _require(self.claim_key == claim_key(self.text, self.cell, self.version),
                 "Delta claim_key is not claim_key(text, cell, version)")


EFFECT_KINDS = ("SUPERSEDE", "OVERRULE", "REVOKE", "DISTINGUISHES")


@dataclass(frozen=True)
class AuthoredEffect:
    scope: str
    kind: str
    c1: bytes
    c2: bytes               # b"" unless SUPERSEDE (old c1 -> new c2) or DISTINGUISHES (c1 < c2)
    target_cell: str        # cell of c1, so authorized(e) reads e and M_pol only
    author: str
    author_cell: str
    author_auth: int
    created_at: int
    mac: str = ""

    def __post_init__(self):
        _check(self)
        _require(self.kind in EFFECT_KINDS, f"kind {self.kind!r}")
        _require(self.author_auth >= 0, "author_auth is a rank >= 0")
        pair = self.kind in ("SUPERSEDE", "DISTINGUISHES")
        _require(bool(self.c2) == pair, f"c2 is set iff kind is SUPERSEDE or DISTINGUISHES")
        _key(self.c1, *([self.c2] if pair else []))
        _require(self.c1 < self.c2 if self.kind == "DISTINGUISHES" else self.c1 != self.c2,
                 "DISTINGUISHES needs c1 < c2; SUPERSEDE needs c1 != c2")


@dataclass(frozen=True)
class SignalRecord:
    scope: str
    c_x: bytes              # c_x < c_y bytewise; premise of pay_xy is c_x
    c_y: bytes
    m_nli: bytes
    pay_xy: Payload
    pay_yx: Payload
    issuer: str
    hw: str                 # issuer's hardware class
    mac: str = ""

    def __post_init__(self):
        _check(self)
        _key(self.c_x, self.c_y)
        _require(self.c_x < self.c_y, "SignalRecord needs c_x < c_y bytewise")
        for p in (self.pay_xy, self.pay_yx):
            _require(p is None or (type(p) is tuple and len(p) == 3 and all(map(_u16, p))),
                     f"payload {p!r} is not None or a uint16 triple")


@dataclass(frozen=True)
class CalRecord:
    scope: str
    c_x: bytes              # the tested pair, c_x < c_y bytewise
    c_y: bytes
    bin: int
    s_q: int                # single-observation score, fixed point
    hw: str
    obs: int                # the epoch builder refuses obs != 1
    label: str
    subtype: str
    source: str
    item: str               # source-item id
    author: str
    created_at: int
    sigma_lo_q: int         # the selector the pair was drawn under: uint16 floor,
    m: int                  # top-m,
    m_pol: bytes            # and M_pol (rank map)
    n_prior: int            # store size at the draw; bin is carried as given and not checked
    mac: str = ""           # here: the epoch builder recomputes it from n_prior and B_0

    def __post_init__(self):
        _check(self)
        _key(self.c_x, self.c_y)
        _require(_u16(self.sigma_lo_q) and self.m >= 1 and len(self.m_pol) == 16
                 and self.n_prior >= 0, "Cal selector fields out of range")
        _require(self.label == "NON_MERGEABLE" and self.subtype in ("CONTRADICTION", "OTHER")
                 and self.source in ("SEED", "AUDIT"), "Cal label/subtype/source")
        _require(self.c_x < self.c_y and _u16(self.s_q) and self.bin >= 0 and self.obs >= 1,
                 "Cal c_x < c_y, s_q, bin or obs out of range")


def _row_ok(row) -> bool:
    if type(row) is not tuple or not all(type(v) is int for v in row):
        return False
    g, n, n_c, th, th_c = row          # ValueError unless five members
    return g >= 0 and 0 <= n_c <= n and all(_u16(t) or t == INF16 for t in (th, th_c))


@dataclass(frozen=True)
class EpochRecord:
    scope: str
    seq: int
    m_nli: bytes
    m_pol: bytes
    policy: str             # SEED or AUDIT
    alpha: int              # ppm (ALPHA_ONE = 1)
    alpha_c: int
    mondrian_c: bool
    cal_root: bytes
    cal_hw: str
    bins: Tuple[int, str]   # (B_0, "doubling")
    table: Tuple[Tuple[int, int, int, int, int], ...]  # (g, n_g, n_c_g, theta_q, theta_c_q)
    sigma_lo_q: int
    sigma_c_q: int
    m: int
    author: str
    created_at: int
    mac: str = ""

    def __post_init__(self):
        _check(self)
        _require(self.policy in ("SEED", "AUDIT"), f"policy {self.policy!r}")
        _require(0 < self.alpha < ALPHA_ONE and 0 < self.alpha_c < ALPHA_ONE, "alpha range")
        b = self.bins
        _require(len(b) == 2 and type(b[0]) is int and b[0] > 0 and b[1] == "doubling",
                 "bins must be (B_0 > 0, 'doubling')")
        _require(len(self.table) > 0 and all(map(_row_ok, self.table)),
                 "table rows are (g, n_g, n_c_g <= n_g, theta_q, theta_c_q)")
        gs = [row[0] for row in self.table]
        _require(gs == sorted(set(gs)), "one table row per bin, sorted")


def _b64(v):
    """bytes, also inside tuples (a certificate's roots), as standard padded base64."""
    if isinstance(v, bytes):
        return base64.b64encode(v).decode()
    return [_b64(x) for x in v] if isinstance(v, tuple) else v


def _unb64(t: str, v):
    if t == "bytes":
        return base64.b64decode(v, validate=True)
    if t == "Tuple[bytes, ...]":
        return tuple(base64.b64decode(x, validate=True) for x in v)
    return _tuples(v)


def _body(obj) -> dict:
    body = {f.name: _b64(getattr(obj, f.name)) for f in fields(obj) if f.name != "mac"}
    body["t"] = type(obj).__name__
    return body


def canonical_bytes(obj) -> bytes:
    """Sorted-key compact JSON of the tagged body without the MAC; the bytes the MAC covers."""
    return _canonical(_body(obj))


def object_id(received: bytes) -> bytes:
    """Content address: H over the canonical bytes as received (delta_id, eid, kid)."""
    return h16(received)


def from_bytes(cls, received: bytes, mac: str = ""):
    """Parse received canonical bytes into ``cls``. Any failure, a wrong type tag, or a
    non-canonical encoding raises ValueError."""
    try:
        d = json.loads(received)
        obj = cls(mac=mac, **{f.name: _unb64(f.type, d[f.name])
                              for f in fields(cls) if f.name != "mac"})
        # the re-encoding carries cls's tag and fields only, so this also refuses a wrong
        # or missing tag and extra members
        _require(canonical_bytes(obj) == received, "not the canonical encoding")
        return obj
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as e:
        raise ValueError(f"refused {cls.__name__} bytes: {e}") from e


def _group_key(key: bytes) -> bytes:
    _require(bool(key), "empty HMAC key")
    return key


def attach_mac(obj, key: bytes):
    return replace(obj, mac=sign_delta(_body(obj), _group_key(key))["sig"])


def signed(obj, key: bytes) -> Tuple[bytes, str]:
    """(canonical bytes, MAC): an object as it travels and enters a receive gate."""
    o = attach_mac(obj, key)
    return canonical_bytes(o), o.mac


def verify_mac(obj, key: bytes) -> bool:
    return verify_delta({**_body(obj), "sig": obj.mac}, _group_key(key))


def join_signal_payloads(a: Payload, b: Payload) -> Payload:
    """Semilattice join of one direction's evidence: None is bottom; (max p_c, min p_e, min p_n)."""
    if a is None:
        return b
    if b is None:
        return a
    return (max(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]))


def contradicts(p: Payload) -> bool:
    return p is not None and p[0] > max(p[1], p[2])


def cal_pair(r: CalRecord) -> Tuple[bytes, bytes]:
    return (r.c_x, r.c_y)


def sig_key(r: SignalRecord) -> Tuple[bytes, bytes, bytes]:
    return (r.c_x, r.c_y, r.m_nli)


def payload(r: SignalRecord, premise: bytes) -> Payload:
    """The payload whose premise is the named claim key."""
    if premise == r.c_x:
        return r.pay_xy
    _require(premise == r.c_y, "premise is not a key of this record")
    return r.pay_yx


def score_s(pay_xy: Payload, pay_yx: Payload) -> Optional[int]:
    """s = min(p_e(x -> y), p_e(y -> x)); None unless both directions are evaluated."""
    return None if pay_xy is None or pay_yx is None else min(pay_xy[1], pay_yx[1])


def m_nli(encoder: str, nli_model: str, quant: str, canon_version: str) -> bytes:
    """Ids should name the weights (e.g. 'model@sha')."""
    return h16(_canonical({"encoder": encoder, "nli": nli_model, "quant": quant,
                           "canon": canon_version}))


def m_pol(rank_map: dict, scope_partition: dict, auth_policy: dict, step_version: int,
          **params) -> bytes:
    """``params`` (the fold's policy-borne inputs, e.g. r_min and bootstrap_selector) enter the
    hash when given; without them the hash is the four-field one."""
    return h16(_canonical({"rank_map": rank_map, "scopes": scope_partition,
                           "auth_policy": auth_policy, "step_version": step_version, **params}))


def m_pol_of(pol: dict) -> bytes:
    """M_pol of a policy dict: the four fields plus the fold inputs it carries."""
    return m_pol(pol["rank_map"], pol["scopes"], pol["auth_policy"], pol["step_version"],
                 **{k: pol[k] for k in ("r_min", "bootstrap_selector") if k in pol})


def unit_to_delta(unit, claim, rank_map: Dict[str, int]) -> Delta:
    """A facade MemoryUnit and one of its Claims as a Delta: float retention to fixed point,
    string authority to rank (KeyError if unranked), owner_did as hex origin."""
    return Delta(unit_id=unit.id, claim_key=claim.claim_key, text=claim.text, cell=unit.cell,
                 version=claim.version, auth=rank_map[unit.authority],
                 status=claim.claim_status, retention=round(unit.retention * FIXED_ONE),
                 created_at=unit.created_at, origin=unit.owner_did.hex(),
                 encoder=unit.encoder, q=unit.q)


def scope_partition(pol: dict) -> Dict[str, str]:
    """cell -> scope from an M_pol-like ``{"scopes": {scope: [cells]}}``; a cell in two
    scopes (or twice in one) is refused. Cells absent from the map belong to no scope."""
    scope_of: Dict[str, str] = {}
    for scope, cells in pol["scopes"].items():
        _require(type(cells) is list, f"cells of scope {scope!r} must be a list")
        for cell in cells:
            _require(cell not in scope_of, f"cell {cell!r} assigned twice")
            scope_of[cell] = scope
    return scope_of


def effect_valid(e: AuthoredEffect, scope_of: Dict[str, str], cell_of: Dict[bytes, str]) -> bool:
    """§2.0/§2.1, c2 clause removed: e is invalid if its scope is not its target cell's
    scope, or if a held c1 sits in another cell than target_cell. c2 plays no part.

    Validity is monotone on nodes: auth is read only for held nodes, and a key's cell is part
    of the key, so the c1 clause gives the same verdict at every time the node is held."""
    if scope_of.get(e.target_cell) != e.scope:
        return False
    return not (e.c1 in cell_of and cell_of[e.c1] != e.target_cell)


__all__ = [
    "FIXED_ONE", "INF16", "ALPHA_ONE", "Delta", "AuthoredEffect", "SignalRecord", "CalRecord",
    "EpochRecord", "EFFECT_KINDS", "canonical_bytes", "object_id", "from_bytes", "attach_mac", "signed",
    "verify_mac", "join_signal_payloads", "contradicts", "cal_pair", "sig_key", "payload", "score_s",
    "m_nli", "m_pol", "m_pol_of", "unit_to_delta", "scope_partition", "effect_valid", "h16",
]
