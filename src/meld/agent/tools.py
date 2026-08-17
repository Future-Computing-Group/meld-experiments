"""Agent tools (smolagents-shaped: ``name``, ``description``, ``__call__``).

The tool surface is PUBLIC and open. The original two tools read/terminate the
ReAct loop:

  recall[query]        read the agent's MELD-brain memory (graph-walk read).
  final_answer[answer] emit the FINAL ANSWER and stop.

The ``remember`` verb is handled by the agent loop directly (it writes through
SMPMemory), so an observation always traverses the merge decision procedure.

For GAIA L2 ACHIEVABILITY we add the standard open tool surface GAIA L2 needs:

  OfflineWebSearchTool  web search with an OFFLINE cache/replay mode (CSC compute
                        nodes have NO internet -- prefetch on a networked node,
                        replay on the compute node). LIVE mode is an explicit,
                        gated path.
  FileReaderTool        a document reader (PDF text-layer / CSV / plain text;
                        image->caption is an optional gated extension).
  CalculatorTool        a sandboxed arithmetic evaluator (AST whitelist; NO
                        ``eval``/``exec`` of arbitrary code, no names/calls/attrs).

LOAD-BEARING MELD DECISION (documented in the campaign plan): the EVIDENCE-reading
tools (web, file) feed their retrieved content THROUGH the MELD Brain as a merged
observation (PUBLISH -> merge-on-receive -> Patch), not only as a raw tool string.
So GAIA attached-file / web content accrues in the brain and a later recall can
retrieve it -- this is the MELD-on-message choice (the tool is also an MELD author).
The calculator is pure computation and does NOT write to the brain.

The tool interface mirrors HF ``smolagents.Tool`` (name/description/forward) so the
same tool objects are handed to the off-the-shelf framework (see registry.py)
without a rewrite. ``forward`` is provided as an alias of ``__call__``.
"""
from __future__ import annotations

import ast
import csv
import operator
import os
import re
from typing import Callable, Dict, List, Optional

from meld.agent.memory import SMPMemory


class _ToolBase:
    """Minimal smolagents-shaped tool base: name/description/__call__, plus a
    ``forward`` alias and ``inputs``/``output_type`` metadata so the adapter in
    registry.py can wrap it as a real ``smolagents.Tool`` if needed."""

    name: str = "tool"
    description: str = ""
    inputs: Dict = {"arg": {"type": "string", "description": "the argument"}}
    output_type: str = "string"

    def __call__(self, arg: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def forward(self, arg: str) -> str:
        return self.__call__(arg)


class RecallTool(_ToolBase):
    name = "recall"
    description = ("recall[query]: read the agent's MELD-brain memory via a "
                   "graph-walk read; returns remembered facts most relevant to "
                   "the query.")

    def __init__(self, memory: SMPMemory, k: int = 3):
        self.memory = memory
        self.k = k

    def __call__(self, query: str) -> str:
        hits: List[str] = self.memory.recall(query, k=self.k)
        if not hits:
            return "(memory empty for that query)"
        return " | ".join(hits)


class FinalAnswerTool(_ToolBase):
    name = "final_answer"
    description = ("final_answer[answer]: emit the FINAL ANSWER and stop. The "
                   "answer must be a short string, number, or comma-separated "
                   "list (GAIA answer format).")

    def __call__(self, answer: str) -> str:
        return answer


# ---------------------------------------------------------------------------
# CalculatorTool: sandboxed arithmetic (AST whitelist; no eval/exec)
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate ONLY a whitelisted arithmetic AST. Anything else (names, calls,
    attribute access, subscripts, comprehensions, ...) raises -- so there is no
    path to imports, builtins, or filesystem access."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left),
                                       _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"refused: {type(node).__name__} is not a permitted "
                     "arithmetic node")


class CalculatorTool(_ToolBase):
    name = "calculator"
    description = ("calculator[expr]: evaluate a numeric arithmetic expression "
                   "(+, -, *, /, //, %, **, parentheses). Sandboxed: no names, "
                   "calls, imports, or attribute access.")

    def __call__(self, expr: str) -> str:
        try:
            tree = ast.parse((expr or "").strip(), mode="eval")
            val = _safe_eval(tree)
        except Exception as e:  # noqa: BLE001 (tool boundary returns a message)
            return f"error: {e}"
        # integral results print without a trailing .0 (GAIA number format)
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return ("%g" % val) if isinstance(val, float) else str(val)


# ---------------------------------------------------------------------------
# FileReaderTool: PDF (text layer) / CSV / plain text -> THROUGH the MELD brain
# ---------------------------------------------------------------------------


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_csv(path: str, max_rows: int = 200) -> str:
    rows: List[str] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.reader(f)):
            if i >= max_rows:
                rows.append("... (truncated)")
                break
            rows.append(", ".join(row))
    return "\n".join(rows)


def _read_pdf_text(path: str) -> str:
    """Best-effort PDF text extraction. Prefer ``pypdf`` if available (lazy
    import, gated-run upgrade path); otherwise fall back to a minimal pure-stdlib
    text-layer scrape sufficient for the local wiring smoke. NEVER OCR here (that
    is an optional image->caption extension for the gated run)."""
    try:  # gated-run upgrade path; not required offline
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(path)
        return "\n".join((pg.extract_text() or "") for pg in reader.pages)
    except Exception:
        pass
    # Minimal stdlib fallback: pull text from BT...ET text objects' (...) Tj/TJ.
    raw = open(path, "rb").read()
    chunks = re.findall(rb"\((?:\\.|[^()\\])*\)", raw)
    text = " ".join(
        c[1:-1].replace(b"\\(", b"(").replace(b"\\)", b")").decode(
            "latin-1", "replace") for c in chunks)
    return text.strip()


class FileReaderTool(_ToolBase):
    name = "read_file"
    description = ("read_file[path]: read a local document (PDF text-layer, CSV, "
                   "or plain text) and return its text. The content is also "
                   "merged into the agent's MELD memory as an observation, so a "
                   "later recall can retrieve it.")

    def __init__(self, memory: SMPMemory, max_chars: int = 4000,
                 chunk_chars: int = 500):
        self.memory = memory
        self.max_chars = max_chars
        self.chunk_chars = chunk_chars

    def _dispatch(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return _read_csv(path)
        if ext == ".pdf":
            return _read_pdf_text(path)
        return _read_text(path)

    def __call__(self, path: str) -> str:
        path = (path or "").strip()
        if not path or not os.path.isfile(path):
            return f"error: file not found: {path!r}"
        try:
            text = self._dispatch(path)
        except Exception as e:  # noqa: BLE001 (tool boundary)
            return f"error: could not read {path!r}: {e}"
        text = (text or "").strip()
        if not text:
            return f"error: no extractable text in {path!r}"
        text = text[: self.max_chars]
        # MELD-on-message: feed the evidence THROUGH the brain as merged
        # observations (chunked so each chunk is a recallable claim), not just a
        # raw tool string. This is the load-bearing decision (see plan G3).
        for chunk in _chunks(text, self.chunk_chars):
            self.memory.write(chunk)
        return text


def _chunks(text: str, size: int) -> List[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    return [text[i:i + size].strip() for i in range(0, len(text), size)
            if text[i:i + size].strip()]


# ---------------------------------------------------------------------------
# OfflineWebSearchTool: replay from a prefetched cache (CSC has no internet)
# ---------------------------------------------------------------------------


class OfflineWebSearchTool(_ToolBase):
    """Web search with an OFFLINE cache/replay mode.

    CSC compute nodes have NO internet, so live web tools cannot run there. The
    design: PREFETCH the pages a bounded GAIA L2 slice needs on a login/networked
    node into a query->page cache, ship the cache to the compute node, and REPLAY
    here (mode='replay', the default). A cache MISS is reported explicitly (never
    fabricate a page -- that would be a hallucinated source). LIVE mode is an
    explicit, gated path that requires an injected ``fetcher`` callable; without
    one it REFUSES (the offline/test path never touches the network).

    Retrieved page content is merged into the MELD brain as an observation (the
    MELD-on-message decision), so a later recall can retrieve it.
    """

    name = "web_search"
    description = ("web_search[query]: search the web. Offline/replay mode "
                   "returns a prefetched cached page for the query (CSC compute "
                   "nodes have no internet); the result is merged into MELD "
                   "memory. A cache miss is reported, never fabricated.")

    def __init__(self, memory: SMPMemory, cache: Optional[Dict[str, str]] = None,
                 mode: str = "replay",
                 fetcher: Optional[Callable[[str], str]] = None,
                 chunk_chars: int = 500):
        self.memory = memory
        self.cache = {self._key(k): v for k, v in (cache or {}).items()}
        self.mode = mode
        self.fetcher = fetcher
        self.chunk_chars = chunk_chars

    @staticmethod
    def _key(q: str) -> str:
        return re.sub(r"\s+", " ", (q or "").strip().lower())

    def __call__(self, query: str) -> str:
        key = self._key(query)
        if self.mode == "live":
            if self.fetcher is None:
                return ("error: live web mode requested but no fetcher injected "
                        "-- refused (offline/compute path has no internet; "
                        "prefetch + replay instead).")
            page = self.fetcher(query)  # pragma: no cover - gated/networked path
        else:
            if key not in self.cache:
                return (f"cache miss: query {query!r} was not prefetched. In "
                        "offline/replay mode only prefetched pages are served "
                        "(no live fetch on compute nodes; no fabrication).")
            page = self.cache[key]
        page = (page or "").strip()
        if not page:
            return f"(empty page for {query!r})"
        for chunk in _chunks(page, self.chunk_chars):
            self.memory.write(chunk)
        return page


__all__ = ["RecallTool", "FinalAnswerTool", "CalculatorTool",
           "FileReaderTool", "OfflineWebSearchTool"]
