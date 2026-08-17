"""M-1 memory-condition harness: measure the merge-protocol DELTA, not agent F1.

MELD's contribution is distributed memory MANAGEMENT (the merge-on-receive
protocol), NOT the agent. The competence gate driver (``run_hotpot_gate.py``)
feeds ALL of a question's context into ONE brain -- that is the *centralized*
condition; it exercises recall but NOT the merge across sovereign brains, which is
the whole point of MELD. This module reframes the HotpotQA experiment as the M-1
memory-condition delta: the SAME held-out questions are answered under five memory
conditions, and the headline numbers are the DIFFERENCES between conditions.

Conditions (``MEMORY_CONDITION``)
---------------------------------
- ``none``              question only, no evidence (the FLOOR).
- ``full_context``      all paragraphs inlined in the prompt, no brain (the ORACLE
                        CEILING -- the evidence is right there to read).
- ``centralized``       all paragraphs -> ONE MELD brain; recall from it (the
                        current ``run_hotpot_gate_csc.py`` behaviour: recall, no
                        cross-brain merge).
- ``distributed_merge`` paragraphs split across N=3 SOVEREIGN brains; each PUBLISHes
                        its deltas; a GLOBAL brain runs ``merge_on_receive`` (MELD)
                        on every received delta -> recall from the global brain.
                        THIS is the MELD path -- the merge across sovereign stores.
- ``naive_union``       split across N; the global brain is the raw UNION of the
                        per-brain deltas with NO merge decision procedure (the
                        ablation: dump everything, never merge/relate/conflict).

The load-bearing comparison is ``distributed_merge`` vs ``naive_union`` (does the
merge protocol beat dumping) and ``distributed_merge`` vs ``centralized`` (is the
merge lossless relative to a single store). ``none`` and ``full_context`` bound
the measurable headroom (the "gate" reframed as a competence FLOOR: full_context
>> none means a delta is *measurable* at all).

Answering modes
---------------
- ``agentic``      the registry agent recalls + reasons (realistic; agent-strategy
                   variance is IN the number).
- ``fixed_recall`` a DETERMINISTIC top-k graph-walk read from the brain assembled
                   into ONE LLM answer call -- removes agent-strategy variance, so
                   the residual difference between conditions is the CLEAN memory
                   delta. The headline uses ``fixed_recall``.

Overlap injection (the merge is trivial without it)
---------------------------------------------------
A clean disjoint partition gives every sovereign brain unique sentences, so the
global merge only ever inserts novel concepts -- no ``same-enough``/``overlaps``
edge ever fires and ``distributed_merge`` collapses to ``naive_union``. So the
split is NOT disjoint: each gold supporting sentence is assigned to >= 1 brain,
and a controlled fraction is DUPLICATED across brains and lightly PARAPHRASED, so
the dense-similarity (sigma) gate fires a MERGE-via-overlap (or RELATE) Patch when
the duplicate is received. An optional CONFLICT variant injects one contradictory
sentence so the ``chi=contradicts`` path emits a first-class CONTRADICTS edge.

This module is offline/transport-free: it builds brains and merges; the LLM call
is the caller's (``run_hotpot_memdelta.py`` injects ``MockLLM`` locally or
``OpenWeightLLM`` on the gated host). NO billable run here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from meld.agent.hotpot_loader import HotpotTask
from meld.agent.memory import SMPMemory
from meld.config import Thresholds

MEMORY_CONDITIONS = (
    "none", "full_context", "centralized", "distributed_merge", "naive_union",
)
ANSWER_MODES = ("agentic", "fixed_recall")
DEFAULT_N_BRAINS = 3


# ---------------------------------------------------------------------------
# overlap-injecting split across N sovereign brains
# ---------------------------------------------------------------------------


def _paragraph_sentences(task: HotpotTask) -> List[Tuple[str, str]]:
    """Flatten the item context to (title, sentence) observation strings' parts,
    preserving order (gold + distractors interleaved as in the corpus)."""
    out: List[Tuple[str, str]] = []
    for title, sentences in task.context:
        for sent in sentences:
            out.append((title, sent))
    return out


def _is_supporting(title: str, sent_id: int,
                   supporting: List[List[object]]) -> bool:
    for st, si in supporting:
        if st == title and int(si) == int(sent_id):
            return True
    return False


def _gold_sentences(task: HotpotTask) -> List[Tuple[str, str]]:
    """The gold SUPPORTING sentences (title, sentence), per supporting_facts."""
    gold: List[Tuple[str, str]] = []
    for title, sentences in task.context:
        for sid, sent in enumerate(sentences):
            if _is_supporting(title, sid, task.supporting_facts):
                gold.append((title, sent))
    return gold


_PARAPHRASE_RULES = [
    (r"\bwas directed by\b", "was made by director"),
    (r"\bdirected by\b", "made by director"),
    (r"\bis the capital( city)? of\b", "is the capital city of"),
    (r"\bwas born in\b", "was born in the city of"),
    (r"\bis a\b", "was a"),
    (r"\bHe\b", "This person"),
    (r"\bIt\b", "This work"),
    (r"\bflows through\b", "runs through"),
]


def light_paraphrase(text: str) -> str:
    """Apply a small, deterministic, MEANING-PRESERVING surface rewrite so a
    duplicated sentence is not byte-identical (which would trip the cheap kappa
    exact-key path) yet stays dense-similar (sigma high) -- exactly the regime
    where the ``same-enough`` MERGE / ``overlaps`` RELATE gate must fire. If no
    rule matches, append a benign connective so the text still differs."""
    out = text
    for pat, repl in _PARAPHRASE_RULES:
        new = re.sub(pat, repl, out)
        if new != out:
            out = new
            break
    if out == text:
        out = text.rstrip(".") + ", as recorded."
    return out


@dataclass
class SplitPlan:
    """The per-brain observation assignment for one task, with audit counts so a
    test/report can assert overlaps were actually injected."""
    per_brain: List[List[str]] = field(default_factory=list)   # brain -> obs texts
    n_gold: int = 0
    n_duplicated: int = 0          # gold sentences placed on >= 2 brains
    n_paraphrased: int = 0         # duplicates that were lightly paraphrased
    n_conflicts_injected: int = 0  # contradictory sentences injected


def split_across_brains(task: HotpotTask, *, n_brains: int = DEFAULT_N_BRAINS,
                        overlap_fraction: float = 0.5,
                        inject_conflict: bool = False) -> SplitPlan:
    """Split the task's context across ``n_brains`` SOVEREIGN brains with
    controlled, load-bearing overlap (NOT a clean disjoint partition):

      * every gold supporting sentence is assigned to >= 1 brain (round-robin);
      * a controlled ``overlap_fraction`` of the gold sentences is ALSO placed on
        the NEXT brain, lightly PARAPHRASED, so the global merge sees a
        same-enough duplicate (sigma gate -> MERGE/RELATE Patch);
      * distractors are spread round-robin (single-assigned), so the brains are
        genuinely sovereign partial views;
      * if ``inject_conflict``: one gold sentence's value is swapped (e.g. a place
        swap) and placed on a DIFFERENT brain than its original, so the global
        merge sees a contradiction (chi=contradicts -> CONTRADICTS edge).

    The observation text mirrors ``feed_context_through_brain`` ("title: sentence")
    so the brain receives the same shape it does in ``centralized``.
    """
    if n_brains < 1:
        raise ValueError("n_brains must be >= 1")
    per_brain: List[List[str]] = [[] for _ in range(n_brains)]
    plan = SplitPlan(per_brain=per_brain)

    flat = _paragraph_sentences(task)
    gold = _gold_sentences(task)
    gold_set = set(gold)
    plan.n_gold = len(gold)

    # gold first (round-robin) so every gold sentence lands on >= 1 brain; a
    # fraction are duplicated (paraphrased) onto the next brain.
    n_dup_target = int(round(overlap_fraction * len(gold)))
    for i, (title, sent) in enumerate(gold):
        b = i % n_brains
        per_brain[b].append(f"{title}: {sent}".strip())
        if i < n_dup_target:
            para = light_paraphrase(sent)
            per_brain[(b + 1) % n_brains].append(f"{title}: {para}".strip())
            plan.n_duplicated += 1
            if para != sent:
                plan.n_paraphrased += 1

    # distractors round-robin (single-assigned), offset so they don't all pile on
    # brain 0; this keeps each brain a sovereign PARTIAL view.
    d = 0
    for title, sent in flat:
        if (title, sent) in gold_set:
            continue
        per_brain[d % n_brains].append(f"{title}: {sent}".strip())
        d += 1

    if inject_conflict and gold:
        title, sent = gold[0]
        conflicting = _make_conflicting(sent)
        if conflicting is not None:
            # place the contradiction on a brain that does NOT already hold the
            # original, so the global merge (not a local brain) sees the conflict.
            orig_brain = 0  # gold[0] went to brain 0 above
            target = (orig_brain + 1) % n_brains
            per_brain[target].append(f"{title}: {conflicting}".strip())
            plan.n_conflicts_injected += 1

    return plan


# a tiny, deterministic place/value swap table for the conflict variant. Each
# pair is an INCOMPATIBLE value swap on otherwise-identical text, which the
# lexical NLI flags as a contradiction (the general single-slot-swap heuristic).
_CONFLICT_SWAPS = [
    ("London", "Paris"), ("Paris", "Rome"), ("France", "Italy"),
    ("Germany", "Spain"), ("British", "French"), ("2010", "2011"),
    ("Nolan", "Spielberg"),
]


def _make_conflicting(sentence: str) -> Optional[str]:
    """Return ``sentence`` with the FIRST present swap-pair value replaced by its
    incompatible counterpart, else ``None`` if no swap applies (so we never inject
    a non-contradiction)."""
    for a, b in _CONFLICT_SWAPS:
        if re.search(rf"\b{re.escape(a)}\b", sentence):
            return re.sub(rf"\b{re.escape(a)}\b", b, sentence, count=1)
    return None


# ---------------------------------------------------------------------------
# building the global brain per condition
# ---------------------------------------------------------------------------


@dataclass
class ConditionBuild:
    """The memory state for one (task, condition): a recall source + audit."""
    condition: str
    brain: Optional[SMPMemory]          # None for none / full_context
    inline_context: List[str]           # non-empty only for full_context
    n_merge: int = 0                    # MERGE Patches that fired (overlap path)
    n_relate: int = 0                   # RELATE Patches (overlaps)
    n_conflict: int = 0                 # CONFLICT Patches (contradiction edges)
    n_total_patches: int = 0
    n_overlap_merges: int = 0           # MERGE Patches via the sigma (non-novel) gate
    stored_units: int = 0               # distinct units held in the store (provenance)
    live_units: int = 0                 # status-live (recallable) units -- the footprint
    split: Optional[SplitPlan] = None


def _footprint(mem: SMPMemory) -> Dict[str, int]:
    """Store footprint for the overhead/compactness claim: distinct stored units
    vs status-LIVE (recallable) units. A merge deprecates redundant duplicates, so
    live < stored < naive-union; this is the metric, not the audit-patch count."""
    return {"stored": len(mem.brain._units), "live": mem.brain.live_unit_count()}


def _new_memory(encoder: str, nli: str, cfg: Thresholds,
                cache_dir: Optional[str] = None) -> SMPMemory:
    # A shared cache_dir lets every brain (sovereign + global) and every condition reuse the
    # same on-disk sentence embeddings, so a sweep embeds each sentence once instead of
    # re-embedding it per build (a redundant-load slowness).
    return SMPMemory(encoder=encoder, nli=nli, cfg=cfg, cache_dir=cache_dir)


def _audit_counts(mem: SMPMemory) -> Dict[str, int]:
    """Count Patch decisions in a brain's audit log, separating a MERGE that fired
    via the SIGMA (same-enough/overlap) gate from a MERGE that was a plain novel
    insert -- the former is the load-bearing 'the merge actually did something'."""
    n = {"INSERT": 0, "MERGE": 0, "RELATE": 0, "CONFLICT": 0, "REJECT": 0,
         "overlap_merge": 0}
    for p in mem.brain.audit_log:
        n[p.decision] = n.get(p.decision, 0) + 1
        if p.decision == "MERGE" and "sigma" in p.gates_that_fired:
            n["overlap_merge"] += 1
    return n


def build_condition(task: HotpotTask, condition: str, *, encoder: str, nli: str,
                    cfg: Thresholds, n_brains: int = DEFAULT_N_BRAINS,
                    overlap_fraction: float = 0.5,
                    inject_conflict: bool = False,
                    cache_dir: Optional[str] = None) -> ConditionBuild:
    """Materialize the memory state for ONE (task, condition).

    - ``none``: no brain, no inline context.
    - ``full_context``: all paragraphs inlined; no brain.
    - ``centralized``: every observation -> ONE brain (merge-on-receive, but all
      from one author, so overlaps only arise from corpus-internal repeats).
    - ``distributed_merge``: overlap-split across N sovereign brains; each brain's
      EXTRACTED DELTA is received by a GLOBAL brain via ``merge_on_receive`` (the
      MELD path); recall from the global brain.
    - ``naive_union``: same split, but the global brain is the raw UNION of the
      per-brain claim texts with NO merge decision procedure (re-inserted as
      independent novel units; no MERGE/RELATE/CONFLICT ever fires).
    """
    if condition not in MEMORY_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; "
                         f"choose from {MEMORY_CONDITIONS}")

    if condition == "none":
        return ConditionBuild(condition=condition, brain=None, inline_context=[])

    if condition == "full_context":
        inline = [f"{title}: {sent}"
                  for title, sentences in task.context for sent in sentences]
        return ConditionBuild(condition=condition, brain=None,
                              inline_context=inline)

    if condition == "centralized":
        mem = _new_memory(encoder, nli, cfg, cache_dir)
        for title, sentences in task.context:
            for sent in sentences:
                mem.write(f"{title}: {sent}".strip())
        c = _audit_counts(mem)
        f = _footprint(mem)
        return ConditionBuild(condition=condition, brain=mem, inline_context=[],
                              n_merge=c["MERGE"], n_relate=c["RELATE"],
                              n_conflict=c["CONFLICT"],
                              n_overlap_merges=c["overlap_merge"],
                              n_total_patches=len(mem.brain.audit_log),
                              stored_units=f["stored"], live_units=f["live"])

    # distributed conditions share the same overlap-injecting split
    split = split_across_brains(task, n_brains=n_brains,
                                overlap_fraction=overlap_fraction,
                                inject_conflict=inject_conflict)

    if condition == "naive_union":
        # the ABLATION: dump every per-brain observation into a global store with
        # NO merge decision procedure. We bypass merge-on-receive by writing each
        # text into a brain that has been emptied between writes, so each becomes a
        # standalone novel unit -- i.e. a raw union, never a merge/relate/conflict.
        return _build_naive_union(split, encoder, nli, cfg, cache_dir)

    # distributed_merge: each sovereign brain ingests its own observations, then
    # PUBLISHes its delta to a GLOBAL brain that runs merge-on-receive on it.
    return _build_distributed_merge(split, encoder, nli, cfg, cache_dir)


def _build_distributed_merge(split: SplitPlan, encoder: str, nli: str,
                             cfg: Thresholds,
                             cache_dir: Optional[str] = None) -> ConditionBuild:
    sovereign: List[SMPMemory] = []
    for obs_list in split.per_brain:
        mem = _new_memory(encoder, nli, cfg, cache_dir)
        for text in obs_list:
            mem.write(text)
        sovereign.append(mem)

    global_brain = _new_memory(encoder, nli, cfg, cache_dir)
    # each sovereign brain PUBLISHes its extracted delta; the global brain runs
    # merge_on_receive on every received claim (SMPMemory.write does exactly the
    # merge-on-receive -> Patch -> apply path). Receiving the duplicated,
    # paraphrased sentences is what fires the same-enough / overlaps gate.
    for mem in sovereign:
        for unit in mem.brain.extract_delta():
            for claim in getattr(unit, "claims", []) or []:
                global_brain.write(claim.text)

    c = _audit_counts(global_brain)
    f = _footprint(global_brain)
    return ConditionBuild(
        condition="distributed_merge", brain=global_brain, inline_context=[],
        n_merge=c["MERGE"], n_relate=c["RELATE"], n_conflict=c["CONFLICT"],
        n_overlap_merges=c["overlap_merge"],
        n_total_patches=len(global_brain.brain.audit_log),
        stored_units=f["stored"], live_units=f["live"], split=split)


def _build_naive_union(split: SplitPlan, encoder: str, nli: str,
                       cfg: Thresholds,
                       cache_dir: Optional[str] = None) -> ConditionBuild:
    """The global store = raw union of per-brain observations, NO merge. Each
    observation is admitted as an independent unit (we clear the brain's local
    candidate set before each write so merge-on-receive always sees an empty local
    set -> a novel insert -> no MERGE-via-overlap / RELATE / CONFLICT ever fires).
    """
    global_brain = _new_memory(encoder, nli, cfg, cache_dir)
    for obs_list in split.per_brain:
        for text in obs_list:
            text = (text or "").strip()
            if not text:
                continue
            # bypass the merge decision procedure: insert as a standalone novel
            # unit by emptying the local candidate set seen by merge-on-receive.
            _raw_insert(global_brain, text)
    c = _audit_counts(global_brain)
    f = _footprint(global_brain)
    return ConditionBuild(
        condition="naive_union", brain=global_brain, inline_context=[],
        n_merge=c["MERGE"], n_relate=c["RELATE"], n_conflict=c["CONFLICT"],
        n_overlap_merges=c["overlap_merge"],
        n_total_patches=len(global_brain.brain.audit_log),
        stored_units=f["stored"], live_units=f["live"], split=split)


def _raw_insert(mem: SMPMemory, text: str) -> None:
    """Insert ``text`` as a standalone unit WITHOUT the merge decision procedure
    relating it to existing claims -- the naive-union ablation. We construct the
    unit and a novel INSERT Patch directly (mirroring merge_on_receive's
    no-candidate branch, D1) and apply it, so EVERY naive-union write is a plain
    insert and ``overlap_merge``/RELATE/CONFLICT counts stay zero by construction."""
    from meld.agent.memory import _unit_from_text
    from meld.protocol.merge_on_receive import _make_patch
    unit = _unit_from_text(text)
    for claim in unit.claims:
        patch = _make_patch("INSERT", unit, claim, gates=["novel"])
        mem.brain.apply(patch)


# ---------------------------------------------------------------------------
# answer assembly per mode
# ---------------------------------------------------------------------------


def assemble_recall(build: ConditionBuild, question: str, *, encoder: str,
                    k: int = 5) -> List[str]:
    """Deterministic top-k evidence read for ``fixed_recall`` and the agentic
    prompt context. For brain-backed conditions this is the brain graph-walk read;
    for ``full_context`` it is the inlined paragraphs; for ``none`` it is empty."""
    if build.condition == "none":
        return []
    if build.condition == "full_context":
        return list(build.inline_context)
    assert build.brain is not None
    return build.brain.recall(question, k=k)


def fixed_recall_prompt(question: str, evidence: List[str]) -> str:
    """The deterministic single-call answer prompt: top-k evidence + the question
    + the GAIA-tight answer contract. NO agent loop, so the residual difference
    between conditions is the clean MEMORY delta (not agent-strategy variance)."""
    from meld.agent.react_agent import GAIA_SYSTEM
    ev = ("\n".join(f"- {e}" for e in evidence) if evidence
          else "(no evidence available)")
    return (f"{GAIA_SYSTEM}\n"
            f"Evidence:\n{ev}\n\n"
            f"Question: {question}\n"
            "Answer with EXACTLY one line 'FINAL ANSWER: <answer>' in the short "
            "form.\nFINAL ANSWER:")


__all__ = [
    "MEMORY_CONDITIONS", "ANSWER_MODES", "DEFAULT_N_BRAINS",
    "SplitPlan", "ConditionBuild",
    "split_across_brains", "light_paraphrase", "build_condition",
    "assemble_recall", "fixed_recall_prompt",
]
