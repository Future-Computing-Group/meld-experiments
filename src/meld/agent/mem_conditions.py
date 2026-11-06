"""M-1 memory-condition harness: measure the merge-protocol DELTA, not agent F1.

MELD's contribution is distributed memory MANAGEMENT (the merge protocol), NOT the agent. The competence gate driver (``run_hotpot_gate.py``)
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
                        its signed deltas; a GLOBAL brain admits every received delta
                        (protocol/admission.py) -> recall from the global brain.
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
the duplicate is a tested pair when received (MERGE under an epoch whose threshold it
clears, RELATE otherwise; without an epoch no similarity merge happens). An optional CONFLICT variant injects one contradictory
sentence so the ``chi=contradicts`` path emits a first-class CONTRADICTS edge.

This module is offline/transport-free: it builds brains and merges; the LLM call
is the caller's (``run_hotpot_memdelta.py`` injects ``MockLLM`` locally or
``OpenWeightLLM`` on the gated host). NO billable run here.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from meld.agent.hotpot_loader import HotpotTask
from meld.agent.memory import DEFAULT_SCOPE, SMPMemory
from meld.protocol.objects import EpochRecord

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


# The copy generator is purely surface: it never rewrites a word of the source (lexical rules such
# as "is a" -> "was a" or "It" -> "This work" change tense, reference or granularity for some
# input). A text whose last sentence is a plain declarative (ends in a period after a lowercase
# letter, a digit or a closing parenthesis, not an abbreviation) gets the v-th connective on that
# sentence; any other text (a question, an exclamation, a closing quote, an abbreviation, no
# terminal punctuation) gets the v-th leading label and is otherwise untouched.
# ponytail: nine variants cap the duplication degree at 10; add suffixes and labels to go higher
_SUFFIXES = (", as recorded.", ", as noted.", ", as reported.", ", as stated.", ", as logged.",
             ", as observed.", ", as written.", ", as mentioned.", ", as described.")
_LABELS = ("Note: ", "Noted: ", "Recorded: ", "Logged: ", "Entry: ", "Remark: ", "Message: ",
           "Record: ", "Memo: ")
_ABBREV = {"etc", "vs", "approx", "incl", "inc", "ltd", "co", "corp", "jr", "sr", "dr", "mr", "mrs",
           "ms", "st", "al", "no", "fig", "vol", "ed", "eds"}
_DECLARATIVE_END = re.compile(r"(?:^|\s)([^\s.]*[a-z0-9)])\.$")


def light_paraphrase(text: str, variant: int = 0) -> str:
    """A small, deterministic, meaning-preserving surface rewrite so a duplicated text is not
    byte-identical (which would trip the cheap kappa exact-key path) yet stays dense-similar (sigma
    high), the regime where the ``same-enough`` MERGE / ``overlaps`` RELATE gate must fire. No word
    of the source changes: a plain declarative ending gets the ``variant``-th connective on its
    last sentence, any other text the ``variant``-th leading label (rules above). Variants 0..8 of
    one text differ from each other and from the text."""
    t = text.rstrip()
    m = _DECLARATIVE_END.search(t)
    if m and m.group(1).rstrip(")").lower() not in _ABBREV:
        return t[:-1] + _SUFFIXES[variant]
    return _LABELS[variant] + t


@dataclass
class SplitPlan:
    """The per-brain observation assignment for one task, with audit counts so a
    test/report can assert overlaps were actually injected."""
    per_brain: List[List[str]] = field(default_factory=list)   # brain -> obs texts
    n_gold: int = 0
    n_duplicated: int = 0          # gold sentences placed on >= 2 brains
    n_paraphrased: int = 0         # duplicates that were lightly paraphrased
    n_conflicts_injected: int = 0  # contradictory sentences injected
    source: dict = field(default_factory=dict)  # placed text -> original text (copies)
    when: dict = field(default_factory=dict)    # placed text -> world time (copies share it)


def split_across_brains(task: HotpotTask, *, n_brains: int = DEFAULT_N_BRAINS,
                        overlap_fraction: float = 0.5,
                        inject_conflict: bool = False, dup_degree: int = 2) -> SplitPlan:
    """Split the task's context across ``n_brains`` SOVEREIGN brains with
    controlled, load-bearing overlap (NOT a clean disjoint partition):

      * every gold supporting sentence is assigned to >= 1 brain (round-robin);
      * a controlled ``overlap_fraction`` of the gold sentences is ALSO placed on
        the next ``dup_degree - 1`` brains (copy j on brain b + j, paraphrase variant
        j - 1), so the global merge sees same-enough duplicates (tested pairs: MERGE or
        RELATE); ``dup_degree <= n_brains``, and 2 is the original single-copy split;
      * distractors are spread round-robin (single-assigned), so the brains are
        genuinely sovereign partial views;
      * if ``inject_conflict``: one gold sentence's value is swapped (e.g. a place
        swap) and placed on a DIFFERENT brain than its original, so the global
        merge sees a contradiction (chi=contradicts -> CONTRADICTS edge).

    The observation text mirrors ``feed_context_through_brain`` ("title: sentence")
    so the brain receives the same shape it does in ``centralized``.

    World time (``plan.when``) is the position of the source sentence in the context (a copy
    shares its source's, the conflict variant that of the sentence it contradicts), and each
    brain's list is in world-time order, so every condition folds one canonical order.
    """
    if n_brains < 1:
        raise ValueError("n_brains must be >= 1")
    if not 1 <= dup_degree <= max(n_brains, 2):
        raise ValueError("dup_degree must be in [1, n_brains]")
    per_brain: List[List[str]] = [[] for _ in range(n_brains)]
    plan = SplitPlan(per_brain=per_brain)

    flat = _paragraph_sentences(task)
    pos: dict = {}
    for i, (title, sent) in enumerate(flat):
        pos.setdefault(f"{title}: {sent}".strip(), i)
    gold = _gold_sentences(task)
    gold_set = set(gold)
    plan.n_gold = len(gold)

    # gold first (round-robin) so every gold sentence lands on >= 1 brain; a
    # fraction are duplicated (paraphrased) onto the next brain.
    n_dup_target = int(round(overlap_fraction * len(gold)))
    for i, (title, sent) in enumerate(gold):
        b = i % n_brains
        per_brain[b].append(f"{title}: {sent}".strip())
        if i < n_dup_target and dup_degree > 1:
            for j in range(1, dup_degree):
                para = light_paraphrase(sent, variant=j - 1)
                per_brain[(b + j) % n_brains].append(f"{title}: {para}".strip())
                plan.source.setdefault(f"{title}: {para}".strip(), f"{title}: {sent}".strip())
            plan.n_duplicated += 1
            plan.n_paraphrased += para != sent

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
            plan.when[f"{title}: {conflicting}".strip()] = pos[f"{title}: {sent}".strip()]
            plan.n_conflicts_injected += 1

    for b in per_brain:
        for t in b:
            if t not in plan.when:
                plan.when[t] = pos[plan.source.get(t, t)]
        b.sort(key=plan.when.__getitem__)
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


def pool_tasks(tasks: List[HotpotTask], per_store: int) -> List[Tuple[HotpotTask, List[HotpotTask]]]:
    """Consecutive groups of ``per_store`` tasks, each as (one task whose context and gold are
    the group's concatenated, the group's tasks), so many brains still hold a non-trivial
    slice of one persistent store; recall stays scored per member question."""
    out = []
    for i in range(0, len(tasks), per_store):
        group = tasks[i:i + per_store]
        out.append((HotpotTask(task_id="+".join(t.task_id for t in group), question="", answer="",
                               level="pool", qtype="pool",
                               context=[p for t in group for p in t.context],
                               supporting_facts=[f for t in group for f in t.supporting_facts]),
                    group))
    return out


# ---------------------------------------------------------------------------
# building the global brain per condition
# ---------------------------------------------------------------------------


@dataclass
class ConditionBuild:
    """The memory state for one (task, condition): a recall source + audit."""
    condition: str
    brain: Optional[SMPMemory]          # None for none / full_context
    inline_context: List[str]           # non-empty only for full_context
    n_merge: int = 0                    # MERGE decisions (similarity merges)
    n_relate: int = 0                   # RELATE decisions (overlaps, abstentions)
    n_conflict: int = 0                 # CONFLICT decisions (contradiction edges)
    n_total_patches: int = 0            # admission decisions
    n_overlap_merges: int = 0           # MERGE decisions (every MERGE is a similarity merge)
    stored_units: int = 0               # nodes held in the store (provenance)
    live_units: int = 0                 # live (recallable) nodes -- the footprint
    split: Optional[SplitPlan] = None


def _new_memory(encoder: str, nli: str, epoch: Optional[EpochRecord],
                cache_dir: Optional[str] = None, *, origin: str = "agent",
                clock=None) -> SMPMemory:
    # A shared cache_dir lets every brain (sovereign + global) and every condition reuse the
    # same on-disk sentence embeddings, so a sweep embeds each sentence once instead of
    # re-embedding it per build (a redundant-load slowness). A per-build counter as the clock
    # makes every delta id, hence every build, reproducible.
    return SMPMemory(encoder=encoder, nli=nli, epoch=epoch, cache_dir=cache_dir,
                     origin=origin, clock=clock or itertools.count(1).__next__)


def _fed(texts, when, write: str, encoder, nli, epoch, cache_dir, origin: str = "agent"):
    """A memory that writes (``write``: "write" or "publish") each text in the given order with
    its world time + 1 as created_at, as eval/topology does."""
    cur = [0]
    mem = _new_memory(encoder, nli, epoch, cache_dir, origin=origin, clock=lambda: cur[0])
    for t in texts:
        cur[0] = when[t] + 1
        getattr(mem, write)(t)
    return mem


def _counts(condition: str, mem: SMPMemory, split=None) -> ConditionBuild:
    """The build record: the decision of every node as the store now holds them (after any
    refold an out-of-order arrival caused; exact copies are one node) and the footprint (a
    merge deprecates redundant duplicates, so live < stored)."""
    decisions = mem.admission.decisions(DEFAULT_SCOPE)
    n = {"MERGE": 0, "RELATE": 0, "CONFLICT": 0}
    for d in decisions:
        n[d.decision] = n.get(d.decision, 0) + 1
    f = mem.footprint()
    return ConditionBuild(condition=condition, brain=mem, inline_context=[],
                          n_merge=n["MERGE"], n_relate=n["RELATE"], n_conflict=n["CONFLICT"],
                          n_overlap_merges=n["MERGE"], n_total_patches=len(decisions),
                          stored_units=f["stored"], live_units=f["live"], split=split)


def build_condition(task: HotpotTask, condition: str, *, encoder: str, nli: str,
                    epoch: Optional[EpochRecord] = None, n_brains: int = DEFAULT_N_BRAINS,
                    overlap_fraction: float = 0.5,
                    inject_conflict: bool = False,
                    cache_dir: Optional[str] = None, policy=None,
                    dup_degree: int = 2) -> ConditionBuild:
    """Materialize the memory state for ONE (task, condition).

    - ``none``: no brain, no inline context.
    - ``full_context``: all paragraphs inlined; no brain.
    - ``centralized``: every observation -> ONE brain (admission, but all from one
      author, so overlaps only arise from corpus-internal repeats).
    - ``distributed_merge``: overlap-split across N sovereign brains; every signed
      delta of each is received and admitted by a GLOBAL brain (the MELD path);
      recall from the global brain.
    - ``naive_union``: same split, but the global brain holds the raw UNION of the
      per-brain deltas with NO admission (exact copies are one node; no MERGE,
      RELATE or CONFLICT ever fires, no link exists).

    ``epoch`` is the scope's epoch record (see SMPMemory); without one no similarity
    merge happens in any condition.

    ``policy`` (a baseline, see meld.baselines): with ``distributed_merge`` only, the global
    brain is a ``BaselineMemory`` that applies the policy to every received delta instead of
    admission; the split and the sovereign brains are unchanged. The build is labelled
    ``policy.name``.
    """
    if condition not in MEMORY_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; "
                         f"choose from {MEMORY_CONDITIONS}")
    if policy is not None and condition != "distributed_merge":
        raise ValueError("a baseline policy replaces the global brain's admission: "
                         "distributed_merge only")

    if condition == "none":
        return ConditionBuild(condition=condition, brain=None, inline_context=[])

    if condition == "full_context":
        inline = [f"{title}: {sent}"
                  for title, sentences in task.context for sent in sentences]
        return ConditionBuild(condition=condition, brain=None,
                              inline_context=inline)

    if condition == "centralized":
        one = split_across_brains(task, n_brains=1, dup_degree=1, overlap_fraction=0.0)
        return _counts(condition, _fed(one.per_brain[0], one.when, "write", encoder, nli, epoch,
                                       cache_dir))

    # distributed conditions share the same overlap-injecting split
    split = split_across_brains(task, n_brains=n_brains,
                                overlap_fraction=overlap_fraction,
                                inject_conflict=inject_conflict, dup_degree=dup_degree)

    if condition == "naive_union":
        # the ABLATION: every per-brain observation enters the global store's D with no
        # admission -- a raw union, never a merge/relate/conflict.
        mem = _fed([t for b in split.per_brain for t in b], split.when, "publish", encoder, nli,
                   epoch, cache_dir)
        return _counts("naive_union", mem, split)

    # distributed_merge: each sovereign brain ingests its own observations, then
    # PUBLISHes its signed deltas to a GLOBAL brain that admits each on receipt.
    return _build_distributed_merge(split, encoder, nli, epoch, cache_dir, policy)


def _build_distributed_merge(split: SplitPlan, encoder: str, nli: str,
                             epoch: Optional[EpochRecord],
                             cache_dir: Optional[str] = None, policy=None) -> ConditionBuild:
    sovereign = [_fed(obs, split.when, "write", encoder, nli, epoch, cache_dir, origin=f"agent{i}")
                 for i, obs in enumerate(split.per_brain)]

    if policy is None:
        global_brain = _new_memory(encoder, nli, epoch, cache_dir)
    else:
        from meld.baselines.memory import BaselineMemory
        global_brain = BaselineMemory(policy, encoder=encoder, nli=nli, epoch=epoch,
                                      cache_dir=cache_dir, clock=itertools.count(1).__next__)
    # each sovereign brain PUBLISHes the original signed bytes of its deltas; the global
    # brain's receive gate verifies and admits each (the duplicated, paraphrased
    # sentences are what reach the test). Delivery is in canonical order (created_at, delta
    # id), so every admission is canonical-order last and none refolds.
    held = [(mem.store.objects.get("D", did).created_at, did, mem.store.objects.raw(did)[:2])
            for mem in sovereign for did in mem.store.objects.ids("D", DEFAULT_SCOPE)]
    for _, _, raw in sorted(held):
        global_brain.receive(*raw)
    return _counts(policy.name if policy else "distributed_merge", global_brain, split)


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
    "split_across_brains", "light_paraphrase", "build_condition", "pool_tasks",
    "assemble_recall", "fixed_recall_prompt",
]
