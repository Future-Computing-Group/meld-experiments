"""Source loaders: each turns one public dataset into claim-claim Pairs.

Every loader sets ``source_item_id`` at the grain that must not straddle splits:
a PAWS item is a connected component of pairs sharing a sentence, a VitaminC
item is a Wikipedia page, a SICK item is a component of pairs sharing a sentence, a LongMemEval item is a component of knowledge-update
questions sharing an answer session, a SciFact item is the abstract both claims
cite, a gold item is a seed. A source label carries a claim only where it is
verified; everything else goes to the annotation panel:

  S1 PAWS label 1          MERGE (source)
  S2 PAWS label 0          non-mergeable, verdict to annotate, ADVERSARIAL_NONPARA
  S3 VitaminC not factual  MERGE pre-label (source_filtered): survives the noise, number,
                           entity, new-content and vandalism filters; annotators confirm it
  S4 VitaminC factual      non-mergeable, verdict and subtype to annotate
  S5 LongMemEval KU        CONFLICT / SUPERSESSION (source) only when the newer sentence
                           contains the answer and the older one a superseded value;
                           otherwise to annotate, or dropped when neither holds
  S6 SciFact               to annotate (opposite-stance claims on one abstract are often
                           about different attributes, so no label is asserted)
  S0 constructed gold      its own labels (construction)
  S8 SICK                 both-way entailment MERGE, A entails B with B neutral to A RELATE (text_a
                          entails text_b), both-way contradiction CONFLICT, all source; other pairs dropped

``dedupe_pairs`` runs over the pooled pairs of all sources: it drops pairs whose two
sides are identical after normalisation, drops every copy of an edit that occurs
with two different labels, and keeps one copy of a repeated pair (either side order).
"""
from __future__ import annotations

import csv
import difflib
import json
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from meld.bench.schema import Pair, make_pair

_TOK = re.compile(r"\w+")
STOP = frozenset((
    "a an the of in on at to for from by with and or but nor is was are were be been being has have had do does "
    "did it its this that these those as he she they we you i his her their our your him them us me my mine who "
    "whom which what when where while why how also into than then there here over under after before about up "
    "down out not no yes so such only very more most other some any all each both own same just can could will "
    "would shall should may might must").split())
PROFANITY = frozenset((
    "fuck fucking fucked fucker shit shitty ass asshole dumbass bitch cunt dick cock penis vagina poop crap suck "
    "sucks sucked stupid idiot moron retard fag faggot nigger whore slut lol lmao haha hahaha").split())
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update(thirty="30", forty="40", fifty="50", sixty="60", seventy="70", eighty="80", ninety="90",
                    hundred="100", thousand="1000", million="1000000", billion="1000000000")
MAX_NOVEL_CONTENT = 0      # new content words allowed in an S3 edit (typo fixes are not new words)
NEGATIONS = frozenset("not no never none nobody nothing neither nor without".split())
# S5 pairs removed by hand after reading: the rule fired on a number unrelated to the answer.
S5_EXCLUDE = {"309bb0266f88b0c7": "old-value rule fired on an unrelated number (hotel category)"}


def _components(node_keys: Dict[str, Iterable[str]]) -> Dict[str, str]:
    """Union nodes that share any key; map each node to the smallest node of its component."""
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for node, keys in node_keys.items():
        for k in keys:
            ra, rb = find("n:" + node), find("k:" + k)
            if ra != rb:
                parent[rb] = ra
    members = defaultdict(list)
    for node in node_keys:
        members[find("n:" + node)].append(node)
    return {n: min(ms) for ms in members.values() for n in ms}


def _dedupe(pairs: List[Pair]) -> List[Pair]:
    return list({p.pair_id: p for p in pairs}.values())


def _n2(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def dedupe_pairs(pairs: Sequence[Pair], stats: Optional[Counter] = None) -> List[Pair]:
    stats = Counter() if stats is None else stats
    groups = defaultdict(list)
    for p in sorted(pairs, key=lambda p: p.pair_id):
        a, b = _n2(p.text_a), _n2(p.text_b)
        if a == b:
            stats["identical_sides"] += 1
            continue
        groups[frozenset((a, b))].append(p)
    out = []
    for g in groups.values():
        if len({(p.stratum, p.label) for p in g}) > 1:
            stats["label_conflict"] += len(g)
            continue
        out.append(g[0])
        stats["repeat"] += len(g) - 1
    return out


def _numbers(t: str) -> List[str]:
    """Digit runs and cardinal number words, in order of appearance."""
    out = []
    for w in re.findall(r"\d+|[A-Za-z]+", t):
        if w.isdigit():
            out.append(w)
        elif w.lower() in NUMBER_WORDS:
            out.append(NUMBER_WORDS[w.lower()])
    return out


# ---------------------------------------------------------------- PAWS (S1, S2)

def detok_paws(t: str) -> str:
    t = " ".join(t.replace("``", '"').replace("''", '"').replace(" -- ", " \u2013 ").split())
    t = re.sub(r"\s+([,.;:!?)\]%])", r"\1", t)
    t = re.sub(r"([(\[$])\s+", r"\1", t)
    t = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b", r"\1", t)
    return re.sub(r'"\s*(.*?)\s*"', r'"\1"', t)


def load_paws(path: str, *, origin_split: str) -> List[Pair]:
    import pandas as pd

    df = pd.read_parquet(path)
    rows = [(str(r.id), detok_paws(r.sentence1), detok_paws(r.sentence2), int(r.label))
            for r in df.itertuples(index=False)]
    comp = _components({rid: (a, b) for rid, a, b, _ in rows})
    out = []
    for rid, a, b, lab in rows:
        common = dict(text_a=a, text_b=b, source="paws", origin_split=origin_split,
                      source_item_id=f"paws:{origin_split}:{comp[rid]}")
        if lab == 1:
            out.append(make_pair(stratum="S1_paws_para", label="MERGE", label_provenance="source",
                                 mergeable=True, **common))
        else:
            out.append(make_pair(stratum="S2_paws_nonpara", label=None, label_provenance="to_annotate",
                                 mergeable=False, subtype="ADVERSARIAL_NONPARA", **common))
    return _dedupe(out)


# ----------------------------------------------------------- VitaminC (S3, S4)

_DIFF = re.compile(r"\{([+\-*])\|(.*?)\|\}")
_INTERLANG = re.compile(r"\b[a-z]{2,3}(?:-[a-z]+)?:[^\W\d_]")


def vitaminc_keep(row: dict) -> bool:
    """Noise filter: drop punctuation/whitespace-only diffs, sides under 6 tokens, and
    edits touching more than half of either side's tokens (rewrites, vandalism)."""
    a, b = row["sent_a"], row["sent_b"]
    if re.sub(r"\W+", "", a.lower()) == re.sub(r"\W+", "", b.lower()):
        return False
    na, nb = len(a.split()), len(b.split())
    if min(na, nb) < 6:
        return False
    old = new = 0
    for op, body in _DIFF.findall(row.get("diff_viz") or ""):
        if op == "*":
            o, _, n = body.partition(" -> ")
            old, new = old + len(o.split()), new + len(n.split())
        elif op == "-":
            old += len(body.split())
        else:
            new += len(body.split())
    return max(old / na, new / nb) <= 0.5


def clean_wiki(t: str) -> Optional[str]:
    """Strip wiki markup and CSV quote doubling; None when the text is unrecoverable
    (replacement characters, interlanguage links, unclosed ref tags)."""
    if "\ufffd" in t or "\u00ef\u00bf\u00bd" in t or _INTERLANG.search(t):
        return None
    t = re.sub(r"<ref[^>]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t)
    if re.search(r"</?ref|ref>", t):
        return None
    for junk in ("{{", "}}", "[[", "]]"):
        t = t.replace(junk, "")
    if '""' in t:                                  # CSV-style field: outer quotes, inner quotes doubled
        if len(t) > 1 and t[0] == t[-1] == '"' and t[1] != '"':
            t = t[1:-1]
        t = t.replace('""', '"')
        if t.count('"') % 2 and t.startswith('"'):
            t = t[1:]
        if t.count('"') % 2 and t.endswith('"'):
            t = t[:-1]
    t = " ".join(t.split())
    return t or None


def _caps(t: str) -> List[str]:
    """Capitalised tokens in order; a sentence-initial one counts unless it is a function word."""
    out, initial = [], True
    for w in re.findall(r"\w[\w'’-]*|[.!?]", t):
        if w in ".!?":
            initial = True
            continue
        if w[0].isupper() and not (initial and w.lower() in STOP):
            out.append(w)
        initial = False
    return out


def _content(t: str) -> set:
    return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 2 and w not in STOP}


def _novel(x: set, y: set) -> List[str]:
    """Words of x absent from y that are not a spelling variant of a word that left y."""
    gone = y - x
    return [w for w in x - y if not any(difflib.SequenceMatcher(None, w, u).ratio() >= 0.8 for u in gone)]


def _negations(t: str) -> List[str]:
    t = re.sub(r"n[’']t\b", " not", t.lower()).replace("cannot", "can not")
    return sorted(w for w in _TOK.findall(t) if w in NEGATIONS)


def s3_reject_reason(a: str, b: str) -> Optional[str]:
    """Why a 'not factual' revision is not trusted as a paraphrase, or None. Checks in order:
    vandalism (a profanity token on one side only), negation parity (the multiset of negation
    markers, with n't and cannot read as not), any change in the ordered numbers (digits
    and number words), any change or swap in the ordered capitalised tokens (a cheap named-entity
    proxy), and new content words beyond MAX_NOVEL_CONTENT in either direction."""
    wa, wb = set(_TOK.findall(a.lower())), set(_TOK.findall(b.lower()))
    if (wa ^ wb) & PROFANITY:
        return "vandalism"
    if _negations(a) != _negations(b):
        return "negation"
    if _numbers(a) != _numbers(b):
        return "number"
    if _caps(a) != _caps(b):
        return "entity"
    ca, cb = _content(a), _content(b)
    if max(len(_novel(ca, cb)), len(_novel(cb, ca))) > MAX_NOVEL_CONTENT:
        return "content"
    return None


def load_vitaminc(path: str, *, origin_split: str, stats: Optional[Counter] = None) -> List[Pair]:
    stats = Counter() if stats is None else stats
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not vitaminc_keep(r):
                stats["noise"] += 1
                continue
            a, b = clean_wiki(r["sent_a"]), clean_wiki(r["sent_b"])
            if a is None or b is None:
                stats["unclean"] += 1
                continue
            common = dict(text_a=a, text_b=b, source="vitaminc", origin_split=origin_split,
                          source_item_id=f"vitaminc:page:{r['page']}")
            if r["label"] == "not factual":
                why = s3_reject_reason(a, b)
                stats[f"s3_{why or 'kept'}"] += 1
                if why is None:
                    out.append(make_pair(stratum="S3_vitc_nonfactual", label="MERGE",
                                         label_provenance="source_filtered", mergeable=True, **common))
            elif r["label"] == "factual":
                stats["s4_kept"] += 1
                out.append(make_pair(stratum="S4_vitc_factual", label=None,
                                     label_provenance="to_annotate", mergeable=False, **common))
    return _dedupe(out)


# --------------------------------------------------------------- LongMemEval (S5)

_SENT = re.compile(r"(?<=[.!?])\s+")


def _sentences(turn: str) -> List[str]:
    return [s for s in _SENT.split(turn.strip()) if len(s.split()) >= 4] or [turn.strip()]


def _jac(x: str, y: str) -> float:
    a, b = set(_TOK.findall(x.lower())), set(_TOK.findall(y.lower()))
    return len(a & b) / len(a | b) if a | b else 0.0


def _best_pair(olds: Sequence[str], news: Sequence[str], question: str):
    """The (old, new) pair maximising token Jaccard with the question plus with each other.
    Model-free on purpose: the benchmark's content must not depend on the encoder it evaluates."""
    cands = [(o, n) for o in olds for n in news]
    cands = [p for p in cands if p[0].lower() != p[1].lower()] or cands   # a repeated sentence carries no update
    return max(cands, key=lambda p: _jac(p[0], question) + _jac(p[1], question) + _jac(*p))


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def load_lme_ku(path: str, stats: Optional[Counter] = None) -> List[Pair]:
    """Knowledge-update items with exactly two has_answer turns -> (older, newer) sentences.

    Verified (CONFLICT / SUPERSESSION, source) when the newer sentence contains the answer
    (all its numbers when it has any, else all its content tokens) and the older sentence
    contains a number that is in neither the answer nor the question (the superseded value).
    The superseded value is derivable only for numeric answers; a pair meeting one condition
    is to_annotate, a pair meeting neither is dropped."""
    stats = Counter() if stats is None else stats
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)
    clean = {}
    for it in items:
        if it["question_type"] != "knowledge-update":
            continue
        turns = []
        sess = sorted(zip(it["haystack_dates"], range(len(it["haystack_sessions"])),
                          it["haystack_session_ids"], it["haystack_sessions"]))
        for date, _, sid, ts in sess:
            turns += [(date, sid, t["content"]) for t in ts if t.get("has_answer")]
        if len(turns) == 2:
            clean[it["question_id"]] = (it, turns)
        else:
            stats["not_two_turns"] += 1
    comp = _components({q: [sid for _, sid, _ in t] for q, (_, t) in clean.items()})
    out = []
    for q, (it, ((d_old, _, old), (d_new, _, new))) in clean.items():
        question, answer = it["question"], str(it["answer"])
        q_num = set(_numbers(question))
        a_num = set(_numbers(answer)) - q_num
        a_words = {_stem(w) for w in _TOK.findall(answer.lower()) if w not in STOP and not w.isdigit()}

        def has_new(s):
            if a_num:
                return a_num <= set(_numbers(s))
            return bool(a_words) and a_words <= {_stem(w) for w in _TOK.findall(s.lower())}

        def has_old(s):
            return bool(a_num) and bool(set(_numbers(s)) - a_num - q_num)

        olds, news = _sentences(old), _sentences(new)
        o_ok, n_ok = [s for s in olds if has_old(s)], [s for s in news if has_new(s)]
        if not (o_ok or n_ok):
            stats["dropped"] += 1
            continue
        a, b = _best_pair(o_ok or olds, n_ok or news, question)
        common = dict(text_a=a, text_b=b, source="longmemeval", source_item_id=f"lme:{comp[q]}",
                      stratum="S5_lme_ku", badges_a={"time": d_old.split()[0]}, badges_b={"time": d_new.split()[0]})
        if make_pair(label=None, label_provenance="to_annotate", mergeable=None, **common).pair_id in S5_EXCLUDE:
            stats["excluded"] += 1
            continue
        if o_ok and n_ok:
            stats["verified"] += 1
            lab = dict(label="CONFLICT", subtype="SUPERSESSION", label_provenance="source", mergeable=False)
        else:
            stats["new_only" if n_ok else "old_only"] += 1
            lab = dict(label=None, label_provenance="to_annotate", mergeable=None)
        out.append(make_pair(**common, **lab))
    return _dedupe(out)


def assign_sessions(session_ids: Sequence[str], *, n_agents: int, q: float, seed: int) -> Dict[str, List[int]]:
    """Session split for independently evolving memories: sessions (in the given, time order)
    get owners round-robin over a seeded agent permutation; every other agent also holds a
    session with probability q, so expected copies are 1 + q (n_agents - 1)."""
    rng = random.Random(seed)
    perm = list(range(n_agents))
    rng.shuffle(perm)
    out = {}
    for i, sid in enumerate(session_ids):
        owner = perm[i % n_agents]
        out[sid] = sorted([owner] + [a for a in range(n_agents) if a != owner and rng.random() < q])
    return out


# ------------------------------------------------------------------ SciFact (S6)

def load_scifact_conflicts(paths: Sequence[str]) -> List[Pair]:
    """Claim-claim pairs: a claim the abstract SUPPORTs against a claim it CONTRADICTs, all
    to annotate. Claims are paired only within one file (one official split)."""
    out = []
    for path in paths:
        by_doc = defaultdict(lambda: {"SUPPORT": [], "CONTRADICT": []})
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                for doc, groups in (r.get("evidence") or {}).items():
                    stance = groups[0].get("label") if groups else None
                    if stance in ("SUPPORT", "CONTRADICT"):
                        by_doc[doc][stance].append((r["id"], r["claim"].strip()))
        for doc in sorted(by_doc, key=int):
            for _, a in sorted(by_doc[doc]["SUPPORT"]):
                for _, b in sorted(by_doc[doc]["CONTRADICT"]):
                    out.append(make_pair(text_a=a, text_b=b, source="scifact",
                                         source_item_id=f"scifact:doc:{doc}", stratum="S6_scifact_conflict",
                                         label=None, label_provenance="to_annotate", mergeable=None))
    return _dedupe(out)


# ---------------------------------------------------------------------- SICK (S8)

SICK_CLASSES = {("A_entails_B", "B_entails_A"): dict(stratum="S8_sick_para", label="MERGE", mergeable=True),
                ("A_entails_B", "B_neutral_A"): dict(stratum="S8_sick_oneway", label="RELATE", mergeable=False),
                ("A_contradicts_B", "B_contradicts_A"): dict(stratum="S8_sick_contra", label="CONFLICT",
                                                             mergeable=False)}


def load_sick(path: str) -> List[Pair]:
    """SICK.txt (tab-separated, no quoting) -> S8 pairs. A source item is a connected component of
    kept pairs sharing a sentence (case and whitespace folded): SICK reuses sentences across pairs."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
                if (r["entailment_AB"], r["entailment_BA"]) in SICK_CLASSES]
    norm = lambda t: " ".join(t.lower().split())  # noqa: E731
    comp = _components({r["pair_ID"]: (norm(r["sentence_A"]), norm(r["sentence_B"])) for r in rows})
    return _dedupe([make_pair(text_a=r["sentence_A"].strip(), text_b=r["sentence_B"].strip(), source="sick",
                              source_item_id=f"sick:{comp[r['pair_ID']]}", label_provenance="source",
                              **SICK_CLASSES[(r["entailment_AB"], r["entailment_BA"])])
                    for r in rows])


# ----------------------------------------------------------- constructed gold (S0)

def load_gold() -> List[Pair]:
    from meld.merge.m2_gold import build_gold_set

    return [make_pair(text_a=r["a"], text_b=r["b"], source="meld_gold", source_item_id=f"gold:{r['seed_id']}",
                      stratum="S0_gold", label=r["label"], label_provenance="construction",
                      mergeable=r["label"] == "MERGE")
            for r in build_gold_set()]
