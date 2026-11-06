# Copyright 2026 Caura (https://caura.ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications: excerpted from four upstream files (below). The excerpted lines are
# byte-identical to upstream, except that the two env-tunable contradiction constants are
# given their upstream defaults as literals (read_float_env / read_int_env not vendored);
# only this header, the docstring, the imports and the logger line were added.
"""Caura write-gate and contradiction-judge prompts and parsers, vendored verbatim from
caura-ai/caura at commit fd1da78b563b21f357fe5a7060a79954c9df94c0 (tag backend-v3.34.5).

Source: https://codeload.github.com/caura-ai/caura/tar.gz/fd1da78b563b21f357fe5a7060a79954c9df94c0
  tarball sha256 d8347126b147b839fecb18f11416c11b4255058002e5aed1e6d6c6334698b4d3 (vendor/caura/PIN)
  common/constants.py sha256 9fc38194803733214826c8347824e0813ba2a6f8ee0c4cee885c02f6adb746af
    lines 93-98 SEMANTIC_DEDUP_AUTO_THRESHOLD, SEMANTIC_DEDUP_JUDGE_THRESHOLD; line 123
    CONTRADICTION_SIMILARITY_THRESHOLD (default 0.45), line 137 CONTRADICTION_CANDIDATE_WINDOW (default 20)
  core-api/src/core_api/services/dedup_judge.py sha256 6d7cdf92e1c913aa15cd12343de3d5661031d3f7bde6e3b335835ff836b2e20b
    lines 36-38 confidences, 41-61 DEDUP_PROMPT, 64-92 _judge_dedup, _fake_dedup_check,
    139-142 DEDUP_JUDGE_CONFIDENCE_THRESHOLD
  core-api/src/core_api/services/dedup_identifier_filter.py sha256 8ccfaa57bd2ba2bb752f64c35459333ace536ca29a468051ecc3fea8954e3c59
    lines 27-79 _IDENTIFIER_PATTERNS, _content_is_identifier_bearing
  core-api/src/core_api/services/contradiction_detector.py sha256 0b5427e721e7bd1319f95cd80eb7892c5f148cdee4eaf77718cbad7f55918eb3
    lines 1251-1360 CONTRADICTION_PROMPT, 1510-1594 NON_CONFLICT_REASONS, _parse_contradiction_response,
    1600-1611 confidences (_CONF_CLEAN 0.90 and _CONF_FALLBACK 0.50 equal dedup_judge's),
    1636-1666 _judge_contradiction, 1743-1839 BATCH_CONTRADICTION_PROMPT, _expand_sparse_batch
The calling logic (pipeline/steps/write/check_semantic_duplicate.py CheckSemanticDuplicate.execute,
dedup_judge._llm_dedup_check, contradiction_detector._detect semantic path,
_llm_contradiction_check, _llm_contradiction_check_batch, and the OpenAI-compatible
complete_json transport) is re-implemented in meld.baselines.caura_policy;
tests/test_caura_baseline.py pins the rendered prompt hashes and, when vendor/caura/src is
present, the byte identity of every excerpt.
"""
import logging
import re

logger = logging.getLogger(__name__)

SEMANTIC_DEDUP_AUTO_THRESHOLD = 0.97  # auto-reject band — tighter than the
# legacy 0.95 so refinements (same first sentence + extra detail) drop into
# the judge band rather than getting silently rejected.
SEMANTIC_DEDUP_JUDGE_THRESHOLD = 0.85  # broad enough to catch refinements,
# tight enough that the LLM judge isn't swamped with unrelated memories that
# happen to share vocabulary.
CONTRADICTION_SIMILARITY_THRESHOLD = 0.45
CONTRADICTION_CANDIDATE_WINDOW = 20


# Confidence rubric. See module docstring for the rationale.
_CONF_CLEAN = 0.90
_CONF_FALLBACK = 0.50


DEDUP_PROMPT = """\
You are a deduplication judge for a personal memory system.

Two short statements are about to be stored. Decide whether they
encode the SAME memory (i.e. one is a duplicate or trivial paraphrase
of the other) or DIFFERENT memories (even if they overlap in topic).

Statement A (NEW): {new_content}
Statement B (EXISTING): {old_content}

Return STRICT JSON with no surrounding prose:
{{
  "is_duplicate": true|false,
  "reason": "<one short clause>"
}}

A is a duplicate of B iff a reasonable reader would consider the two
interchangeable carriers of the same fact. Refinements that ADD new
information are NOT duplicates. Different subjects are NOT duplicates
even if their predicates match.
"""


def _judge_dedup(raw) -> tuple[bool, float]:
    """Parse the LLM response into ``(is_duplicate, confidence)``.

    Conservative defaults: anything that doesn't look like a well-formed
    ``{is_duplicate: bool, reason: str}`` dict → ``(False, 0.50)``.
    Downstream callers gate 409 on confidence, so this floor means a
    flaky response will not 409 a legitimate write.
    """
    if not isinstance(raw, dict):
        return False, _CONF_FALLBACK

    is_dup_raw = raw.get("is_duplicate")
    reason_raw = raw.get("reason")
    if not isinstance(is_dup_raw, bool):
        return False, _CONF_FALLBACK
    if not isinstance(reason_raw, str) or not reason_raw.strip():
        return False, _CONF_FALLBACK
    return is_dup_raw, _CONF_CLEAN


def _fake_dedup_check(new_content: str, old_content: str) -> bool:
    """Heuristic fallback when no LLM credentials are available.

    Conservative: only flags as duplicate when the two strings are
    near-identical after whitespace normalisation. Real dedup judgement
    is best deferred to the LLM; this exists so the pipeline doesn't
    raise on test/CI environments without credentials.
    """
    return " ".join(new_content.split()).lower() == " ".join(old_content.split()).lower()


# Below this, callers must NOT 409 — even if ``is_duplicate=True``.
# Set at gate-1 / gate-2 confidence floor so heuristic fallbacks and
# malformed responses don't sink legitimate writes.
DEDUP_JUDGE_CONFIDENCE_THRESHOLD = 0.60


# Each pattern matches a single token shape. None overlap with prose-
# common shapes (years, prices, decimal counts) — verified by the
# unit tests in ``tests/test_a1_identifier_prefilter.py``.
_IDENTIFIER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # UUIDs (8-4-4-4-12 hex).
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    # PR refs: ``PR#123``, ``pr-456``, ``#789`` (with surrounding text).
    re.compile(r"\b[Pp][Rr][-#]\d+\b"),
    # ``#nnnn`` (issue/PR shorthand) — require ≥ 3 digits to dodge ``#5``-style emphasis.
    re.compile(r"#\d{3,}\b"),
    # ``word#nnn`` — any non-digit word followed by ``#`` + digits.
    # This is unambiguously an identifier shape (issue, ticket, build,
    # internal ref) and covers the ``X#42`` motivating case where
    # only 2 digits follow the ``#``.
    re.compile(r"\b[A-Za-z]\w*#\d+\b"),
    # Build numbers: ``build-123``, ``build#123``, ``build 1024`` (case-insensitive).
    re.compile(r"\b[Bb]uild[-# ]\d+\b"),
    # Semver: ``v1.2.3`` / ``1.2.3`` / ``v2.0.0-rc4`` / ``2.0.0-rc4``. Requires
    # leading ``v`` OR a hyphen-suffixed pre-release tag so plain prose
    # decimals (``$1.99``) don't trip it.
    re.compile(r"\bv\d+\.\d+\.\d+(?:[-+][\w.]+)?\b"),
    re.compile(r"\b\d+\.\d+\.\d+[-+][\w.]+\b"),
    # Ticket refs: ``PROJECT-123`` (2+ uppercase letters + hyphen + digits).
    re.compile(r"\b[A-Z]{2,}-\d+\b"),
    # ``sha256:`` and similar content-addressed prefixes.
    re.compile(r"\b(?:sha\d*|md5):[0-9a-fA-F]{6,}\b", re.IGNORECASE),
    # Standalone short git SHAs — require explicit context word
    # (``commit``, ``sha``, ``hash``) before the hex to avoid matching
    # everyday hex-looking words (``decade``, ``acceded``, ``café``
    # is filtered out by the \b anchors + the context word).
    re.compile(
        r"\b(?:commit|sha|hash|rev|ref)\s+[0-9a-f]{7,40}\b",
        re.IGNORECASE,
    ),
    # Full 40-char git SHAs — long enough that they can stand alone
    # without context (no English word is 40 hex chars).
    re.compile(r"\b[0-9a-f]{40}\b", re.IGNORECASE),
)


def _content_is_identifier_bearing(content: str) -> bool:
    """Return True iff ``content`` carries any identifier-shaped token.

    See module docstring for the rationale. Used by
    ``CheckSemanticDuplicate`` to short-circuit semantic dedup when
    the identifier IS the disambiguator and the surrounding template
    is what the embedder sees.

    Returns False (don't pre-filter) for any non-string or empty input.
    """
    if not isinstance(content, str) or not content:
        return False
    return any(pat.search(content) for pat in _IDENTIFIER_PATTERNS)


CONTRADICTION_PROMPT = """\
You are a contradiction detector for a business memory system.

Two statements contradict ONLY IF they make incompatible claims about the
SAME real-world subject. Different subjects -> NOT a contradiction, even if
the predicates look opposite or the statements look semantically similar.

Statement A (NEW): {new_content}

Statement B (EXISTING): {old_content}

Follow these steps in order:

1. Extract subject_a: the entity Statement A is primarily about
   (person, company, project, product, etc.). Use a short noun phrase.
2. Extract subject_b: the entity Statement B is primarily about.
3. Decide same_subject. Set true ONLY when subject_a and subject_b refer
   to the SAME real-world entity. Treat these as same_subject=true:
     - exact name match
     - known alias / nickname / abbreviation of the same entity
     - role description and proper name referring to the same individual
       in context (e.g., "the CEO" and "Sarah Johnson" when context makes
       it unambiguous)
     - pronoun resolved unambiguously to the other statement's subject
   Treat these as same_subject=false:
     - two different people who share a first name or last name
     - two different companies, products, projects, or teams
     - any case where you are not confident the subjects are the same entity
4. Decide non_conflict_reason. Even when same_subject is true, certain
   shapes describe two claims that BOTH hold and so are NOT a
   contradiction. Pick at most one value; pick "none" when the two
   statements really do assert mutually exclusive states.
     - "temporal_supersession": BOTH statements are records of PAST
       events or completed milestones that stay true as history
       ("hired in 2020" / "promoted in 2023"; "v1 shipped in March" /
       "v2 shipped in June"). Records of what happened never compete.
       This reason does NOT apply when the statements assert a
       subject's CURRENT state or attribute and the newer one changes
       it ("status: planned" vs "status: shipped", "lives in X" vs
       "lives in Y", "runs at 01:00" vs "runs at 03:00") — a state
       change is an update, and updates ARE contradictions
       (contradicts=true, reason "none"): flagging them is how the
       stale value gets retired.
     - "list_valued_predicate": the two statements describe attributes
       of the same subject that do not compete for a single slot.
       Two shapes both qualify:
       (a) one predicate that naturally holds multiple values at the
       same time — "supports English" / "supports French"; speaks
       multiple languages; "reports_to" in matrix orgs; "works_on"
       parallel projects;
       (b) two entirely different attributes of the same subject —
       e.g., "Alice was promoted to Senior Engineer" (her title)
       and "Alice is on the platform team" (her team) are
       complementary facts; both hold simultaneously. Different
       attributes do not exclude each other.
     - "refinement": one statement is a more specific version of the
       other ("Europe" vs "Munich"; "tech" vs "Google"; "Q3" vs
       "September 15"). Both hold; finer granularity does not negate
       coarser.
     - "scope_mismatch": the statements describe the same subject with
       different implicit qualifiers — whole vs part (parent company
       vs division), different time windows (annual vs quarterly), or
       different context qualifiers (weekday vs weekend, work vs
       residence). Both can hold simultaneously.
     - "same_name_distinct_subject": subject_a and subject_b share a
       surface name but plausibly refer to different real-world
       instances (two different builds of "the nightly build", two
       different days of "today's standup", two different people both
       called "John" without disambiguator). This is the symmetric
       complement to same_subject=false for cases where the names
       happen to match.
     - "conditional_unrealized": one statement is conditional /
       hypothetical / irrealis ("if X then Y", "would", "could",
       "might"), and the other is a realised state. The conditional
       does not assert a claim that can contradict.
     - "event_restatement": the two statements describe the SAME
       event with different tense, aspect, or synonymous verbs
       ("acquired" / "is acquiring" the same deal; "was hired" /
       "joined"). They restate the same fact, not different facts.
     - "none": none of the above applies. Use this when the two
       statements really do make incompatible claims about the same
       subject at the same time frame (e.g., "X lives in Tel Aviv"
       vs "X lives in Haifa" as undated current-state claims).
   Two state claims about the same subject are also NOT a
   contradiction when BOTH statements explicitly reference
   non-overlapping past time periods (e.g., "X lived in Tel Aviv
   from 2010 to 2014" vs "X lived in Haifa from 2015 to 2018").
   In that case set non_conflict_reason="scope_mismatch".
5. Decide contradicts:
   - If same_subject is false, contradicts MUST be false.
   - If non_conflict_reason is not "none", contradicts MUST be false.
   - Otherwise contradicts is true only when the two statements assert
     mutually exclusive states about that subject in the same time
     frame. Updates / corrections about the same subject ARE
     contradictions (e.g., "X lives in Tel Aviv" vs "X lives in
     Haifa"). Do not speculate that one statement might describe a
     future state that resolves the conflict. Choose
     temporal_supersession ONLY when both statements read as records
     of past events — never merely because two states could have held
     one after the other; states that replace each other are updates,
     i.e. contradictions.

Reply with ONLY a JSON object, no prose, no markdown fences:
{{"subject_a": "<short noun phrase>",
  "subject_b": "<short noun phrase>",
  "same_subject": true/false,
  "non_conflict_reason": "none|temporal_supersession|list_valued_predicate|refinement|scope_mismatch|same_name_distinct_subject|conditional_unrealized|event_restatement",
  "contradicts": true/false,
  "reason": "one short phrase referencing the subjects and the conflict (or its absence)"}}
"""


NON_CONFLICT_REASONS: frozenset[str] = frozenset(
    {
        "list_valued_predicate",
        "refinement",
        "scope_mismatch",
        "same_name_distinct_subject",
        "conditional_unrealized",
        "event_restatement",
    }
)


def _parse_contradiction_response(raw: dict) -> bool:
    """Apply the structured-output safety gates.

    Two gates run, in order:

    1. **Cross-subject gate (CAURA-111).** The prompt requires the model
       to commit to ``same_subject`` before ``contradicts``. If
       ``same_subject`` is false (or missing), ``contradicts`` MUST be
       false regardless of what the model emitted.

    2. **Within-subject FP gate (CAURA-124).** Even when same_subject is
       true, certain shapes describe two claims that both hold and
       must not be flagged. The model classifies the shape into
       ``non_conflict_reason``; any value in ``NON_CONFLICT_REASONS``
       forces ``contradicts=false``. ``"none"`` (or absent / unknown
       value) leaves ``contradicts`` untouched.

    Missing keys and non-boolean values are treated conservatively
    (False for booleans; None for non_conflict_reason, which has the
    same effect as "none" — neither fires Gate 2). ``bool("false")``
    is True in Python, so a model returning the *string* "false"
    instead of the boolean would have silently bypassed the gate —
    both gates use identity-against-True comparisons to avoid that
    trap.
    """
    if not isinstance(raw, dict):
        return False

    same_subject = raw.get("same_subject") is True
    contradicts = raw.get("contradicts") is True

    # Gate 1 — cross-subject (CAURA-111).
    if contradicts and not same_subject:
        logger.warning(
            "Contradiction model returned contradicts=true with same_subject=false; "
            "overriding to false. subject_a=%r subject_b=%r reason=%r",
            raw.get("subject_a"),
            raw.get("subject_b"),
            raw.get("reason"),
        )
        return False

    # Gate 2 — within-subject FP shapes (CAURA-124). Only fires when
    # the model both flagged a contradiction AND named a recognised
    # non-conflict shape — otherwise it's a no-op. Locals are named
    # ``non_conflict_reason`` (not ``reason``) so they don't collide
    # with the model's free-text ``raw["reason"]`` field that the
    # logger calls pass through verbatim.
    raw_ncr = raw.get("non_conflict_reason")
    non_conflict_reason = raw_ncr if isinstance(raw_ncr, str) else None
    if contradicts and non_conflict_reason in NON_CONFLICT_REASONS:
        # WARNING (not INFO): this branch fires only when the model
        # returned an internally inconsistent response — contradicts=true
        # alongside a recognised non_conflict_reason, which the prompt
        # explicitly forbids in step 5. Same severity as Gate 1's
        # cross-subject override so model regressions surface at the
        # standard WARNING level rather than getting buried in INFO.
        logger.warning(
            "Contradiction model returned contradicts=true with "
            "non_conflict_reason=%r; overriding to false. "
            "subject_a=%r subject_b=%r reason=%r",
            non_conflict_reason,
            raw.get("subject_a"),
            raw.get("subject_b"),
            raw.get("reason"),
        )
        return False

    # Both gates passed (or contradicts is already False) — return the
    # model's verdict.
    return contradicts




# Confidence rubric (see ``tests/test_a4_12_contradiction_judge_confidence.py``
# for the per-branch pin):
#   0.90 — Clean LLM agreement (both gates aligned with the verdict).
#   0.85 — Gate 2 fired (model itself named a non-conflict pattern,
#          parser overrode contradicts=true to false).
#   0.60 — Gate 1 fired (model's same_subject contradicted its
#          contradicts=true; parser overrode to false).
#   0.50 — Malformed / unparseable response (parser conservative-default).
_CONF_CLEAN = 0.90
_CONF_GATE2 = 0.85
_CONF_GATE1 = 0.60
_CONF_FALLBACK = 0.50


def _judge_contradiction(raw) -> tuple[bool, float]:
    """A4 #12 — wrap ``_parse_contradiction_response`` with a confidence
    score derived from the model's own coherence.

    Callers (A4 #13 retraction's confidence-weighted veto, A1 #16
    dedup danger-zone judge) use the score to decide whether the
    final verdict is trustworthy enough to act on. Verdict semantics
    are identical to ``_parse_contradiction_response``; this is purely
    additive on top.
    """
    if not isinstance(raw, dict) or not raw:
        return False, _CONF_FALLBACK

    verdict = _parse_contradiction_response(raw)
    same_subject = raw.get("same_subject") is True
    model_contradicts = raw.get("contradicts") is True
    raw_ncr = raw.get("non_conflict_reason")
    non_conflict_reason = raw_ncr if isinstance(raw_ncr, str) else None

    if model_contradicts and not same_subject:
        # Gate 1 fired — model said contradicts=true but its own
        # same_subject said false. Internal inconsistency → lower trust.
        return verdict, _CONF_GATE1
    if model_contradicts and non_conflict_reason in NON_CONFLICT_REASONS:
        # Gate 2 fired — model recognised a non-conflict pattern AND
        # said contradicts=true. The pattern-recognition signal is
        # what we trust — high confidence in NOT-a-contradiction.
        return verdict, _CONF_GATE2
    # Either contradicts=False with consistent ancillary fields, or
    # contradicts=True with both gates aligned. Clean case.
    return verdict, _CONF_CLEAN


BATCH_CONTRADICTION_PROMPT = """\
You are a contradiction detector for a business memory system.

A NEW statement is compared against several EXISTING candidate statements.
Judge EACH candidate INDEPENDENTLY against the NEW statement — one candidate's
verdict must NOT influence another's.

Two statements contradict ONLY IF they make incompatible claims about the SAME
real-world subject. Different subjects -> NOT a contradiction, even if the
predicates look opposite.

NEW statement:
{new_content}

Candidate statements (judge each, by index):
{candidates_block}

For EACH candidate index decide, applying the rules in order:
1. same_subject (bool): true ONLY when the candidate and the NEW statement refer
   to the SAME real-world entity (exact name / known alias / role+name in
   context / unambiguously resolved pronoun). false for two people sharing a
   name, different companies/products/projects/teams, or any uncertainty.
2. non_conflict_reason (exactly one): "none"; "temporal_supersession"
   (BOTH statements are records of PAST events that stay true as history —
   "hired in 2020" / "promoted in 2023"; NOT for current-state claims where
   the newer statement changes the state, e.g. "lives in X" vs "lives in Y"
   or "runs at 01:00" vs "runs at 03:00" — a state change is an update and
   updates ARE contradictions, reason "none");
   "list_valued_predicate" (a multi-valued predicate or two different attributes
   that both hold); "refinement" (one is a more specific version of the other);
   "scope_mismatch" (same subject under different implicit qualifiers — time
   window / whole vs part / context; both hold); "same_name_distinct_subject"
   (same surface name, plausibly different instances); "conditional_unrealized"
   (one is hypothetical/irrealis); "event_restatement" (same event restated).
3. contradicts (bool): true ONLY when same_subject is true AND
   non_conflict_reason is "none" AND the two assert mutually exclusive states in
   the same time frame. Updates / corrections about the same subject ARE
   contradictions ("X lives in Tel Aviv" vs "X lives in Haifa").

Reply with ONLY a JSON object, no prose, no markdown fences, with exactly two
keys. EVERY candidate index from 0 to N-1 must appear in exactly one of them:
- "clean": array of every candidate index (integers) where contradicts is false.
- "hits": object mapping each contradicting candidate index (as a STRING) to
  its judgment with the three fields above.
{{"clean": [1, 2, 4], "hits": {{"0": {{"same_subject": true, "non_conflict_reason": "none", "contradicts": true}}, "3": {{"same_subject": true, "non_conflict_reason": "none", "contradicts": true}}}}}}
Calibration: in most batches FEW OR NO candidates contradict — an empty "hits"
is the normal answer, not a failure to find something. Apply the three rules to
every candidate independently, exactly as if you were judging that one pair on
its own. A candidate goes in "hits" ONLY when all three rules pass decisively;
uncertainty, partial overlap, related-but-compatible claims, or any applicable
non_conflict_reason put it in "clean". Never add a hit because a batch "should"
contain one. If NO candidate contradicts, "hits" is {{}} and "clean" lists every
index.
"""


def _expand_sparse_batch(obj: object, n: int) -> list[dict]:
    """Expand a batch judge reply into one raw judgment dict per candidate,
    aligned to input order (E4).

    Primary shape is the sparse contract — ``{"clean": [indices], "hits":
    {"idx": {judgment}}}`` — where only contradicting candidates carry a full
    judgment. Every index that is not a well-formed ``hits`` entry (listed
    clean, listed nowhere, or malformed) expands to ``{"contradicts": False}``:
    byte-identical to what the dense parser produced for a non-hit, so
    ``_judge_contradiction`` scores it exactly as before. Hits pass through
    verbatim so Gate 1 / Gate 2 still veto an incoherent one.

    The pre-E4 DENSE shape — index keys at the top level — is still accepted:
    a fallback provider (or a stale prompt cache) that answers in the old
    format must degrade to old behaviour, never to "everything is clean".
    Detection order matters: ``clean``/``hits`` keys mark the sparse shape;
    otherwise top-level index keys are read densely; anything else expands to
    the safe default for every candidate.
    """
    if not isinstance(obj, dict):
        return [{"contradicts": False} for _ in range(n)]

    if "hits" in obj or "clean" in obj:
        hits_raw = obj.get("hits")
        hits = hits_raw if isinstance(hits_raw, dict) else {}
        out: list[dict] = []
        for i in range(n):
            entry = hits.get(str(i))
            if entry is None:
                entry = hits.get(i)  # tolerate int keys
            out.append(entry if isinstance(entry, dict) else {"contradicts": False})
        return out

    # Dense back-compat: the pre-E4 per-index mapping.
    out = []
    for i in range(n):
        entry = obj.get(str(i))
        if entry is None:
            entry = obj.get(i)  # tolerate int keys
        out.append(entry if isinstance(entry, dict) else {"contradicts": False})
    return out
