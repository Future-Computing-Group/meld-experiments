"""G3 real-LLM-contract regression: the GAIA final-answer must be format-TIGHT
(a short string / number / list), matching the GAIA quasi-exact-match scorer.

Root cause (caught on the local real-model ext-smoke): the ReAct baseline
answered "Alice owns a dog named Rex." when gold was "dog" -- a correct fact in
a full sentence that the quasi-exact-match scorer rightly counts wrong. Two
levers: (1) the GAIA system prompt must INSTRUCT the short form (no sentences,
no extra words, strip units/articles unless required); (2) a normalization
safety net (``tighten_gaia_answer``) handles the safe, unambiguous over-verbose
patterns it CAN strip (trailing period, a leading "the answer is" preamble)
without guessing -- it must NEVER fabricate the short answer from a sentence
(that would create false positives), so the prompt is the primary fix.
"""
from __future__ import annotations

from meld.agent.gaia_scorer import tighten_gaia_answer
from meld.agent.react_agent import GAIA_SYSTEM


# --------------------------------------------------------------------------
# the prompt instructs the short GAIA form (primary fix)
# --------------------------------------------------------------------------


def test_gaia_system_prompt_instructs_short_form():
    sys = GAIA_SYSTEM.lower()
    # must still carry the FINAL ANSWER template
    assert "final answer:" in sys
    # must instruct brevity / the short answer types (not a full sentence)
    assert "as few words as possible" in sys or "as short as possible" in sys
    assert "number" in sys and "string" in sys
    assert "sentence" in sys  # explicitly tells the model not to write a sentence


# --------------------------------------------------------------------------
# normalization safety net (only the unambiguous, non-guessing strips)
# --------------------------------------------------------------------------


def test_tighten_strips_trailing_period():
    assert tighten_gaia_answer("Paris.") == "Paris"


def test_tighten_strips_leading_answer_preamble():
    assert tighten_gaia_answer("The answer is dog") == "dog"
    assert tighten_gaia_answer("the answer is: 42") == "42"


def test_tighten_does_not_fabricate_short_answer_from_a_sentence():
    """It must NOT guess 'dog' out of a full descriptive sentence -- that would
    be unsafe (false positives). The sentence is returned (cleaned of trailing
    punctuation only); fixing this case is the prompt's job, not the scorer's."""
    out = tighten_gaia_answer("Alice owns a dog named Rex.")
    assert out == "Alice owns a dog named Rex"


def test_tighten_is_idempotent_and_safe_on_short_answers():
    assert tighten_gaia_answer("dog") == "dog"
    assert tighten_gaia_answer("Monday, up") == "Monday, up"
    assert tighten_gaia_answer("") == ""
    assert tighten_gaia_answer(None) == ""


def test_tighten_strips_trailing_react_continuation():
    # the model answered, then kept emitting ReAct scaffolding on a new line; keep
    # only the answer span before the first marker (the memdelta qwen confound).
    assert tighten_gaia_answer("no\nAction: recall[query]") == "no"
    assert tighten_gaia_answer("Nelson Rockefeller\nThought: done") == \
        "Nelson Rockefeller"
    # a string that is WHOLLY an action wrapper still unwraps to its payload
    # (start==0 -> not cut here, flows to the verb-wrap rule).
    assert tighten_gaia_answer("Action: final_answer[no]") == "no"
    # a genuine short answer with no marker is untouched.
    assert tighten_gaia_answer("New Action Heroes") == "New Action Heroes"
