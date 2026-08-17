"""Real-LLM-contract regressions surfaced by the HotpotQA local real-model
ext-smoke (cross-family on qwen2.5:7b + mistral:7b).

Two NEW failure modes the GAIA fixture never exposed (its oracle always scripts a
clean ``final_answer[gold]``), caught only once a real model answered open
HotpotQA questions over brain-fed evidence:

1. **bare-verb leak.** On a forced-synthesis turn a model replied with a bare
   action verb ``final_answer[no]`` instead of a ``FINAL ANSWER: no`` line, so the
   synthesis fallback returned the literal wrapper ``"final_answer[no]"`` as the
   answer (react/qwen). The wrapper must be stripped: the intended answer is the
   bracketed payload.
2. **template echo.** A model echoed the prompt placeholder ``[YOUR FINAL
   ANSWER]`` verbatim (reflexion/mistral); that is a non-answer and must normalize
   to empty (so the fallback drops to the best recalled fact, never returns the
   literal template).

Both strips are SAFE and unambiguous (they do not guess a short answer out of a
sentence), so they live in ``tighten_gaia_answer`` and apply to GAIA too.
"""
from __future__ import annotations

from meld.agent.gaia_scorer import tighten_gaia_answer


def test_tighten_strips_bare_action_verb_wrapper():
    assert tighten_gaia_answer("final_answer[no]") == "no"
    assert tighten_gaia_answer("final_answer[Paris]") == "Paris"
    # with the optional Action: prefix and surrounding whitespace
    assert tighten_gaia_answer("Action: final_answer[London]") == "London"
    # a recall wrapper likewise reduces to its payload (a leaked read verb)
    assert tighten_gaia_answer("recall[capital of France]") == "capital of France"


def test_tighten_drops_template_placeholder_echo():
    assert tighten_gaia_answer("[YOUR FINAL ANSWER]") == ""
    assert tighten_gaia_answer("FINAL ANSWER: [YOUR FINAL ANSWER]") == ""
    assert tighten_gaia_answer("[your answer here]") == ""


def test_tighten_still_safe_on_real_short_answers():
    # a real bracketed list answer is NOT a verb wrapper -> left intact
    assert tighten_gaia_answer("yes") == "yes"
    assert tighten_gaia_answer("Christopher Nolan") == "Christopher Nolan"
    # a genuine comma list with no verb wrapper is untouched
    assert tighten_gaia_answer("Paris, London") == "Paris, London"
