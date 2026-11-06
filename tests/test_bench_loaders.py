"""Claim-pair benchmark loaders and filters, each on a tiny hand-made fixture (no dataset needed)."""
import json
from collections import Counter

import pandas as pd
import pytest

from meld.bench import loaders as L


# ------------------------------------------------------------------------- PAWS

def test_paws_maps_labels_and_groups_shared_sentences_into_one_item(tmp_path):
    f = tmp_path / "train.parquet"
    pd.DataFrame({
        "id": [1, 2, 3],
        "sentence1": ["A met B in 1990 .", "A met B in 1990 .", "C left D ."],
        "sentence2": ["B met A in 1990 .", "In 1990 , A met B .", "D was left by C ."],
        "label": [0, 1, 1],
    }).to_parquet(f)
    ps = L.load_paws(str(f), origin_split="dev")
    by = {p.text_b: p for p in ps}
    non, para, other = by["B met A in 1990."], by["In 1990, A met B."], by["D was left by C."]
    assert (non.stratum, non.label, non.label_provenance, non.mergeable, non.subtype) == \
        ("S2_paws_nonpara", None, "to_annotate", False, "ADVERSARIAL_NONPARA")
    assert (para.stratum, para.label, para.mergeable) == ("S1_paws_para", "MERGE", True)
    assert non.source_item_id == para.source_item_id != other.source_item_id
    assert all(p.origin_split == "dev" and p.source == "paws" for p in ps)


@pytest.mark.parametrize("raw,clean", [
    ("His father emigrated to Missouri in 1868 , but returned .", "His father emigrated to Missouri in 1868, but returned."),
    ("`` Love '' is a song ( released 1950 ) ; it charted : twice !", '"Love" is a song (released 1950); it charted: twice!'),
    ("John 's car was n't red ?", "John's car wasn't red?"),
    ("The town -- a small port -- grew fast .", "The town \u2013 a small port \u2013 grew fast."),
])
def test_paws_detokenise(raw, clean):
    assert L.detok_paws(raw) == clean


# --------------------------------------------------------------------- VitaminC

@pytest.mark.parametrize("raw,clean", [
    ('}}Michael Oldfield is an English musician<ref name="x">cite web</ref> and composer.',
     "Michael Oldfield is an English musician and composer."),
    ('"Some of his hits include ""Love Generation"" (with Gary Pine) and more."',
     'Some of his hits include "Love Generation" (with Gary Pine) and more.'),
    ("The match ended at 5�4 after a penalty shoot-out.", None),                       # encoding junk
    ("A 1\u00ef\u00bf\u00bd0 away win against the visiting side.", None),                 # UTF-8 mojibake of U+FFFD
    ("it:Carlo XIII di Svezianl:Karel XIII van Zweden", None),                               # interlanguage links
    ("The band formed in 1995 in Glasgow.<ref>Smith 2001, p. 4", None),                      # unclosed ref
])
def test_clean_wiki(raw, clean):
    assert L.clean_wiki(raw) == clean


@pytest.mark.parametrize("a,b,reason", [
    ("The team won 3 games in the regular season .", "The team won 4 games in the regular season .", "number"),
    ("They lost 1–2 at home to the visiting club last week .", "They lost 2–1 at home to the visiting club last week .", "number"),
    ("She owned two horses on the farm near the river .", "She owned three horses on the farm near the river .", "number"),
    ("He moved to Paris with his family in the spring .", "He moved to Lyon with his family in the spring .", "entity"),
    ("In the final round Smith beat Jones on points again .", "In the final round Jones beat Smith on points again .", "entity"),
    ("The film received reviews from many critics at the time .", "The film received glowing reviews from many critics at the time .", "content"),
    ("He is an American rapper and singer from Florida state .", "He is an American rapper and dumbass from Florida state .", "vandalism"),
    ("The drug is effective in adults over sixty .", "The drug is not effective in adults over sixty .", "negation"),
    ("He left the club with a transfer fee paid .", "He left the club without a transfer fee paid .", "negation"),
    ("They never won the title at home .", "They had never won the title at home .", None),
    ("His record made him the most succesful coach in club history .", "His record made him the most successful coach in club history .", None),
    ("He had firstly applied for a postponement of the trial .", "He had first applied for a postponement of the trial .", None),
])
def test_s3_filter_reasons(a, b, reason):
    assert L.s3_reject_reason(a, b) == reason


_V = [
    # kept as S4: one-word factual edit
    {"label": "factual", "page": "P1", "sent_a": "The film received 39 reviews from critics in total .",
     "sent_b": "The film received 41 reviews from critics in total .",
     "diff_viz": "The film received{*| 39 -> 41 |}reviews from critics in total ."},
    # kept as S3: stylistic edit, same page
    {"label": "not factual", "page": "P1", "sent_a": "He firstly worked in Paris for the paper .",
     "sent_b": "He first worked in Paris for the paper .",
     "diff_viz": "He{*| firstly -> first |}worked in Paris for the paper ."},
    # dropped by S3 filter: number change labelled not factual
    {"label": "not factual", "page": "P5", "sent_a": "The bridge is 300 metres long and opened to traffic .",
     "sent_b": "The bridge is 350 metres long and opened to traffic .",
     "diff_viz": "The bridge is{*| 300 -> 350 |}metres long and opened to traffic ."},
    # dropped: punctuation-only diff
    {"label": "not factual", "page": "P2", "sent_a": "She was born in Oslo in 1950, Norway.",
     "sent_b": "She was born in Oslo in 1950 Norway.",
     "diff_viz": "She was born in Oslo in{*| 1950, -> 1950 |}Norway."},
    # dropped: too short
    {"label": "factual", "page": "P3", "sent_a": "It is red .", "sent_b": "It is blue .",
     "diff_viz": "It is{*| red -> blue |}."},
    # dropped: rewrite of more than half the tokens
    {"label": "factual", "page": "P4", "sent_a": "Alpha beta gamma delta epsilon zeta eta theta .",
     "sent_b": "One two three four five six eta theta .",
     "diff_viz": "{*| Alpha beta gamma delta epsilon zeta -> One two three four five six |}eta theta ."},
    # dropped: encoding junk
    {"label": "factual", "page": "P6", "sent_a": "The score was 5�4 at the end of the match .",
     "sent_b": "The score was 5�3 at the end of the match .", "diff_viz": "{*| 4 -> 3 |}"},
]


def test_vitaminc_filters_label_map_and_stats(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in _V))
    stats = Counter()
    ps = L.load_vitaminc(str(f), origin_split="test", stats=stats)
    assert len(ps) == 2
    fact = next(p for p in ps if p.stratum == "S4_vitc_factual")
    style = next(p for p in ps if p.stratum == "S3_vitc_nonfactual")
    assert (fact.label, fact.label_provenance, fact.mergeable) == (None, "to_annotate", False)
    assert (style.label, style.label_provenance, style.mergeable) == ("MERGE", "source_filtered", True)
    assert fact.source_item_id == style.source_item_id == "vitaminc:page:P1"
    assert stats["s3_number"] == 1 and stats["s3_kept"] == 1 and stats["unclean"] == 1


def _mk(a, b, stratum, item, **kw):
    lab = dict(label="MERGE", label_provenance="source", mergeable=True) if stratum in ("S1_paws_para",) \
        else dict(label=None, label_provenance="to_annotate", mergeable=False)
    lab.update(kw)
    return L.make_pair(text_a=a, text_b=b, source="vitaminc", source_item_id=item, stratum=stratum, **lab)


def test_dedupe_drops_label_conflicts_identical_sides_and_keeps_one_repeat():
    s3 = dict(label="MERGE", label_provenance="source_filtered", mergeable=True)
    ps = [
        _mk("He played for Bordeaux on loan .", "He played for FC Bordeaux on loan .", "S3_vitc_nonfactual", "p:1", **s3),
        _mk("He played for FC Bordeaux on loan.", "He played for Bordeaux on loan.", "S4_vitc_factual", "p:1"),  # same edit, other label
        _mk("Same words here, really.", "same words here really", "S4_vitc_factual", "p:2"),                  # identical after normalisation
        _mk("Drug X lowers risk.", "Drug X raises risk.", "S4_vitc_factual", "p:3"),
        _mk("Drug X raises risk.", "Drug X lowers risk.", "S4_vitc_factual", "p:4"),                          # repeat, sides swapped
    ]
    stats = Counter()
    out = L.dedupe_pairs(ps, stats=stats)
    assert len(out) == 1 and out[0].text_a in ("Drug X lowers risk.", "Drug X raises risk.")
    assert stats == Counter(label_conflict=2, identical_sides=1, repeat=1)


# ------------------------------------------------------------------ LongMemEval

def _lme_item(qid, answer, sessions, question="What is my personal best 5K time?", qtype="knowledge-update"):
    """sessions: list of (date, session_id, [(content, has_answer)])."""
    return {"question_id": qid, "question_type": qtype, "question": question, "answer": answer,
            "haystack_dates": [d for d, _, _ in sessions],
            "haystack_session_ids": [s for _, s, _ in sessions],
            "answer_session_ids": [s for _, s, ts in sessions if any(h for _, h in ts)],
            "haystack_sessions": [[{"role": "user", "content": c, **({"has_answer": True} if h else {})}
                                   for c, h in ts] for _, _, ts in sessions]}


_OLD5K = "I set a personal best 5K time of 27:12. Do you have tips on how to run a faster race this year?"


def _load(tmp_path, items):
    f = tmp_path / "lme.json"
    f.write_text(json.dumps(items))
    stats = Counter()
    return L.load_lme_ku(str(f), stats=stats), stats


def test_lme_verified_pair_new_has_answer_old_has_superseded_value(tmp_path):
    ps, stats = _load(tmp_path, [
        _lme_item("q1", "25 minutes and 50 seconds (or 25:50)", [   # newer session listed first
            ("2023/05/30 (Tue) 13:53", "a2", [("Do you have tips on how to run a faster race this year? My best 5K is now 25:50.", True)]),
            ("2023/05/23 (Tue) 13:01", "a1", [(_OLD5K, True)]),
            ("2023/05/01 (Mon) 10:00", "d1", [("Unrelated chat about cooking pasta tonight.", False)])]),
        _lme_item("q4", "4", [
            ("2023/01/01", "b1", [("Thanks. I own two bikes right now, a road bike and a commuter.", True)]),
            ("2023/02/01", "b2", [("Great trip planned. I now have four bikes in total after the purchase.", True)])],
            question="How many bikes do I currently own?"),
    ])
    p = next(p for p in ps if "25:50" in p.text_b)
    assert "27:12" in p.text_a
    assert p.badges_a == {"time": "2023/05/23"} and p.badges_b == {"time": "2023/05/30"}
    assert (p.stratum, p.label, p.subtype, p.label_provenance, p.mergeable) == \
        ("S5_lme_ku", "CONFLICT", "SUPERSESSION", "source", False)
    q = next(p for p in ps if "four bikes" in p.text_b)
    assert "two bikes" in q.text_a and q.label_provenance == "source"
    assert stats["verified"] == 2


def test_lme_unverifiable_old_value_is_to_annotate_and_unmatchable_dropped(tmp_path):
    ps, stats = _load(tmp_path, [
        # non-numeric answer: the new sentence carries it, the old value cannot be derived -> to_annotate
        _lme_item("q2", "Friday", [
            ("2023/01/01", "c1", [("By the way, my cocktail class is on Thursday this term.", True)]),
            ("2023/03/01", "c2", [("Hello again. I moved my cocktail-making class to Fridays now.", True)])],
            question="What day of the week do I take a cocktail-making class?"),
        # yes/no answer: neither condition can be met -> dropped
        _lme_item("q3", "Yes.", [
            ("2023/01/01", "e1", [("I like the new grocery list app a lot.", True)]),
            ("2023/03/01", "e2", [("My mom uses the same grocery list app as me now.", True)])],
            question="Is my mom using the same grocery list method as me?"),
        # three has_answer turns: not a clean pair, skipped
        _lme_item("q5", "3", [("2023/01/01", "f1", [("x one two three", True)]), ("2023/01/02", "f2", [("y one two three", True)]),
                              ("2023/01/03", "f3", [("z one two three", True)])]),
        _lme_item("q6", "tea", [("2023/01/01", "g1", [("I like tea very much.", True)])], qtype="single-session-user"),
    ])
    assert len(ps) == 1
    p = ps[0]
    assert "Fridays" in p.text_b and (p.label, p.label_provenance, p.mergeable) == (None, "to_annotate", None)
    assert stats["new_only"] == 1 and stats["dropped"] == 1


def test_lme_numeric_answer_needs_answer_in_new_and_other_value_in_old(tmp_path):
    q = "How many pre-1920 American coins do I have?"
    ps, stats = _load(tmp_path, [
        # old sentence has no superseded value -> only the new condition holds
        _lme_item("c1", "38", [("2023/01/01", "h1", [("My collection of pre-1920 American coins keeps growing every month.", True)]),
                               ("2023/02/01", "h2", [("I now have 38 pre-1920 American coins in my collection.", True)])], question=q),
        # new sentence lacks the answer -> only the old condition holds
        _lme_item("c2", "38", [("2023/01/01", "i1", [("I have 37 pre-1920 American coins at home right now.", True)]),
                               ("2023/02/01", "i2", [("I just added a 1915-S Barber quarter to my pre-1920 American coins.", True)])],
                  question=q),
    ])
    assert len(ps) == 2 and all(p.label_provenance == "to_annotate" for p in ps)
    assert stats["new_only"] == 1 and stats["old_only"] == 1 and stats["verified"] == 0


def test_lme_excluded_pair_is_dropped(tmp_path, monkeypatch):
    items = [_lme_item("q1", "25:50", [("2023/05/23", "a1", [(_OLD5K, True)]),
                                       ("2023/05/30", "a2", [("My best 5K is now 25:50 after training.", True)])])]
    ps, _ = _load(tmp_path, items)
    monkeypatch.setattr(L, "S5_EXCLUDE", {ps[0].pair_id: "test exclusion"})
    ps2, stats = _load(tmp_path, items)
    assert ps2 == [] and stats["excluded"] == 1


def test_lme_items_sharing_an_answer_session_share_a_source_item(tmp_path):
    ps, _ = _load(tmp_path, [
        _lme_item("q1", "25:50", [("2023/05/23", "a1", [(_OLD5K, True)]),
                                  ("2023/05/30", "a2", [("My best 5K is now 25:50 after training.", True)])]),
        _lme_item("q4", "24:59", [("2023/05/23", "a1", [(_OLD5K, True)]),
                                  ("2023/06/01", "a3", [("My personal best 5K time dropped to 24:59 this month.", True)])]),
    ])
    assert len(ps) == 2 and len({p.source_item_id for p in ps}) == 1


def test_assign_sessions_one_owner_and_expected_copies():
    sids = [f"s{i}" for i in range(4000)]
    a = L.assign_sessions(sids, n_agents=8, q=0.1, seed=3)
    assert a == L.assign_sessions(sids, n_agents=8, q=0.1, seed=3)
    assert set(a) == set(sids) and all(1 <= len(h) <= 8 for h in a.values())
    mean = sum(len(h) for h in a.values()) / len(sids)
    assert abs(mean - (1 + 0.1 * 7)) < 0.05
    solo = L.assign_sessions(sids[:16], n_agents=4, q=0.0, seed=0)
    assert all(len(h) == 1 for h in solo.values())
    assert sorted(h[0] for h in solo.values()) == sorted(list(range(4)) * 4)   # round-robin owners


# ---------------------------------------------------------------------- SciFact

def test_scifact_pairs_are_claim_claim_and_all_to_annotate(tmp_path):
    f = tmp_path / "claims_train.jsonl"
    rows = [
        {"id": 1, "claim": "Drug X lowers risk.", "evidence": {"7": [{"label": "SUPPORT", "sentences": [0]}],
                                                               "8": [{"label": "SUPPORT", "sentences": [1]}]}},
        {"id": 2, "claim": "Drug X raises risk.", "evidence": {"7": [{"label": "CONTRADICT", "sentences": [0]}],
                                                               "8": [{"label": "CONTRADICT", "sentences": [3]}]}},
        {"id": 3, "claim": "Drug X is cheap.", "evidence": {"7": [{"label": "SUPPORT", "sentences": [2]}]}},
        {"id": 4, "claim": "Gene Y is expressed.", "evidence": {"9": [{"label": "SUPPORT", "sentences": [1]}]}},
        {"id": 5, "claim": "No evidence here.", "evidence": {}},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    ps = L.dedupe_pairs(L.load_scifact_conflicts([str(f)]))
    assert sorted((p.text_a, p.text_b) for p in ps) == [("Drug X is cheap.", "Drug X raises risk."),
                                                        ("Drug X lowers risk.", "Drug X raises risk.")]
    assert all((p.stratum, p.label, p.label_provenance, p.mergeable) ==
               ("S6_scifact_conflict", None, "to_annotate", None) for p in ps)


# ------------------------------------------------------------------------- gold

def test_gold_is_its_own_stratum_and_swapped_no_relation_repeats_are_deduplicated():
    ps = L.dedupe_pairs(L.load_gold())
    assert len(ps) == 91 and {p.stratum for p in ps} == {"S0_gold"}
    assert Counter(p.label for p in ps) == Counter(MERGE=26, RELATE=26, CONFLICT=26, NO_RELATION=13)
    assert all(p.label_provenance == "construction" and p.mergeable == (p.label == "MERGE") for p in ps)


# ------------------------------------------------------------------------- SICK

_SICK_HEAD = ("pair_ID\tsentence_A\tsentence_B\tentailment_label\trelatedness_score\tentailment_AB\tentailment_BA\t"
              "sentence_A_original\tsentence_B_original\tsentence_A_dataset\tsentence_B_dataset\tSemEval_set")


def _sick(tmp_path, rows):
    f = tmp_path / "SICK.txt"
    f.write_text("\n".join([_SICK_HEAD] + ["\t".join([str(i), a, b, "X", "3", ab, ba, a, b, "FLICKR", "FLICKR", "TRAIN"])
                                          for i, a, b, ab, ba in rows]) + "\n", encoding="utf-8")
    return str(f)


def test_sick_label_mapping_keeps_three_classes_and_parses_quotes_verbatim(tmp_path):
    ps = L.load_sick(_sick(tmp_path, [
        (1, "A man is playing a guitar", "A guitar is being played by a man", "A_entails_B", "B_entails_A"),
        (2, "A man is playing a guitar", "A person is playing an instrument", "A_entails_B", "B_neutral_A"),
        (3, "A dog runs on the beach", "No dog is running on the beach", "A_contradicts_B", "B_contradicts_A"),
        (4, "A cat sleeps", "A woman is cooking", "A_neutral_B", "B_neutral_A"),              # unrelated: dropped
        (5, "A boy jumps", "Nobody is jumping", "A_contradicts_B", "B_neutral_A"),            # one-way contradiction: dropped
        (6, "A girl sings", "A girl is not singing", "A_entails_B", "B_contradicts_A"),       # inconsistent: dropped
        (7, '"Quoted" kids play', 'Kids play "outside"', "A_entails_B", "B_entails_A"),
    ]))
    by = {p.text_b: p for p in ps}
    assert len(ps) == 4
    para, one, con = by["A guitar is being played by a man"], by["A person is playing an instrument"], \
        by["No dog is running on the beach"]
    assert (para.stratum, para.label, para.label_provenance, para.mergeable) == ("S8_sick_para", "MERGE", "source", True)
    assert (one.stratum, one.label, one.label_provenance, one.mergeable) == ("S8_sick_oneway", "RELATE", "source", False)
    assert one.text_a == "A man is playing a guitar"                       # text_a is the entailing side
    assert (con.stratum, con.label, con.label_provenance, con.mergeable) == ("S8_sick_contra", "CONFLICT", "source", False)
    assert by['Kids play "outside"'].text_a == '"Quoted" kids play'          # no csv quote processing
    assert all(p.source == "sick" and p.origin_split is None for p in ps)


def test_sick_pairs_sharing_a_sentence_share_a_source_item(tmp_path):
    ps = L.load_sick(_sick(tmp_path, [
        (1, "A man is playing a guitar", "A guitar is being played by a man", "A_entails_B", "B_entails_A"),
        (2, "a man is  playing a guitar", "A person plays an instrument", "A_entails_B", "B_neutral_A"),  # case/space variant
        (3, "A person plays an instrument", "Nobody plays an instrument", "A_contradicts_B", "B_contradicts_A"),  # chain
        (4, "A dog runs on the beach", "A dog is running on the sand", "A_entails_B", "B_entails_A"),
        (5, "A cat sleeps", "A dog runs on the beach", "A_neutral_B", "B_neutral_A"),          # dropped pair joins nothing
        (6, "A cat sleeps", "A cat is sleeping", "A_entails_B", "B_entails_A"),
    ]))
    item = {p.text_b: p.source_item_id for p in ps}
    assert item["A guitar is being played by a man"] == item["A person plays an instrument"] \
        == item["Nobody plays an instrument"]
    assert len({item["A guitar is being played by a man"], item["A dog is running on the sand"],
                item["A cat is sleeping"]}) == 3
