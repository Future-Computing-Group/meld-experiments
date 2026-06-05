"""E1: the enlarged, balanced M-2 gold set.

A reproducible 104-pair benchmark (26 seeds x 4 classes) replacing a smaller
hand-built smoke set. Built from a curated seed bank of facts, balanced across
the four decision classes; each seed contributes one pair to each class:

  MERGE    canonical vs a paraphrase (same claim, reworded)
  RELATE   canonical vs an overlapping but distinct, non-contradicting fact
  CONFLICT canonical vs a contradiction (same subject, incompatible value)
  REJECT   canonical vs an unrelated fact (deterministic cross-domain pairing)

Each seed carries a difficulty tag. ``hard`` seeds have low-lexical-overlap
paraphrases and/or lexically-divergent contradictions -- the hardest case for
a similarity gate (low conflict recall on lexically divergent pairs). The set is balanced
(26 per class) and deterministic; no model is invoked here.

This is a controlled DIAGNOSTIC benchmark (a calibration / sensitivity
instrument with known ground truth by construction), not a naturalistic claim
distribution; Section 7 frames it as such. Reproduced by
tests/test_m2_gold_v2.py.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Each seed: (id, difficulty, canonical, paraphrase[MERGE], related[RELATE],
#             contradiction[CONFLICT]).
_SEEDS: List[Tuple[str, str, str, str, str, str]] = [
    ("geo-paris", "easy",
     "Paris is the capital of France",
     "The capital city of France is Paris",
     "France is a country in western Europe",
     "Paris is the capital of Italy"),
    ("geo-everest", "easy",
     "Mount Everest is the tallest mountain on Earth",
     "Earth's highest mountain is Mount Everest",
     "Mount Everest lies on the border of Nepal and China",
     "K2 is the tallest mountain on Earth"),
    ("lit-1984", "easy",
     "The novel 1984 was written by George Orwell",
     "George Orwell is the author of the novel 1984",
     "1984 is a dystopian novel about surveillance",
     "The novel 1984 was written by Aldous Huxley"),
    ("chem-water", "easy",
     "Water boils at 100 degrees Celsius at sea level",
     "At sea level, the boiling point of water is 100 degrees Celsius",
     "Water freezes at 0 degrees Celsius at sea level",
     "Water boils at 90 degrees Celsius at sea level"),
    ("astro-sun", "hard",
     "The Sun is primarily composed of hydrogen and helium",
     "Hydrogen and helium make up most of the Sun's mass",
     "The Sun is the star at the centre of the Solar System",
     "Most of the Sun's mass consists of carbon"),
    ("hist-ww2", "easy",
     "World War II ended in 1945",
     "The year 1945 marked the end of World War II",
     "World War II involved many countries across the globe",
     "World War II ended in 1939"),
    ("bio-photosynth", "hard",
     "Photosynthesis converts sunlight into chemical energy in plants",
     "In plants, light energy is transformed into chemical energy through photosynthesis",
     "Photosynthesis releases oxygen as a byproduct",
     "In plants, photosynthesis turns chemical energy back into sunlight"),
    ("tech-http", "easy",
     "HTTP is a protocol used to transfer web pages",
     "Web pages are transferred using the HTTP protocol",
     "HTTPS adds encryption on top of HTTP",
     "HTTP is a protocol used to encrypt hard drives"),
    ("math-prime", "easy",
     "A prime number has exactly two distinct positive divisors",
     "A prime number is divisible only by one and itself",
     "Two is the smallest prime number",
     "A prime number has exactly three distinct positive divisors"),
    ("geo-nile", "hard",
     "The Nile is one of the longest rivers in the world",
     "Among the world's rivers, the Nile ranks among the longest",
     "The Nile flows through Egypt into the Mediterranean Sea",
     "The Nile is one of the shortest rivers in the world"),
    ("sport-olympics", "easy",
     "The modern Olympic Games are held every four years",
     "Every four years, the modern Olympic Games take place",
     "The Olympic Games feature athletes from around the world",
     "The modern Olympic Games are held every year"),
    ("music-beethoven", "hard",
     "Beethoven composed nine symphonies",
     "There are nine symphonies in Beethoven's body of work",
     "Beethoven kept composing after losing his hearing",
     "Beethoven composed only three symphonies"),
    ("phys-light", "easy",
     "Light travels faster than sound",
     "The speed of light exceeds the speed of sound",
     "Light from the Sun takes about eight minutes to reach Earth",
     "Sound travels faster than light"),
    ("bio-heart", "easy",
     "The human heart has four chambers",
     "There are four chambers in the human heart",
     "The heart pumps blood throughout the body",
     "The human heart has two chambers"),
    ("geo-amazon", "hard",
     "The Amazon rainforest produces a large share of Earth's oxygen",
     "A substantial portion of the planet's oxygen comes from the Amazon rainforest",
     "The Amazon rainforest is home to millions of species",
     "The Amazon rainforest consumes most of Earth's oxygen"),
    ("hist-moon", "easy",
     "Apollo 11 landed humans on the Moon in 1969",
     "In 1969, the Apollo 11 mission put humans on the Moon",
     "Neil Armstrong was the first person to walk on the Moon",
     "Apollo 11 landed humans on the Moon in 1972"),
    ("tech-python", "easy",
     "Python is a high-level programming language",
     "Python is a programming language operating at a high level of abstraction",
     "Python is widely used for data science and machine learning",
     "Python is a low-level assembly language"),
    ("chem-gold", "hard",
     "Gold is a chemical element with the symbol Au",
     "The chemical element gold is denoted by the symbol Au",
     "Gold is a dense, malleable metal often used in jewellery",
     "Gold is a chemical element with the symbol Ag"),
    ("geo-sahara", "easy",
     "The Sahara is the largest hot desert in the world",
     "The world's largest hot desert is the Sahara",
     "The Sahara spans much of northern Africa",
     "The Sahara is the smallest hot desert in the world"),
    ("lit-hamlet", "hard",
     "Shakespeare wrote the play Hamlet",
     "Hamlet is a play penned by Shakespeare",
     "Hamlet is set in the kingdom of Denmark",
     "Christopher Marlowe wrote the play Hamlet"),
    ("phys-gravity", "easy",
     "Objects fall toward Earth because of gravity",
     "Gravity causes objects to fall toward Earth",
     "Gravity also governs the orbits of the planets",
     "Objects fall away from Earth because of gravity"),
    ("bio-dna", "hard",
     "DNA carries the genetic instructions of living organisms",
     "The hereditary information of living organisms is stored in DNA",
     "DNA is structured as a double helix",
     "DNA holds none of the genetic information in living organisms"),
    ("geo-pacific", "easy",
     "The Pacific is the largest ocean on Earth",
     "Earth's largest ocean is the Pacific",
     "The Pacific Ocean lies between Asia and the Americas",
     "The Atlantic is the largest ocean on Earth"),
    ("tech-tcp", "hard",
     "TCP provides reliable, ordered delivery of data",
     "Data sent over TCP arrives reliably and in order",
     "TCP runs on top of the IP layer",
     "TCP provides unreliable, unordered delivery of data"),
    ("hist-berlin", "easy",
     "The Berlin Wall fell in 1989",
     "1989 was the year the Berlin Wall fell",
     "The Berlin Wall divided East and West Berlin",
     "The Berlin Wall fell in 1979"),
    ("astro-mars", "easy",
     "Mars is the fourth planet from the Sun",
     "The fourth planet from the Sun is Mars",
     "Mars is often called the Red Planet",
     "Mars is the second planet from the Sun"),
]


def build_gold_set() -> List[Dict]:
    """Materialize the gold set as a list of dict rows with metadata.

    Returns rows ordered MERGE, RELATE, CONFLICT, then REJECT (the REJECT block
    pairs each canonical with the canonical n/2 seeds away -- a deterministic
    cross-domain, unrelated pairing)."""
    rows: List[Dict] = []
    for sid, diff, canon, para, rel, conf in _SEEDS:
        rows.append({"label": "MERGE", "a": canon, "b": para,
                     "difficulty": diff, "seed_id": sid})
    for sid, diff, canon, para, rel, conf in _SEEDS:
        rows.append({"label": "RELATE", "a": canon, "b": rel,
                     "difficulty": diff, "seed_id": sid})
    for sid, diff, canon, para, rel, conf in _SEEDS:
        rows.append({"label": "CONFLICT", "a": canon, "b": conf,
                     "difficulty": diff, "seed_id": sid})
    n = len(_SEEDS)
    for i, (sid, diff, canon, para, rel, conf) in enumerate(_SEEDS):
        other = _SEEDS[(i + n // 2) % n][2]  # canonical of the seed n/2 away
        rows.append({"label": "REJECT", "a": canon, "b": other,
                     "difficulty": "easy", "seed_id": sid})
    return rows


# Flat (label, a, b) view -- drop-in compatible with evaluate_m2 / evaluate_baseline.
M2_GOLD_V2: List[Tuple[str, str, str]] = [
    (r["label"], r["a"], r["b"]) for r in build_gold_set()
]

__all__ = ["build_gold_set", "M2_GOLD_V2", "_SEEDS"]
