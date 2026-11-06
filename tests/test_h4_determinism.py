"""H4 replica-determinism instrument (scripts/run_h4_determinism.py): gold workload, hashing encoder,
lexical NLI probabilities as the fake model; a second cell perturbs the floats below and across the
quantisation steps. No model download."""
from __future__ import annotations

import os
from collections import Counter
import sys

import numpy as np

from meld.merge.signals import _HashingEncoder, _lexical_probs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_h4_determinism as H  # noqa: E402

KEY = b"test-only-h4-key"
ENC, NLI_NAME = "hashing", "lexical"


def _embed(jitter=0.0):
    enc = _HashingEncoder()

    def embed(texts, batch):
        E = np.stack([enc.encode_one(t) for t in texts]).astype(np.float32)
        return E + np.float32(jitter) * (np.arange(E.shape[1], dtype=np.float32) % 3 - 1)
    return embed


def _nli(jitter=0.0):
    def nli(pairs, batch):
        out = []
        for a, b in pairs:
            p = list(_lexical_probs(a, b))
            if jitter:          # move mass from entailment to neutral: fewer tested scores pass theta
                d = min(jitter, p[1])
                p[1], p[2] = p[1] - d, p[2] + d
            out.append(tuple(p))
        return out
    return nli


def _ref():
    return H.build_ref(["gold"], encoder=ENC, nli=NLI_NAME, key=KEY, manifest=None, epoch_dir=None,
                       dev_theta=0.3)


def _cell(ref, name, ej=0.0, nj=0.0, batch=1, table=None):
    return H.run_cell(ref, name=name, embed=_embed(ej), nli=_nli(nj), batch=batch, key=KEY,
                      encoder=ENC, nli_name=NLI_NAME, table=table)


def _p2(ref, cells, names):
    return H.run_pass2(ref, cells, names, key=KEY, encoder=ENC, nli_name=NLI_NAME,
                       provider_for=lambda n: (lambda a, b: H.to_fixed16(_nli()([(a, b)], 1)[0])))


def test_identical_cells_agree_on_everything():
    ref = _ref()
    cells = {"a": _cell(ref, "a"), "b": _cell(ref, "b")}
    rep = H.report(ref, cells, _p2(ref, cells, ["a", "b"]), "a", key=KEY, encoder=ENC, nli_name=NLI_NAME)["gold"]
    c = rep["cells"]["b"]
    assert rep["fingerprint1_agreement"] == rep["fingerprint2_agreement"] == 1.0
    assert c["emb_rows_bit_identical"] == c["emb_rows"] == 104
    assert c["nli_float_bit_identical"] == c["nli_dirs_common"] > 0
    assert c["q_rows_differing"] == c["fixed16_differing"] == c["theta_flips"] == 0
    assert c["selection_flips_own_q"] == 0 and c["pass2_misses"] == 0
    assert c["theta_tested"] > 0


def test_planted_float_noise_is_counted_and_the_exchange_restores_agreement():
    ref = _ref()
    cells = {"a": _cell(ref, "a"), "b": _cell(ref, "b", ej=0.02, nj=0.5)}
    rep = H.report(ref, cells, _p2(ref, cells, ["a", "b"]), "a", key=KEY, encoder=ENC, nli_name=NLI_NAME)["gold"]
    c = rep["cells"]["b"]
    assert c["emb_rows_bit_identical"] < c["emb_rows"] and c["emb_max_abs_diff"] > 0.01
    assert c["q_rows_differing"] > 0 and c["q_elements_differing"] > 0
    assert c["fixed16_differing"] > 0 and c["nli_max_abs_diff"] > 0.1
    assert c["theta_flips"] > 0 and c["decisions1_differing"] > 0
    assert rep["fingerprint1_distinct"] == 2                  # the cells disagree before the exchange
    assert rep["fingerprint2_agreement"] == 1.0 and rep["fingerprint2_distinct"] == 1
    assert c["new_merges_without_release"] == 0          # Lemma 7b


def test_batched_cell_serves_the_reference_read_set_from_its_table():
    ref = _ref()
    a = _cell(ref, "a")
    b = _cell(ref, "b32", batch=32, table={"gold": {"nli_dirs": a["gold"]["nli_dirs"]}})
    assert b["gold"]["batch1_calls"] == 0
    assert b["gold"]["fingerprint1"] == a["gold"]["fingerprint1"]


def test_gate_releases_counts_a_released_group_gate():
    d1 = [["k1", "INSERT", [], None, None, None], ["k2", "CONFLICT", ["group_contradiction"], ["k1", "k2"], 9, 5]]
    d2 = [["k1", "INSERT", [], None, None, None], ["k2", "MERGE", [], ["k1", "k2"], 9, 5]]
    g = H.gate_releases(d1, d2)
    assert g == {"decision_changes": {"CONFLICT->MERGE": 1}, "group_gate_releases": 1, "new_merges": 1,
                 "new_merges_without_release": 0}
    d1[1][2] = []
    assert H.gate_releases(d1, d2)["new_merges_without_release"] == 1


# -- own-issuance arm: every cell issues its own deltas (own int8 q) from the same texts --------------

from meld.agent.hotpot_loader import HotpotTask  # noqa: E402

_TASK = HotpotTask(task_id="t0", question="q", answer="a", level="easy", context=[
    ["Alpha", ["Alpha is a small town in France.", "Alpha has an old river port."]],
    ["Beta", ["Beta is a large city in Spain.", "Beta was founded in 1200."]],
    ["Gamma", ["Gamma is a village in Italy.", "Gamma lies on a hill."]]],
    supporting_facts=[["Alpha", 0], ["Beta", 0], ["Gamma", 1]])


def _ref_own():
    """gold plus one HotpotQA-shaped world at overlap 1.0: every gold sentence and its copy share a
    created_at, so the canonical order breaks those ties on delta id (which hashes q)."""
    from run_nbrains import dev_epoch
    return H.build_ref(["gold", "hotpot"], encoder=ENC, nli=NLI_NAME, key=KEY, manifest=None, epoch_dir=None,
                       dev_theta=0.3, hotpot_tasks=[(1.0, 0, _TASK)],
                       hotpot_epoch=dev_epoch(ENC, NLI_NAME, 0.3))


def _flip_embed(text, j):
    """_embed() with one int8 step added to element j of ``text``'s row (a planted int8 flip)."""
    base = _embed()

    def embed(texts, batch):
        E = base(texts, batch)
        for i, t in enumerate(texts):
            if t == text:
                E[i, j] += np.abs(E[i]).max() / 127
        return E
    return embed


def _own(ref, name, embed=None):
    return H.run_cell(ref, name=name, embed=embed or _embed(), nli=_nli(), batch=1, key=KEY, encoder=ENC,
                      nli_name=NLI_NAME, own=True)


def test_hotpot_workload_is_one_scope_per_task_and_overlap_with_tied_copies():
    ref = _ref_own()
    assert set(ref) == {"gold", "hotpot-o1.0-t000"}
    s = H.replica("hotpot-o1.0-t000", ref, KEY, ENC)
    ds = [d for _, d in s.nodes().values()]
    assert len(ds) == 9                                   # 6 sentences + 3 paraphrased gold copies
    ca = Counter(d.created_at for d in ds)
    assert sorted(v for v in ca.values() if v > 1) == [2, 2, 2]
    assert {d.origin for d in ds} == {"agent0", "agent1", "agent2"}


def test_tie_swaps_counts_reordered_pairs_with_equal_created_at_only():
    ca = {"a": 1, "b": 1, "c": 2, "d": 3, "e": 3}
    assert H.tie_swaps(list("abcde"), list("abcde"), ca) == (2, 0)
    assert H.tie_swaps(list("abcde"), list("bacde"), ca) == (2, 1)
    assert H.tie_swaps(list("abcde"), list("baced"), ca) == (2, 2)


def test_own_issuance_identical_cells_report_zero():
    ref = _ref_own()
    cells = {"a": _own(ref, "a"), "b": _own(ref, "b")}
    rep = H.report_own(ref, cells, "a", key=KEY, encoder=ENC, nli_name=NLI_NAME)
    for wl in ref:
        c = rep[wl]["cells"]["b"]
        assert c["q_rows_differing"] == c["q_elements_differing"] == c["delta_ids_differing"] == 0
        assert c["tie_swaps"] == 0 and c["decisions_differing"] == 0 and c["tested_differing"] == 0
        assert c["fingerprint_equal"] and c["live_diff"] == 0
        assert rep[wl]["fingerprint1_distinct"] == 1
    assert rep["hotpot-o1.0-t000"]["cells"]["b"]["tie_pairs"] == 3
    s = rep["_summary"]["hotpot"]["b"]
    assert s["workloads"] == 1 and s["q_rows_differing"] == 0 and s["workloads_fingerprint_differing"] == 0


def test_own_issuance_with_the_issuers_q_folds_the_reference_deltas():
    """A cell whose q equals the issued q holds exactly the reference set: same fold as the shared arm.
    Its embedding rows are the issued int8 vectors as floats (quantize_int8 maps them back exactly)."""
    ref = _ref_own()
    issued = {d.text: np.frombuffer(d.q, np.int8).astype(np.float32)
              for wl in ref for _, d in H.replica(wl, ref, KEY, ENC).nodes().values()}
    own = _own(ref, "a", lambda texts, batch: np.stack([issued[t] for t in texts]))
    shared = _cell(ref, "a")
    for wl in ref:
        assert own[wl]["fingerprint1"] == shared[wl]["fingerprint1"]
        assert own[wl]["decisions1"] == shared[wl]["decisions1"]
        assert [k for k, _ in own[wl]["order"]] == [k.hex() for k in H.replica(wl, ref, KEY, ENC).nodes()]


def test_planted_int8_flip_is_counted_and_reorders_its_tied_pair():
    ref = _ref_own()
    wl = "hotpot-o1.0-t000"
    s = H.replica(wl, ref, KEY, ENC)
    ds = [d for _, d in s.nodes().values()]
    tied = [d for d in ds if sum(e.created_at == d.created_at for e in ds) > 1]
    a = _own(ref, "a")
    E = _embed()([d.text for d in tied], 1)
    swapped = None
    for d, v in zip(tied, E):                       # deterministic search for a flip that reorders its pair
        for j in np.argsort(np.abs(v))[:8]:
            b = _own(ref, "b", _flip_embed(d.text, int(j)))
            c = H.report_own(ref, {"a": a, "b": b}, "a", key=KEY, encoder=ENC, nli_name=NLI_NAME)[wl]["cells"]["b"]
            assert c["q_rows_differing"] == 1 and c["q_elements_differing"] == 1 and c["delta_ids_differing"] == 1
            assert c["tie_swaps"] in (0, 1)
            if c["tie_swaps"] == 1:
                swapped = c
                break
        if swapped:
            break
    assert swapped is not None, "no planted flip reordered a tied pair"
    by_a = {x[0]: x[1] for x in a[wl]["decisions1"]}
    by_b = {x[0]: x[1] for x in b[wl]["decisions1"]}
    assert swapped["decisions_differing"] == sum(by_a[k] != by_b.get(k) for k in by_a)
    assert swapped["fingerprint_equal"] == (a[wl]["fingerprint1"] == b[wl]["fingerprint1"])
