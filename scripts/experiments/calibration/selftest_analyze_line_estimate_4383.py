#!/usr/bin/env python3
"""Planted-answer self-test for ``analyze_line_estimate_4383.py``.

Checks what the construction of planted corpora fixes:

* **the oracle** is the deepest cut at precision >= P by brute force, and its
  best F1 is the best of every cut;
* **a perfectly separated corpus** (every positive on top): the oracle keeps
  ``floor(n_pos / P)`` items; today's ``fixed`` keeps 32 whatever n_pos is;
  ``grow`` walks down band by band to the oracle's band, so on a corpus ten
  times larger it keeps ten times more while ``fixed`` does not;
* **a corpus with no positives:** every rule returns a set of at least one
  item (best effort), precision 0, recall NaN, and the oracle keeps nothing;
* **audits are uniform within their band** (each band's picks lie inside it,
  and their labels are the corpus's own);
* **the logit shift** recovers a planted offset: a posterior that is too
  optimistic by a constant logit is pulled back onto the truth, and the
  recalibrated curve crosses P where the truth does;
* **band metrics are exact:** the band-stratified estimate of a union whose
  every band was censused is its true precision;
* the per-cell pipeline runs on a planted frame and is deterministic under
  its seed.

    python selftest_analyze_line_estimate_4383.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_line_estimate_4383 as L


def separated(n: int, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(1.0, 0.0, n)
    y = np.zeros(n, dtype=np.int64)
    y[:n_pos] = 1
    return s, y


def test_oracle_brute_force() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 300
        y = (rng.random(n) < 0.1).astype(np.int64)
        s = rng.normal(size=n) + y
        y = y[np.argsort(-s, kind="stable")]
        n_pos = int(y.sum())
        for p in L.FLOORS:
            k, rec, best_f1 = L.oracle(y, p)
            want_k, want_f1 = 0, 0.0
            for kk in range(1, n + 1):
                right = int(y[:kk].sum())
                if right / kk >= p:
                    want_k = kk
                want_f1 = max(want_f1, 2 * right / (kk + n_pos))
            assert k == want_k, (p, k, want_k)
            assert abs(rec - (y[:k].sum() / n_pos if k else 0.0)) < 1e-12
            assert abs(best_f1 - want_f1) < 1e-12


def test_separated_corpus_scales_and_fixed_does_not() -> None:
    rng = np.random.default_rng(2)
    for n, n_pos in ((2000, 40), (20000, 400)):
        s, y = separated(n, n_pos)
        k, _, _ = L.oracle(y, 0.5)
        assert k == n_pos * 2, k
        assert L.rule_fixed(y, 0.5)[0] == 32
        audits = L.draw_audits(y, rng)
        k_grow, votes, est = L.rule_grow(audits, 0.5, n, lb=False)
        # The oracle's 2 * n_pos sits between band edges; grow returns the deepest
        # band edge whose union still meets 50%, which is the edge at or below it.
        edges = L.band_edges(n)
        want = int(edges[edges <= 2 * n_pos].max())
        assert k_grow == want, (n, k_grow, want)
        assert votes >= L.M_PICKS
        k_iso, _, _ = L.rule_bands_iso(audits, 0.5, n, False, rng)
        assert 0.5 * want <= k_iso <= 2.5 * n_pos, (n, k_iso)
    # A seeded search from a model's guess lands on the same band, whichever side it starts from.
    s3, y3 = separated(4000, 100)
    fine = L.draw_audits(y3, rng, L.BASE_FINE)
    want = int(L.band_edges(4000, L.BASE_FINE)[L.band_edges(4000, L.BASE_FINE) <= 200].max())
    assert L.rule_grow(fine, 0.5, 4000, False, start_k=20)[0] == want
    assert L.rule_grow(fine, 0.5, 4000, False, start_k=3000)[0] == want
    # Fine bands can return fewer than 32: a corpus of 40 with 3 positives on top.
    s4, y4 = separated(40, 3)
    fine4 = L.draw_audits(y4, rng, L.BASE_FINE)
    assert L.rule_grow(fine4, 0.5, 40, False, start_k=3)[0] == 8
    # Ten times the corpus, ten times the set: what the fixed count cannot do.
    s1, y1 = separated(2000, 40)
    s2, y2 = separated(20000, 400)
    a1, a2 = L.draw_audits(y1, rng), L.draw_audits(y2, rng)
    assert L.rule_grow(a2, 0.5, 20000, False)[0] >= 8 * L.rule_grow(a1, 0.5, 2000, False)[0]
    assert L.rule_fixed(y2, 0.5)[0] == L.rule_fixed(y1, 0.5)[0]


def test_no_positives_is_best_effort() -> None:
    rng = np.random.default_rng(3)
    y = np.zeros(500, dtype=np.int64)
    assert L.oracle(y, 0.5)[0] == 0
    audits = L.draw_audits(y, rng)
    for k, _, _ in (
        L.rule_fixed(y, 0.5),
        L.rule_check(y, 0.5, rng),
        L.rule_grow(audits, 0.5, 500, False),
        L.rule_bands_iso(audits, 0.5, 500, False, rng),
    ):
        m = L.score_set(y, k, 0.5)
        assert m["k"] >= 1 and m["precision"] == 0.0 and np.isnan(m["recall"]) and m["shortfall"] == 0.5


def test_audits_are_uniform_within_bands() -> None:
    rng = np.random.default_rng(4)
    y = (rng.random(5000) < 0.3).astype(np.int64)
    a = L.draw_audits(y, rng)
    assert a.edges[0] == 0 and a.edges[-1] == 5000 and a.edges[1] == L.BASE
    for b, (lo, hi) in enumerate(zip(a.edges[:-1], a.edges[1:], strict=True)):
        assert np.all((a.ranks[b] >= lo) & (a.ranks[b] < hi))
        assert len(a.ranks[b]) == min(L.M_PICKS, hi - lo)
        assert np.array_equal(a.labels[b], y[a.ranks[b]])
    # A censused union's stratified estimate is exact.
    y2 = np.array([1, 1, 0, 1] * 8 + [0] * 20)  # 32 then 20
    a2 = L.Audits(np.array([0, 32, 52]), [np.arange(32), np.arange(32, 52)], [y2[:32], y2[32:52]])
    assert abs(a2.union_estimate(1) - 24 / 32) < 1e-12
    assert abs(a2.union_estimate(2) - 24 / 52) < 1e-12


def test_shift_recovers_a_planted_offset() -> None:
    rng = np.random.default_rng(5)
    n = 4000
    truth = 1.0 / (1.0 + np.exp(np.linspace(-4, 12, n)))  # decreasing in rank, 4 logits per 1000 ranks
    y = (rng.random(n) < truth).astype(np.int64)
    logit = np.log(truth / (1 - truth))
    optimistic = 1.0 / (1.0 + np.exp(-(logit + 2.0)))  # too sure by 2 logits: the crossing moves ~500 ranks
    k_opt, _ = L.k_from_curve(optimistic, 0.5)
    k_true, _ = L.k_from_curve(truth, 0.5)
    assert k_opt - k_true >= 400, (k_opt, k_true)
    a = L.draw_audits(y, rng)
    ranks, labels = a.upto(a.n_bands)
    fixed = L.fit_shift(optimistic, ranks, labels)
    k_fix, _ = L.k_from_curve(fixed, 0.5)
    assert abs(k_fix - k_true) < 0.5 * k_true, (k_fix, k_true, k_opt)


def _planted_frame(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = 3000
    test_y = (rng.random(n) < 0.02).astype(np.uint8)
    test_s = (rng.normal(0.3, 0.08, n) + 0.3 * test_y).astype(np.float32)
    pool_s = rng.normal(0.3, 0.08, n).astype(np.float32)
    vote_s = np.concatenate([pool_s[:20] + 0.3, pool_s[20:60]]).astype(np.float32)
    vote_y = np.concatenate([np.ones(20), np.zeros(40)]).astype(np.uint8)
    arrays = {
        "t150/t": np.int32(150),
        "t150/threshold": np.float32(0.5),
        "t150/test_scores": test_s,
        "t150/test_labels": test_y,
        "t150/pool_scores": pool_s,
        "t150/vote_scores": vote_s,
        "t150/vote_labels": vote_y,
        "t150/fold_cal_scores": vote_s,
        "t150/fold_cal_labels": vote_y,
        "t150/fold_cal_fold": np.array([0] * 30 + [1] * 30, dtype=np.uint8),
        "t150/fold_cal_phase": np.array(["hard"] * 60),
        "t150/fold_cal_vote": np.arange(60),
        "t150/fold_hay_scores": np.concatenate([pool_s, pool_s]).astype(np.float32),
        "t150/fold_hay_fold": np.array([0] * n + [1] * n, dtype=np.uint8),
        "t150/style": np.str_("whole_image"),
    }
    np.savez(path / "task_0000__pframes.npz", **arrays)
    pd.DataFrame([{"seed": 0, "dataset": "coco_better", "category": "cat@large", "t": 150}]).to_csv(
        path / "task_0000.csv", index=False
    )


def test_pipeline_runs_and_is_deterministic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cells = Path(tmp) / "cells"
        cells.mkdir()
        _planted_frame(cells)
        job = ("0.44%", str(cells / "task_0000__pframes.npz"), str(cells / "task_0000.csv"), (150,), 1, 7)
        a = pd.DataFrame(L.cell_rows(job))
        b = pd.DataFrame(L.cell_rows(job))
        assert len(a) == len(L.SIZES) * len(L.FLOORS) * len(L.RULES)
        pd.testing.assert_frame_equal(a, b)
        assert set(a["rule"]) == set(L.RULES)
        assert (a["k"] >= 1).all()
        full = a[(a["size"] == "full") & (a["floor"] == 0.5)].set_index("rule")
        assert full.loc["fixed", "k"] == 32 and full.loc["fixed", "votes"] == 0
        assert full.loc["gmm", "votes"] == 0 and full.loc["post", "votes"] == 0
        assert full.loc["gmm+shift5", "votes"] == 5 and full.loc["post+shift", "votes"] > 5
        summ = L.summarise(a)
        assert {"recall_share", "f1_share", "se_f1"} <= set(summ.columns)


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
