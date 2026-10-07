#!/usr/bin/env python3
"""Planted-answer self-test for ``ceiling_line_4490``.

Writes synthetic withheld-half snapshots in the layout ``run_cells.py`` saves and checks that the analysis:

* replays the shipped line exactly as :func:`fit_labels_line` draws it from the same folds;
* reads a census of Bads (a random sample of the corpus's negatives) as representative and a session's
  selected Bads (drawn from the top of the corpus) as not, so the candidate rule only moves the first;
* on a corpus whose negatives carry a heavy upper tail, keeps the shipped line too deep and the candidate near the
  planted positives;
* scores F-beta and the best cut in closed form on a planted ranking.

    python selftest_ceiling_line_4490.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402

import ceiling_line_4490 as C  # noqa: E402
from vtscore.training.thresholds import fit_labels_line  # noqa: E402
from vtscore.training.thresholds.labels_line import class_score_model  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def negatives(rng: np.random.Generator, n: int) -> np.ndarray:
    """A normal bulk with a heavy upper tail (a Student-t), on the logit scale."""
    return -2.0 + 0.4 * rng.standard_t(3, size=n)


def write(path: Path, key: str, scores, labels, fold_scores, fold_labels, beta=float("nan")) -> dict:
    folds = [(fold_scores[i::2], fold_labels[i::2]) for i in range(2)]
    model = class_score_model(folds)
    fs = np.concatenate([f[0] for f in folds])
    fl = np.concatenate([f[1] for f in folds]).astype(np.int8)
    fi = np.concatenate([np.full(f[0].size, i, dtype=np.int16) for i, f in enumerate(folds)])
    meta = {"t": 0 if key == "ceiling" else 150, "phase": key, "beta": beta, "model": asdict(model)}
    np.savez_compressed(
        path,
        **{
            f"siglip/whole_image/{key}/scores": np.asarray(scores, dtype=np.float64),
            f"siglip/whole_image/{key}/labels": np.asarray(labels, dtype=np.int8),
            f"siglip/whole_image/{key}/fold_scores": fs,
            f"siglip/whole_image/{key}/fold_labels": fl,
            f"siglip/whole_image/{key}/fold_index": fi,
            f"siglip/whole_image/{key}/meta": np.array(json.dumps(meta)),
        },
    )
    return {"folds": folds}


def main() -> int:
    rng = np.random.default_rng(4490)
    n_neg, n_pos = 10_000, 50
    corpus = np.concatenate([negatives(rng, n_neg), 1.0 + 0.4 * rng.standard_normal(n_pos)])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    scores = sigmoid(corpus)
    goods = sigmoid(1.0 + 0.4 * rng.standard_normal(30))
    census_bads = sigmoid(negatives(rng, 6000))
    # A session's Bads: drawn from the top of a corpus of negatives, as active learning picks them.  And a mix like
    # the sessions the first gate misfired on: a selected minority among mostly uniform Bads (20 + 100).
    pool = np.sort(negatives(rng, n_neg))[::-1]
    session_bads = sigmoid(pool[rng.choice(300, 120, replace=False)])
    mixed_bads = sigmoid(np.concatenate([pool[rng.choice(300, 20, replace=False)], negatives(rng, 100)]))

    with tempfile.TemporaryDirectory() as tmp:
        cells = Path(tmp)
        census = write(cells / "task_0000__testscores.npz", "ceiling", scores, labels,
                       np.concatenate([goods, census_bads]), np.concatenate([np.ones(30), np.zeros(6000)]))  # fmt: skip
        write(cells / "task_0001__testscores.npz", "last", scores, labels,
              np.concatenate([goods, session_bads]), np.concatenate([np.ones(30), np.zeros(120)]), beta=1.0)  # fmt: skip
        write(cells / "task_0002__testscores.npz", "final", scores, labels,
              np.concatenate([goods, mixed_bads]), np.concatenate([np.ones(30), np.zeros(120)]), beta=1.0)  # fmt: skip

        snap = C.load(cells / "task_0000__testscores.npz")["siglip/whole_image/ceiling"]
        line = fit_labels_line(census["folds"], scores, list(range(scores.size)), {})
        for b in C.BETAS:
            check(f"shipped replay == fit_labels_line at beta {b}", C.shipped(snap, scores)(b) == line.threshold(b))

        sess = C.load(cells / "task_0001__testscores.npz")["siglip/whole_image/last"]
        share_c = C.bads_share_above_median(snap, scores)
        share_s = C.bads_share_above_median(sess, scores)
        check("a census of Bads reads as representative", C.representative(snap, scores), f"share {share_c:.2f}")
        check("a session's selected Bads do not", not C.representative(sess, scores), f"share {share_s:.2f}")
        mixed = C.load(cells / "task_0002__testscores.npz")["siglip/whole_image/final"]
        check("a selected minority among uniform Bads is not representative either", not C.representative(mixed, scores),
              f"top-5% enrichment {C.top_enrichment(mixed, scores):.1f}")  # fmt: skip
        check("the median test passes that mix, which is why the gate reads the top",
              C.representative_by_median(mixed, scores), f"share {C.bads_share_above_median(mixed, scores):.2f}")  # fmt: skip
        check("the census sits near enrichment 1", abs(C.top_enrichment(snap, scores) - 1.0) < 0.5,
              f"{C.top_enrichment(snap, scores):.2f}")  # fmt: skip
        check("so the candidate keeps the session's shipped line",
              C.rep_kde(sess, scores)(1.0) == C.shipped(sess, scores)(1.0))  # fmt: skip

        k_ship = int((scores >= C.shipped(snap, scores)(1.0)).sum())
        k_fix = int((scores >= C.rep_kde(snap, scores)(1.0)).sum())
        f_ship = C.kept(scores, labels, C.shipped(snap, scores)(1.0), 1.0)["f"]
        f_fix = C.kept(scores, labels, C.rep_kde(snap, scores)(1.0), 1.0)["f"]
        check("a heavy negative tail sends the shipped line deep", k_ship > 2 * n_pos, f"keeps {k_ship}")
        check("the candidate cuts near the planted positives", n_pos // 2 <= k_fix <= 2 * n_pos, f"keeps {k_fix}")
        check("and scores higher", f_fix > f_ship + 0.05, f"F1 {f_ship:.2f} -> {f_fix:.2f}")

        rows = C.price([str(cells)], [], ["shipped", "rep_kde"], jobs=1)
        check("price scores the ceiling at every preset and a session at its own",
              sorted(rows[rows.key == "ceiling"].beta.unique()) == list(C.BETAS)
              and rows[rows.key == "last"].beta.unique().tolist() == [1.0])  # fmt: skip

    # A KDE of one or two Bads (a session's first clicks) is a density on the grid, not a crash.
    for xn in (np.array([-1.0]), np.array([-1.0, -1.0]), np.array([-1.0, -0.2])):
        pdf = C.bads_kde(xn)
        x = np.linspace(-6, 4, 2001)
        mass = float(np.trapezoid(pdf(x), x))
        check(f"a KDE of {xn.size} Bad(s) integrates to 1", abs(mass - 1.0) < 1e-3, f"{mass:.4f}")

    # Closed form: ranking [1, 0, 1, 0, 0], two positives.
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    check("F1 of the top 3 is 2*2/(2+3)", abs(C.kept(s, y, 0.65, 1.0)["f"] - 0.8) < 1e-12)
    bc = C.best_cut(s, y, 1.0)
    check("the best F1 cut keeps 3 (0.8) over 1 (0.667)", bc["k_best"] == 3 and abs(bc["f_best"] - 0.8) < 1e-12)
    check("a cut that keeps nothing scores 0", C.kept(s, y, 2.0, 1.0)["f"] == 0.0)

    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
