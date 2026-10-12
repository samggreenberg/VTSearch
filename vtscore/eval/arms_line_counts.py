"""#4786's line counts: at every click Test gives the trained head, what each candidate line would keep.

At beta 4 the first trained head's labels line keeps too many faces: from four
starting photos a median of 16 at vote 8 against about 3 of the person's withheld
photos, and with the head started from the centroid (#4785) still 5, while F-beta
falls 0.78 -> 0.62 at the hand-over (#4776's ``dip.csv``).  The ranking is nearly the
centroid's there (AP 0.86 vs 0.88), so the rest of the drop is the line.

A rule for the first trained clicks moves the line, not the ranking, so every count
rule can be priced on the same sessions once each such click also records, on the
withheld half (the corpus Test cuts):

* where its positives sit in the head's ranking (``test_pos_ranks``, 0-based
  positions in :class:`~vtscore.training.thresholds.LineRanking` order, as #4357's
  rank frames record them), so keeping the top ``k`` finds the positives ranked
  below ``k``;
* what the labels line keeps there at each preset (``k_labels_<b>``; at the run's
  beta, the set Test returns today);
* what the Goods' centroid's line keeps there (``k_centroid``: the app's rule at the
  run's beta, #4732/#4759/#4783), and the centroid's own Test at that line;
* what that count rule keeps when it reads the head's own logit scores instead of the
  centroid's cosines (``k_head_count``), a count of what stands out with no class model;
* the labels line's class model and the positives it estimates in the corpus, which
  say why a line keeps what it keeps;
* what the class model's own cut keeps under each of #4798's candidate fixes to its
  positives total (``k_cut_<name>``, :mod:`vtscore.eval.arms_cut_total`), whether or
  not the head's count serves at that click, and the 3-part chances' sum.

The row is recorded only: nothing here picks, trains or moves a line, so the session
is the app's.  One :data:`LINE_COUNT_COLUMNS` row per such click, in the
``__linecounts.csv`` side frame (``CALIB_LINE_COUNTS=1``).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from vtscore.eval.arms_cut_total import CUT_VARIANTS, chances_sum, variant_counts
from vtscore.eval.label_curve import _average_precision

#: The side frame's columns, one row per click Test gives the trained head.
LINE_COUNT_COLUMNS = [
    "seed",
    "dataset",
    "category",
    "calibration_seed",
    "style",
    "t",
    "phase",
    "n_good",
    "n_bad",
    "beta",
    "n_test",
    "n_test_pos",
    # Space-separated 0-based positions of the withheld positives in the head's ranking.
    "test_pos_ranks",
    # What Test keeps today: the labels line on the withheld half at the run's beta (or the fallback cut).
    "k_test",
    "k_labels_b025",
    "k_labels_b1",
    "k_labels_b4",
    # The Goods' centroid's line on the withheld half at the run's beta, and its own Test there.
    "k_centroid",
    "centroid_fbeta",
    "centroid_ap",
    # The centroid's count rule (the app's, at the run's beta) drawn on the head's own logit scores.
    "k_head_count",
    # The labels line's class model (held-out Goods / Bads, d') and the positives it estimates
    # among the withheld half: NaN with no class model this click (the fallback cut).
    "model_n_pos",
    "model_n_neg",
    "separation",
    "est_positives",
    # #4798: the class model's cut under each candidate (the gate aside), and the 3-part chances' sum.
    *[f"k_cut_{name}" for name in CUT_VARIANTS],
    "chances_sum",
]


def line_count_row(
    ident: dict[str, Any],
    t: int,
    phase: str,
    n_good: int,
    n_bad: int,
    beta: float,
    test_ids: Sequence[int],
    test_scores: Any,
    test_labels: Any,
    test_threshold: float,
    find_on_test: Any,
    centroid_scores: Any,
    centroid_threshold: float,
) -> dict[str, Any]:
    """One :data:`LINE_COUNT_COLUMNS` row for a click Test gave the trained head.

    *test_scores* / *test_labels* are the head's scores of the withheld half and their
    truth, in *test_ids* order; *test_threshold* is the cut Test applied (the labels
    line's at the run's *beta*, or the fallback with no class model).  *find_on_test*
    is that labels line with its corpus side fitted on the withheld half (``None``
    with no class model: the fallback keeps the same set at every preset).
    *centroid_scores* are the Goods' centroid head's scores of the same media, in the
    same order, cut at *centroid_threshold* where its line falls.
    """
    from vtscore.detectors.centroid_head import centroid_cut  # noqa: PLC0415
    from vtscore.eval.calibration_metrics import fbeta_metrics  # noqa: PLC0415
    from vtscore.eval.voting_iterations import _labels_line_counts, _rank_frame  # noqa: PLC0415

    nan = float("nan")
    s = np.asarray(test_scores, dtype=np.float64)
    y = np.asarray(test_labels, dtype=np.float64)
    frame = _rank_frame("step", t, list(test_ids), s, y)
    counts = _labels_line_counts(s, find_on_test, None if find_on_test is not None else test_threshold)
    c = np.asarray(centroid_scores, dtype=np.float64)
    ok = np.isfinite(s) & (s > 0.0) & (s < 1.0)
    logits = np.log(s[ok]) - np.log1p(-s[ok])
    k_head = int(np.count_nonzero(logits >= centroid_cut(logits, beta=beta))) if logits.size else -1
    model = find_on_test.model if find_on_test is not None else None
    est = nan
    if find_on_test is not None and find_on_test.unvoted_share is not None and find_on_test.unvoted_scores is not None:
        est = float(find_on_test.unvoted_share) * int(find_on_test.unvoted_scores.size)
    return {
        **ident,
        "t": int(t),
        "phase": phase,
        "n_good": int(n_good),
        "n_bad": int(n_bad),
        "beta": float(beta),
        "n_test": frame["n_test"],
        "n_test_pos": frame["n_test_pos"],
        "test_pos_ranks": frame["test_pos_ranks"],
        "k_test": int(np.count_nonzero(s >= test_threshold)),
        "k_labels_b025": counts[0.25],
        "k_labels_b1": counts[1.0],
        "k_labels_b4": counts[4.0],
        "k_centroid": int(np.count_nonzero(c >= centroid_threshold)),
        "centroid_fbeta": float(fbeta_metrics(c, y, centroid_threshold, beta)["fbeta"]),
        "centroid_ap": float(_average_precision(c, y)) if y.size and (y > 0.5).any() else nan,
        "k_head_count": k_head,
        "model_n_pos": model.n_pos if model is not None else nan,
        "model_n_neg": model.n_neg if model is not None else nan,
        "separation": float(find_on_test.separation) if find_on_test is not None else nan,
        "est_positives": est,
        **{f"k_cut_{name}": k for name, k in variant_counts(find_on_test, s, beta).items()},
        "chances_sum": chances_sum(find_on_test),
    }


__all__ = ["LINE_COUNT_COLUMNS", "line_count_row"]
