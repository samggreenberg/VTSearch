"""Goodness-of-fit rows for the mixture the cut is read off (issue #3329).

One side frame, emitted on a stride rather than every step: the fitted mixture
moves slowly against the vote count, so a row per step would multiply the fit
cost by the horizon to resolve a curve a fifth of the points already resolves.
Its schema is :data:`~vtscore.eval.voting_columns.FIT_QUALITY_ROW_COLUMNS`.
"""

from __future__ import annotations

from typing import Any


from vtscore.eval.fit_quality import fit_quality_row
from vtscore.training.thresholds import FOLD_ANCHOR_WEIGHT


def _fit_quality_rows(
    base_row: dict[str, Any],
    safe_cut: Any,
    sim_scores_by_geometry: dict[str, Any],
    sim_labels_by_geometry: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    """Goodness-of-fit rows for one step (issue #3329).

    Two scopes, deliberately not pooled - see :data:`FIT_QUALITY_ROW_COLUMNS`.

    The **fold** scopes read ``safe_cut.fits`` against ``safe_cut.fold_haystacks``,
    which is the shipped estimator scored against its own data.  Nothing else in
    the eval tier reads those fits at all, so a fold row is the only place the
    threshold the app actually computes is compared to the distribution it claims
    to describe.  They are label-free: a fold haystack is the unlabelled
    remainder under *that fold's* model, and the sim labels belong to the final
    model's score scale, so attaching them here would compare two scalings.

    The **sim** scopes fit the unanchored mixture to the labelled sim scores and
    carry the shape and identification statistics.  Both pooling geometries are
    emitted because ``pooled`` and ``image`` are the same media under the same
    model with only the pooling changed - the exactly-paired form of "does
    max-pooling make the Bad mode non-Gaussian?".
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import fit_score_gmm, gmm_fit_array  # noqa: PLC0415

    rows: list[dict[str, Any]] = []

    if safe_cut is not None:
        fits = getattr(safe_cut, "fits", ()) or ()
        haystacks = getattr(safe_cut, "fold_haystacks", ()) or ()
        anchor_counts = getattr(safe_cut, "anchor_counts", ()) or ()
        for i, (fit, hay) in enumerate(zip(fits, haystacks, strict=False)):
            arr = np.asarray(hay, dtype=np.float64).ravel()
            # The votes THIS fold anchored on - its held-out share - which is
            # what the mass fraction needs.  The obvious-looking `n_anchored` is
            # a count of FOLDS (0/1/2), and passing it produced a mass share
            # flat at 2.9e-4 across every click of the first real run, off by
            # the fold's vote count (#3329).  Recorded on the cut rather than
            # recomputed here so the two cannot disagree.
            n_anchors = int(anchor_counts[i]) if i < len(anchor_counts) else 0
            # The counterfactual the H3 delta is against: the same sample, the
            # same estimator, no anchors.  `fold_haystacks[i]` IS the array the
            # shipped fit was fitted to (sorted), so the two fits are comparable
            # point for point.
            unanchored = fit_score_gmm(arr)
            fq = fit_quality_row(
                arr,
                fit,
                cut=threshold,
                n_anchors=n_anchors,
                anchor_weight=float(getattr(safe_cut, "anchor_weight", FOLD_ANCHOR_WEIGHT)),
                unanchored_fit=unanchored,
            )
            rows.append(
                {
                    **base_row,
                    "scope": f"fold{i}",
                    "fold_index": i,
                    "n_folds": len(fits),
                    "n_fit_sample": int(arr.size),
                    "fit_ok": fit is not None,
                    "fq_w_lo": fit.w_lo,
                    "fq_mu_lo": fit.mu_lo,
                    "fq_var_lo": fit.var_lo,
                    "fq_w_hi": fit.w_hi,
                    "fq_mu_hi": fit.mu_hi,
                    "fq_var_hi": fit.var_hi,
                    "fq_cut": float(threshold),
                    **fq,
                }
            )

    for geometry, scores in sim_scores_by_geometry.items():
        if scores is None:
            continue
        labels = sim_labels_by_geometry.get(geometry)
        arr = gmm_fit_array(np.asarray(scores, dtype=np.float64).ravel())
        fit = fit_score_gmm(arr)
        fq = fit_quality_row(
            arr,
            fit,
            cut=threshold,
            labels=labels,
            label_scores=np.asarray(scores, dtype=np.float64).ravel() if labels is not None else None,
        )
        rows.append(
            {
                **base_row,
                "scope": f"sim:{geometry}",
                "fold_index": -1,
                "n_folds": 0,
                "n_fit_sample": int(arr.size),
                "fit_ok": fit is not None,
                "fq_w_lo": None if fit is None else fit.w_lo,
                "fq_mu_lo": None if fit is None else fit.mu_lo,
                "fq_var_lo": None if fit is None else fit.var_lo,
                "fq_w_hi": None if fit is None else fit.w_hi,
                "fq_mu_hi": None if fit is None else fit.mu_hi,
                "fq_var_hi": None if fit is None else fit.var_hi,
                "fq_cut": float(threshold),
                **fq,
            }
        )
    return rows
