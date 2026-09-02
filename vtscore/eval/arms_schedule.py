"""The mix-in schedule arms (issue #2841).

One extra row per candidate blend schedule, on the production cut.  Deliberately
independent of the cut-variant arms in :mod:`vtscore.eval.arms_safe_gmm`: the
schedule screen only needs the pooled simulation scores the blend actually fits,
so it runs on a binary control arm too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vtscore.eval.row_metrics import operating_metrics, round6
from vtscore.training.blend_schedules import BlendContext


def _schedule_variant_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    sim_pooled_scores: list[float],
    inclusion: int,
    n_pool_rows: float,
    schedules: list[str],
) -> list[dict[str, Any]]:
    """One metric row per mix-in **schedule** (issue #2841).

    The dual of :func:`_safe_gmm_variant_rows`, which holds the schedule fixed
    and varies the GMM *cut*: here the cut is production's (midpoint of the
    component means, fitted on the style's inference pool) and the *schedule*
    varies.  Every schedule re-combines the same two candidate cuts from the
    same per-step model against the same held-out test scores, so the rows are
    paired within a step by construction and differ only in the mix-in rule.

    This is the study's **screen**, not its verdict.  Holding the trajectory
    fixed is precisely what makes it cheap - one simulation scores every
    schedule - but the blended threshold also feeds acquisition (Autopilot's
    Hard phase picks the item nearest the decision boundary), so schedules that
    would have labelled *different items* cannot show that here.  The A/B runs
    exist to measure what this screen structurally cannot see.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import (  # noqa: PLC0415
        blend_gmm_threshold,
        fit_gmm_threshold,
        safe_blend_weight,
    )

    xcal = float(details["xcal_threshold"])
    n_votes = int(details["n_votes"])
    pre_blend_provenance = str(details.get("pre_blend_provenance", "conformal"))
    fold_orderings = details.get("fold_orderings") or []
    cal_scores = np.array([s for scores, _ in fold_orderings for s in scores]) if fold_orderings else None
    cal_labels = np.array([lb for _, labels_ in fold_orderings for lb in labels_]) if fold_orderings else None

    # Required, not defaulted: the ``rare``/``pos`` families ramp on the class
    # split, so a guessed split would silently mis-score two whole families
    # rather than fail.  The caller sets both keys alongside ``n_votes``.
    ctx = BlendContext(n_labels=n_votes, n_good=int(details["n_good"]), n_bad=int(details["n_bad"]))
    # One fit, re-combined under every schedule.  The corridor schedules need
    # the component means, so the fit object rides along with the cut.
    gmm_cut, gmm_fit = fit_gmm_threshold(sim_pooled_scores)

    rows: list[dict[str, Any]] = []
    for name in schedules:
        threshold = blend_gmm_threshold(xcal, gmm_cut, ctx, schedule=name, fit=gmm_fit)
        weight = safe_blend_weight(ctx, name)
        row = operating_metrics(
            base_scores,
            base_labels,
            threshold,
            inclusion,
            cal_scores,
            cal_labels,
            pool_variant="max",
            provenance="gmm_blend" if weight < 1.0 else pre_blend_provenance,
            n_pool_rows=n_pool_rows,
        )
        row["gmm_variant"] = ""
        row["schedule"] = name
        row["xcal_threshold"] = round6(xcal)
        row["gmm_cut"] = round6(gmm_cut)
        row["blend_weight"] = round6(weight)
        rows.append(row)
    return rows
