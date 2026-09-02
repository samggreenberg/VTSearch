"""Shaping one result row out of a scored test set.

:func:`operating_metrics` is the shared tail of every arm the harness emits:
given held-out scores, labels and a threshold it produces the cost / FPR / FNR /
oracle / regret block that each row carries, so the shipped row and a dozen
experiment arms are always priced by the same code.  The rest of the module is
what that computation leans on - the 6-dp rounding every emitted float goes
through, the fold-count reader for a provenance string, and a small memo in
front of the cross-fitted honest oracle.

It lives apart from :mod:`vtscore.eval.voting_iterations` because the per-study
``arms_*`` modules all need it and none of them should need the harness.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from vtscore.eval.voting_columns import SKYLINE_COLUMNS
from vtscore.training.thresholds import CUT_KIND_INTERIOR

#: ``fold_anchored[2/4]`` / ``fold_conformal_qmean[3/4]`` - *a* of *k*.
_PROVENANCE_USED_RE = re.compile(r"\[(\d+)/(\d+)\]$")


def folds_used(provenance: str, k: int) -> float:
    """How many of the *k* folds contributed a cut, read off *provenance*.

    NaN for the arms where the question has no answer: the pooled conformal cut
    and the blend take one quantile over every fold's scores at once, so no fold
    ever "contributes a cut" that could be counted or dropped.  Reporting *k*
    there would look like agreement with the combining arms and hide exactly the
    asymmetry #3115 is about - a single-class fold is silently *in* the pool and
    explicitly *out* of a mean.
    """
    m = _PROVENANCE_USED_RE.search(provenance)
    return float(m.group(1)) if m else float("nan")


def round6(x: float) -> float:
    """Round to 6 dp when finite, else pass NaN/inf through unchanged."""
    import math  # noqa: PLC0415

    return round(x, 6) if math.isfinite(x) else x


#: Memo for :func:`honest_oracle`, keyed on a digest of its exact inputs.
#: Bounded because the only reuse that matters is *within* a step, where every
#: variant row measures the same ``(base_scores, base_labels)`` at the same
#: inclusion: a step emits dozens of rows and the cross-fitted oracle is five
#: sorts, so recomputing it per row would be the dominant cost of the
#: decomposition.  Across steps the key changes and the old entry is dead, so a
#: handful of slots is all this ever needs.
_HONEST_ORACLE_MEMO: "dict[bytes, tuple[float, float]]" = {}


_HONEST_ORACLE_MEMO_MAX = 8


def honest_oracle(scores: "np.ndarray", labels: "np.ndarray", wf: float, wn: float) -> tuple[float, float]:
    """Memoized :func:`~vtscore.eval.transfer_rules.honest_test_oracle`.

    Safe to memoize because the estimator is a pure function of its arguments -
    its own resampling is seeded from a digest of the score array rather than
    from global RNG state (see :func:`vtscore.eval.transfer_rules._rng`), so two
    calls on equal inputs return bit-identical results with or without the cache.
    The key is a digest of both arrays *and* the cost weights, because the same
    test scores are decomposed at more than one inclusion in a single run.
    """
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from vtscore.eval.transfer_rules import honest_test_oracle  # noqa: PLC0415

    s = np.ascontiguousarray(scores, dtype=np.float64)
    lb = np.ascontiguousarray(labels, dtype=np.float64)
    h = hashlib.blake2b(s.tobytes(), digest_size=16)
    h.update(lb.tobytes())
    h.update(np.asarray([wf, wn], dtype=np.float64).tobytes())
    key = h.digest()
    hit = _HONEST_ORACLE_MEMO.get(key)
    if hit is not None:
        return hit
    out = honest_test_oracle(s, lb, wf, wn)
    if len(_HONEST_ORACLE_MEMO) >= _HONEST_ORACLE_MEMO_MAX:
        _HONEST_ORACLE_MEMO.clear()
    _HONEST_ORACLE_MEMO[key] = out
    return out


def operating_metrics(
    scores: "np.ndarray",
    labels: "np.ndarray",
    threshold: float,
    inclusion: int,
    cal_scores: "np.ndarray | None",
    cal_labels: "np.ndarray | None",
    *,
    pool_variant: str,
    provenance: str,
    n_pool_rows: float,
) -> dict[str, Any]:
    """Full per-step calibration metrics for one pooling (issue #2781).

    ``scores``/``labels`` are the held-out test scores+labels under *pool_variant*
    at the trained *threshold*.  Computes the trained cost, the oracle cost (best
    cut on the test scores) and the resulting **regret**, plus the
    calibration-set oracle that splits regret into *rule inefficiency*
    (trained-vs-best-use-of-calibration) and *calibration→test shift* (best cut
    on calibration vs. best cut on test).  ``cal_scores``/``cal_labels`` are the
    pooled calibration fold orderings under the same pooling; ``None`` skips the
    decomposition (leaves those columns NaN).

    **Two reference points, because the naive one is optimistic** (#3116, #3248).
    ``oracle_cost`` is the minimum of the empirical cost over the very test
    sample it is then scored on - :func:`~vtscore.eval.calibration_metrics.oracle_cut`'s
    own docstring calls it a lower bound on achievable cost rather than a rule -
    so every gap measured against it is inflated by however much that minimum
    overfits.  #2883 measured that optimism directly and found it was the *whole*
    of the sibling ``transfer`` term (+0.041 naive, −0.001 cross-fitted).
    ``oracle_cost_honest`` is the same quantity cross-fitted
    (:func:`~vtscore.eval.transfer_rules.honest_test_oracle`: cut chosen on K−1
    folds, paid on the held-out one), and the pair **brackets** the population
    optimum rather than pinning it - naive from below, honest from above, since
    the honest cut sees only ``(K−1)/K`` of the sample.  ``regret`` and
    ``calibration_shift`` therefore ship beside ``regret_honest`` and
    ``calibration_shift_honest``; read the two as an interval, and prefer the
    honest one whenever a *level* rather than a paired contrast is being quoted.
    ``rule_inefficiency`` is untouched by the choice - it never references the
    test oracle - so both decompositions telescope exactly:

    * ``rule_inefficiency + calibration_shift        == regret``
    * ``rule_inefficiency + calibration_shift_honest == regret_honest``

    **The split's reference moves with anything that feeds ``cal_scores``**
    (#3116).  ``c_thr`` is estimated *from the calibration set*, so a study that
    sweeps a knob changing that set's size or content - ``calibrate_count`` is
    the case on record - is moving the yardstick it measures against.  As the
    calibration set grows, ``c_thr`` converges on the test-oracle cut, which
    shrinks ``calibration_shift`` and widens ``rule_inefficiency`` **from one
    cause, in opposite directions**, with their sum pinned to ``regret`` by
    construction.  #2897 read exactly that anti-correlation as a finding; it is
    algebra.  Do not report the two terms as independent effects of such a knob
    without a reference held fixed across the arms - and note that
    ``rule_inefficiency`` is a signed cost gap between two cuts, **not** a
    variance: it is routinely negative (the trained cut beating a
    calibration-set "oracle" that overfits a handful of scores), and a study
    asking whether the *threshold* got less variable wants ``sd(threshold)``
    across seeds, which ``analyze_folds_2897.py`` reports.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import (  # noqa: PLC0415
        detection_metrics,
        inclusion_weights,
        is_degenerate,
        operating_cost,
        oracle_cut,
        threshold_percentile,
    )
    from vtscore.eval.label_curve import _auroc, _average_precision  # noqa: PLC0415

    wf, wn = inclusion_weights(inclusion)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    cost, fpr, fnr = operating_cost(scores, labels, threshold, wf, wn)
    o_thr, o_cost, o_fpr, o_fnr = oracle_cut(scores, labels, wf, wn)
    regret = cost - o_cost
    # The honest half of the bracket; NaN on a test sample too small or too
    # one-sided to cross-fit, which leaves the honest columns empty rather than
    # silently falling back to the optimistic reference they exist to correct.
    o_cost_honest, o_thr_honest = honest_oracle(scores, labels, wf, wn)
    regret_honest = cost - o_cost_honest

    nan = float("nan")
    if cal_scores is not None and np.asarray(cal_scores).size > 0:
        cal_scores = np.asarray(cal_scores, dtype=np.float64)
        cal_labels = np.asarray(cal_labels, dtype=np.float64)
        c_thr, _, _, _ = oracle_cut(cal_scores, cal_labels, wf, wn)
        cal_oracle_cost, _, _ = operating_cost(scores, labels, c_thr, wf, wn)
        rule_inefficiency = cost - cal_oracle_cost
        calibration_shift = cal_oracle_cost - o_cost
        calibration_shift_honest = cal_oracle_cost - o_cost_honest
    else:
        c_thr = nan
        cal_oracle_cost = nan
        rule_inefficiency = nan
        calibration_shift = nan
        calibration_shift_honest = nan

    return {
        "pool_variant": pool_variant,
        # Safe-threshold study columns (issue #2799): defaults here; the base
        # row and the per-variant rows overwrite them where they apply.
        "gmm_variant": "",
        "schedule": "",
        "xcal_threshold": round6(float(threshold)),
        "gmm_cut": nan,
        "blend_weight": nan,
        # Fold-count study columns (issue #2897); only the fold-count arms set
        # them.  ``n_cal_scores`` is the pooled calibration-set size the
        # conformal quantile is taken over, which is what K actually buys.
        "fold_count": nan,
        "fold_seconds": nan,
        # #3314: `fold_seconds` counts the fold *fits* and the conformal rule's
        # overhead only.  A live run at K also scores the sim set once per fold
        # and fits production's anchored mixture over the prefix, both of which
        # scale with K and neither of which lands in `train_seconds`,
        # `xcal_seconds`, `pool_score_seconds` or `test_score_seconds`.
        # `cal_seconds` is the sum of all four and is what a cost ceiling reads.
        "fold_fit_seconds": nan,
        "fold_score_seconds": nan,
        "anchored_seconds": nan,
        "cal_seconds": nan,
        "n_cal_scores": nan,
        # #3115: how many of the K folds actually contributed a cut, parsed off
        # the arm's own provenance.  A combine rule and a pooled quantile weight
        # a degenerate (single-class) fold completely differently, so a contrast
        # between them is only readable next to the count of folds that were
        # dropped rather than averaged.
        "n_folds_used": nan,
        # Cut-rule study columns (issue #2836); only the variant rows set them.
        # ``cut_fallback_kind`` says *what was substituted* where ``cut_fallback``
        # only says *that* something was, which the two emitting families answer
        # differently on the same fits (issue #2900).
        "cut_fallback": 0,
        "cut_fallback_kind": CUT_KIND_INTERIOR,
        "cut_fail_reason": "",
        "raw_cut_cost": nan,
        "raw_cut_fpr": nan,
        "raw_cut_fnr": nan,
        "threshold": round6(float(threshold)),
        "threshold_provenance": provenance,
        "degenerate": 1 if is_degenerate(scores, threshold) else 0,
        "threshold_percentile": round6(threshold_percentile(scores, threshold)),
        "cost": round6(cost),
        "fpr": round6(fpr),
        "fnr": round6(fnr),
        **{k: round6(v) for k, v in detection_metrics(scores, labels, threshold).items()},
        "auroc": round6(float(_auroc(scores, labels))),
        "average_precision": round6(float(_average_precision(scores, labels))),
        "oracle_threshold": round6(float(o_thr)),
        "oracle_cost": round6(o_cost),
        "oracle_fpr": round6(o_fpr),
        "oracle_fnr": round6(o_fnr),
        "regret": round6(regret),
        # The cross-fitted reference and the two terms it re-bases (#3116).
        # Bracket, not replacement: `oracle_cost` bounds the population optimum
        # from below and `oracle_cost_honest` from above, so a level quoted from
        # either alone is one end of an interval.
        "oracle_threshold_honest": round6(float(o_thr_honest)),
        "oracle_cost_honest": round6(o_cost_honest),
        "regret_honest": round6(regret_honest),
        "cal_oracle_threshold": round6(float(c_thr)),
        "cal_oracle_cost": round6(cal_oracle_cost),
        "rule_inefficiency": round6(rule_inefficiency),
        "calibration_shift": round6(calibration_shift),
        "calibration_shift_honest": round6(calibration_shift_honest),
        # The #3322 decomposition (see `SKYLINE_COLUMNS`).  Filled in one pass
        # after the run, from the skyline arm's own row - a skyline is
        # vote-independent, so there is nothing per-step to compute here, and
        # NaN is the honest value on a run that asked for no skyline.
        **dict.fromkeys(SKYLINE_COLUMNS, nan),
        "n_pool_rows": round6(float(n_pool_rows)),
    }
