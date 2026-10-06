"""The Test arm: Test mode's autopilot run on the withheld half, read against the truth (#4523).

Test mode (``vtscore/docs/packages/training.md``) measures a detector's line on a corpus
it never trained on, from uniform picks within rank bands, and stops when its
ranges are narrow enough or its budgets are spent
(:mod:`vtscore.training.thresholds.line_test`).  Whether those ranges hold,
and what the stop costs, are claims about a simulated user over thousands of
sessions; this module is the harness side of pricing them.

The harness already measures *train on X, Find on Y*: the withheld half is
the Test corpus and its ground truth is in hand.  After a training run's last
ordinary click the arm takes the withheld half as Find sees it - the final
model's scores, the labels line with its corpus side fitted there
(``find_on_test``) - draws the line at the run's balance, and runs the Test
autopilot on that frozen ranking with every pick answered from the truth.
Test votes never train, so the training run is unchanged by the arm; what it
records is the Test's own cost and accuracy, per session:

* picks to Done, per phase, and which rule ended each phase;
* the precision, recall and F-beta ranges at Done, and whether each held the
  truth;
* the F-beta the verdict reads against the truth at the same line;
* the verdict's reading (:func:`read_verdict`: ship / lean / retrain) against
  what the truth says (:func:`oracle_verdict`) under the same rule;
* at every band edge, whether the re-estimated ranges held, since *Lean the
  Threshold* reads those.

Everything here is a pure function of ``(ranking's truth, line count, beta,
model posteriors, budgets, seed)``, so the same code replays a saved
withheld-half snapshot (``task_NNNN__testscores.npz``) over a grid of targets
and budgets without retraining anything: that is how #4523's study prices the
stop rule.  :func:`simulate_line_test` is the arm; :func:`line_test_inputs`
turns a scored corpus into its inputs; :func:`line_test_row` is the two
together, as the harness and the study both call it.

The verdict rule is the study's proposal, not app behaviour: nothing in the
app reads it yet (that is #4524's slice, which takes its thresholds from the
study's report).  Its two constants are named here so the study can move them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vtscore.training.thresholds.line_test import (
    DEFAULT_BUDGETS,
    PHASE_NOTHING,
    LineBudgets,
    LineEstimates,
    LineTest,
    found_words,
    line_bands,
)
from vtscore.training.thresholds.spot_check import BALANCE_PRESETS, fbeta_score

#: How much better another band edge's F-beta must read than the line's for
#: the verdict to say *Lean the Threshold* (a proposal for #4523 to price).
LEAN_GAIN = 0.05

#: The F-beta a line must likely reach: a range whose upper end is under this
#: reads *Add Corrections and retrain* (a proposal for #4523 to price).
RETRAIN_BAR = 0.5

#: The verdict's readings.  ``nothing`` is Test's *Nothing to test*.
VERDICT_SHIP = "ship"
VERDICT_LEAN = "lean"
VERDICT_RETRAIN = "retrain"
VERDICT_NOTHING = "nothing"
VERDICTS = (VERDICT_SHIP, VERDICT_LEAN, VERDICT_RETRAIN, VERDICT_NOTHING)

#: Where the line the arm tests came from: the labels line with its corpus
#: side fitted on the withheld half (Find's own fit), the retrain's fallback
#: cut (no class model that step), or no line at all.
LINE_LABELS = "labels"
LINE_FALLBACK = "fallback"
LINE_NONE = "none"

#: One NaN object, so two rows reporting "no value" compare equal (``nan != nan`` otherwise).
_NAN = float("nan")


@dataclass(frozen=True)
class LineTestInputs:
    """A scored corpus as the Test sees it: its truth in rank order, the line's count, the model's posteriors."""

    #: ``True`` where the item at that rank is a positive.
    truth: np.ndarray
    line_count: int
    #: The labels line's chance of each ranked item being a positive, aligned
    #: with *truth*; ``None`` with no class model.
    posteriors: np.ndarray | None
    line_source: str
    #: How many items the labels line keeps at each balance preset: what the
    #: verdict's *Lean the Threshold* re-estimates (#4540).  Empty with no class model.
    preset_counts: Mapping[float, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.truth.size)

    @property
    def positives(self) -> int:
        return int(self.truth.sum())


def line_test_inputs(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    find_on_test: Any = None,
    fallback_threshold: float | None = None,
    *,
    beta: float,
) -> LineTestInputs:
    """The withheld half as Find ranks it, with the line Find draws on it at *beta*.

    The ranking is score descending, ties by position (a stable sort), which
    is the order the labels line's own corpus fit keeps, so its item
    posteriors (``unvoted_posteriors``) align with the ranking directly when
    every score is finite and in [0, 1]; otherwise they are read off the fit
    by score.  The line keeps every item scoring at or above
    ``find_on_test.threshold(beta)`` - the headline's returned set - or the
    retrain's *fallback_threshold* with no class model; with neither there is
    no line, and the count is 0 (*Nothing to test*).
    """
    a = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if a.shape != y.shape:
        raise ValueError("scores and labels must align")
    order = np.argsort(-a, kind="stable")
    truth = y[order] >= 0.5
    thr: float | None
    source = LINE_NONE
    post: np.ndarray | None = None
    if find_on_test is not None:
        thr = float(find_on_test.threshold(float(beta)))
        source = LINE_LABELS
        fit_scores = getattr(find_on_test, "unvoted_scores", None)
        fit_post = getattr(find_on_test, "unvoted_posteriors", None)
        if fit_scores is not None and fit_post is not None:
            ranked = a[order]
            if fit_scores.shape == ranked.shape and np.allclose(fit_scores, ranked, equal_nan=False):
                post = np.asarray(fit_post, dtype=np.float64)
            else:
                # Some scores were not scorable (dropped by the fit): read each
                # ranked item's posterior off the fit at its score, 0 where the
                # score itself is unusable.
                xp = np.asarray(fit_scores, dtype=np.float64)[::-1]
                fp = np.asarray(fit_post, dtype=np.float64)[::-1]
                usable = np.isfinite(ranked) & (ranked >= 0.0) & (ranked <= 1.0)
                post = np.zeros_like(ranked)
                post[usable] = np.interp(ranked[usable], xp, fp)
    elif fallback_threshold is not None and math.isfinite(float(fallback_threshold)):
        thr = float(fallback_threshold)
        source = LINE_FALLBACK
    else:
        thr = None
    count = int(np.count_nonzero(a >= thr)) if thr is not None and math.isfinite(thr) else 0
    presets: dict[float, int] = {}
    if find_on_test is not None:
        for b in BALANCE_PRESETS:
            t_b = float(find_on_test.threshold(float(b)))
            presets[float(b)] = int(np.count_nonzero(a >= t_b)) if math.isfinite(t_b) else 0
    return LineTestInputs(truth, count, post, source, presets)


def truth_at(truth: np.ndarray, count: int, beta: float) -> tuple[float, float, float]:
    """``(precision, recall, F-beta)`` of the top *count* of a ranking whose positives are *truth*."""
    k = max(0, min(int(count), int(truth.size)))
    tp = int(np.count_nonzero(truth[:k]))
    total = int(np.count_nonzero(truth))
    precision = tp / k if k > 0 else 0.0
    recall = tp / total if total > 0 else 0.0
    return precision, recall, float(fbeta_score(tp, k, total, beta))


def best_cut(truth: np.ndarray, beta: float) -> tuple[int, float]:
    """The deepest-is-not-best oracle: the count whose F-beta is the highest any cut of the ranking reaches."""
    total = int(np.count_nonzero(truth))
    if total == 0 or truth.size == 0:
        return 0, 0.0
    tp = np.cumsum(truth.astype(np.float64))
    k = np.arange(1, truth.size + 1, dtype=np.float64)
    b2 = beta * beta
    fb = (1.0 + b2) * tp / (b2 * total + k)
    i = int(np.argmax(fb))
    return i + 1, float(fb[i])


def read_verdict(est: LineEstimates, line_count: int, *, lean_gain: float = LEAN_GAIN, bar: float = RETRAIN_BAR) -> str:
    """What the verdict page says of the ranges at Done.

    *Lean* when another band edge's F-beta (from the same draws) reads at
    least *lean_gain* above the line's; *retrain* when the line's F-beta range
    cannot reach *bar*; else *ship*.  The rule is #4523's proposal.
    """
    others = [e.fbeta.point for e in est.at_edges if e.count != line_count]
    if others and max(others) - est.fbeta.point >= lean_gain:
        return VERDICT_LEAN
    if est.fbeta.hi < bar:
        return VERDICT_RETRAIN
    return VERDICT_SHIP


def oracle_verdict(
    truth: np.ndarray,
    line_count: int,
    edges: Sequence[int],
    beta: float,
    *,
    lean_gain: float = LEAN_GAIN,
    bar: float = RETRAIN_BAR,
) -> str:
    """What the truth says under :func:`read_verdict`'s rule, over the same band edges."""
    _, _, at_line = truth_at(truth, line_count, beta)
    others = [truth_at(truth, e, beta)[2] for e in edges if e != line_count]
    if others and max(others) - at_line >= lean_gain:
        return VERDICT_LEAN
    if at_line < bar:
        return VERDICT_RETRAIN
    return VERDICT_SHIP


def _nan_estimate() -> dict[str, float]:
    return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}


def _preset_columns(test: LineTest | None, truth: np.ndarray, counts: Mapping[float, int]) -> dict[str, Any]:
    """The ranges *Lean the Threshold* shows for each balance preset, beside the truth (#4540).

    Per preset (``preset_b025_*`` / ``b1`` / ``b4``): its line count, and for
    precision, recall and F-beta at its own beta the range's point, the
    truth, and whether the range held it.  -1 / NaN with no test (nothing to
    test) or no class model to draw the preset's line.
    """
    from vtscore.eval.voting_columns import beta_tag  # noqa: PLC0415

    out: dict[str, Any] = {}
    for b in BALANCE_PRESETS:
        tag = f"preset_{beta_tag(b)}"
        count = counts.get(float(b))
        out[f"{tag}_count"] = -1 if count is None else int(count)
        truths = truth_at(truth, count, b) if count is not None else (_NAN,) * 3
        edge = test.estimate_at(count, b) if (test is not None and count is not None and count > 0) else None
        for name, t in zip(("precision", "recall", "fbeta"), truths):
            e = None if edge is None else getattr(edge, name)
            out[f"{tag}_{name}_point"] = _NAN if e is None else e.point
            out[f"{tag}_{name}_true"] = t
            out[f"{tag}_{name}_held"] = -1 if e is None else int(e.holds(t))
    return out


def simulate_line_test(
    truth: np.ndarray,
    line_count: int,
    beta: float,
    *,
    posteriors: np.ndarray | None = None,
    budgets: LineBudgets = DEFAULT_BUDGETS,
    seed: int = 0,
    preset_counts: Mapping[float, int] | None = None,
) -> dict[str, Any]:
    """Run the Test autopilot to Done on a frozen ranking, every pick answered from *truth*.

    The ranking's ids are its ranks.  Returns one flat record (the
    :data:`~vtscore.eval.voting_columns.LINE_TEST_COLUMNS` payload, less the
    identity and the inputs' provenance): the picks and stops per phase, the
    ranges at Done beside the truth, the verdict beside the oracle's, and
    the band-edge coverage.  A line keeping fewer items than one round is
    *Nothing to test*: no picks, every estimate NaN.  With *preset_counts*
    (each balance preset's line count), it also records the ranges the
    verdict's *Lean the Threshold* shows for each preset - ``estimate_at``
    at that count and beta, as the app's route draws them - beside the truth
    (#4540).
    """
    truth = np.asarray(truth, dtype=bool)
    n = int(truth.size)
    test = LineTest.start(range(n), line_count, beta, posteriors=posteriors, budgets=budgets, seed=seed)
    while True:
        picks = test.draw()
        if not picks:
            break
        test.record({cid: bool(truth[cid]) for cid in picks})
    report = test.phase()
    k = test.line_count
    total = int(np.count_nonzero(truth))
    p_true, r_true, f_true = truth_at(truth, k, beta)
    best_k, best_f = best_cut(truth, beta)
    edges = [b.hi for b in line_bands(n, k)]
    gains_true = [truth_at(truth, e, beta)[2] - f_true for e in edges if e != k]
    out: dict[str, Any] = {
        "phase": report.phase,
        "line_count": k,
        "n_test": n,
        "n_test_pos": total,
        "rounds": test.rounds,
        "picks_above": report.picks_above,
        "picks_below": report.picks_below,
        "picks_total": report.picks_above + report.picks_below,
        "matches_stop": report.matches_stop or "",
        "misses_stop": report.misses_stop or "",
        "precision_true": p_true,
        "recall_true": r_true,
        "fbeta_true": f_true,
        "found_true": found_words(r_true),
        "fbeta_best_cut_true": best_f,
        "best_cut_true": best_k,
        "edge_gain_true": max(gains_true) if gains_true else float("nan"),
        "n_edges": len(edges),
        "bands_below": len(test.below),
        "bands_below_reached": sum(1 for b in test.below if test.audited(b.index)),
    }
    out.update(_preset_columns(test if report.phase != PHASE_NOTHING else None, truth, preset_counts or {}))
    if report.phase == PHASE_NOTHING:
        for name in ("precision", "recall", "fbeta", "positives_above", "positives_below"):
            out.update({f"{name}_{key}": value for key, value in _nan_estimate().items()})
        out.update(
            {
                "precision_held": -1,
                "recall_held": -1,
                "fbeta_held": -1,
                "found": "",
                "found_match": -1,
                "positives_above_true": int(np.count_nonzero(truth[:k])),
                "positives_below_true": total - int(np.count_nonzero(truth[:k])),
                "tail_positives_model": float("nan"),
                "tail_positives_true": -1,
                "tail_from_model": -1,
                "edge_gain_est": float("nan"),
                "best_edge_est": -1,
                "fbeta_at_best_edge_est_true": float("nan"),
                "edges_precision_held": -1,
                "edges_recall_held": -1,
                "edges_fbeta_held": -1,
                "verdict": VERDICT_NOTHING,
                "oracle_verdict": VERDICT_NOTHING,
                "verdict_match": 1,
            }
        )
        return out
    est = test.estimates()
    above_true = int(np.count_nonzero(truth[:k]))
    unreached = [b for b in test.below if not test.audited(b.index)]
    tail_true = sum(int(np.count_nonzero(truth[b.lo : b.hi])) for b in unreached)
    for name, e, t in (
        ("precision", est.precision, p_true),
        ("recall", est.recall, r_true),
        ("fbeta", est.fbeta, f_true),
        ("positives_above", est.positives_above, float(above_true)),
        ("positives_below", est.positives_below, float(total - above_true)),
    ):
        out[f"{name}_point"], out[f"{name}_lo"], out[f"{name}_hi"] = e.point, e.lo, e.hi
        if name in ("precision", "recall", "fbeta"):
            out[f"{name}_held"] = int(e.holds(t))
    # The edges the verdict's *Lean* exit reads: the same draws re-estimating the line at each.
    others = [e for e in est.at_edges if e.count != k]
    best_edge = max(others, key=lambda e: e.fbeta.point) if others else None
    held = {"precision": 0, "recall": 0, "fbeta": 0}
    for e in est.at_edges:
        p, r, f = truth_at(truth, e.count, beta)
        held["precision"] += int(e.precision.holds(p))
        held["recall"] += int(e.recall.holds(r))
        held["fbeta"] += int(e.fbeta.holds(f))
    verdict = read_verdict(est, k)
    oracle = oracle_verdict(truth, k, edges, beta)
    out.update(
        {
            "found": est.found,
            "found_match": int(est.found == out["found_true"]),
            "positives_above_true": above_true,
            "positives_below_true": total - above_true,
            "tail_positives_model": est.tail_positives,
            "tail_positives_true": tail_true,
            "tail_from_model": int(est.tail_from_model),
            "edge_gain_est": (best_edge.fbeta.point - est.fbeta.point) if best_edge is not None else float("nan"),
            "best_edge_est": best_edge.count if best_edge is not None else -1,
            "fbeta_at_best_edge_est_true": (
                truth_at(truth, best_edge.count, beta)[2] if best_edge is not None else float("nan")
            ),
            "edges_precision_held": held["precision"],
            "edges_recall_held": held["recall"],
            "edges_fbeta_held": held["fbeta"],
            "verdict": verdict,
            "oracle_verdict": oracle,
            "verdict_match": int(verdict == oracle),
        }
    )
    return out


def line_test_row(
    t: int,
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    beta: float,
    *,
    find_on_test: Any = None,
    fallback_threshold: float | None = None,
    budgets: LineBudgets = DEFAULT_BUDGETS,
    seed: int = 0,
) -> dict[str, Any]:
    """One session's Test on the withheld half as it stood at click *t*: inputs, budgets and outcome in one record.

    What the harness appends to its ``line_test_sink`` after the last ordinary
    click, and what the study's replay computes from a saved snapshot; the
    two agree by construction, which the study checks.
    """
    inputs = line_test_inputs(scores, labels, find_on_test, fallback_threshold, beta=beta)
    outcome = simulate_line_test(
        inputs.truth,
        inputs.line_count,
        beta,
        posteriors=inputs.posteriors,
        budgets=budgets,
        seed=seed,
        preset_counts=inputs.preset_counts,
    )
    return {
        "t": int(t),
        "beta": float(beta),
        "line_source": inputs.line_source,
        "has_model": int(inputs.posteriors is not None),
        **{key: value for key, value in budgets.as_dict().items() if key not in ("alpha", "draws")},
        "test_seed": int(seed),
        **outcome,
    }


def row_from_snapshot(
    snapshot: Mapping[str, Any],
    *,
    budgets: LineBudgets = DEFAULT_BUDGETS,
    seed: int = 0,
    keep: np.ndarray | None = None,
) -> dict[str, Any]:
    """The arm replayed on a saved withheld-half snapshot (a ``test_score_sink`` entry, or an npz's ``last``).

    *snapshot* carries ``scores``, ``labels``, ``t``, ``beta`` and ``model``
    (the labels' class model as a dict, or ``None``) and ``train_threshold``;
    the labels line's corpus side is re-fitted on the snapshot's scores, as
    Find fits it on a new corpus.  *keep* restricts the corpus to a boolean
    mask of its items (the study's thinner scenarios); the fit, the line and
    the Test then all see only the kept items.
    """
    from vtscore.training.thresholds.labels_line import ClassScoreModel, LabelsLine  # noqa: PLC0415

    scores = np.asarray(snapshot["scores"], dtype=np.float64)
    labels = np.asarray(snapshot["labels"], dtype=np.float64)
    if keep is not None:
        scores, labels = scores[keep], labels[keep]
    beta = float(snapshot["beta"])
    if not math.isfinite(beta):
        raise ValueError("the snapshot has no balance: the Test is the balance line's")
    model = snapshot.get("model")
    find_on_test = None
    if model is not None:
        # ``on_corpus`` reads only the class model; the prevalence here is a placeholder.
        find_on_test = LabelsLine(ClassScoreModel(**model), 0.5).on_corpus(scores)
    fallback = snapshot.get("train_threshold")
    return line_test_row(
        int(snapshot["t"]),
        scores,
        labels,
        beta,
        find_on_test=find_on_test,
        fallback_threshold=None if fallback is None else float(fallback),
        budgets=budgets,
        seed=seed,
    )


__all__ = [
    "LEAN_GAIN",
    "LINE_FALLBACK",
    "LINE_LABELS",
    "LINE_NONE",
    "RETRAIN_BAR",
    "VERDICTS",
    "VERDICT_LEAN",
    "VERDICT_NOTHING",
    "VERDICT_RETRAIN",
    "VERDICT_SHIP",
    "LineTestInputs",
    "best_cut",
    "line_test_inputs",
    "line_test_row",
    "oracle_verdict",
    "read_verdict",
    "row_from_snapshot",
    "simulate_line_test",
    "truth_at",
]
