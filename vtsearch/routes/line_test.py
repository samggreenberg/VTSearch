"""Test mode's test of the line (#4524): draw picks, take the votes, report what the line would ship.

Find scores a corpus the detector never trained on and draws the balance's
line on it.  A **test** measures that line: uniform picks from rank bands on
both sides of it, the user votes each one, and the picks say, as likely
ranges, what share of what the line ships would be right and what share of
the real matches it would ship (``vtscore/docs/packages/training.md``).  The rule lives
in :mod:`vtscore.training.thresholds.line_test`; this blueprint is its
lifecycle for the active detector's Find session:

* ``POST /api/line-test/start`` freezes the ranking off the Find pass's scores
  and the line at its threshold, and deals the first round;
* ``POST /api/line-test/votes`` records votes on the round's picks as session
  votes (provenance ``test``, verified in Find mode, never a training label:
  ``find_mode`` keeps them out of the labelset), and deals the next round when
  the round is whole;
* ``POST /api/line-test/unvote`` takes one vote of the current round back (the
  ↓ key);
* ``POST /api/line-test/cancel`` abandons a running test (the votes cast stay
  session votes);
* ``POST /api/line-test/forget`` forgets the verdict the detector keeps for
  this dataset (#4526): a reset for the screenshot harness, not a user action;
* ``GET /api/line-test`` reports the running test, or the last finished one.

Unlike the spot check (``vtsearch/routes/precision_check.py``) a test never
moves the line and never trains: the ranking is frozen for the whole test,
which is what makes the band design valid.  The result goes ``stale`` once
**Add Corrections** folds the session's votes into the detector (the detector
has now seen the test set), and ``moved`` once the line no longer keeps the
set the test measured.  A finished test stays on the context until the next
Find pass, vote clear or dataset switch.

The verdict outlives the session (#4526): the vote that finishes a test keeps
its verdict on the detector's JSON, one per tested dataset
(:mod:`vtscore.detectors.line_verdicts`), where the detector's Stats and the
Dashboard's AutoRun tab read it; and a test started over a ranking and line a
kept verdict was drawn from resumes from its picks rather than dealing new
ones.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from flask_smorest import Blueprint, abort

from vtsearch.routes._context import require_detector_header
from vtsearch.schemas.line_test import LineTestResponseSchema, LineTestUnvoteRequestSchema, LineTestVotesRequestSchema

log = logging.getLogger(__name__)

line_test_bp = Blueprint(
    "line_test",
    __name__,
    description="Test mode's test of the line: draw picks, take the votes, report what the line would ship.",
)


def _ranking(det_ctx) -> tuple[list[int], list[float]]:
    """The Find pass's frozen scores in rank order: ``(ids, scores)``, best first."""
    from vtscore.state.votes import _find_ranked_scores  # noqa: PLC0415

    ranked = _find_ranked_scores(det_ctx)
    return [int(cid) for cid, _ in ranked], [float(s) for _, s in ranked]


def _line_count(det_ctx, scores: list[float]) -> int:
    """How many of the ranked items the line keeps: those scoring at or above the detector's threshold."""
    return sum(1 for s in scores if s >= det_ctx.threshold)


def _preset_count(det_ctx, scores: list[float], beta: float) -> int | None:
    """How many items the labels' line would keep at *beta*, or ``None`` with no class model to draw one."""
    line = det_ctx.labels_line
    if line is None:
        return None
    threshold = float(line.threshold(beta))
    return sum(1 for s in scores if s >= threshold)


def _tested_dataset(det_ctx) -> tuple[str, str]:
    """The id and name of the dataset the Find pass scored: the one a test's verdict is kept for.

    The dataset the detector's votes are keyed in, which is the one the frozen
    scores' ids belong to; the request's dataset when that is not stamped yet.
    """
    from vtscore.datasets.registry import get_dataset  # noqa: PLC0415
    from vtscore.state.core import get_active_context, is_request_missing_dataset_context  # noqa: PLC0415

    dataset_id = det_ctx.votes_dataset_id
    if not dataset_id:
        ds_ctx = get_active_context()
        dataset_id = "" if is_request_missing_dataset_context(ds_ctx) else ds_ctx.dataset_id
    entry = get_dataset(dataset_id) if dataset_id else None
    return dataset_id, (entry or {}).get("name", "") or ""


def _keep(det_ctx, test) -> None:
    """Keep a finished test's verdict on the detector.  A failed write is logged: the vote that finished it stands."""
    from vtscore.detectors.line_verdicts import keep_verdict  # noqa: PLC0415
    from vtscore.detectors.store import DetectorWriteError  # noqa: PLC0415

    dataset_id, dataset_name = _tested_dataset(det_ctx)
    try:
        keep_verdict(det_ctx, test, dataset_id=dataset_id, dataset_name=dataset_name)
    except (OSError, DetectorWriteError):
        log.warning("could not keep the test verdict of detector %s", det_ctx.detector_id, exc_info=True)


def _payload() -> dict[str, Any]:
    """The balance's state, the line, the test (running, else the last finished one), and the presets."""
    from vtscore.state.core import detector_balance_state, get_active_detector_context  # noqa: PLC0415
    from vtscore.training.thresholds import BALANCE_PRESETS  # noqa: PLC0415
    from vtsearch.state import get_beta  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    ids, scores = _ranking(det_ctx) if det_ctx.find_mode else ([], [])
    line_count = _line_count(det_ctx, scores)
    test = det_ctx.line_test
    moved = test is not None and (test.line_count != line_count or list(test.ranking_ids) != ids)
    presets: list[dict[str, Any]] = []
    if test is not None and not test.nothing_to_test:
        for beta in BALANCE_PRESETS:
            count = _preset_count(det_ctx, scores, beta)
            if count is None:
                presets = []
                break
            edge = test.estimate_at(count, beta)
            presets.append({"beta": beta, **edge.as_dict()})
    return {
        "balance": detector_balance_state(det_ctx, get_beta()),
        "threshold": det_ctx.threshold if det_ctx.find_mode else None,
        "line_count": line_count,
        "test": test.as_dict() if test is not None else None,
        "stale": bool(test is not None and det_ctx.find_eval_stale),
        "moved": bool(moved),
        "presets": presets,
    }


@line_test_bp.route("/api/line-test", methods=["GET"])
@line_test_bp.response(200, LineTestResponseSchema)
def get_line_test():
    """The active detector's test of the line (running, else the last finished) and the balance's state."""
    return _payload()


@line_test_bp.route("/api/line-test/start", methods=["POST"])
@require_detector_header
@line_test_bp.response(200, LineTestResponseSchema)
@line_test_bp.alt_response(409, description="No Find pass to test: score the dataset first.")
def start_line_test():
    """Start a test of the line the Find pass drew.

    The ranking is the pass's frozen scores, best first, and the line keeps
    the items at or above the detector's threshold; both are fixed for the
    whole test.  The labels line's chance per item is the auxiliary below the
    line, when the detector has one.  Deals the first round.  A test already
    running is replaced; a finished test of the same line is returned as it
    is, since its picks already say what a new one would.  So is a verdict
    the detector keeps for this dataset (#4526), when it is not stale and its
    picks were drawn from this very ranking and line: the test resumes from
    them (``kept_at`` says when they were taken), and they are session votes
    again, in the Review tab's piles, as they were when they were cast.
    """
    from vtscore.config import SPOT_CHECK_SEED  # noqa: PLC0415
    from vtscore.detectors.line_verdicts import kept_verdict  # noqa: PLC0415
    from vtscore.state.core import get_active_detector_context  # noqa: PLC0415
    from vtscore.state.votes import record_vote_provenance  # noqa: PLC0415
    from vtscore.training.thresholds import TEST_PROVENANCE, LineTest  # noqa: PLC0415
    from vtsearch.state import get_beta, set_vote  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    if not det_ctx.find_mode or not det_ctx.find_scores:
        abort(409, message="Nothing to test yet: score the dataset first.")
    ids, scores = _ranking(det_ctx)
    line_count = _line_count(det_ctx, scores)
    beta = get_beta()
    last = det_ctx.line_test
    if (
        last is not None
        and last.phase().done
        and list(last.ranking_ids) == ids
        and last.line_count == line_count
        and last.beta == beta
    ):
        return _payload()
    posteriors = None
    if det_ctx.labels_line is not None:
        line = det_ctx.labels_line
        posteriors = np.asarray(line.model.posterior(np.asarray(scores, dtype=np.float64), line.prevalence))
    kept = kept_verdict(det_ctx, _tested_dataset(det_ctx)[0])
    labels = kept[0].kept_labels(ids, line_count) if kept is not None and not kept[1] else None
    # Unseeded unless VTSEARCH_SPOT_CHECK_SEED is set, which only the
    # screenshot harness does, so a refresh frames the same picks.
    test = LineTest.start(ids, line_count, beta, posteriors=posteriors, seed=SPOT_CHECK_SEED, labels=labels)
    if labels is not None and kept is not None:
        test.kept_at = kept[0].tested_at
        for cid, right in labels.items():
            # A pick the user has already verified this session keeps the
            # session's vote; the test keeps its own label either way.
            if cid in det_ctx.verified_ids:
                continue
            set_vote(cid, "good" if right else "bad", provenance=dict(TEST_PROVENANCE))
            record_vote_provenance(cid, dict(TEST_PROVENANCE))
    test.draw()
    det_ctx.line_test = test
    return _payload()


@line_test_bp.route("/api/line-test/votes", methods=["POST"])
@require_detector_header
@line_test_bp.arguments(LineTestVotesRequestSchema)
@line_test_bp.response(200, LineTestResponseSchema)
@line_test_bp.alt_response(400, description="A vote on an item that is not one of this round's picks.")
@line_test_bp.alt_response(409, description="No test is running.")
def vote_line_test(body: dict):
    """Record votes on the running test's picks.

    Each vote is a session vote on the item (provenance ``test``): verified in
    Find mode, so it lands in the Review tab's piles, and kept out of the
    labelset like every Find vote.  The test records it against the pick's
    band and the ranges move.  A partial round waits for the rest; once the
    round is whole the next round is dealt, or the test is done.  The vote
    that finishes it keeps its verdict on the detector (#4526).
    """
    from vtscore.state.core import get_active_detector_context  # noqa: PLC0415
    from vtscore.state.votes import record_vote_provenance  # noqa: PLC0415
    from vtscore.training.thresholds import PHASE_DONE, TEST_PROVENANCE  # noqa: PLC0415
    from vtsearch.state import set_vote  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    test = det_ctx.line_test
    if test is None or test.phase().done or not test.pending:
        abort(409, message="No test is running.")
    votes = {int(v["id"]): v["label"] == "good" for v in body["votes"]}
    stray = [cid for cid in votes if cid not in test.pending]
    if stray:
        abort(400, message=f"Not this round's picks: {stray}")
    for cid, right in votes.items():
        set_vote(cid, "good" if right else "bad", provenance=dict(TEST_PROVENANCE))
        # An idempotent re-vote keeps its original provenance; the pick was
        # dealt all the same, so the test's tag goes on regardless.
        record_vote_provenance(cid, dict(TEST_PROVENANCE))
    if test.record(votes):
        test.draw()
    # The stop rule reads the ranges as they stand, so a test can finish part
    # way through a round; either way this is the vote that finished it.
    if test.phase().phase == PHASE_DONE:
        _keep(det_ctx, test)
    return _payload()


@line_test_bp.route("/api/line-test/unvote", methods=["POST"])
@require_detector_header
@line_test_bp.arguments(LineTestUnvoteRequestSchema)
@line_test_bp.response(200, LineTestResponseSchema)
@line_test_bp.alt_response(400, description="Not a vote of the current round.")
@line_test_bp.alt_response(409, description="No test is running.")
def unvote_line_test(body: dict):
    """Take back one vote of the current round (the ↓ key): the pick goes back to pending, and its session vote is lifted."""
    from vtscore.state.core import get_active_detector_context  # noqa: PLC0415
    from vtsearch.state import set_vote  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    test = det_ctx.line_test
    if test is None or test.phase().done or not test.pending:
        abort(409, message="No test is running.")
    cid = int(body["id"])
    try:
        test.unrecord(cid)
    except ValueError as exc:
        abort(400, message=str(exc))
    set_vote(cid, "none")
    return _payload()


@line_test_bp.route("/api/line-test/cancel", methods=["POST"])
@require_detector_header
@line_test_bp.response(200, LineTestResponseSchema)
def cancel_line_test():
    """Abandon the running test.  Its votes so far stay session votes; a finished test is left as it is."""
    from vtscore.state.core import get_active_detector_context  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    test = det_ctx.line_test
    if test is not None and not test.phase().done:
        det_ctx.line_test = None
    return _payload()


@line_test_bp.route("/api/line-test/forget", methods=["POST"])
@require_detector_header
@line_test_bp.response(200, LineTestResponseSchema)
@line_test_bp.alt_response(500, description="The detector file could not be rewritten.")
def forget_line_test():
    """Forget the verdict the detector keeps for this dataset (#4526), so the next start deals a fresh test.

    A reset, not a user action: the app never calls it.  A user has no need
    to, because a retrain already marks the verdict stale, a changed ranking
    already deals a fresh test (whose verdict replaces the kept one at Done),
    and on an unchanged ranking a second test of the same budget measures
    nothing the first did not.  The screenshot harness calls it between shots,
    so a shot that reaches Done does not leave the next one resuming on the
    same pair.  A finished test in memory goes with it; its picks stay session
    votes, as a cancelled test's do.  A running test is left as it is.  200
    whether or not a verdict was kept; 500 only when the detector file cannot
    be rewritten.
    """
    from vtscore.detectors.line_verdicts import forget_verdict  # noqa: PLC0415
    from vtscore.detectors.store import DetectorWriteError  # noqa: PLC0415
    from vtscore.state.core import get_active_detector_context  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    try:
        forget_verdict(det_ctx, _tested_dataset(det_ctx)[0])
    except (OSError, DetectorWriteError):
        log.warning("could not forget the test verdict of detector %s", det_ctx.detector_id, exc_info=True)
        abort(500, message="Could not forget the kept test.")
    test = det_ctx.line_test
    if test is not None and test.phase().done:
        det_ctx.line_test = None
    return _payload()
