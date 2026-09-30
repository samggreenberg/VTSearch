"""The precision floor's spot check (#4272): draw picks, take the votes, report the line's state.

The floor *P* is a share of what the line returns that should be right.  A
**spot check** measures it by walking the unvoted ranking in bands (#4388):
the user votes on uniform random picks from each band (the top 8, the next 8,
then 16, 32, ...), the walk goes deeper while the band-weighted share of right
answers meets *P* and shallower while it does not, and the line keeps the
deepest set that met it.  The rule lives in
:mod:`vtscore.training.thresholds.spot_check`; this blueprint is its lifecycle
for the active detector:

* ``POST /api/precision-check/start`` fixes the unvoted ranking off the
  detector's current one and deals the first band's picks;
* ``POST /api/precision-check/votes`` records the user's votes on them as
  ordinary labels (provenance ``check``), so they train the model like any
  other vote, and either deals the next band or ends the check;
* ``POST /api/precision-check/cancel`` abandons a running check, leaving the
  floor's state as it was (the votes already cast stay votes);
* ``GET /api/precision-check`` reports the running check, or the last finished
  one, beside the floor's state.

A finished check is kept on the detector and its result decides where the line
sits - the set the walk ended on - until a new check replaces it.  Later votes
retrain the model and the line follows the new ranking at the same count; the
result's range then reports ``stale``.  Starting a new check needs a changed
ranking: there is no redraw on the same list (any vote, the check's own
included, changes it).
"""

from __future__ import annotations

import logging

from flask_smorest import Blueprint, abort

from vtsearch.routes._context import require_detector_header
from vtsearch.schemas.sorting import PrecisionCheckResponseSchema, PrecisionCheckVotesRequestSchema

log = logging.getLogger(__name__)

precision_check_bp = Blueprint(
    "precision_check",
    __name__,
    description="The precision floor's spot check: draw picks, take the votes, report how close the line got.",
)


def _payload() -> dict:
    """The floor's state and the check (running, else the last finished one) for the active detector."""
    from vtscore.state.core import detector_floor_state, get_active_detector_context  # noqa: PLC0415
    from vtsearch.state import get_min_precision  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    check = det_ctx.precision_check_run or det_ctx.precision_check
    return {
        "floor": detector_floor_state(det_ctx, get_min_precision()),
        "check": check.as_dict() if check is not None else None,
    }


@precision_check_bp.route("/api/precision-check", methods=["GET"])
@precision_check_bp.response(200, PrecisionCheckResponseSchema)
def get_precision_check():
    """The active detector's spot check (running, else the last finished) and the floor's state."""
    return _payload()


@precision_check_bp.route("/api/precision-check/start", methods=["POST"])
@require_detector_header
@precision_check_bp.response(200, PrecisionCheckResponseSchema)
@precision_check_bp.alt_response(
    409, description="No ranking to draw from, nothing unvoted in it, or the same list as the last check."
)
def start_precision_check():
    """Start a spot check of the active detector's floor over its current ranking.

    The walk is over the unvoted items of the ranking the detector last
    scored, in rank order, fixed for the whole check, and starts at the bands
    that hold the floor's starting count (32 at 50% and above, 128 at 10%).
    Deals the first band's picks.  A check already running is replaced.  The
    last finished result stays in force until this check ends.
    """
    from vtscore.config import SPOT_CHECK_SEED  # noqa: PLC0415
    from vtscore.state.core import get_active_detector_context, human_voted_ids  # noqa: PLC0415
    from vtscore.training.thresholds import SpotCheck  # noqa: PLC0415
    from vtsearch.state import get_min_precision  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    ranking = det_ctx.line_ranking
    if ranking is None:
        abort(409, message="No ranking to check: run a learned sort or a Find pass first.")
    floor = get_min_precision()
    if floor is None:
        abort(409, message="The detector has no precision floor to check.")
    unvoted = tuple(int(i) for i in ranking.unvoted_ids(human_voted_ids(det_ctx)))
    if not unvoted:
        abort(409, message="Nothing is left unvoted to check.")
    last = det_ctx.precision_check
    if last is not None and last.ranking_ids == unvoted:
        abort(409, message="This list was already checked; vote on something first, or re-sort.")
    # Unseeded unless VTSEARCH_SPOT_CHECK_SEED is set, which only the
    # screenshot harness does, so a refresh frames the same picks (#4330).
    det_ctx.precision_check_run = SpotCheck.start(unvoted, floor, seed=SPOT_CHECK_SEED)
    return _payload()


@precision_check_bp.route("/api/precision-check/votes", methods=["POST"])
@require_detector_header
@precision_check_bp.arguments(PrecisionCheckVotesRequestSchema)
@precision_check_bp.response(200, PrecisionCheckResponseSchema)
@precision_check_bp.alt_response(400, description="A vote on an item that is not one of this band's picks.")
@precision_check_bp.alt_response(409, description="No check is running.")
def vote_precision_check(body: dict):
    """Record votes on the running check's picks.

    Each vote is an ordinary label on the item (provenance ``check``): it
    trains the model, persists to the labelset, and in Find mode verifies the
    item.  Once every pick of the band is labelled the walk moves on: the next
    band the set under test still owes is dealt, or the set is decided and the
    walk goes deeper (a new band is dealt), shallower, or ends.  The response
    carries the check's new state and the floor's, whose line moves to the set
    a finished check ended on.
    """
    from vtscore.state.core import get_active_detector_context, human_voted_ids  # noqa: PLC0415
    from vtscore.state.votes import record_vote_provenance  # noqa: PLC0415
    from vtscore.training.thresholds import CHECK_PROVENANCE  # noqa: PLC0415
    from vtsearch.state import get_min_precision, set_vote  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    check = det_ctx.precision_check_run
    if check is None or not check.running:
        abort(409, message="No spot check is running.")
    votes = {int(v["id"]): v["label"] == "good" for v in body["votes"]}
    stray = [cid for cid in votes if cid not in check.pending]
    if stray:
        abort(400, message=f"Not this band's picks: {stray}")

    for cid, right in votes.items():
        set_vote(cid, "good" if right else "bad", provenance=dict(CHECK_PROVENANCE))
        # A pick the machine had already labelled the same way is an idempotent
        # vote to ``set_vote`` (verified, but its provenance untouched); the
        # pick was dealt all the same, so the check's tag goes on regardless.
        record_vote_provenance(cid, dict(CHECK_PROVENANCE))
    _persist_votes()

    finished = check.record(votes)
    if finished and check.finished:
        # The set the line keeps from now on, as it stands with the check's
        # own votes cast: what a later vote or retrain is compared against.
        check.fingerprint = det_ctx.line_ranking.fingerprint(check.k, human_voted_ids(det_ctx))
        det_ctx.precision_check = check
        det_ctx.precision_check_run = None
        _move_line(det_ctx, get_min_precision())
    return _payload()


@precision_check_bp.route("/api/precision-check/cancel", methods=["POST"])
@require_detector_header
@precision_check_bp.response(200, PrecisionCheckResponseSchema)
def cancel_precision_check():
    """Abandon the running check.  Its votes so far stay ordinary votes; the floor's state is as it was."""
    from vtscore.state.core import get_active_detector_context  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    check = det_ctx.precision_check_run
    if check is not None:
        check.cancel()
        det_ctx.precision_check_run = None
    return _payload()


def _persist_votes() -> None:
    """Persist the check's votes exactly as ``/api/medias/<id>/vote`` persists one."""
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector  # noqa: PLC0415
    from vtscore.labels.sync import sync_to_labelset_source  # noqa: PLC0415

    try:
        sync_labels_to_loaded_detector()
    except Exception as exc:
        log.exception("precision check: detector label sync failed")
        abort(500, message=f"Failed to persist votes to detector store: {exc}")
    try:
        sync_to_labelset_source()
    except Exception:
        log.exception("precision check: labelset source scheduling failed")


def _move_line(det_ctx, floor: float | None) -> None:
    """Move the line to the set the finished check ended on, and in Find mode re-split the unverified items."""
    from vtscore.state.core import recut_detector_threshold  # noqa: PLC0415
    from vtscore.state.votes import rethreshold_unverified_find_items  # noqa: PLC0415

    threshold = recut_detector_threshold(det_ctx, min_precision=floor)
    if threshold is not None:
        det_ctx.threshold = threshold
        rethreshold_unverified_find_items()
