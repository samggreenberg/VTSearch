"""Schemas for the sorting / voting / balance APIs.

Covers the routes in ``vtsearch/routes/sorting.py``:

* ``POST /api/sort``                          -> :class:`SortRequestSchema` ->
                                                :class:`SortResponseSchema`
* ``POST /api/learned-sort``                  -> :class:`LearnedSortRequestSchema` ->
                                                :class:`LearnedSortResponseSchema`
* ``GET  /api/learned-sort/result``           -> :class:`LearnedSortResultQuerySchema` ->
                                                :class:`LearnedSortResponseSchema`
* ``GET  /api/votes``                         -> :class:`VotesResponseSchema`
* ``POST /api/votes/clear``                   -> :class:`OkResponseSchema`
* ``GET  /api/textsort-suggestions``          -> :class:`TextsortSuggestionsResponseSchema`
* ``POST /api/textsort-suggestions``          -> :class:`TextsortSuggestionRequestSchema` ->
                                                :class:`OkResponseSchema`
* ``GET  /api/balance``                       -> :class:`BalanceResponseSchema`
* ``POST /api/balance``                       -> :class:`BalanceRequestSchema` ->
                                                :class:`BalanceResponseSchema`

and the balance's spot check in ``vtsearch/routes/precision_check.py``
(#4272, #4413):

* ``GET  /api/precision-check``                -> :class:`PrecisionCheckResponseSchema`
* ``POST /api/precision-check/start``          -> :class:`PrecisionCheckResponseSchema`
* ``POST /api/precision-check/votes``          -> :class:`PrecisionCheckVotesRequestSchema` ->
                                                :class:`PrecisionCheckResponseSchema`
* ``POST /api/precision-check/cancel``         -> :class:`PrecisionCheckResponseSchema`
* ``POST /api/example-sort``                  (multipart upload) ->
                                                :class:`SortResponseSchema`
* ``POST /api/label-file-sort``               (multipart upload) ->
                                                :class:`LabelFileSortResponseSchema`
* ``GET|POST /api/coverage-atlas/next``       -> :class:`CoverageAtlasNextRequestSchema` ->
                                                :class:`CoverageAtlasNextResponseSchema`

The sort result items use ``fields.Dict()`` rather than nested schemas
because the inner shape varies: text/example sort produces
``{id, similarity[, best_region]}`` while learned sort produces
``{id, score}``. Keeping both as plain dicts lets the same response
schema serialise either path without coercing keys.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


class OkResponseSchema(Schema):
    """``{"ok": true}`` response used by clear / seed-style endpoints."""

    ok = fields.Boolean(required=True)


from vtscore.training.thresholds.spot_check import (
    BALANCE_CHECKED,
    BALANCE_STATES,
    CHECK_CANCELLED,
    CHECK_RUNNING,
    CHECK_SHAPES,
)

#: The states a spot check can be in: its rounds still being voted on, ended
#: on the balance's checked state, or abandoned.
PRECISION_CHECK_STATES = (CHECK_RUNNING, BALANCE_CHECKED, CHECK_CANCELLED)


class LikelyRangeSchema(Schema):
    """How much of the set the line keeps is likely right, from a spot check's picks (#4272).

    A Clopper-Pearson interval from the check's labels inside the set, each
    tail at the level the check's rounds were tested at; exact when the labels
    cover the set.
    """

    lo = fields.Float(required=True)
    hi = fields.Float(required=True)
    # How many of the set's items the check labelled, and how many were right.
    labelled = fields.Integer(required=True)
    right = fields.Integer(required=True)
    # True once the ranking under the result has moved since the check (later
    # votes retrained the model): the range describes the list as it was.
    stale = fields.Boolean(required=False)


class CheckScheduleSchema(Schema):
    """What a spot check at this balance costs: its starting candidate, rounds and picks a round."""

    candidate = fields.Integer(required=True)
    rounds = fields.Integer(required=True)
    picks = fields.Integer(required=True)


class BalanceStateSchema(Schema):
    """What the balance says about the line a response carries (#4413).

    Built by :func:`vtscore.state.core.detector_balance_state`.  ``unchecked``:
    no balance walk has run at this beta, and the line keeps the mixture's
    F-beta argmax under the balance's cap; ``checked``: a walk has, and under
    the ``trim`` shape (beta above 1) the line keeps its end, under ``advisory``
    (beta 1 and below) the walk's ranges inform the line and the count stays
    the unchecked rule's (#4427).
    """

    # The detector's balance: F-beta's beta.
    beta = fields.Float(required=True)
    status = fields.String(required=True, validate=validate.OneOf(BALANCE_STATES))
    # How many unvoted items the line keeps.
    count = fields.Integer(required=True)
    # The walk's likely ranges for the kept set's precision and recall, and
    # its F-beta estimate; ``null`` while unchecked.
    precision = fields.Nested(LikelyRangeSchema, required=True, allow_none=True)
    recall = fields.Nested(LikelyRangeSchema, required=True, allow_none=True)
    fbeta = fields.Float(required=True, allow_none=True)
    # The balance's cap and what a walk from it costs.
    schedule = fields.Nested(CheckScheduleSchema, required=True)
    # How a check treats the line at this beta (#4427): ``advisory`` or ``trim``.
    shape = fields.String(required=True, validate=validate.OneOf(CHECK_SHAPES))
    # The set the last check audited (the walk's end); under ``advisory`` not the
    # set the line keeps.  ``null`` while unchecked.
    audited = fields.Integer(required=True, allow_none=True)


# ---------------------------------------------------------------------------
# /api/sort, /api/example-sort
# ---------------------------------------------------------------------------


class SortRequestSchema(Schema):
    """Body for ``POST /api/sort``."""

    text = fields.String(required=True)


#: Windowing metadata carried alongside a full ``results`` list so a client can
#: page deeper via ``GET /api/sort/page`` without holding the whole ranking.
#: Additive and optional — every sort route still returns the full ``results``
#: today (frontend windowing lands separately; see scalability.md S3/S17/S19).
_WINDOW_META_FIELDS = {
    # Opaque handle for /api/sort/page; also the sort-generation token.
    "sort_token": fields.String(required=False),
    # Full ranking length (>= len(results): ``results`` may be a head window).
    "total": fields.Integer(required=False),
    # Rows scoring at or above ``threshold`` across the whole ranking.
    "above_threshold": fields.Integer(required=False),
    # True when ``results`` is a head window and more rows follow (page them via
    # /api/sort/page). False when the full ranking was transmitted.
    "has_more_below": fields.Boolean(required=False),
    # The rank position Autopilot's Hard / New picks sample around, which since
    # #2876 sits *above* the reporting ``threshold`` - the two cuts do different
    # jobs (see vtscore.state.core.detector_acquisition_threshold).  ``None`` on
    # sorts with no detector behind them; the client falls back to ``threshold``.
    "acq_threshold": fields.Float(required=False, allow_none=True),
}


class SortResponseSchema(Schema):
    """Response for ``POST /api/sort`` and ``POST /api/example-sort``."""

    results = fields.List(fields.Dict(), required=True)
    threshold = fields.Float(required=True)
    sort_token = _WINDOW_META_FIELDS["sort_token"]
    total = _WINDOW_META_FIELDS["total"]
    above_threshold = _WINDOW_META_FIELDS["above_threshold"]
    has_more_below = _WINDOW_META_FIELDS["has_more_below"]
    acq_threshold = _WINDOW_META_FIELDS["acq_threshold"]


class SortPageQuerySchema(Schema):
    """Query for ``GET /api/sort/page``."""

    token = fields.String(required=True)
    offset = fields.Integer(load_default=0, validate=validate.Range(min=0))
    limit = fields.Integer(load_default=200, validate=validate.Range(min=1, max=2000))

    class Meta:
        # Tolerate request-context params (dataset_id / detector_id) smuggled in
        # as query args by browser-native requests, matching the learned-sort
        # result query schema.
        unknown = "exclude"


class SortPageResponseSchema(Schema):
    """Response for ``GET /api/sort/page`` — one window of a cached ranking."""

    results = fields.List(fields.Dict(), required=True)
    offset = fields.Integer(required=True)
    limit = fields.Integer(required=True)
    total = fields.Integer(required=True)
    threshold = fields.Float(required=True, allow_none=True)
    has_more = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# /api/learned-sort and /api/learned-sort/result
# ---------------------------------------------------------------------------


class LearnedSortRequestSchema(Schema):
    """Body for ``POST /api/learned-sort``."""

    wait = fields.Boolean(
        load_default=False,
        metadata={
            "description": (
                "If true, block until the background job completes and return the result inline. "
                "Used by tests; production clients poll ``/api/learned-sort/result`` instead."
            )
        },
    )


class LearnedSortResultQuerySchema(Schema):
    """Query for ``GET /api/learned-sort/result``."""

    job_id = fields.String(required=True)

    class Meta:
        # Tolerate request-context headers smuggled in as query params on
        # some clients (e.g. dataset_id / detector_id).
        unknown = "exclude"


class LearnedSortResponseSchema(Schema):
    """Combined response for the learned-sort start + poll endpoints.

    The response varies by status: ``running`` carries ``current``/``total``,
    ``done`` carries ``results``/``threshold``, ``error`` carries an
    ``error`` message. Declared permissively so a single schema covers
    every state.
    """

    job_id = fields.String(required=True)
    status = fields.String(
        required=True,
        validate=validate.OneOf(["running", "done", "error", "cancelled", "missing"]),
    )
    results = fields.List(fields.Dict())
    threshold = fields.Float()
    current = fields.Integer()
    total = fields.Integer()
    error = fields.String()
    # Windowing metadata on the ``done`` payload (see scalability.md S3/S17/S19).
    sort_token = fields.String(required=False)
    above_threshold = fields.Integer(required=False)
    has_more_below = fields.Boolean(required=False)
    # The acquisition cut Autopilot samples around; this is the only sort with a
    # detector behind it, so the only one that carries one.
    acq_threshold = _WINDOW_META_FIELDS["acq_threshold"]
    # What the balance says about ``threshold`` (#4247, #4413), on ``done``.
    balance = fields.Nested(BalanceStateSchema, required=False, allow_none=True)


class LearnedSortCancelResponseSchema(Schema):
    """Response for ``POST /api/learned-sort/cancel/<job_id>``."""

    ok = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# /api/votes (+ clear)
# ---------------------------------------------------------------------------


class VotesResponseSchema(Schema):
    """Response for ``GET /api/votes``."""

    good = fields.List(fields.Integer(), required=True)
    bad = fields.List(fields.Integer(), required=True)
    # Find-mode: ids the human has explicitly verified this session.  Lets the
    # frontend split verified (right panel) from unverified (left work queue).
    # Empty outside Find mode.
    verified = fields.List(fields.Integer(), required=True)
    click_times = fields.Dict(keys=fields.String(), values=fields.Float(), required=True)
    learned_scores = fields.Dict(keys=fields.String(), values=fields.Float(), required=True)
    labelset_good_count = fields.Integer(required=True)
    labelset_bad_count = fields.Integer(required=True)
    # Per-media normalised region boxes ([x0, y0, x1, y1]) for good votes cast
    # by drawing a box on an image.  Keyed by media id (string).  Lets the Good
    # pile request a cropped thumbnail of just the voted region.  Only good
    # votes that carry a box appear; empty otherwise.
    good_region_boxes = fields.Dict(
        keys=fields.String(),
        values=fields.List(fields.Float()),
        required=True,
    )


# ---------------------------------------------------------------------------
# /api/textsort-suggestions
# ---------------------------------------------------------------------------


class TextsortSuggestionsResponseSchema(Schema):
    """Response for ``GET /api/textsort-suggestions``."""

    suggestions = fields.List(fields.String(), required=True)


class TextsortSuggestionRequestSchema(Schema):
    """Body for ``POST /api/textsort-suggestions``."""

    text = fields.String(required=True)


# ---------------------------------------------------------------------------
# /api/precision-check
# ---------------------------------------------------------------------------


class CheckBandSchema(Schema):
    """The band a walk is auditing: its index from the top and its rank positions (1-based, inclusive)."""

    index = fields.Integer(required=True)
    lo = fields.Integer(required=True)
    hi = fields.Integer(required=True)


class PrecisionCheckStateSchema(Schema):
    """A balance walk of the active detector's line: the band being audited, the set under test, labels and ranges (#4272, #4388, #4413)."""

    status = fields.String(required=True, validate=validate.OneOf(PRECISION_CHECK_STATES))
    # The balance the walk runs at, its F-beta estimate for the set under test
    # (``null`` while a band of it is still unaudited) and the set's likely
    # recall range (``null`` before any label).
    beta = fields.Float(required=True)
    fbeta = fields.Float(required=True, allow_none=True)
    recall = fields.Nested(LikelyRangeSchema, required=True, allow_none=True)
    # The round being voted on (1-based; one round per band audited) and how
    # many bands the ranking has in all.
    round = fields.Integer(required=True)
    rounds = fields.Integer(required=True)
    # How many fresh picks each band is audited with.
    picks_per_round = fields.Integer(required=True)
    # The set under test's size (the top ``bands`` bands; the kept set's once
    # the check has finished), and the size the walk started from.
    candidate = fields.Integer(required=True)
    start_candidate = fields.Integer(required=True)
    # How many bands from the top the set under test spans, the band whose
    # picks are pending (``null`` between bands), and which way the walk last
    # moved: ``start``, ``deeper`` or ``shallower``.
    bands = fields.Integer(required=True)
    band = fields.Nested(CheckBandSchema, required=True, allow_none=True)
    direction = fields.String(required=True, validate=validate.OneOf(["start", "deeper", "shallower"]))
    # The band-weighted share of the set under test that its picks say is
    # right; ``null`` while a band of it is still unaudited.
    estimate = fields.Float(required=True, allow_none=True)
    # The picks awaiting the user's vote this round, in draw order (random).
    # They are a check, not the ranking: a client must not show them as the
    # top of the sort.
    picks = fields.List(fields.Integer(), required=True)
    # Labels inside the set under test so far, and how many were right.
    labelled = fields.Integer(required=True)
    right = fields.Integer(required=True)
    # The set's likely range from those labels; ``null`` before any.
    range = fields.Nested(LikelyRangeSchema, required=True, allow_none=True)


class PrecisionCheckResponseSchema(Schema):
    """Response for every ``/api/precision-check`` verb: the balance's state and the check, if any."""

    # The balance's state for the line (#4413), exactly as every other carrier
    # reports it.
    balance = fields.Nested(BalanceStateSchema, required=True)
    # The running check, or the last finished one; ``null`` when there is
    # neither.
    check = fields.Nested(PrecisionCheckStateSchema, required=True, allow_none=True)


class PrecisionCheckVoteSchema(Schema):
    """One vote on a pick: the media id and whether it is Good (right) or Bad."""

    id = fields.Integer(required=True)
    label = fields.String(required=True, validate=validate.OneOf(["good", "bad"]))


class PrecisionCheckVotesRequestSchema(Schema):
    """Body for ``POST /api/precision-check/votes``."""

    # Votes on this round's picks.  A partial round is accepted and waits for
    # the rest; an id that is not one of the round's picks is a 400.
    votes = fields.List(fields.Nested(PrecisionCheckVoteSchema), required=True)


class BalanceResponseSchema(BalanceStateSchema):
    """Response for ``GET|POST /api/balance``: the balance's state, plus the line it draws (#4413)."""

    # The line the detector draws (``null`` when no detector has a threshold),
    # and how many items of its last ranking sit at or above it.
    threshold = fields.Float(required=True, allow_none=True)
    n_returned = fields.Integer(required=True, allow_none=True)


class BalanceRequestSchema(Schema):
    """Body for ``POST /api/balance``."""

    # F-beta's beta, clamped to ``[0.25, 4]`` (presets 0.5 / 1 / 2).  ``null``,
    # a boolean, a non-numeric string and ``NaN`` / ``Infinity`` are refused.
    beta = fields.Float(required=True)


# ---------------------------------------------------------------------------
# /api/label-file-sort (multipart)
# ---------------------------------------------------------------------------


class LabelFileSortResponseSchema(Schema):
    """Response for ``POST /api/label-file-sort``."""

    results = fields.List(fields.Dict(), required=True)
    threshold = fields.Float(required=True)
    loaded = fields.Integer(required=True)
    skipped = fields.Integer(required=True)
    sort_token = _WINDOW_META_FIELDS["sort_token"]
    total = _WINDOW_META_FIELDS["total"]
    above_threshold = _WINDOW_META_FIELDS["above_threshold"]
    has_more_below = _WINDOW_META_FIELDS["has_more_below"]
    acq_threshold = _WINDOW_META_FIELDS["acq_threshold"]


# ---------------------------------------------------------------------------
# /api/coverage-atlas/next
# ---------------------------------------------------------------------------


class CoverageAtlasNextRequestSchema(Schema):
    """Body for ``POST /api/coverage-atlas/next``.

    Both fields are optional. ``scores`` keys are media ids encoded as
    strings (JSON object keys can't be ints); the handler converts them
    back to ints. Declared as a permissive dict so the handler keeps
    ownership of the int-key coercion + error path.
    """

    scores = fields.Dict(keys=fields.String(), values=fields.Float(), load_default=None)
    threshold = fields.Float(load_default=None, allow_none=True)


class CoverageAtlasNextResponseSchema(Schema):
    """Response for ``GET|POST /api/coverage-atlas/next``."""

    id = fields.Integer(allow_none=True, required=True)
    coverage_level = fields.Integer(required=True)
    exhausted = fields.Boolean(required=True)


__all__ = [
    "CoverageAtlasNextRequestSchema",
    "CoverageAtlasNextResponseSchema",
    "LabelFileSortResponseSchema",
    "LearnedSortCancelResponseSchema",
    "LearnedSortRequestSchema",
    "LearnedSortResponseSchema",
    "LearnedSortResultQuerySchema",
    "OkResponseSchema",
    "SortPageQuerySchema",
    "SortPageResponseSchema",
    "SortRequestSchema",
    "SortResponseSchema",
    "TextsortSuggestionRequestSchema",
    "TextsortSuggestionsResponseSchema",
    "VotesResponseSchema",
]
