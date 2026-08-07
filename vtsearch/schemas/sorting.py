"""Schemas for the sorting / voting / inclusion APIs.

Covers the routes in ``vtsearch/routes/sorting.py``:

* ``POST /api/sort``                          -> :class:`SortRequestSchema` ->
                                                :class:`SortResponseSchema`
* ``POST /api/learned-sort``                  -> :class:`LearnedSortRequestSchema` ->
                                                :class:`LearnedSortResponseSchema`
* ``GET  /api/learned-sort/result``           -> :class:`LearnedSortResultQuerySchema` ->
                                                :class:`LearnedSortResponseSchema`
* ``GET  /api/votes``                         -> :class:`VotesResponseSchema`
* ``POST /api/votes/clear``                   -> :class:`OkResponseSchema`
* ``POST /api/votes/seed-from-examples``      -> :class:`SeedFromExamplesRequestSchema` ->
                                                :class:`SeedFromExamplesResponseSchema`
* ``GET  /api/textsort-suggestions``          -> :class:`TextsortSuggestionsResponseSchema`
* ``POST /api/textsort-suggestions``          -> :class:`TextsortSuggestionRequestSchema` ->
                                                :class:`OkResponseSchema`
* ``GET  /api/inclusion``                     -> :class:`InclusionResponseSchema`
* ``POST /api/inclusion``                     -> :class:`InclusionRequestSchema` ->
                                                :class:`InclusionResponseSchema`
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

from marshmallow import Schema, ValidationError, fields, validate


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


class OkResponseSchema(Schema):
    """``{"ok": true}`` response used by clear / seed-style endpoints."""

    ok = fields.Boolean(required=True)


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


class LearnedSortCancelResponseSchema(Schema):
    """Response for ``POST /api/learned-sort/cancel/<job_id>``."""

    ok = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# /api/votes (+ clear, seed-from-examples)
# ---------------------------------------------------------------------------


class VotesResponseSchema(Schema):
    """Response for ``GET /api/votes``."""

    good = fields.List(fields.Integer(), required=True)
    bad = fields.List(fields.Integer(), required=True)
    # Find-mode: ids the human has explicitly verified this session.  Lets the
    # frontend split verified (right panel) from unverified (left work queue).
    # Empty outside Find mode.  See docs/plans/find-verification-workflow.md.
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


class SeedFromExamplesRequestSchema(Schema):
    """Body for ``POST /api/votes/seed-from-examples``."""

    examples = fields.List(fields.Dict(), required=True)


class SeedFromExamplesResponseSchema(Schema):
    """Response for ``POST /api/votes/seed-from-examples``."""

    seeded = fields.Integer(required=True)
    skipped = fields.Integer(required=True)


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
# /api/inclusion
# ---------------------------------------------------------------------------


class InclusionResponseSchema(Schema):
    """Response for ``GET|POST /api/inclusion``."""

    inclusion = fields.Integer(required=True)
    # The cutoff that this inclusion resolves to over the active detector's
    # cached fold orderings.  Returned so the Find slider can move the
    # green/red line over the frozen scores without re-scoring.  ``None`` when
    # no detector context has computed a threshold yet.
    threshold = fields.Float(required=False, allow_none=True)


def _validate_numeric(value):
    """Reject booleans and non-numeric values for the inclusion field.

    Booleans are a subclass of ``int`` in Python; without this guard
    ``true`` / ``false`` would sneak through as ``1`` / ``0``.
    Declared as a plain validator rather than ``fields.Integer(strict=True)``
    so that ``3.7`` continues to round to ``3`` in the handler
    (preserving the pre-migration coercion behavior).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("Must be a number.")


class InclusionRequestSchema(Schema):
    """Body for ``POST /api/inclusion``."""

    inclusion = fields.Raw(required=True, validate=_validate_numeric)


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
    "InclusionRequestSchema",
    "InclusionResponseSchema",
    "LabelFileSortResponseSchema",
    "LearnedSortCancelResponseSchema",
    "LearnedSortRequestSchema",
    "LearnedSortResponseSchema",
    "LearnedSortResultQuerySchema",
    "OkResponseSchema",
    "SeedFromExamplesRequestSchema",
    "SeedFromExamplesResponseSchema",
    "SortPageQuerySchema",
    "SortPageResponseSchema",
    "SortRequestSchema",
    "SortResponseSchema",
    "TextsortSuggestionRequestSchema",
    "TextsortSuggestionsResponseSchema",
    "VotesResponseSchema",
]
