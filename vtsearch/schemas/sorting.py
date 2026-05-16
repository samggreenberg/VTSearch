"""Schemas for the sorting / voting / inclusion APIs.

Covers the routes in ``vtsearch/routes/sorting.py``:

* ``POST /api/sort``                          — :class:`SortRequestSchema` →
                                                :class:`SortResponseSchema`
* ``POST /api/learned-sort``                  — :class:`LearnedSortRequestSchema` →
                                                :class:`LearnedSortResponseSchema`
* ``GET  /api/learned-sort/result``           — :class:`LearnedSortResultQuerySchema` →
                                                :class:`LearnedSortResponseSchema`
* ``GET  /api/votes``                         — :class:`VotesResponseSchema`
* ``POST /api/votes/clear``                   — :class:`OkResponseSchema`
* ``POST /api/votes/seed-from-examples``      — :class:`SeedFromExamplesRequestSchema` →
                                                :class:`SeedFromExamplesResponseSchema`
* ``GET  /api/textsort-suggestions``          — :class:`TextsortSuggestionsResponseSchema`
* ``POST /api/textsort-suggestions``          — :class:`TextsortSuggestionRequestSchema` →
                                                :class:`OkResponseSchema`
* ``GET  /api/inclusion``                     — :class:`InclusionResponseSchema`
* ``POST /api/inclusion``                     — :class:`InclusionRequestSchema` →
                                                :class:`InclusionResponseSchema`
* ``GET  /api/safe-thresholds``               — :class:`SafeThresholdsResponseSchema`
* ``POST /api/safe-thresholds``               — :class:`SafeThresholdsRequestSchema` →
                                                :class:`SafeThresholdsResponseSchema`
* ``POST /api/example-sort``                  — multipart upload →
                                                :class:`SortResponseSchema`
* ``POST /api/label-file-sort``               — multipart upload →
                                                :class:`LabelFileSortResponseSchema`
* ``GET|POST /api/diversity-tree/next``       — :class:`DiversityTreeNextRequestSchema` →
                                                :class:`DiversityTreeNextResponseSchema`

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


class SortResponseSchema(Schema):
    """Response for ``POST /api/sort`` and ``POST /api/example-sort``."""

    results = fields.List(fields.Dict(), required=True)
    threshold = fields.Float(required=True)


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

    The response varies by status — ``running`` carries ``current``/``total``,
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


# ---------------------------------------------------------------------------
# /api/votes (+ clear, seed-from-examples)
# ---------------------------------------------------------------------------


class VotesResponseSchema(Schema):
    """Response for ``GET /api/votes``."""

    good = fields.List(fields.Integer(), required=True)
    bad = fields.List(fields.Integer(), required=True)
    click_times = fields.Dict(keys=fields.String(), values=fields.Float(), required=True)
    learned_scores = fields.Dict(keys=fields.String(), values=fields.Float(), required=True)
    labelset_good_count = fields.Integer(required=True)
    labelset_bad_count = fields.Integer(required=True)


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


def _validate_numeric(value):
    """Reject booleans and non-numeric values for the inclusion field.

    Booleans are a subclass of ``int`` in Python — without this guard
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
# /api/safe-thresholds
# ---------------------------------------------------------------------------


class SafeThresholdsResponseSchema(Schema):
    """Response for ``GET|POST /api/safe-thresholds``."""

    safe_thresholds = fields.Boolean(required=True)


def _validate_bool(value):
    """Reject non-boolean values for boolean fields.

    Restricting ``fields.Boolean.truthy / falsy`` to ``{True} / {False}``
    isn't enough on its own — Python treats ``1 == True`` and ``0 == False``
    when checking set membership, so numeric inputs sneak through. A
    plain ``isinstance(value, bool)`` check is the only way to require
    strictly-typed booleans.
    """
    if not isinstance(value, bool):
        raise ValidationError("Must be a boolean.")


class SafeThresholdsRequestSchema(Schema):
    """Body for ``POST /api/safe-thresholds``.

    Declared as ``fields.Raw`` + a custom validator to preserve the
    pre-migration "must be a boolean" behavior — string forms
    (``"yes"`` / ``"no"`` / ``"true"``) and numeric forms (``0`` / ``1``)
    are rejected as 422.
    """

    safe_thresholds = fields.Raw(required=True, validate=_validate_bool)


# ---------------------------------------------------------------------------
# /api/label-file-sort (multipart)
# ---------------------------------------------------------------------------


class LabelFileSortResponseSchema(Schema):
    """Response for ``POST /api/label-file-sort``."""

    results = fields.List(fields.Dict(), required=True)
    threshold = fields.Float(required=True)
    loaded = fields.Integer(required=True)
    skipped = fields.Integer(required=True)


# ---------------------------------------------------------------------------
# /api/diversity-tree/next
# ---------------------------------------------------------------------------


class DiversityTreeNextRequestSchema(Schema):
    """Body for ``POST /api/diversity-tree/next``.

    Both fields are optional. ``scores`` keys are media ids encoded as
    strings (JSON object keys can't be ints); the handler converts them
    back to ints. Declared as a permissive dict so the handler keeps
    ownership of the int-key coercion + error path.
    """

    scores = fields.Dict(keys=fields.String(), values=fields.Float(), load_default=None)
    threshold = fields.Float(load_default=None, allow_none=True)


class DiversityTreeNextResponseSchema(Schema):
    """Response for ``GET|POST /api/diversity-tree/next``."""

    id = fields.Integer(allow_none=True, required=True)
    diversity_level = fields.Integer(required=True)
    exhausted = fields.Boolean(required=True)


__all__ = [
    "DiversityTreeNextRequestSchema",
    "DiversityTreeNextResponseSchema",
    "InclusionRequestSchema",
    "InclusionResponseSchema",
    "LabelFileSortResponseSchema",
    "LearnedSortRequestSchema",
    "LearnedSortResponseSchema",
    "LearnedSortResultQuerySchema",
    "OkResponseSchema",
    "SafeThresholdsRequestSchema",
    "SafeThresholdsResponseSchema",
    "SeedFromExamplesRequestSchema",
    "SeedFromExamplesResponseSchema",
    "SortRequestSchema",
    "SortResponseSchema",
    "TextsortSuggestionRequestSchema",
    "TextsortSuggestionsResponseSchema",
    "VotesResponseSchema",
]
