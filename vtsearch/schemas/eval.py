"""Schemas for the eval / labeling-progress API.

Covers four routes in ``vtsearch/routes/eval.py``:

* ``POST /api/labeling-progress``                -> :class:`LabelingProgressResponseSchema`
* ``GET  /api/labeling-status``                  -> :class:`LabelingStatusResponseSchema`
* ``GET  /api/indicator-score-history``          -> :class:`IndicatorScoreHistoryQuerySchema`
                                                    -> :class:`IndicatorScoreHistoryResponseSchema`
* ``POST /api/eval/train-and-score``             -> :class:`EvalTrainAndScoreRequestSchema`
                                                    -> :class:`EvalTrainAndScoreResponseSchema`
* ``GET  /api/eval/train-and-score/result``      -> :class:`EvalTrainAndScoreResultQuerySchema`
                                                    -> :class:`EvalTrainAndScoreResponseSchema`

The per-step ``error_cost`` / ``stability`` / ``diversity`` lists are
declared as ``fields.List(fields.Dict())`` rather than fully nested
schemas; the inner shapes are computed by
``vtscore.detectors.labeling_progress`` and round-trip cleanly as
plain dicts.

The two train-and-score endpoints share a single response schema with
``unknown = "include"``: the metric-specific data key
(``error_cost`` / ``stability`` / ``diversity``) is computed at runtime
and flows through without per-key declarations.
"""

from __future__ import annotations

from marshmallow import Schema, fields, post_dump, validate


_METRIC_VALIDATOR = validate.OneOf(["smart", "stable", "diverse"])


# ---------------------------------------------------------------------------
# /api/labeling-progress
# ---------------------------------------------------------------------------


class LabelingProgressResponseSchema(Schema):
    """Response for ``POST /api/labeling-progress``."""

    error_cost_over_time = fields.List(fields.Dict(), required=True)
    stability_over_time = fields.List(fields.Dict(), required=True)
    diversity_level_over_time = fields.List(fields.Dict(), required=True)
    total_labels = fields.Integer(required=True)
    total_medias = fields.Integer(required=True)


# ---------------------------------------------------------------------------
# /api/labeling-status
# ---------------------------------------------------------------------------


class StatusIndicatorSchema(Schema):
    """One ``smart`` / ``stable`` / ``span`` indicator in the labeling-status
    response.  ``status`` is the red/yellow/green flag every indicator emits;
    ``reason`` is the human-readable explanation.  Metric-specific keys
    (``cost``, ``flips``, ``diversity_level``, ``avg_flip_rate``, …) flow
    through unchanged via ``unknown = "include"`` plus :meth:`_include_extras`
    the :mod:`vtscore.detectors.labeling_progress` analyzer remains the
    source of truth for that shape."""

    status = fields.String(required=True)
    reason = fields.String()

    class Meta:
        unknown = "include"

    @post_dump(pass_original=True)
    def _include_extras(self, data: dict, original: dict, **_: object) -> dict:
        # ``unknown = "include"`` only affects ``load``; declared schemas drop
        # unknown keys on ``dump``.  Re-merge the original dict's metric-
        # specific keys so callers receive the full indicator payload.
        if isinstance(original, dict):
            for k, v in original.items():
                if k not in data:
                    data[k] = v
        return data


class LabelingStatusResponseSchema(Schema):
    """Response for ``GET /api/labeling-status``."""

    good_count = fields.Integer(required=True)
    bad_count = fields.Integer(required=True)
    total_count = fields.Integer(required=True)
    smart = fields.Nested(StatusIndicatorSchema, required=True)
    stable = fields.Nested(StatusIndicatorSchema, required=True)
    span = fields.Nested(StatusIndicatorSchema, required=True)


# ---------------------------------------------------------------------------
# /api/indicator-score-history
# ---------------------------------------------------------------------------


class IndicatorScoreHistoryQuerySchema(Schema):
    """Query for ``GET /api/indicator-score-history``."""

    metric = fields.String(
        required=True,
        validate=_METRIC_VALIDATOR,
        metadata={"description": "Which metric history to return: ``smart``, ``stable``, or ``diverse``."},
    )

    class Meta:
        # Tolerate unrelated query params (e.g. dataset/detector ids
        # added by the request-context middleware on some clients).
        unknown = "exclude"


class IndicatorScoreHistoryResponseSchema(Schema):
    """Response for ``GET /api/indicator-score-history``."""

    metric = fields.String(required=True, validate=_METRIC_VALIDATOR)
    history = fields.List(fields.Dict(), required=True)


# ---------------------------------------------------------------------------
# /api/eval/train-and-score (start) and /result (poll)
# ---------------------------------------------------------------------------


class EvalTrainAndScoreRequestSchema(Schema):
    """Body for ``POST /api/eval/train-and-score``."""

    metric = fields.String(required=True, validate=_METRIC_VALIDATOR)
    wait = fields.Boolean(
        load_default=False,
        metadata={
            "description": (
                "If true, block until the job completes and return the metric data inline. "
                "Used by tests; production clients poll ``/result`` instead."
            )
        },
    )


class EvalTrainAndScoreResultQuerySchema(Schema):
    """Query for ``GET /api/eval/train-and-score/result``."""

    job_id = fields.String(required=True)

    class Meta:
        unknown = "exclude"


class EvalTrainAndScoreResponseSchema(Schema):
    """Combined response for the start + poll train-and-score routes.

    The response shape varies with status (``running`` vs ``done`` vs
    ``error`` vs ``cancelled``) and metric (``error_cost`` vs
    ``stability`` vs ``diversity`` data keys). Declared as a permissive
    schema so the metric-specific data key flows through unchanged.
    """

    job_id = fields.String(required=True)
    status = fields.String(
        required=True,
        validate=validate.OneOf(["running", "done", "error", "cancelled", "missing"]),
    )
    metric = fields.String()
    current = fields.Integer()
    total = fields.Integer()
    error_cost = fields.List(fields.Dict())
    stability = fields.List(fields.Dict())
    diversity = fields.List(fields.Dict())
    error = fields.String()

    class Meta:
        # Future metrics may add new data keys; let them pass through
        # without per-key declarations.
        unknown = "include"


class EvalTrainAndScoreCancelResponseSchema(Schema):
    """Response for ``POST /api/eval/train-and-score/cancel/<job_id>``."""

    ok = fields.Boolean(required=True)


__all__ = [
    "EvalTrainAndScoreCancelResponseSchema",
    "EvalTrainAndScoreRequestSchema",
    "EvalTrainAndScoreResponseSchema",
    "EvalTrainAndScoreResultQuerySchema",
    "IndicatorScoreHistoryQuerySchema",
    "IndicatorScoreHistoryResponseSchema",
    "LabelingProgressResponseSchema",
    "LabelingStatusResponseSchema",
    "StatusIndicatorSchema",
]
