"""Schemas for Test mode's test of the line (#4524).

Every ``/api/line-test`` verb (``vtsearch/routes/line_test.py``) answers with
:class:`LineTestResponseSchema`: the balance's state and the line it draws,
the test (running, else the last finished one) as
:meth:`vtscore.training.thresholds.LineTest.as_dict` reports it, whether the
result is stale, and the line each balance preset would ship, re-estimated
from the picks already taken.

* ``GET  /api/line-test``        -> :class:`LineTestResponseSchema`
* ``POST /api/line-test/start``  -> :class:`LineTestResponseSchema`
* ``POST /api/line-test/votes``  -> :class:`LineTestVotesRequestSchema` -> :class:`LineTestResponseSchema`
* ``POST /api/line-test/unvote`` -> :class:`LineTestUnvoteRequestSchema` -> :class:`LineTestResponseSchema`
* ``POST /api/line-test/cancel`` -> :class:`LineTestResponseSchema`
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

from vtscore.training.thresholds import PHASES, SIDES, STOP_REASONS
from vtsearch.schemas.sorting import BalanceStateSchema, LikelyRangeSchema, PrecisionCheckVoteSchema


class LineTestEstimateSchema(Schema):
    """A point and its central 95% range, from the test's joint draws."""

    point = fields.Float(required=True)
    lo = fields.Float(required=True)
    hi = fields.Float(required=True)


class LineTestEdgeSchema(Schema):
    """What the line would ship if it kept the top ``count``: the ranges at that count, from the same draws."""

    count = fields.Integer(required=True)
    side = fields.String(required=True, validate=validate.OneOf(SIDES))
    precision = fields.Nested(LineTestEstimateSchema, required=True)
    recall = fields.Nested(LineTestEstimateSchema, required=True)
    fbeta = fields.Nested(LineTestEstimateSchema, required=True)
    # The recall range in words, as the spot check reads one.
    found = fields.String(required=True)


class LineTestEstimatesSchema(Schema):
    """Every number a test reports (:class:`vtscore.training.thresholds.LineEstimates`)."""

    beta = fields.Float(required=True)
    precision = fields.Nested(LineTestEstimateSchema, required=True)
    recall = fields.Nested(LineTestEstimateSchema, required=True)
    fbeta = fields.Nested(LineTestEstimateSchema, required=True)
    found = fields.String(required=True)
    positives_above = fields.Nested(LineTestEstimateSchema, required=True)
    positives_below = fields.Nested(LineTestEstimateSchema, required=True)
    # The model's count in the bands below the line the walk never reached,
    # folded into ``positives_below`` as a point and flagged.
    tail_positives = fields.Float(required=True)
    tail_from_model = fields.Boolean(required=True)
    # The picks the ranges rest on.
    labelled = fields.Integer(required=True)
    # The line re-estimated at every band edge on both sides.
    at_edges = fields.List(fields.Nested(LineTestEdgeSchema), required=True)


class LineTestRoundBandSchema(Schema):
    """The band a round was drawn from: its index, which side of the line, and its ranks (1-based, inclusive)."""

    index = fields.Integer(required=True)
    side = fields.String(required=True, validate=validate.OneOf(SIDES))
    lo = fields.Integer(required=True)
    hi = fields.Integer(required=True)


class LineTestBandSchema(LineTestRoundBandSchema):
    """One band of the sample, with its picks so far."""

    labelled = fields.Integer(required=True)
    right = fields.Integer(required=True)
    # The band's own likely range from its picks; ``null`` before any.
    range = fields.Nested(LikelyRangeSchema, required=True, allow_none=True)


class LineTestReportSchema(Schema):
    """Where the test stands (:class:`vtscore.training.thresholds.PhaseReport`): the phase, and why each finished phase ended."""

    phase = fields.String(required=True, validate=validate.OneOf(PHASES))
    matches_stop = fields.String(required=True, allow_none=True, validate=validate.OneOf(STOP_REASONS))
    misses_stop = fields.String(required=True, allow_none=True, validate=validate.OneOf(STOP_REASONS))
    # The ranges' current widths, for the phase lights.
    matches_width = fields.Float(required=True, allow_none=True)
    misses_width = fields.Float(required=True, allow_none=True)
    picks_above = fields.Integer(required=True)
    picks_below = fields.Integer(required=True)


class LineTestBudgetsSchema(Schema):
    """What the test may spend and when a phase is narrow enough (:class:`vtscore.training.thresholds.LineBudgets`)."""

    matches_width = fields.Float(required=True)
    misses_width = fields.Float(required=True)
    matches_picks = fields.Integer(required=True)
    misses_picks = fields.Integer(required=True)
    picks_per_round = fields.Integer(required=True)
    dry_run_share = fields.Float(required=True)
    model_weight = fields.Float(required=True)
    alpha = fields.Float(required=True)


class LineTestStateSchema(Schema):
    """A test of the line, as :meth:`vtscore.training.thresholds.LineTest.as_dict` reports it."""

    phase = fields.String(required=True, validate=validate.OneOf(PHASES))
    report = fields.Nested(LineTestReportSchema, required=True)
    beta = fields.Float(required=True)
    # The top ``line_count`` of the ``size`` ranked items are what the line keeps.
    line_count = fields.Integer(required=True)
    size = fields.Integer(required=True)
    # The round being voted on (1-based) and the picks a round deals.
    round = fields.Integer(required=True)
    picks_per_round = fields.Integer(required=True)
    # The band the pending picks were drawn from; ``null`` between rounds.
    band = fields.Nested(LineTestRoundBandSchema, required=True, allow_none=True)
    # The picks awaiting a vote, in draw order (random).  They are a sample,
    # not the ranking: a client must not show them with a rank.
    picks = fields.List(fields.Integer(), required=True)
    labelled = fields.Integer(required=True)
    bands = fields.List(fields.Nested(LineTestBandSchema), required=True)
    # ``null`` when there is nothing to test.
    estimates = fields.Nested(LineTestEstimatesSchema, required=True, allow_none=True)
    budgets = fields.Nested(LineTestBudgetsSchema, required=True)
    # When the test resumed from a verdict the detector keeps (#4526): the
    # epoch seconds its picks were taken at.  ``null`` for a test begun afresh.
    kept_at = fields.Float(required=True, allow_none=True)


class LineTestPresetSchema(Schema):
    """The line one balance preset would draw on this corpus, and what the picks say it would ship."""

    beta = fields.Float(required=True)
    count = fields.Integer(required=True)
    precision = fields.Nested(LineTestEstimateSchema, required=True)
    recall = fields.Nested(LineTestEstimateSchema, required=True)
    fbeta = fields.Nested(LineTestEstimateSchema, required=True)
    found = fields.String(required=True)


class LineTestResponseSchema(Schema):
    """Response for every ``/api/line-test`` verb."""

    # The balance's state for the line, as every carrier reports it.
    balance = fields.Nested(BalanceStateSchema, required=True)
    # The line the Find pass drew, and how many scored items it keeps.
    threshold = fields.Float(required=True, allow_none=True)
    line_count = fields.Integer(required=True)
    # The running test, or the last finished one; ``null`` when there is neither.
    test = fields.Nested(LineTestStateSchema, required=True, allow_none=True)
    # ``stale``: corrections were folded into the detector since the test
    # (``Add Corrections``: the detector has now seen the test set).
    # ``moved``: the line no longer keeps the set the test measured (the
    # balance changed, or a vote re-split the line).
    stale = fields.Boolean(required=True)
    moved = fields.Boolean(required=True)
    # The line each balance preset would draw, re-estimated from the picks
    # already taken (``Lean the Threshold``); empty before a test, or when
    # the detector has no class model to draw a line at another balance with.
    presets = fields.List(fields.Nested(LineTestPresetSchema), required=True)


class LineTestVerdictSchema(Schema):
    """A finished test's verdict as the detector keeps it (#4526), summarised for a reader.

    One per tested dataset (:mod:`vtscore.detectors.line_verdicts`): the
    ranges and F-beta at Done, the balance and line they were measured at, and
    how many picks they rest on.  ``stale``: the detector was retrained since
    (its labels changed), so the ranking the picks were drawn from no longer
    exists; the verdict stays, flagged.
    """

    dataset_id = fields.String(required=True)
    # The dataset's registered name now, else the one kept with the verdict.
    dataset_name = fields.String(required=True)
    # Epoch seconds the test finished at.
    tested_at = fields.Float(required=True)
    beta = fields.Float(required=True)
    # The line kept the top ``line_count`` of the ``size`` items scored.
    line_count = fields.Integer(required=True)
    size = fields.Integer(required=True)
    # The picks the ranges rest on.
    labelled = fields.Integer(required=True)
    precision = fields.Nested(LineTestEstimateSchema, required=True)
    recall = fields.Nested(LineTestEstimateSchema, required=True)
    fbeta = fields.Nested(LineTestEstimateSchema, required=True)
    # The recall range in the spot check's words.
    found = fields.String(required=True)
    stale = fields.Boolean(required=True)


class LineTestVotesRequestSchema(Schema):
    """Body for ``POST /api/line-test/votes``: votes on this round's picks, a partial round accepted."""

    votes = fields.List(fields.Nested(PrecisionCheckVoteSchema), required=True)


class LineTestUnvoteRequestSchema(Schema):
    """Body for ``POST /api/line-test/unvote``: take back one vote of the current round (the ↓ key)."""

    id = fields.Integer(required=True)
