"""Schemas for the usage-metrics API (``POST /api/metrics``).

The SPA's default metrics provider batches events into one request; see
:mod:`vtsearch.metrics` for what the server does with them and the shape of
the line each event becomes.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

#: Most events one request may carry. The SPA flushes well below this.
MAX_EVENTS_PER_BATCH: int = 200


class MetricsEventSchema(Schema):
    """One usage event."""

    group = fields.String(
        required=True,
        validate=validate.Regexp(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$"),
        metadata={"description": "Event name, e.g. ``view``, ``resultsExported``, ``settingChanged``, ``click``."},
    )
    attributes = fields.Dict(
        keys=fields.String(),
        values=fields.Raw(allow_none=True),
        load_default=dict,
        metadata={
            "description": "Flat event details. Scalar values are kept (strings cut to 200 characters); "
            "nested values are dropped."
        },
    )
    client_ts = fields.Float(
        load_default=None,
        allow_none=True,
        metadata={"description": "When the browser logged the event, in epoch milliseconds."},
    )


class MetricsBatchSchema(Schema):
    """Body of ``POST /api/metrics``."""

    events = fields.List(
        fields.Nested(MetricsEventSchema),
        required=True,
        validate=validate.Length(max=MAX_EVENTS_PER_BATCH),
    )
    session = fields.String(
        load_default=None,
        allow_none=True,
        validate=validate.Length(max=64),
        metadata={"description": "Opaque id of the page load that logged the events."},
    )


class MetricsBatchResponseSchema(Schema):
    """Response of ``POST /api/metrics``."""

    recorded = fields.Integer(required=True, metadata={"description": "How many events were written."})
    enabled = fields.Boolean(
        required=True,
        metadata={"description": "Whether the server records metrics; the SPA stops sending once this is false."},
    )
