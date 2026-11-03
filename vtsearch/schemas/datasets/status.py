"""Schemas for ``/api/dataset/status`` and ``/api/dataset/cancel`` (``vtsearch/routes/datasets/status.py``)."""

from __future__ import annotations

from marshmallow import Schema, fields


class DatasetStatusResponseSchema(Schema):
    """Response for ``GET /api/dataset/status``."""

    loaded = fields.Boolean(required=True)
    num_medias = fields.Integer(required=True)
    has_votes = fields.Boolean(required=True)
    media_type = fields.String(allow_none=True, required=True)
    display_name = fields.String(required=True)
    num_dupes = fields.Integer(required=True)


class CancelDatasetLoadResponseSchema(Schema):
    """Response for ``POST /api/dataset/cancel`` and ``/cancel/<task_id>``.

    Cancellation is cooperative, so ``ok`` reports whether the flag actually
    reached something that can act on it — not merely that it was set.  The
    lists say which operations did what; see
    :func:`vtscore.concurrency.progress.cancel_dataset_progress`.  A request
    that reached nothing answers ``409`` with ``ok: false``.
    """

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    #: Everything that claimed to be working when the cancel arrived.
    targets = fields.List(fields.String(), required=True)
    #: Targets that reached a terminal state within the grace period.
    acknowledged = fields.List(fields.String(), required=True)
    #: Targets still running, whose live worker will observe the flag.
    pending = fields.List(fields.String(), required=True)
    #: Targets whose progress claimed work no live thread was doing. Stale
    #: trackers, now cleared — not operations that were stopped.
    unresponsive = fields.List(fields.String(), required=True)


__all__ = [
    "CancelDatasetLoadResponseSchema",
    "DatasetStatusResponseSchema",
]
