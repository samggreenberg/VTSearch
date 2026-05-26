"""Schemas for the Recent Sessions API (``/api/sessions/recent``).

A "session" here is a ``(dataset_id, detector_id)`` pair the user has
recently been working on. The user-facing surface (the burger-menu
submenu) lists the N most recent pairs so the user can jump back into
work in one click. Entries whose ids no longer resolve in the dataset
or detector registry are filtered out on read.
"""

from __future__ import annotations

from marshmallow import Schema, fields


class RecentSessionSchema(Schema):
    """One hydrated recent-session entry returned by the API.

    ``last_activity`` is epoch seconds (float). The ``dataset_name`` /
    ``detector_name`` fields are resolved against the current registry
    at read time; entries pointing at deleted ids are dropped before
    we get here, so these are always populated.
    """

    dataset_id = fields.String(required=True)
    detector_id = fields.String(required=True)
    dataset_name = fields.String(required=True)
    detector_name = fields.String(required=True)
    last_activity = fields.Float(required=True)


class RecentSessionsResponseSchema(Schema):
    """Response shape for ``GET`` and ``POST /api/sessions/recent``."""

    sessions = fields.List(fields.Nested(RecentSessionSchema), required=True)


class BumpRecentSessionRequestSchema(Schema):
    """Body shape for ``POST /api/sessions/recent``."""

    dataset_id = fields.String(required=True)
    detector_id = fields.String(required=True)


__all__ = [
    "BumpRecentSessionRequestSchema",
    "RecentSessionSchema",
    "RecentSessionsResponseSchema",
]
