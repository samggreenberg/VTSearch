"""Schemas for the achievements API (``/api/achievements/*``).

Models the shape returned by :func:`vtsearch.achievements.get_full_state`
and the small request bodies accepted by the acknowledge / check-phrase
endpoints. See ``vtsearch/achievements.py`` for the source-of-truth
shape these schemas mirror.
"""

from __future__ import annotations

from marshmallow import Schema, fields


class _AchievementEntrySchema(Schema):
    id = fields.String(required=True)
    name = fields.String(required=True)
    description = fields.String(required=True)
    icon = fields.String(required=True)
    tiers = fields.List(fields.Integer(), required=True)
    counter = fields.Integer(required=True)
    tier_idx = fields.Integer(required=True, metadata={"description": "Current tier index; -1 = locked."})
    next_threshold = fields.Integer(
        allow_none=True,
        required=True,
        metadata={"description": "Counter value needed to reach the next tier, or null at max tier."},
    )


class _PendingAnnouncementSchema(Schema):
    id = fields.String(required=True)
    name = fields.String(required=True)
    icon = fields.String(required=True)
    tier_idx = fields.Integer(required=True)
    tier_name = fields.String(required=True)
    threshold = fields.Integer(required=True)


class _DocEntrySchema(Schema):
    id = fields.String(required=True)
    name = fields.String(required=True)
    path = fields.String(required=True)
    read = fields.Boolean(required=True)


class _MediaTypeEntrySchema(Schema):
    id = fields.String(required=True)
    name = fields.String(required=True)
    seen = fields.Boolean(required=True, metadata={"description": "Whether the user has voted on this media type."})


class _HourEntrySchema(Schema):
    hour = fields.Integer(required=True, metadata={"description": "Hour of day, UTC (0-23)."})
    seen = fields.Boolean(required=True, metadata={"description": "Whether the user has voted in this hour bucket."})


class AchievementStateSchema(Schema):
    """Response for ``GET /api/achievements``."""

    tier_names = fields.List(fields.String(), required=True)
    achievements = fields.List(fields.Nested(_AchievementEntrySchema), required=True)
    pending_announcements = fields.List(fields.Nested(_PendingAnnouncementSchema), required=True)
    docs = fields.List(fields.Nested(_DocEntrySchema), required=True)
    media_types = fields.List(fields.Nested(_MediaTypeEntrySchema), required=True)
    hours = fields.List(fields.Nested(_HourEntrySchema), required=True)


class AcknowledgeRequestSchema(Schema):
    tier_idx = fields.Integer(required=True, strict=True)


class AcknowledgeResponseSchema(Schema):
    ok = fields.Boolean(required=True)
    changed = fields.Boolean(required=True)


class CheckPhraseRequestSchema(Schema):
    phrase = fields.String(required=True)


class CheckPhraseResponseSchema(Schema):
    matched = fields.Boolean(required=True)
    doc_id = fields.String(allow_none=True, required=True)
    doc_name = fields.String(allow_none=True, required=True)
    already_read = fields.Boolean(required=True)


__all__ = [
    "AchievementStateSchema",
    "AcknowledgeRequestSchema",
    "AcknowledgeResponseSchema",
    "CheckPhraseRequestSchema",
    "CheckPhraseResponseSchema",
]
