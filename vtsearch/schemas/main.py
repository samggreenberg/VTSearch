"""Schemas for the small ``main`` blueprint (``/api/version``).

The static-file / SPA-serving routes in :mod:`vtsearch.routes.main` are
not JSON APIs and do not have schemas; only ``GET /api/version`` is
modelled here.
"""

from __future__ import annotations

from marshmallow import Schema, fields


class VersionSchema(Schema):
    """Response for ``GET /api/version``."""

    version = fields.String(
        required=True,
        metadata={"description": "UTC timestamp of HEAD's commit (ISO 8601, Z-terminated)."},
    )


__all__ = ["VersionSchema"]
