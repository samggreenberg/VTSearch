"""Schemas for the server file-browser API (``GET /api/browse``).

Single endpoint, single query schema, single response schema. The
response intentionally omits any absolute filesystem path: only
relative paths within the user's allowed root are exposed.
"""

from __future__ import annotations

from marshmallow import Schema, fields


class BrowseQuerySchema(Schema):
    """Query string for ``GET /api/browse``."""

    path = fields.String(
        load_default="",
        metadata={"description": "Relative path within the allowed root. Empty string means the root itself."},
    )
    extensions = fields.String(
        load_default="",
        metadata={
            "description": (
                "Comma-separated list of file extensions to show (e.g. ``.csv,.json``). "
                "When omitted, all files are listed. Leading dots are optional."
            )
        },
    )

    class Meta:
        # Tolerate unrelated query params (e.g. dataset/detector ids
        # added by the request-context middleware on some clients).
        unknown = "exclude"


class BrowseDirectoryEntrySchema(Schema):
    """A directory entry in the browse response."""

    name = fields.String(required=True)
    path = fields.String(
        required=True,
        metadata={"description": "Path relative to the browse root."},
    )
    modified_at = fields.String(
        required=True,
        metadata={"description": "Modification time as ``YYYY-MM-DD HH:MM`` (empty on stat failure)."},
    )


class BrowseFileEntrySchema(BrowseDirectoryEntrySchema):
    """A file entry in the browse response (directory entry + size)."""

    size_bytes = fields.Integer(
        required=True,
        metadata={"description": "File size in bytes (0 on stat failure)."},
    )


class BrowseResponseSchema(Schema):
    """Response for ``GET /api/browse``.

    Note: ``root`` is intentionally omitted from the response — exposing
    the server's absolute filesystem path is a leak that the
    multi-user-isolation tests assert against.
    """

    directories = fields.List(fields.Nested(BrowseDirectoryEntrySchema), required=True)
    files = fields.List(fields.Nested(BrowseFileEntrySchema), required=True)
    current_path = fields.String(required=True)


__all__ = [
    "BrowseDirectoryEntrySchema",
    "BrowseFileEntrySchema",
    "BrowseQuerySchema",
    "BrowseResponseSchema",
]
