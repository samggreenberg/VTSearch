"""Face clippers - pass-through face media.

Faces are already the atomic unit (one crop per detected face, produced by the
``image2face`` converter), so the only clipper is the canonical no-op default
that keeps every embeddable media type's clipper list non-empty.
"""

from __future__ import annotations

from typing import Any

from vtscore.media.clipper import MediaClipper


class FaceDefaultClipper(MediaClipper):
    """Returns the face media unchanged."""

    @property
    def name(self) -> str:
        return "face_default"

    @property
    def display_name(self) -> str:
        return "None"

    @property
    def media_type(self) -> str:
        return "face"

    @property
    def description(self) -> str:
        return "Keep each face crop as-is, without splitting."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [media]
