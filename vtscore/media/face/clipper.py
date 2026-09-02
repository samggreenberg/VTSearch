"""Face clippers - pass-through face media.

Faces are already the atomic unit (one crop per detected face, produced by the
``image2face`` converter), so the only clipper is the canonical no-op default
that keeps every embeddable media type's clipper list non-empty.
"""

from __future__ import annotations


from vtscore.media.clipper import DefaultClipper


class FaceDefaultClipper(DefaultClipper):
    """Returns the face media unchanged."""

    def __init__(self) -> None:
        super().__init__("face_default", "face", "Keep each face crop as-is, without splitting.")
