"""A face space with a ball, for the tests that need the face crops' ball drawn (#5497).

Every test starts with no ``face_ball.json`` (:func:`tests_shared.state_reset.no_face_models`), so a
face corpus keeps the line from before the ball.  :func:`face_ball_space` writes buffalo_s's entry,
the floors #5478 shipped, to a file of the test's own; :func:`tag_faces` makes a fixture corpus face
crops in that space.
"""

from __future__ import annotations

from typing import Any

import pytest

#: buffalo_s's face space, and the floors #5478 fitted there (``docs/experiments/2026-10-30-face-ball-line-5478``).
COMPAT = "oss-arcface-mbf:1"
FLOORS = {1.0: 0.4425, 4.0: 0.385}
SLOPE = 0.25
CENTRE = 0.6371


def face_ball(compat: str = COMPAT, *, use: bool = True):
    """buffalo_s's ball, in *compat*'s space."""
    from vtscore.media.face.face_ball import FaceBall

    return FaceBall(compat=compat, use=use, betas={b: (f, SLOPE) for b, f in FLOORS.items()}, centre=CENTRE)


@pytest.fixture
def face_ball_space(tmp_path, monkeypatch):
    """A ``face_ball.json`` holding buffalo_s's entry; yields the :class:`~vtscore.media.face.face_ball.FaceBall`."""
    from vtscore.media.face import face_ball as module

    path = tmp_path / "face-ball" / module.FACE_BALL_RELPATH
    monkeypatch.setattr(module, "face_ball_path", lambda: path)
    ball = face_ball()
    module.write_entry(ball, path)
    module.reset_for_tests()
    yield ball
    module.reset_for_tests()


def tag_faces(snap: dict[Any, dict], compat: str | None = COMPAT) -> dict[Any, dict]:
    """*snap*'s medias as face crops embedded in *compat*'s space (no tag for ``None``)."""
    from vtscore.media.face.imagekit_bridge import COMPAT_METADATA_KEY

    for media in snap.values():
        media["media_type"] = "face"
        meta = media.setdefault("custom_metadata", {})
        if compat is None:
            meta.pop(COMPAT_METADATA_KEY, None)
        else:
            meta[COMPAT_METADATA_KEY] = compat
    return snap
