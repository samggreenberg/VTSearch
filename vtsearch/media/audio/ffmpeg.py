"""Locate the ffmpeg binary.

Checks the system ``$PATH`` first, then falls back to the static binary
bundled by the ``imageio-ffmpeg`` Python package (installed automatically
as part of the video plugin dependencies).
"""

from __future__ import annotations

import shutil


def get_ffmpeg_exe() -> str:
    """Return the path to an ffmpeg executable.

    Resolution order:
    1. System ``ffmpeg`` on ``$PATH``
    2. Static binary from ``imageio-ffmpeg``

    Raises ``FileNotFoundError`` if neither is available.
    """
    system = shutil.which("ffmpeg")
    if system:
        return system

    try:
        import imageio_ffmpeg  # noqa: PLC0415

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    raise FileNotFoundError("ffmpeg not found. Install it via your OS package manager or 'pip install imageio-ffmpeg'.")
