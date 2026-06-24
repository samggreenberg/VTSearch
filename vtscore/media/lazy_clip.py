"""Lazy clip resolution: derive a clip's bytes from its source on demand.

A *lazy clip* is a derived sub-media that stores **no** materialized
``media_bytes`` of its own.  Instead it keeps a reference to the original
source file (``media_path``) plus a *recipe* (the clip boundaries recorded in
``origin.params``) and reproduces its bytes on demand by slicing/cropping the
source.  This is how reference (thin) imports avoid duplicating the source
bytes once per clip in the dataset pickle: a 5-minute recording tiled into 30
clips stores the recipe 30 times (a few bytes each) instead of 30 copies of
the audio.

Only the media types whose clippers actually *re-slice bytes* participate:

* **audio** - ``clip_start`` / ``clip_end`` seconds, sliced via
  :func:`~vtscore.media.audio.clipper._wav_slice`.
* **image** - ``clip_box`` pixel box, cropped via Pillow.

Text clips keep their (tiny) ``media_string`` materialized, and video clips are
already metadata-only (they share the parent's bytes and the player seeks via
``clip_start`` / ``clip_end``), so neither needs lazy resolution.

Resolved bytes are held in a small process-scoped LRU cache.  Per the
no-persisted-bytes rule, this cache is purely in-memory: nothing here writes a
materialized clip back to disk or into a pickle.
"""

from __future__ import annotations

import io
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["LAZY_CLIP_TYPES", "clip_recipe", "lazy_clip_bytes"]

#: Media types whose clippers re-slice bytes and therefore support lazy clips.
LAZY_CLIP_TYPES = ("audio", "image")

# Process-scoped LRU of resolved clip bytes, keyed by (source, recipe).  Bounded
# by entry count; clip payloads are small (one tile / crop) so a few hundred
# entries stays well within memory.  Never persisted.
_CACHE_MAX = 256
_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
_cache_lock = threading.Lock()


def _parse_box(raw: Any) -> tuple[int, int, int, int] | None:
    """Parse a ``clip_box`` into a 4-int pixel tuple, or ``None`` if malformed.

    Accepts the ``"x1,y1,x2,y2"`` string stored in ``origin.params`` as well
    as a native list/tuple (the in-memory form on a freshly clipped media).
    """
    if isinstance(raw, (list, tuple)):
        parts = list(raw)
    elif isinstance(raw, str):
        parts = [p for p in raw.split(",") if p != ""]
    else:
        return None
    if len(parts) != 4:
        return None
    try:
        return (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3])))
    except (TypeError, ValueError):
        return None


def clip_recipe(media: dict[str, Any]) -> tuple | None:
    """Return a hashable recipe describing *media*'s clip, or ``None``.

    The recipe is read from ``origin.params`` (the channel that survives the
    pickle round-trip), not the top-level ``clip_*`` fields, so it resolves
    identically for a freshly clipped media and one reopened from disk.

    Returns one of:

    * ``("audio", clip_start, clip_end)`` - seconds, both floats.
    * ``("image", (x1, y1, x2, y2))`` - integer pixel box.

    or ``None`` when *media* is not a recognised lazy clip (no recipe, or a
    media type that doesn't re-slice bytes).
    """
    media_type = media.get("media_type")
    origin = media.get("origin")
    params = origin.get("params", {}) if isinstance(origin, dict) else {}

    if media_type == "audio":
        cs = params.get("clip_start")
        ce = params.get("clip_end")
        if cs is None or ce is None:
            return None
        try:
            return ("audio", float(cs), float(ce))
        except (TypeError, ValueError):
            return None

    if media_type == "image":
        box = _parse_box(params.get("clip_box"))
        if box is None:
            return None
        return ("image", box)

    return None


def _apply_recipe(source_bytes: bytes, recipe: tuple) -> bytes | None:
    """Reproduce a clip's bytes by applying *recipe* to *source_bytes*."""
    kind = recipe[0]
    if kind == "audio":
        from vtscore.media.audio.clipper import _wav_slice  # noqa: PLC0415

        return _wav_slice(source_bytes, recipe[1], recipe[2])
    if kind == "image":
        from PIL import Image  # noqa: PLC0415

        box = recipe[1]
        with Image.open(io.BytesIO(source_bytes)) as img:
            fmt = img.format or "PNG"
            cropped = img.crop(box)
            buf = io.BytesIO()
            cropped.save(buf, format=fmt)
        return buf.getvalue()
    return None


def _read_source_bytes(media: dict[str, Any]) -> bytes | None:
    """Read the whole source file (or URL) backing a lazy clip."""
    media_path = media.get("media_path")
    if media_path:
        path = Path(media_path)
        if path.exists():
            return path.read_bytes()
    media_url = media.get("media_url")
    if media_url:
        from vtscore.media.base import _fetch_media_url  # noqa: PLC0415

        return _fetch_media_url(media_url)
    return None


def lazy_clip_bytes(media: dict[str, Any]) -> bytes | None:
    """Return *media*'s clip bytes derived from its source, or ``None``.

    ``None`` means "not a lazy clip" (no recipe) or "source unavailable" - in
    both cases the caller falls back to its existing resolution order
    (``media_path`` whole-file read, then ``media_url``).  A media that already
    carries inline ``media_bytes`` is never lazy, so callers should check that
    first; this function does not.
    """
    recipe = clip_recipe(media)
    if recipe is None:
        return None

    source_key = media.get("media_path") or media.get("media_url")
    cache_key = (source_key, recipe) if source_key else None

    if cache_key is not None:
        with _cache_lock:
            hit = _cache.get(cache_key)
            if hit is not None:
                _cache.move_to_end(cache_key)
                return hit

    source = _read_source_bytes(media)
    if source is None:
        return None

    try:
        out = _apply_recipe(source, recipe)
    except Exception:
        log.warning("lazy_clip: failed to apply recipe %r to %s", recipe, source_key, exc_info=True)
        return None
    if out is None:
        return None

    if cache_key is not None:
        with _cache_lock:
            _cache[cache_key] = out
            _cache.move_to_end(cache_key)
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
    return out
