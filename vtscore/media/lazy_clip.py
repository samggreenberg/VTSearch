"""Lazy clip resolution: derive a clip's bytes from its source on demand.

A *lazy clip* is a derived sub-media that stores **no** materialized
``media_bytes`` of its own.  Instead it keeps a reference to the original
source file (``media_path``) plus a *recipe* (recorded in ``origin.params``)
and reproduces its bytes on demand.  This is how reference (thin) imports avoid
duplicating the source bytes once per derived item in the dataset pickle: a
5-minute recording tiled into 30 clips stores the recipe 30 times (a few bytes
each) instead of 30 copies of the audio.

Two recipe families participate:

* **clip slices** - the media type's clipper re-slices the source bytes:

  * **audio** - ``clip_start`` / ``clip_end`` seconds, sliced via
    :func:`~vtscore.media.audio.clipper._wav_slice`.
  * **image** - ``clip_box`` pixel box, cropped via Pillow.

  Text clips keep their (tiny) ``media_string`` materialized, and video clips
  are already metadata-only (they share the parent's bytes and the player
  seeks via ``clip_start`` / ``clip_end``), so neither needs lazy resolution.

* **converter output** - a converter (``document2image``, ``video2image``, …)
  rendered a derived media (a PDF page PNG, an extracted video frame) from the
  *source* file.  Keyed by the ``converter`` entry in ``origin.params`` (not by
  target media type), the recipe re-runs the converter and re-selects the exact
  sub-output recorded at import time (``converter_out_index`` /
  ``converter_n_out`` / ``converter_content_hash``) via the shared
  :func:`~vtscore.datasets.clipper_chain._select_chain_output` selector.

Resolved bytes are held in process-scoped LRU caches.  Per the
no-persisted-bytes rule, these caches are purely in-memory: nothing here writes
a materialized clip back to disk or into a pickle.  Clip slices use a
count-bounded cache (payloads are small and uniform - one tile / crop);
converter output uses a *byte-bounded* cache (a rendered page or video frame is
1-8 MB and varies by an order of magnitude, so a count bound that is safe for
clips would be unsafe here).
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
#: The converter branch is keyed by ``converter`` in ``origin.params`` instead,
#: so it is *not* gated on this list (a converter can target any media type).
LAZY_CLIP_TYPES = ("audio", "image")

# Process-scoped LRU of resolved clip bytes, keyed by (source, recipe).  Bounded
# by entry count; clip payloads are small (one tile / crop) so a few hundred
# entries stays well within memory.  Never persisted.
_CACHE_MAX = 256
_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
_cache_lock = threading.Lock()


class _ByteBoundedLRU:
    """Process-scoped LRU bounded by *total held bytes*, not entry count.

    Converter output (a rendered PDF page, a decoded video frame) is large and
    non-uniform - 1-8 MB per entry, varying by an order of magnitude across
    documents - so a count bound that keeps clip slices safe could hold 1-2 GB
    here.  Bounding by summed payload size keeps memory predictable regardless
    of output size.  Never persisted; purely an in-memory recompute cache.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max = max_bytes
        self._d: "OrderedDict[tuple, bytes]" = OrderedDict()
        self._total = 0
        self._lock = threading.Lock()

    def get(self, key: tuple) -> bytes | None:
        with self._lock:
            hit = self._d.get(key)
            if hit is not None:
                self._d.move_to_end(key)
            return hit

    def put(self, key: tuple, value: bytes) -> None:
        with self._lock:
            if key in self._d:
                self._total -= len(self._d[key])
            self._d[key] = value
            self._d.move_to_end(key)
            self._total += len(value)
            # Evict oldest until under the ceiling, but never evict the entry we
            # just inserted (a single output larger than the ceiling is kept so
            # the immediate fetch still hits; it is dropped on the next insert).
            while self._total > self._max and len(self._d) > 1:
                _, evicted = self._d.popitem(last=False)
                self._total -= len(evicted)


#: Byte ceiling for the converter-output cache (~256 MB).  Large enough to keep
#: a hot set of rendered pages / frames resident across HTTP range/scrub bursts
#: without an unbounded memory risk.
_CONV_CACHE_MAX_BYTES = 256 * 1024 * 1024
_conv_cache = _ByteBoundedLRU(_CONV_CACHE_MAX_BYTES)


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


def _converter_recipe(params: dict[str, Any]) -> tuple | None:
    """Build a converter recipe from ``origin.params``, or ``None``.

    Returns ``("converter", name, params_items, out_index, n_out,
    content_hash)`` where ``params_items`` is a hashable
    ``tuple(sorted(...))`` of the reconstructed converter params (rebuilt
    from the ``converter_param_<key>`` keys).  Requires at least one
    sub-output disambiguator (``converter_out_index`` or
    ``converter_content_hash``); without one, replay cannot pick the right
    output, so it is treated as "not a lazy converter media" (``None``) and
    the caller falls through to its normal resolution order.
    """
    name = params.get("converter")
    if not name:
        return None
    out_index_raw = params.get("converter_out_index")
    content_hash = params.get("converter_content_hash")
    if out_index_raw is None and content_hash is None:
        return None

    prefix = "converter_param_"
    conv_params = {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}

    def _as_int(raw: Any) -> int | None:
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    return (
        "converter",
        str(name),
        tuple(sorted(conv_params.items())),
        _as_int(out_index_raw),
        _as_int(params.get("converter_n_out")),
        content_hash,
    )


def clip_recipe(media: dict[str, Any]) -> tuple | None:
    """Return a hashable recipe describing *media*'s derivation, or ``None``.

    The recipe is read from ``origin.params`` (the channel that survives the
    pickle round-trip), not the top-level ``clip_*`` fields, so it resolves
    identically for a freshly derived media and one reopened from disk.

    Returns one of:

    * ``("converter", name, params_items, out_index, n_out, content_hash)`` -
      a converter output re-rendered from the source file.
    * ``("audio", clip_start, clip_end)`` - seconds, both floats.
    * ``("image", (x1, y1, x2, y2))`` - integer pixel box.

    or ``None`` when *media* is not a recognised lazy media (no recipe, or a
    media type that doesn't re-slice bytes).

    The converter branch is checked first and is keyed by the ``converter``
    param, not the target media type: a ``document2image`` output has
    ``media_type == "image"`` but is a converter output, not an image *clip*.

    **Archive members never lazy-slice.** A ``local_archive_member`` media may
    carry ``clip_start`` / ``clip_end`` (a windowed import), but its window is
    *display-only*: the byte routes serve the whole member and the player seeks
    within the window.  These corpora are AAC/MP4, which the WAV-only audio
    slicer cannot cut and which we deliberately do not decode server-side, so we
    return ``None`` here and let the caller fall through to whole-member byte
    serving.
    """
    from vtscore.datasets.archive_stream import archive_member_ref  # noqa: PLC0415

    if archive_member_ref(media) is not None:
        return None

    origin = media.get("origin")
    params = origin.get("params", {}) if isinstance(origin, dict) else {}

    if params.get("converter"):
        return _converter_recipe(params)

    media_type = media.get("media_type")

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


def _apply_recipe(source_bytes: bytes, recipe: tuple, media: dict[str, Any]) -> bytes | None:
    """Reproduce a derived media's bytes by applying *recipe* to *source_bytes*."""
    kind = recipe[0]
    if kind == "converter":
        return _apply_converter_recipe(source_bytes, recipe, media)
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


def _apply_converter_recipe(source_bytes: bytes, recipe: tuple, media: dict[str, Any]) -> bytes | None:
    """Re-run a converter on the source file and re-select the recorded output.

    Rebuilds the converter's ``params`` from the recipe, hands it a minimal
    source-media dict (the whole source bytes plus filename / path), and
    selects the exact sub-output via the shared
    :func:`~vtscore.datasets.clipper_chain._select_chain_output` so the
    content-hash-first, drift-aware, refuse-to-guess semantics match the
    cross-dataset resolver.  Returns ``None`` (cache nothing, fall through)
    when the converter is unknown or no output matches - the same "better no
    bytes than wrong bytes" stance the clip branches take.
    """
    from vtscore.converters import get_converter  # noqa: PLC0415
    from vtscore.datasets.clipper_chain import _select_chain_output  # noqa: PLC0415

    _, converter_name, params_items, out_index, n_out, content_hash = recipe
    conv = get_converter(converter_name)
    if conv is None:
        log.warning("lazy_clip: unknown converter %r; cannot resolve output", converter_name)
        return None

    source_path = media.get("media_path")
    filename = Path(source_path).name if source_path else media.get("filename", "")
    source_media: dict[str, Any] = {"media_bytes": source_bytes, "filename": filename}
    if source_path:
        source_media["media_path"] = source_path

    outputs = conv.convert_normalized(source_media, dict(params_items))

    entry: dict[str, Any] = {"kind": "converter", "name": converter_name, "out_index": out_index, "n_out": n_out}
    if content_hash is not None:
        entry["content_hash"] = content_hash
    picked = _select_chain_output(outputs, entry)
    if picked is None:
        return None
    payload = picked.get("media_bytes")
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    text = picked.get("media_string")
    if isinstance(text, str):
        return text.encode("utf-8")
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

    is_converter = recipe[0] == "converter"
    source_key = media.get("media_path") or media.get("media_url")
    cache_key = (source_key, recipe) if source_key else None

    if cache_key is not None:
        hit = _conv_cache.get(cache_key) if is_converter else _count_cache_get(cache_key)
        if hit is not None:
            return hit

    source = _read_source_bytes(media)
    if source is None:
        return None

    try:
        out = _apply_recipe(source, recipe, media)
    except Exception:
        log.warning("lazy_clip: failed to apply recipe %r to %s", recipe, source_key, exc_info=True)
        return None
    if out is None:
        return None

    if cache_key is not None:
        if is_converter:
            _conv_cache.put(cache_key, out)
        else:
            _count_cache_put(cache_key, out)
    return out


def _count_cache_get(cache_key: tuple) -> bytes | None:
    """Look up a clip slice in the count-bounded LRU."""
    with _cache_lock:
        hit = _cache.get(cache_key)
        if hit is not None:
            _cache.move_to_end(cache_key)
        return hit


def _count_cache_put(cache_key: tuple, value: bytes) -> None:
    """Insert a clip slice into the count-bounded LRU, evicting the oldest."""
    with _cache_lock:
        _cache[cache_key] = value
        _cache.move_to_end(cache_key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
