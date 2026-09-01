"""Audio media type - WAV/MP3/FLAC/OGG/M4A files."""

from __future__ import annotations

import io
import math
import threading
from collections import OrderedDict
from pathlib import Path


from vtscore.media.audio._demo_sources import build_demo_datasets, load_demo_source
from vtscore.media.base import (
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
)

# Thumbnail dimensions (square)
_THUMB_SIZE = 128

# Waveform thumbnails are theme-agnostic alpha masks, not pre-coloured images:
# the wave is painted fully opaque and the background left fully transparent, so
# no colour is baked into the PNG.  The frontend tints the mask to the live
# theme at render time — browse-canvas tiles via an offscreen ``source-in`` fill,
# the top-left now-playing indicator via a CSS ``mask`` — so one PNG serves the
# dark / light / highviz themes equally and recolours instantly on a theme
# switch, with no staleness and nothing theme-specific frozen into demo-dataset
# pickles.  See issue #2369.
_WAVE_FILL = (255, 255, 255, 255)  # opaque; only the alpha channel is used downstream

# Vertical gain applied to the RMS envelope before it's drawn.  RMS of full-scale
# audio is well under 1.0 (a full-scale sine is ~0.71), so a modest boost lets a
# typical clip use a healthy share of the frame's height while quiet passages stay
# thin — without loud columns clamping edge-to-edge.  See ``_render_waveform``.
_WAVE_GAIN = 1.4

# Process-scoped cache of decoded ``(samples, sr)`` keyed by an archive member.
# A windowed archive-member manifest fans one member into many clip windows
# (e.g. ~14 × 10 s CLAP windows per chunk), each its own media; without this
# cache every window's thumbnail would re-stream and re-decode the whole member.
# Bounded to a handful of members (each decoded member is a full float array in
# memory) with LRU eviction. In-memory only - never persisted (decoded audio is
# a derived artifact, re-derived on demand; see the "No Persisted Vectors" rule).
_DECODE_CACHE_MAX = 8
_decode_cache: OrderedDict[tuple[str, str], tuple] = OrderedDict()
_decode_cache_lock = threading.Lock()


def _render_waveform(audio_data, *, size: int = _THUMB_SIZE) -> bytes | None:
    """Draw an RMS amplitude envelope of *audio_data* onto a square PNG.

    Returns PNG bytes, or ``None`` for empty/undrawable input.  Shared by every
    waveform generator so the pixel rendering lives in one place.

    The envelope is the per-column *RMS* (energy), not the min/max peak.  A
    min/max envelope saturates to full height on loud, dense real-world audio —
    a single peak sample in a column's ~thousand-sample span pins that column
    top-to-bottom, so a whole clip (rain, fire, insects…) fills every column and
    the alpha mask tints to a solid rectangle (issue #2555).  RMS follows the
    loudness contour instead: quiet passages stay thin, loud ones grow, and the
    thumbnail reads as a waveform rather than a filled block.
    """
    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except Exception:
        return None

    if audio_data is None or len(audio_data) == 0:
        return None

    # Compute the RMS (energy) envelope across `size` columns.
    samples = len(audio_data)
    step = max(1, samples // size)
    cols = min(size, samples)

    rms = np.empty(cols, dtype=np.float32)
    for i in range(cols):
        start = i * step
        end = min(start + step, samples)
        chunk = audio_data[start:end]
        rms[i] = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))) if len(chunk) else 0.0

    # Normalise to pixel range
    amp = size // 2
    mid = size // 2

    # Transparent background so the frontend can tint the shape to any theme;
    # only the wave strokes are opaque (they form the alpha mask).  The
    # background is transparent *white* (not transparent black): giving the
    # hidden pixels the same RGB as the opaque wave means downscaling only ever
    # averages white-with-white, so shrinking a thumbnail to XS/M can't pull dark
    # RGB out of the "empty" pixels and fringe the wave.  Both render paths
    # (canvas ``source-in`` tint, CSS ``mask``) then key off the alpha channel
    # alone, so the wave stays clean at any size.
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    for i in range(cols):
        # Draw the column symmetric about the midline, its half-height the gained
        # RMS clamped to the frame so loud columns cap at the edges rather than
        # overrun them.
        h = min(1.0, float(rms[i]) * _WAVE_GAIN) * amp
        y_top = int(mid - h)
        y_bot = int(mid + h)
        # Ensure at least 1px line
        if y_top == y_bot:
            y_bot += 1
        draw.line([(i, y_top), (i, y_bot)], fill=_WAVE_FILL)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def generate_waveform_thumbnail(audio_bytes: bytes, *, size: int = _THUMB_SIZE) -> bytes | None:
    """Render a waveform thumbnail as a PNG image from raw audio bytes.

    Decodes the audio with :func:`~vtscore.media.audio.decode.decode_audio`,
    computes the RMS amplitude envelope, and draws it onto a square PIL image.
    Returns PNG bytes, or ``None`` if the audio cannot be decoded.

    Every container decodes straight from the in-memory buffer - libsndfile for
    WAV/FLAC/OGG/MP3, ffmpeg over ``stdin`` for AAC/M4A/MP4 - so no filename
    hint is needed to pick a decoder and nothing spills to disk.
    """
    audio_data, _sr = _decode_audio_bytes(audio_bytes)
    if audio_data is None:
        return None
    return _render_waveform(audio_data, size=size)


def _decode_audio_bytes(audio_bytes: bytes) -> tuple:
    """Decode *audio_bytes* to ``(mono_samples, sr)``, or ``(None, None)``.

    ffmpeg reads AAC/M4A/MP4 buffers straight off ``stdin``, so no container
    needs a filesystem path and there is no temp-file spill on any code path.
    """
    try:
        from vtscore.media.audio.decode import decode_audio  # noqa: PLC0415

        return decode_audio(audio_bytes, sr=None, mono=True)
    except Exception:
        return None, None


def _slice_window(audio_data, sr, clip_start: float | None, clip_end: float | None):
    """Return the ``[clip_start, clip_end]`` slice of *audio_data* (seconds).

    A ``None``/``NaN`` bound is left open (start-of-clip / end-of-clip).  A
    degenerate or out-of-order window falls back to the whole array so the tile
    still shows a waveform rather than nothing.
    """
    total = len(audio_data)
    if not sr or total == 0:
        return audio_data

    def _bound(v, default):
        if v is None:
            return default
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        if math.isnan(f):
            return default
        return int(f * sr)

    start = max(0, _bound(clip_start, 0))
    end = min(total, _bound(clip_end, total))
    if end <= start:
        return audio_data
    return audio_data[start:end]


def _decode_member_cached(cache_key, loader) -> tuple:
    """Return decoded ``(samples, sr)`` for a member, using the LRU decode cache.

    *loader* is a zero-arg callable returning the member's raw bytes; it is only
    invoked on a cache miss, so a cache hit skips both streaming and decoding.
    *cache_key* of ``None`` disables caching (decode every call).
    """
    if cache_key is not None:
        with _decode_cache_lock:
            hit = _decode_cache.get(cache_key)
            if hit is not None:
                _decode_cache.move_to_end(cache_key)
                return hit

    audio_bytes = loader()
    if audio_bytes is None:
        return None, None
    audio_data, sr = _decode_audio_bytes(audio_bytes)
    if audio_data is None:
        return None, None

    if cache_key is not None:
        with _decode_cache_lock:
            _decode_cache[cache_key] = (audio_data, sr)
            _decode_cache.move_to_end(cache_key)
            while len(_decode_cache) > _DECODE_CACHE_MAX:
                _decode_cache.popitem(last=False)
    return audio_data, sr


def generate_waveform_thumbnail_window(
    loader,
    *,
    clip_start: float | None = None,
    clip_end: float | None = None,
    size: int = _THUMB_SIZE,
    cache_key=None,
) -> bytes | None:
    """Render a waveform for one clip window.

    Built for archive-member audio (whose bytes stream from a tar/zip shard and
    whose codec is often AAC/M4A/MP4): decodes the whole member once via
    :func:`_decode_member_cached` (straight from the streamed bytes, ffmpeg-only
    codecs included), slices to ``[clip_start, clip_end]`` so each window shows
    its own waveform, and renders.  Returns ``None`` when the member can't be
    decoded.
    """
    audio_data, sr = _decode_member_cached(cache_key, loader)
    if audio_data is None:
        return None
    windowed = _slice_window(audio_data, sr, clip_start, clip_end)
    return _render_waveform(windowed, size=size)


def _clear_decode_cache() -> None:
    """Drop every cached decoded member (test hook; also frees memory)."""
    with _decode_cache_lock:
        _decode_cache.clear()


def generate_waveform_thumbnail_from_file(file_path: Path, *, size: int = _THUMB_SIZE) -> bytes | None:
    """Generate a waveform thumbnail from an audio file on disk.

    Decodes straight from the path so ffmpeg-only codecs (AAC/M4A/MP4) work,
    not just ``soundfile`` containers.
    """
    try:
        from vtscore.media.audio.decode import decode_audio  # noqa: PLC0415

        audio_data, _sr = decode_audio(str(file_path), sr=None, mono=True)
    except Exception:
        return None
    return _render_waveform(audio_data, size=size)


class AudioMediaType(MediaType):
    """Handles audio medias - file import, HTTP serving, and demo datasets.

    Embedding is handled by
    :class:`~vtscore.media.audio.embedder_clap_general.AudioClapGeneralEmbedder`
    (the default) or any of the other registered audio embedders.
    """

    #: Audio renders a waveform PNG (``generate_waveform_thumbnail``), so it is
    #: a browsable-thumbnail type: square tiles on the VTSBrowse map.
    has_thumbnail = True

    def __init__(self) -> None:
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "audio"

    @property
    def name(self) -> str:
        return "Audio"

    @property
    def icon(self) -> str:
        return "audio"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]

    @property
    def folder_import_name(self) -> str:
        return "audio"

    @property
    def dir_key(self) -> str:
        return "audio_dir"

    # ------------------------------------------------------------------
    # Display metadata
    # ------------------------------------------------------------------

    def display_metadata(self, media: dict) -> dict:
        result: dict = {}
        freq = media.get("frequency")
        if freq:
            result["Frequency"] = freq
        cat = media.get("category")
        if cat and cat not in ("unknown", "custom"):
            result["Category"] = cat
        dur = media.get("duration")
        if dur and dur > 0:
            result["Duration"] = dur
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        result.update({k: v for k, v in super().display_metadata(media).items() if k not in result})
        return result

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    @property
    def demo_datasets(self) -> list:
        return build_demo_datasets()

    def load_demo_source(
        self,
        source,
        categories,
        slice_start,
        slice_end,
        clips,
        on_progress=None,
        embedder=None,
        slice_frac_start=None,
        slice_frac_end=None,
        skip_embedding=False,
        **kwargs,
    ):
        return load_demo_source(
            self,
            source,
            categories,
            slice_start,
            slice_end,
            clips,
            on_progress=on_progress,
            embedder=embedder,
            slice_frac_start=slice_frac_start,
            slice_frac_end=slice_frac_end,
            skip_embedding=skip_embedding,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    @property
    def pickle_extra_fields(self) -> list[str]:
        return ["thumbnail_bytes"]

    def load_media_data(self, file_path: Path, media_bytes: bytes | None = None) -> dict:
        from vtscore.media.audio.decode import decode_audio  # noqa: PLC0415

        if media_bytes is None:
            with open(file_path, "rb") as f:
                media_bytes = f.read()
        try:
            audio_data, sr = decode_audio(str(file_path), sr=None, mono=True)
            duration = len(audio_data) / sr
        except Exception:
            duration = 0.0
        thumbnail = generate_waveform_thumbnail(media_bytes)
        return {"media_bytes": media_bytes, "duration": duration, "thumbnail_bytes": thumbnail}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def ensure_thumbnail_bytes(self, media: dict) -> bytes | None:
        """Return the waveform PNG, rendering it from the media's bytes if absent.

        Memoises the per-window PNG so repeat fetches of this exact window
        (each browse-canvas pan/zoom re-requests it) skip the decode, and so
        the background warm-up pass and the request path share one
        implementation.  In-memory only; never written to disk.
        """
        thumb = media.get("thumbnail_bytes")
        if thumb:
            return thumb
        thumb = self._waveform_for_media(media)
        if thumb:
            media["thumbnail_bytes"] = thumb
        return thumb

    def image_response(self, media: dict) -> MediaResponse | None:
        """Return the waveform thumbnail as a PNG image, or *None*."""
        thumb = self.ensure_thumbnail_bytes(media)
        if not thumb:
            return None
        return MediaResponse(
            data=thumb,
            mimetype="image/png",
            download_name=f"media_{media['id']}_waveform.png",
        )

    def _waveform_for_media(self, media: dict) -> bytes | None:
        """Generate this media's waveform on the fly (window-aware for clips).

        Archive-member audio carries no ``thumbnail_bytes`` at import (the
        importer never reads member bytes), and its bytes stream from a tar/zip
        shard.  Decode the streamed bytes directly (ffmpeg-only codecs
        (AAC/M4A/MP4) included), slice to the clip window so each of a member's
        windows shows its own waveform, and cache the decoded member so its
        windows don't each re-stream and re-decode it.

        The window is applied **only** when the resolved bytes are the whole
        source.  Bytes that already *are* this clip - a clipper's materialized
        slice, a cleaner's trimmed payload, or a lazy clip the resolver cuts
        from the source on demand - carry their own 0-based timeline, so
        re-applying the source-relative ``clip_start`` / ``clip_end`` would
        slice a second time and render the wrong stretch of audio.
        """
        from vtscore.media.lazy_clip import clip_recipe  # noqa: PLC0415

        ref = media.get("archive_member")
        member = ref.get("member", "") if isinstance(ref, dict) else ""
        serves_clip_bytes = media.get("media_bytes") is not None or clip_recipe(media) is not None
        cache_key = (
            None
            if serves_clip_bytes  # a per-clip payload must not be cached under a per-member key
            else (ref["path"], member)
            if isinstance(ref, dict) and ref.get("path")
            else None
        )
        return generate_waveform_thumbnail_window(
            lambda: self._resolve_media_bytes(media),
            clip_start=None if serves_clip_bytes else media.get("clip_start"),
            clip_end=None if serves_clip_bytes else media.get("clip_end"),
            cache_key=cache_key,
        )

    def media_response(self, media: dict) -> MediaResponse:
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype="audio/wav", download_name=f"media_{media['id']}.wav")
        return MediaResponse(
            data=data,
            mimetype="audio/wav",
            download_name=f"media_{media['id']}.wav",
        )
