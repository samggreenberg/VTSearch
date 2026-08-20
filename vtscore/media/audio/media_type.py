"""Audio media type - WAV/MP3/FLAC/OGG/M4A files."""

from __future__ import annotations

import io
import math
import threading
from collections import OrderedDict
from pathlib import Path


from vtscore.config import DATA_DIR
from vtscore.media._toponymy_demo import SOURCE_ID as _TOPONYMY_SOURCE_ID
from vtscore.media._toponymy_demo import TAXONOMY as _TOPONYMY_TAXONOMY
from vtscore.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    demo_slice,
)
from vtscore.utils.hashing import content_md5

# Thumbnail dimensions (square)
_THUMB_SIZE = 128

# Synthetic world-map demo tone: a short mono sine whose pitch identifies the
# leaf city, so 108 cities span an audible, visibly-distinct range of waveforms.
_SYNTH_TONE_SECONDS = 0.6
_SYNTH_TONE_SR = 16000
_SYNTH_TONE_LO_HZ = 180.0
_SYNTH_TONE_HI_HZ = 1400.0


def _synthetic_tone_wav(city_index: int, n_cities: int) -> bytes:
    """Render a leaf city as a short 16-bit-PCM sine-tone WAV (no files, no ffmpeg).

    The pitch rises log-linearly with *city_index* across the taxonomy so each
    city gets a recognisably different tone (and waveform thumbnail); a faint
    second harmonic keeps the render from looking perfectly flat.
    """
    import struct  # noqa: PLC0415
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    frac = city_index / max(1, n_cities - 1)
    freq = _SYNTH_TONE_LO_HZ * (_SYNTH_TONE_HI_HZ / _SYNTH_TONE_LO_HZ) ** frac
    t = np.arange(int(_SYNTH_TONE_SR * _SYNTH_TONE_SECONDS), dtype=np.float64) / _SYNTH_TONE_SR
    wave_f = 0.6 * np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * 2 * freq * t)
    samples = np.clip(wave_f, -1.0, 1.0)
    pcm = (samples * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SYNTH_TONE_SR)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm.tolist()))
    return buf.getvalue()


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

    # Shared categories for all S/M/L audio demo datasets.
    _DEMO_CATEGORIES = [
        "dog",
        "rooster",
        "pig",
        "cow",
        "frog",
        "cat",
        "hen",
        "insects",
        "sheep",
        "crow",
        "rain",
        "sea_waves",
        "crackling_fire",
        "crickets",
        "chirping_birds",
        "water_drops",
        "wind",
        "pouring_water",
        "toilet_flush",
        "thunderstorm",
        "crying_baby",
        "sneezing",
        "clapping",
        "breathing",
        "coughing",
        "footsteps",
        "laughing",
        "brushing_teeth",
        "snoring",
        "drinking_sipping",
        "door_wood_knock",
        "mouse_click",
        "keyboard_typing",
        "door_wood_creep",
        "can_opening",
        "washing_machine",
        "vacuum_cleaner",
        "clock_alarm",
        "clock_tick",
        "glass_breaking",
        "helicopter",
        "chainsaw",
        "siren",
        "car_horn",
        "engine",
        "train",
        "church_bells",
        "airplane",
        "fireworks",
        "hand_saw",
    ]

    _GTZAN_CATEGORIES = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]

    _SPEECH_COMMANDS_CATEGORIES = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "bed",
        "bird",
        "cat",
        "dog",
        "happy",
        "house",
        "marvin",
        "sheila",
        "tree",
        "wow",
        "backward",
        "follow",
        "forward",
        "learn",
        "visual",
    ]

    _URBANSOUND8K_CATEGORIES = [
        "air_conditioner",
        "car_horn",
        "children_playing",
        "dog_bark",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "siren",
        "street_music",
    ]

    # TUT Sound Events 2017 ships uncut ~4-minute street soundscapes.  We don't
    # use its event annotations: every recording goes into one "street" bucket
    # so the user clips the long files themselves.
    _TUT_CATEGORIES = ["street"]

    # Clotho is an audio *captioning* dataset (no class labels), imported as
    # real-world Freesound sound clips in one undifferentiated bucket.  It's the
    # compositional-scene playground for natural-language text->audio search.
    _CLOTHO_CATEGORIES = ["sound"]

    # The three long-form demos below share a shape: hours-long unlabelled
    # recordings in one undifferentiated bucket, where the interesting content
    # is *discrete events scattered through the runtime* rather than a clip
    # whose label is the whole clip.  That is what makes them detector
    # playgrounds - you clip, listen, vote on a handful of hits, and let the
    # ranker find the rest.
    _APOLLO11_CATEGORIES = ["mission_audio"]
    _BIRDVOX_CATEGORIES = ["night_recording"]
    _NIXON_CATEGORIES = ["conversation"]

    @property
    def demo_datasets(self) -> list:
        from vtscore.datasets.downloader import (  # noqa: PLC0415
            APOLLO11_AUDIO_DOWNLOAD_SIZE_MB,
            BIRDVOX_FULL_NIGHT_DOWNLOAD_SIZE_MB,
            CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
            ESC50_DOWNLOAD_SIZE_MB,
            GTZAN_DOWNLOAD_SIZE_MB,
            NIXON_TAPES_DOWNLOAD_SIZE_MB,
            SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
            TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB,
            URBANSOUND8K_DOWNLOAD_SIZE_MB,
        )

        cats = self._DEMO_CATEGORIES
        folder = DATA_DIR / "ESC-50-master" / "audio"
        esc_desc = "Animals, nature, cities, & homes"
        sc_desc = "Spoken keyword utterances"
        us_desc = "Urban recordings"
        # 24 development + 8 evaluation recordings, all one "street" bucket.
        tut_folder = DATA_DIR / "tut_sound_events_2017"
        tut_desc = "Long ~4min street soundscapes (clip them yourself)"
        tut_total = 32
        # Clotho eval split: 1045 real-world Freesound clips, one "sound" bucket.
        clotho_folder = DATA_DIR / "clotho"
        clotho_desc = "Real-world Freesound clips for text search"
        clotho_total = 1045
        # Apollo 11: 103 MP3 tracks, ~174 h of mission loops (median ~85 min).
        apollo_folder = DATA_DIR / "apollo11_audio"
        apollo_desc = "Long NASA mission loops — Quindar beeps, alarms, applause"
        apollo_total = 103
        # BirdVox: 6 ten-hour units, segmented into 10-minute chunks on download.
        birdvox_folder = DATA_DIR / "birdvox_full_night"
        birdvox_desc = "10min chunks of all-night birdsong — sub-second flight calls"
        birdvox_total = 360
        # Nixon: 12 tapes' worth of conversations, one MP3 per conversation.
        nixon_folder = DATA_DIR / "nixon_tapes"
        nixon_desc = "Secret-taping-system conversations (rough audio, by design)"
        nixon_total = 1917
        return [
            DemoDataset(
                id="esc50_s",
                label="ESC-50 (S)",
                description=esc_desc,
                categories=cats,
                source="esc50",
                required_folder=folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=40,
                download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="esc50_m",
                label="ESC-50 (M)",
                description=esc_desc,
                categories=cats,
                source="esc50",
                required_folder=folder,
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=40,
                download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="esc50_l",
                label="ESC-50 (L)",
                description=esc_desc,
                categories=cats,
                source="esc50",
                required_folder=folder,
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=40,
                download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="esc50_a",
                label="ESC-50 (A)",
                description=esc_desc,
                categories=cats,
                source="esc50",
                required_folder=folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=40,
                download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="gtzan_a",
                label="GTZAN Music Genre (A)",
                description="30sec music excerpts",
                categories=self._GTZAN_CATEGORIES,
                source="gtzan",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=100,
                download_size_mb=GTZAN_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="speech_commands_v2_s",
                label="Speech Commands v2 (S)",
                description=sc_desc,
                categories=self._SPEECH_COMMANDS_CATEGORIES,
                source="speech_commands_v2",
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=3000,
                download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="speech_commands_v2_m",
                label="Speech Commands v2 (M)",
                description=sc_desc,
                categories=self._SPEECH_COMMANDS_CATEGORIES,
                source="speech_commands_v2",
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=3000,
                download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="speech_commands_v2_l",
                label="Speech Commands v2 (L)",
                description=sc_desc,
                categories=self._SPEECH_COMMANDS_CATEGORIES,
                source="speech_commands_v2",
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=3000,
                download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="speech_commands_v2_a",
                label="Speech Commands v2 (A)",
                description=sc_desc,
                categories=self._SPEECH_COMMANDS_CATEGORIES,
                source="speech_commands_v2",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=3000,
                download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="urbansound8k_s",
                label="UrbanSound8K (S)",
                description=us_desc,
                categories=self._URBANSOUND8K_CATEGORIES,
                source="urbansound8k",
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=873,
                download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="urbansound8k_m",
                label="UrbanSound8K (M)",
                description=us_desc,
                categories=self._URBANSOUND8K_CATEGORIES,
                source="urbansound8k",
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=873,
                download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="urbansound8k_l",
                label="UrbanSound8K (L)",
                description=us_desc,
                categories=self._URBANSOUND8K_CATEGORIES,
                source="urbansound8k",
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=873,
                download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="urbansound8k_a",
                label="UrbanSound8K (A)",
                description=us_desc,
                categories=self._URBANSOUND8K_CATEGORIES,
                source="urbansound8k",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=873,
                download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="tut_sound_events_2017_s",
                label="TUT Sound Events 2017 (S)",
                description=tut_desc,
                categories=self._TUT_CATEGORIES,
                source="tut_sound_events_2017",
                required_folder=tut_folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=tut_total,
                download_size_mb=TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="tut_sound_events_2017_m",
                label="TUT Sound Events 2017 (M)",
                description=tut_desc,
                categories=self._TUT_CATEGORIES,
                source="tut_sound_events_2017",
                required_folder=tut_folder,
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=tut_total,
                download_size_mb=TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="tut_sound_events_2017_l",
                label="TUT Sound Events 2017 (L)",
                description=tut_desc,
                categories=self._TUT_CATEGORIES,
                source="tut_sound_events_2017",
                required_folder=tut_folder,
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=tut_total,
                download_size_mb=TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="tut_sound_events_2017_a",
                label="TUT Sound Events 2017 (A)",
                description=tut_desc,
                categories=self._TUT_CATEGORIES,
                source="tut_sound_events_2017",
                required_folder=tut_folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=tut_total,
                download_size_mb=TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="clotho_s",
                label="Clotho (S)",
                description=clotho_desc,
                categories=self._CLOTHO_CATEGORIES,
                source="clotho",
                required_folder=clotho_folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=clotho_total,
                download_size_mb=CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="clotho_m",
                label="Clotho (M)",
                description=clotho_desc,
                categories=self._CLOTHO_CATEGORIES,
                source="clotho",
                required_folder=clotho_folder,
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=clotho_total,
                download_size_mb=CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="clotho_l",
                label="Clotho (L)",
                description=clotho_desc,
                categories=self._CLOTHO_CATEGORIES,
                source="clotho",
                required_folder=clotho_folder,
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=clotho_total,
                download_size_mb=CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="clotho_a",
                label="Clotho (A)",
                description=clotho_desc,
                categories=self._CLOTHO_CATEGORIES,
                source="clotho",
                required_folder=clotho_folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=clotho_total,
                download_size_mb=CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="apollo11_audio_s",
                label="Apollo 11 Mission Audio (S)",
                description=apollo_desc,
                categories=self._APOLLO11_CATEGORIES,
                source="apollo11_audio",
                required_folder=apollo_folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 12,
                items_per_category=apollo_total,
                download_size_mb=APOLLO11_AUDIO_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="apollo11_audio_m",
                label="Apollo 11 Mission Audio (M)",
                description=apollo_desc,
                categories=self._APOLLO11_CATEGORIES,
                source="apollo11_audio",
                required_folder=apollo_folder,
                slice_frac_start=1 / 12,
                slice_frac_end=3 / 12,
                items_per_category=apollo_total,
                download_size_mb=APOLLO11_AUDIO_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="apollo11_audio_l",
                label="Apollo 11 Mission Audio (L)",
                description=apollo_desc,
                categories=self._APOLLO11_CATEGORIES,
                source="apollo11_audio",
                required_folder=apollo_folder,
                slice_frac_start=3 / 12,
                slice_frac_end=None,
                items_per_category=apollo_total,
                download_size_mb=APOLLO11_AUDIO_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="apollo11_audio_a",
                label="Apollo 11 Mission Audio (A)",
                description=apollo_desc,
                categories=self._APOLLO11_CATEGORIES,
                source="apollo11_audio",
                required_folder=apollo_folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=apollo_total,
                download_size_mb=APOLLO11_AUDIO_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="birdvox_full_night_s",
                label="BirdVox Full Night (S)",
                description=birdvox_desc,
                categories=self._BIRDVOX_CATEGORIES,
                source="birdvox_full_night",
                required_folder=birdvox_folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 6,
                items_per_category=birdvox_total,
                download_size_mb=BIRDVOX_FULL_NIGHT_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="birdvox_full_night_m",
                label="BirdVox Full Night (M)",
                description=birdvox_desc,
                categories=self._BIRDVOX_CATEGORIES,
                source="birdvox_full_night",
                required_folder=birdvox_folder,
                slice_frac_start=1 / 6,
                slice_frac_end=3 / 6,
                items_per_category=birdvox_total,
                download_size_mb=BIRDVOX_FULL_NIGHT_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="birdvox_full_night_l",
                label="BirdVox Full Night (L)",
                description=birdvox_desc,
                categories=self._BIRDVOX_CATEGORIES,
                source="birdvox_full_night",
                required_folder=birdvox_folder,
                slice_frac_start=3 / 6,
                slice_frac_end=None,
                items_per_category=birdvox_total,
                download_size_mb=BIRDVOX_FULL_NIGHT_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="birdvox_full_night_a",
                label="BirdVox Full Night (A)",
                description=birdvox_desc,
                categories=self._BIRDVOX_CATEGORIES,
                source="birdvox_full_night",
                required_folder=birdvox_folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=birdvox_total,
                download_size_mb=BIRDVOX_FULL_NIGHT_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="nixon_tapes_s",
                label="Nixon White House Tapes (S)",
                description=nixon_desc,
                categories=self._NIXON_CATEGORIES,
                source="nixon_tapes",
                required_folder=nixon_folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 12,
                items_per_category=nixon_total,
                download_size_mb=NIXON_TAPES_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="nixon_tapes_m",
                label="Nixon White House Tapes (M)",
                description=nixon_desc,
                categories=self._NIXON_CATEGORIES,
                source="nixon_tapes",
                required_folder=nixon_folder,
                slice_frac_start=1 / 12,
                slice_frac_end=3 / 12,
                items_per_category=nixon_total,
                download_size_mb=NIXON_TAPES_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="nixon_tapes_l",
                label="Nixon White House Tapes (L)",
                description=nixon_desc,
                categories=self._NIXON_CATEGORIES,
                source="nixon_tapes",
                required_folder=nixon_folder,
                slice_frac_start=3 / 12,
                slice_frac_end=None,
                items_per_category=nixon_total,
                download_size_mb=NIXON_TAPES_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="nixon_tapes_a",
                label="Nixon White House Tapes (A)",
                description=nixon_desc,
                categories=self._NIXON_CATEGORIES,
                source="nixon_tapes",
                required_folder=nixon_folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=nixon_total,
                download_size_mb=NIXON_TAPES_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="synthetic_world_audio",
                label="Synthetic World Map (signposts demo)",
                description=(
                    "Pre-baked 4-level toponymy (Continent → Country → State → City) "
                    "with cheating ground-truth signposts — no download, loads instantly."
                ),
                categories=list(_TOPONYMY_TAXONOMY.keys()),
                source=_TOPONYMY_SOURCE_ID,
                items_per_category=0,
                download_size_mb=0,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def _load_synthetic_toponymy(self, clips, embedder, on_progress):
        """Populate *clips* with the synthetic world-map demo (no model, no download).

        Each item is a leaf city rendered as a short sine tone (a per-city
        frequency), tagged with its ``Continent/Country/State/City`` path and a
        pre-baked hierarchical embedding.  Browsing it lights up the ground-truth
        signpost layer straight from those paths — the friction-free way to eval
        the VTSBrowse sign display.  See :mod:`vtscore.media._toponymy_demo`.
        """

        from vtscore.media._toponymy_demo import generate_items, total_cities  # noqa: PLC0415

        # CLAP's audio/text space is 512-D; match it so the baked vectors slot
        # into the primary embedder's slot and text queries don't dimension-clash.
        items = generate_items(dim=512)
        n_cities = total_cities()
        total = len(items)
        on_progress("loading", f"Generating {total} synthetic clips…", 0, total)

        emb_name = embedder.name if embedder is not None else "clap"
        clip_id = 1
        for i, item in enumerate(items):
            wav_bytes = _synthetic_tone_wav(item.city_index, n_cities)
            thumb = generate_waveform_thumbnail(wav_bytes)
            filename = f"{item.category}/clip{i:04d}.wav"
            clips[clip_id] = {
                "id": clip_id,
                "media_type": self.type_id,
                "embedder": emb_name,
                "duration": _SYNTH_TONE_SECONDS,
                "file_size": len(wav_bytes),
                "md5": content_md5(wav_bytes),
                "embeddings": {emb_name: item.embedding},
                "media_bytes": wav_bytes,
                "thumbnail_bytes": thumb,
                "filename": filename,
                "category": item.category,
                "origin": {"importer": "demo", "params": {}},
                "origin_name": filename,
            }
            clip_id += 1
            if (i + 1) % 100 == 0:
                on_progress("loading", f"Generating synthetic clips… ({i + 1}/{total})", i + 1, total)
        # Bytes ride inline in the pickle — no external media dir.
        return None

    def _collect_longform_audio_files(
        self,
        source: str,
        categories: list,
        slice_start: int,
        slice_end: int | None,
        slice_frac_start: float | None,
        slice_frac_end: float | None,
        on_progress,
    ):
        """Resolve a long-form demo source → ``(audio_files, audio_dir)``, else ``None``.

        Apollo 11, BirdVox-full-night and the Nixon tapes are hours-long
        unlabelled recordings running to 5-10 GB apiece, so they invert the
        order the other demos use: the *manifest* is sliced first and only the
        selected items are downloaded, rather than pulling the whole source and
        slicing afterwards.  Each loads as one undifferentiated bucket - the
        events worth finding are scattered inside the recordings, so there is
        nothing to label at the file level.

        Returns ``None`` when *source* is not one of the three, leaving
        :meth:`_collect_audio_files` to handle it.
        """

        def _bucket(paths: list, default_category: str, audio_dir: Path):
            category = categories[0] if categories else default_category
            return [(p, {"category": category, "path": p}) for p in paths], audio_dir

        def _select(items: list):
            return demo_slice(items, slice_start, slice_end, slice_frac_start, slice_frac_end)

        if source == "apollo11_audio":
            from vtscore.datasets.downloader import (  # noqa: PLC0415
                apollo11_audio_manifest,
                download_apollo11_audio,
            )

            tracks = _select(apollo11_audio_manifest())
            audio_dir = download_apollo11_audio(tracks, on_progress=on_progress)
            paths = [audio_dir / name for name, _size in tracks]
            return _bucket([p for p in paths if p.exists()], "mission_audio", audio_dir)

        if source == "birdvox_full_night":
            from vtscore.datasets.downloader import (  # noqa: PLC0415
                birdvox_full_night_manifest,
                download_birdvox_full_night,
            )

            units = _select(birdvox_full_night_manifest())
            base_dir = download_birdvox_full_night(units, on_progress=on_progress)
            # The download segments each ~10-hour unit into 10-minute chunks.
            paths = sorted(p for unit in units for p in (base_dir / unit).glob("*.flac"))
            return _bucket(paths, "night_recording", base_dir)

        if source == "nixon_tapes":
            from vtscore.datasets.downloader import (  # noqa: PLC0415
                download_nixon_tapes,
                nixon_tape_manifest,
            )

            tapes = _select(nixon_tape_manifest())
            base_dir = download_nixon_tapes(tapes, on_progress=on_progress)
            # NARA serves one MP3 per recorded conversation.
            paths = sorted(p for tape in tapes for p in (base_dir / tape).glob("*.mp3"))
            return _bucket(paths, "conversation", base_dir)

        return None

    def _collect_audio_files(
        self,
        source: str,
        categories: list,
        slice_start: int,
        slice_end: int | None,
        slice_frac_start: float | None,
        slice_frac_end: float | None,
        on_progress,
    ):
        """Resolve a demo source name → (audio_files, audio_dir)."""

        def _sliced_by_category(by_cat: dict[str, list]) -> list:
            out: list = []
            for cat in categories:
                out.extend(
                    demo_slice(
                        by_cat.get(cat, []),
                        slice_start,
                        slice_end,
                        slice_frac_start,
                        slice_frac_end,
                    )
                )
            return out

        if source == "gtzan":
            from vtscore.datasets.downloader import download_gtzan  # noqa: PLC0415
            from vtscore.datasets.loader import load_audio_metadata_from_folders  # noqa: PLC0415

            audio_dir = download_gtzan(on_progress=on_progress)
            metadata = load_audio_metadata_from_folders(audio_dir, categories)
            by_cat = self._group_metadata_by_category(metadata, categories, filter_to_categories=False)
            return _sliced_by_category(by_cat), audio_dir

        if source == "speech_commands_v2":
            from vtscore.datasets.downloader import download_speech_commands_v2  # noqa: PLC0415
            from vtscore.datasets.loader import load_audio_metadata_from_folders  # noqa: PLC0415

            audio_dir = download_speech_commands_v2(on_progress=on_progress)
            metadata = load_audio_metadata_from_folders(audio_dir, categories)
            by_cat = self._group_metadata_by_category(metadata, categories, filter_to_categories=False)
            return _sliced_by_category(by_cat), audio_dir

        if source == "urbansound8k":
            from vtscore.datasets.downloader import download_urbansound8k  # noqa: PLC0415
            from vtscore.datasets.loader import load_urbansound8k_metadata  # noqa: PLC0415

            us8k_dir = download_urbansound8k(on_progress=on_progress)
            metadata = load_urbansound8k_metadata(us8k_dir)
            by_cat = self._group_metadata_by_category(metadata, categories, filter_to_categories=True)
            return _sliced_by_category(by_cat), us8k_dir / "audio"

        if source == "tut_sound_events_2017":
            from vtscore.datasets.downloader import download_tut_sound_events_2017  # noqa: PLC0415

            audio_dir = download_tut_sound_events_2017(on_progress=on_progress)
            # No annotations: every recording is one undifferentiated bucket.
            category = categories[0] if categories else "street"
            by_cat = {category: [(p, {"category": category, "path": p}) for p in sorted(audio_dir.rglob("*.wav"))]}
            return _sliced_by_category(by_cat), audio_dir

        longform = self._collect_longform_audio_files(
            source,
            categories,
            slice_start,
            slice_end,
            slice_frac_start,
            slice_frac_end,
            on_progress,
        )
        if longform is not None:
            return longform

        if source == "clotho":
            from vtscore.datasets.downloader import download_clotho  # noqa: PLC0415

            audio_dir = download_clotho(on_progress=on_progress)
            # Captioning dataset with no class labels: one undifferentiated bucket.
            category = categories[0] if categories else "sound"
            by_cat = {category: [(p, {"category": category, "path": p}) for p in sorted(audio_dir.rglob("*.wav"))]}
            return _sliced_by_category(by_cat), audio_dir

        if not source or source == "esc50":
            from vtscore.datasets.downloader import download_esc50  # noqa: PLC0415
            from vtscore.datasets.loader import load_esc50_metadata  # noqa: PLC0415

            audio_dir = download_esc50(on_progress=on_progress)
            esc_metadata = load_esc50_metadata(audio_dir.parent)
            by_cat = self._esc50_by_category(audio_dir, esc_metadata, categories)
            return _sliced_by_category(by_cat), audio_dir

        raise ValueError(f"Unsupported audio source: {source!r}")

    @staticmethod
    def _esc50_by_category(audio_dir: Path, esc_metadata: dict, categories: list) -> dict[str, list]:
        """Group ESC-50 wav files by their category, keeping only *categories*."""
        by_cat: dict[str, list] = {}
        for audio_path in sorted(audio_dir.glob("*.wav")):
            meta = esc_metadata.get(audio_path.name)
            if meta is not None and meta["category"] in categories:
                by_cat.setdefault(meta["category"], []).append((audio_path, meta))
        return by_cat

    @staticmethod
    def _group_metadata_by_category(
        metadata: dict,
        categories: list,
        *,
        filter_to_categories: bool,
    ) -> dict[str, list]:
        by_cat: dict[str, list] = {}
        for _key, meta in sorted(metadata.items()):
            cat = meta["category"]
            if filter_to_categories and cat not in categories:
                continue
            by_cat.setdefault(cat, []).append((meta["path"], meta))
        return by_cat

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

        if on_progress is None:
            from vtscore.concurrency.progress import update_progress

            on_progress = update_progress

        if embedder is None:
            from vtscore.media import embedders_for_type

            avail = embedders_for_type(self.type_id)
            if not avail:
                raise ValueError(f"No embedders registered for media type {self.type_id!r}")
            embedder = avail[0]

        # Synthetic signposts demo: generated in-memory, no download, no model.
        if source == _TOPONYMY_SOURCE_ID:
            return self._load_synthetic_toponymy(clips, embedder, on_progress)

        audio_files, audio_dir = self._collect_audio_files(
            source,
            categories,
            slice_start,
            slice_end,
            slice_frac_start,
            slice_frac_end,
            on_progress,
        )

        # Load models (skipped when a clipper will re-embed every clip - see
        # skip_embedding in load_demo_dataset).
        if not skip_embedding and getattr(embedder, "_model", None) is None:
            on_progress("loading", "Loading audio embedding model…", 0, 0)
            with embedder.progress_scope(on_progress):
                embedder.load_models()

        clip_id = max(clips.keys(), default=0) + 1
        total = len(audio_files)
        status = "loading" if skip_embedding else "embedding"
        verb = "Loading" if skip_embedding else "Embedding"
        on_progress(status, f"{verb} {total} audio files...", 0, total)
        demo_origin: dict = {"importer": "demo", "params": {}}

        from vtscore.media.embedder import media_from_path  # noqa: PLC0415

        for i, (audio_path, meta) in enumerate(audio_files):
            rel_name = f"{meta['category']}/{audio_path.name}"
            if skip_embedding:
                on_progress("loading", f"Loading {rel_name}", i + 1, total)
                embedding = None
            else:
                on_progress("embedding", f"Embedding {rel_name}", i + 1, total)
                embedding = embedder.embed_media(media_from_path(audio_path))
                if embedding is None:
                    continue

            with open(audio_path, "rb") as f:
                wav_bytes = f.read()

            media_fields = self.load_media_data(audio_path)
            clips[clip_id] = {
                "id": clip_id,
                "media_type": self.type_id,
                "embedder": embedder.name,
                "duration": media_fields["duration"],
                "file_size": len(wav_bytes),
                "md5": content_md5(wav_bytes),
                "embeddings": {} if skip_embedding else {embedder.name: embedding},
                "media_bytes": wav_bytes,
                "thumbnail_bytes": media_fields.get("thumbnail_bytes"),
                "filename": rel_name,
                "category": meta["category"],
                "origin": demo_origin,
                "origin_name": rel_name,
            }
            clip_id += 1

        return str(audio_dir.absolute())

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
