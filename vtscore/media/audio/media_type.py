"""Audio media type - WAV/MP3/FLAC/OGG/M4A files."""

from __future__ import annotations

import io
import math
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path


from vtscore.config import DATA_DIR
from vtscore.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    demo_slice,
)

# Thumbnail dimensions (square)
_THUMB_SIZE = 128

# Waveform colours (dark background, bright waveform)
_BG_COLOR = (30, 30, 30)
_WAVE_COLOR = (0, 180, 255)

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
    """Draw a min/max amplitude envelope of *audio_data* onto a square PNG.

    Returns PNG bytes, or ``None`` for empty/undrawable input.  Shared by every
    waveform generator so the pixel rendering lives in one place.
    """
    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except Exception:
        return None

    if audio_data is None or len(audio_data) == 0:
        return None

    # Compute min/max envelope across `size` columns
    samples = len(audio_data)
    step = max(1, samples // size)
    cols = min(size, samples)

    mins = np.empty(cols, dtype=np.float32)
    maxs = np.empty(cols, dtype=np.float32)
    for i in range(cols):
        start = i * step
        end = min(start + step, samples)
        chunk = audio_data[start:end]
        mins[i] = chunk.min()
        maxs[i] = chunk.max()

    # Normalise to pixel range
    amp = size // 2
    mid = size // 2

    img = Image.new("RGB", (size, size), _BG_COLOR)
    draw = ImageDraw.Draw(img)

    for i in range(cols):
        y_top = int(mid - maxs[i] * amp)
        y_bot = int(mid - mins[i] * amp)
        # Ensure at least 1px line
        if y_top == y_bot:
            y_bot += 1
        draw.line([(i, y_top), (i, y_bot)], fill=_WAVE_COLOR)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def generate_waveform_thumbnail(
    audio_bytes: bytes, *, filename: str = "", size: int = _THUMB_SIZE
) -> bytes | None:
    """Render a waveform thumbnail as a PNG image from raw audio bytes.

    Decodes the audio with librosa, computes the min/max amplitude envelope, and
    draws it onto a square PIL image.  Returns PNG bytes, or ``None`` if the
    audio cannot be decoded.

    ``soundfile``-backed containers (WAV/FLAC/OGG/MP3) decode straight from the
    in-memory buffer.  Codecs that fall back to ``audioread`` + ffmpeg
    (AAC/M4A/MP4) can't be decoded from a ``BytesIO`` at all - librosa only
    reaches the ffmpeg backend for a real filesystem path - so on a buffer-decode
    failure this spills to a temp file via :func:`_decode_audio_file_bytes` and
    retries.  *filename* (when known) lends its extension to that temp file so
    ``audioread`` picks the right backend.
    """
    try:
        import librosa  # noqa: PLC0415
    except Exception:
        return None
    try:
        audio_data, _sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    except Exception:
        audio_data, _sr = _decode_audio_file_bytes(audio_bytes, filename)
    if audio_data is None:
        return None
    return _render_waveform(audio_data, size=size)


def _decode_audio_file_bytes(audio_bytes: bytes, filename: str = "") -> tuple:
    """Decode *audio_bytes* to ``(mono_samples, sr)`` via a temp file.

    Spilling to a temp file (preserving *filename*'s extension so ``audioread``
    can pick the right backend) is what lets AAC/M4A/MP4 members decode at all:
    ``librosa.load`` over an in-memory ``BytesIO`` never reaches ffmpeg, which
    needs a filesystem path.  Returns ``(None, None)`` on any decode error.
    """
    try:
        import librosa  # noqa: PLC0415
    except Exception:
        return None, None

    suffix = Path(filename).suffix or ".bin"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        audio_data, sr = librosa.load(tmp_path, sr=None, mono=True)
        return audio_data, sr
    except Exception:
        return None, None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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


def _decode_member_cached(cache_key, loader, filename: str) -> tuple:
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
    audio_data, sr = _decode_audio_file_bytes(audio_bytes, filename)
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
    filename: str = "",
    clip_start: float | None = None,
    clip_end: float | None = None,
    size: int = _THUMB_SIZE,
    cache_key=None,
) -> bytes | None:
    """Render a waveform for one clip window, decoding via a temp file.

    Built for archive-member audio (whose bytes stream from a tar/zip shard and
    whose codec is often AAC/M4A/MP4): decodes the whole member once via
    :func:`_decode_member_cached` (temp-file path so ffmpeg-only codecs work),
    slices to ``[clip_start, clip_end]`` so each window shows its own waveform,
    and renders.  Returns ``None`` when the member can't be decoded.
    """
    audio_data, sr = _decode_member_cached(cache_key, loader, filename)
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
        import librosa  # noqa: PLC0415
    except Exception:
        return None
    try:
        audio_data, _sr = librosa.load(str(file_path), sr=None, mono=True)
    except Exception:
        return None
    return _render_waveform(audio_data, size=size)


class AudioMediaType(MediaType):
    """Handles audio medias - file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtscore.media.audio.embedder_clap.AudioClapEmbedder`.
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

    @property
    def demo_datasets(self) -> list:
        from vtscore.datasets.downloader import (  # noqa: PLC0415
            ESC50_DOWNLOAD_SIZE_MB,
            GTZAN_DOWNLOAD_SIZE_MB,
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
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

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
        import hashlib  # noqa: PLC0415

        if on_progress is None:
            from vtscore.concurrency.progress import update_progress

            on_progress = update_progress

        if embedder is None:
            from vtscore.media import embedders_for_type

            avail = embedders_for_type(self.type_id)
            if not avail:
                raise ValueError(f"No embedders registered for media type {self.type_id!r}")
            embedder = avail[0]

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
            original_cb = embedder._on_progress
            embedder._on_progress = on_progress
            try:
                embedder.load_models()
            finally:
                embedder._on_progress = original_cb

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
                "md5": hashlib.md5(wav_bytes).hexdigest(),
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
        import librosa  # noqa: PLC0415

        if media_bytes is None:
            with open(file_path, "rb") as f:
                media_bytes = f.read()
        try:
            audio_data, sr = librosa.load(file_path, sr=None, mono=True)
            duration = len(audio_data) / sr
        except Exception:
            duration = 0.0
        thumbnail = generate_waveform_thumbnail(media_bytes, filename=Path(file_path).name)
        return {"media_bytes": media_bytes, "duration": duration, "thumbnail_bytes": thumbnail}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def image_response(self, media: dict) -> MediaResponse | None:
        """Return the waveform thumbnail as a PNG image, or *None*."""
        thumb = media.get("thumbnail_bytes")
        if not thumb:
            thumb = self._waveform_for_media(media)
            if thumb:
                # Memoise the per-window PNG so repeat fetches of this exact
                # window (each browse-canvas pan/zoom re-requests it) skip the
                # decode. In-memory only; never written to disk.
                media["thumbnail_bytes"] = thumb
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
        shard.  Decode via a temp file so ffmpeg-only codecs (AAC/M4A/MP4)
        render, slice to the clip window so each of a member's windows shows its
        own waveform, and cache the decoded member so its windows don't each
        re-stream and re-decode it.
        """
        ref = media.get("archive_member")
        member = ref.get("member", "") if isinstance(ref, dict) else ""
        filename = media.get("filename") or (Path(member).name if member else "")
        cache_key = (ref["path"], member) if isinstance(ref, dict) and ref.get("path") else None
        return generate_waveform_thumbnail_window(
            lambda: self._resolve_media_bytes(media),
            filename=filename,
            clip_start=media.get("clip_start"),
            clip_end=media.get("clip_end"),
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
