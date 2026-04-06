"""Audio media type — WAV/MP3/FLAC/OGG/M4A files."""

from __future__ import annotations

import io
from pathlib import Path


from vtsearch.config import DATA_DIR
from vtsearch.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
)

# Thumbnail dimensions (square)
_THUMB_SIZE = 128

# Waveform colours (dark background, bright waveform)
_BG_COLOR = (30, 30, 30)
_WAVE_COLOR = (0, 180, 255)


def generate_waveform_thumbnail(audio_bytes: bytes, *, size: int = _THUMB_SIZE) -> bytes | None:
    """Render a waveform thumbnail as a PNG image from raw audio bytes.

    Decodes the audio with librosa, computes the min/max amplitude envelope,
    and draws it onto a square PIL image.  Returns PNG bytes, or ``None`` if
    the audio cannot be decoded.
    """
    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except Exception:
        return None

    try:
        audio_data, _sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    except Exception:
        return None

    if len(audio_data) == 0:
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


def generate_waveform_thumbnail_from_file(file_path: Path, *, size: int = _THUMB_SIZE) -> bytes | None:
    """Generate a waveform thumbnail from an audio file on disk."""
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        return generate_waveform_thumbnail(audio_bytes, size=size)
    except Exception:
        return None


class AudioMediaType(MediaType):
    """Handles audio medias — file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtsearch.media.audio.embedder.AudioClapEmbedder`.
    """

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
        return "sounds"

    @property
    def tab_title(self) -> str:
        return "Sounds"

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
        "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
        "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
        "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
        "crying_baby", "sneezing", "clapping", "breathing", "coughing",
        "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",
        "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creep", "can_opening",
        "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",
        "helicopter", "chainsaw", "siren", "car_horn", "engine",
        "train", "church_bells", "airplane", "fireworks", "hand_saw",
    ]

    _GTZAN_CATEGORIES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    ]

    _SPEECH_COMMANDS_CATEGORIES = [
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
        "backward", "follow", "forward", "learn", "visual",
    ]

    _URBANSOUND8K_CATEGORIES = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
    ]

    @property
    def demo_datasets(self) -> list:
        from vtsearch.datasets.downloader import (  # noqa: PLC0415
            ESC50_DOWNLOAD_SIZE_MB,
            GTZAN_DOWNLOAD_SIZE_MB,
            SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
            URBANSOUND8K_DOWNLOAD_SIZE_MB,
        )

        cats = self._DEMO_CATEGORIES
        folder = DATA_DIR / "ESC-50-master" / "audio"
        return [
            DemoDataset(
                id="esc50_s", label="ESC-50 (S)",
                description="Real-world environmental recordings — animals, nature, cities, homes, and people.",
                categories=cats, source="esc50", required_folder=folder,
                slice_start=0, slice_end=7, download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="esc50_m", label="ESC-50 (M)",
                description="Real-world environmental recordings — animals, nature, cities, homes, and people.",
                categories=cats, source="esc50", required_folder=folder,
                slice_start=7, slice_end=20, download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="esc50_l", label="ESC-50 (L)",
                description="Real-world environmental recordings — animals, nature, cities, homes, and people.",
                categories=cats, source="esc50", required_folder=folder,
                slice_start=20, slice_end=40, download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="gtzan_a", label="GTZAN Music Genre (A)",
                description="30-second music excerpts, one per genre.",
                categories=self._GTZAN_CATEGORIES, source="gtzan",
                slice_start=0, slice_end=100, download_size_mb=GTZAN_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="speech_commands_v2_s", label="Speech Commands v2 (S)",
                description="One-second keyword utterances from crowd-sourced speakers.",
                categories=self._SPEECH_COMMANDS_CATEGORIES, source="speech_commands_v2",
                slice_start=0, slice_end=429, download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="speech_commands_v2_m", label="Speech Commands v2 (M)",
                description="One-second keyword utterances from crowd-sourced speakers.",
                categories=self._SPEECH_COMMANDS_CATEGORIES, source="speech_commands_v2",
                slice_start=429, slice_end=1287, download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="speech_commands_v2_l", label="Speech Commands v2 (L)",
                description="One-second keyword utterances from crowd-sourced speakers.",
                categories=self._SPEECH_COMMANDS_CATEGORIES, source="speech_commands_v2",
                slice_start=1287, slice_end=3000, download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="speech_commands_v2_a", label="Speech Commands v2 (A)",
                description="One-second keyword utterances from crowd-sourced speakers.",
                categories=self._SPEECH_COMMANDS_CATEGORIES, source="speech_commands_v2",
                slice_start=0, slice_end=3000, download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="urbansound8k_s", label="UrbanSound8K (S)",
                description="Real urban field recordings, pre-segmented into labeled sounds.",
                categories=self._URBANSOUND8K_CATEGORIES, source="urbansound8k",
                slice_start=0, slice_end=125, download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="urbansound8k_m", label="UrbanSound8K (M)",
                description="Real urban field recordings, pre-segmented into labeled sounds.",
                categories=self._URBANSOUND8K_CATEGORIES, source="urbansound8k",
                slice_start=125, slice_end=374, download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="urbansound8k_l", label="UrbanSound8K (L)",
                description="Real urban field recordings, pre-segmented into labeled sounds.",
                categories=self._URBANSOUND8K_CATEGORIES, source="urbansound8k",
                slice_start=374, slice_end=873, download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="urbansound8k_a", label="UrbanSound8K (A)",
                description="Real urban field recordings, pre-segmented into labeled sounds.",
                categories=self._URBANSOUND8K_CATEGORIES, source="urbansound8k",
                slice_start=0, slice_end=873, download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def load_demo_source(self, source, categories, slice_start, slice_end, clips, on_progress=None, embedder=None):
        import hashlib  # noqa: PLC0415

        if on_progress is None:
            from vtsearch.utils import update_progress

            on_progress = update_progress

        if embedder is None:
            from vtsearch.media import embedders_for_type

            avail = embedders_for_type(self.type_id)
            if not avail:
                raise ValueError(f"No embedders registered for media type {self.type_id!r}")
            embedder = avail[0]

        if source == "gtzan":
            from vtsearch.datasets.downloader import download_gtzan  # noqa: PLC0415
            from vtsearch.datasets.loader import load_audio_metadata_from_folders  # noqa: PLC0415

            genres_dir = download_gtzan(on_progress=on_progress)
            metadata = load_audio_metadata_from_folders(genres_dir, categories)

            by_cat: dict[str, list] = {}
            for _key, meta in sorted(metadata.items()):
                cat = meta["category"]
                by_cat.setdefault(cat, []).append((meta["path"], meta))

            audio_files: list = []
            for cat in categories:
                audio_files.extend(by_cat.get(cat, [])[slice_start:slice_end])

            audio_dir = genres_dir

        elif source == "speech_commands_v2":
            from vtsearch.datasets.downloader import download_speech_commands_v2  # noqa: PLC0415
            from vtsearch.datasets.loader import load_audio_metadata_from_folders  # noqa: PLC0415

            sc_dir = download_speech_commands_v2(on_progress=on_progress)
            metadata = load_audio_metadata_from_folders(sc_dir, categories)

            by_cat = {}
            for _key, meta in sorted(metadata.items()):
                cat = meta["category"]
                by_cat.setdefault(cat, []).append((meta["path"], meta))

            audio_files = []
            for cat in categories:
                audio_files.extend(by_cat.get(cat, [])[slice_start:slice_end])

            audio_dir = sc_dir

        elif source == "urbansound8k":
            from vtsearch.datasets.downloader import download_urbansound8k  # noqa: PLC0415
            from vtsearch.datasets.loader import load_urbansound8k_metadata  # noqa: PLC0415

            us8k_dir = download_urbansound8k(on_progress=on_progress)
            metadata = load_urbansound8k_metadata(us8k_dir)

            by_cat = {}
            for _fname, meta in sorted(metadata.items()):
                cat = meta["category"]
                if cat in categories:
                    by_cat.setdefault(cat, []).append((meta["path"], meta))

            audio_files = []
            for cat in categories:
                audio_files.extend(by_cat.get(cat, [])[slice_start:slice_end])

            audio_dir = us8k_dir / "audio"

        elif not source or source == "esc50":
            from vtsearch.datasets.downloader import download_esc50  # noqa: PLC0415
            from vtsearch.datasets.loader import load_esc50_metadata  # noqa: PLC0415

            audio_dir = download_esc50(on_progress=on_progress)
            esc_metadata = load_esc50_metadata(audio_dir.parent)

            by_cat = {}
            for audio_path in sorted(audio_dir.glob("*.wav")):
                if audio_path.name in esc_metadata:
                    cat = esc_metadata[audio_path.name]["category"]
                    if cat in categories:
                        by_cat.setdefault(cat, []).append((audio_path, esc_metadata[audio_path.name]))

            audio_files = []
            for cat in categories:
                audio_files.extend(by_cat.get(cat, [])[slice_start:slice_end])

        else:
            raise ValueError(f"Unsupported audio source: {source!r}")

        # Load models
        if getattr(embedder, "_model", None) is None:
            on_progress("loading", "Loading audio embedding model…", 0, 0)
            original_cb = embedder._on_progress
            embedder._on_progress = on_progress
            try:
                embedder.load_models()
            finally:
                embedder._on_progress = original_cb

        clip_id = max(clips.keys(), default=0) + 1
        total = len(audio_files)
        on_progress("embedding", f"Starting embedding for {total} audio files...", 0, total)
        demo_origin: dict = {"importer": "demo", "params": {}}

        for i, (audio_path, meta) in enumerate(audio_files):
            rel_name = f"{meta['category']}/{audio_path.name}"
            on_progress("embedding", f"Embedding {rel_name}", i + 1, total)
            embedding = embedder.embed_media(audio_path)
            if embedding is None:
                continue

            with open(audio_path, "rb") as f:
                wav_bytes = f.read()

            media_fields = self.load_media_data(audio_path)
            clips[clip_id] = {
                "id": clip_id,
                "type": self.type_id,
                "embedder": embedder.name,
                "duration": media_fields["duration"],
                "file_size": len(wav_bytes),
                "md5": hashlib.md5(wav_bytes).hexdigest(),
                "embedding": embedding,
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

    def load_media_data(self, file_path: Path) -> dict:
        import librosa  # noqa: PLC0415

        with open(file_path, "rb") as f:
            media_bytes = f.read()
        try:
            audio_data, sr = librosa.load(file_path, sr=None, mono=True)
            duration = len(audio_data) / sr
        except Exception:
            duration = 0.0
        thumbnail = generate_waveform_thumbnail(media_bytes)
        return {"media_bytes": media_bytes, "duration": duration, "thumbnail_bytes": thumbnail}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def image_response(self, media: dict) -> MediaResponse | None:
        """Return the waveform thumbnail as a PNG image, or *None*."""
        thumb = media.get("thumbnail_bytes")
        if thumb:
            return MediaResponse(
                data=thumb,
                mimetype="image/png",
                download_name=f"media_{media['id']}_waveform.png",
            )
        # Fallback: generate on the fly from media bytes
        raw = self._resolve_media_bytes(media)
        if raw:
            thumb = generate_waveform_thumbnail(raw)
            if thumb:
                media["thumbnail_bytes"] = thumb
                return MediaResponse(
                    data=thumb,
                    mimetype="image/png",
                    download_name=f"media_{media['id']}_waveform.png",
                )
        return None

    def media_response(self, media: dict) -> MediaResponse:
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype="audio/wav", download_name=f"media_{media['id']}.wav")
        return MediaResponse(
            data=data,
            mimetype="audio/wav",
            download_name=f"media_{media['id']}.wav",
        )
