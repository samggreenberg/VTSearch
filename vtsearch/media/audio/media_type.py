"""Audio media type — WAV/MP3/FLAC/OGG/M4A files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from vtsearch.config import DATA_DIR, SAMPLE_RATE
from vtsearch.media.base import (
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    intercept_tqdm_progress,
)


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
        return "🔊"

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

    @property
    def legacy_bytes_keys(self) -> list[str]:
        return ["wav_bytes"]

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
            with intercept_tqdm_progress(on_progress):
                embedder.load_models()

        clip_id = 1
        total = len(audio_files)
        on_progress("embedding", f"Starting embedding for {total} audio files...", 0, total)
        demo_origin: dict = {"importer": "demo", "params": {}}

        for i, (audio_path, meta) in enumerate(audio_files):
            rel_name = f"{meta['category']}/{audio_path.name}"
            on_progress("embedding", f"Embedding {rel_name} ({i + 1}/{total})", i + 1, total)
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

    def load_media_data(self, file_path: Path) -> dict:
        import librosa  # noqa: PLC0415

        with open(file_path, "rb") as f:
            media_bytes = f.read()
        try:
            audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            duration = len(audio_data) / sr
        except Exception:
            duration = 0.0
        return {"media_bytes": media_bytes, "duration": duration}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def media_response(self, media: dict) -> MediaResponse:
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype="audio/wav", download_name=f"media_{media['id']}.wav")
        return MediaResponse(
            data=data,
            mimetype="audio/wav",
            download_name=f"media_{media['id']}.wav",
        )
