"""Audio media type — CLAP embeddings, WAV/MP3/FLAC/OGG/M4A files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import CLAP_MODEL_ID, DATA_DIR, ESC50_DOWNLOAD_SIZE_MB, MODELS_CACHE_DIR, SAMPLE_RATE

if TYPE_CHECKING:
    from transformers import ClapModel, ClapProcessor
from vtsearch.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    intercept_tqdm_progress,
)


class AudioMediaType(MediaType):
    """Handles audio medias using the CLAP model (laion/clap-htsat-unfused).

    * Embeds audio files via CLAP's audio encoder + projection head.
    * Embeds text queries via CLAP's text encoder + projection head, so
      queries land in the same 512-dimensional space as audio embeddings.
    * Serves medias as ``audio/wav`` streams.
    """

    def __init__(self) -> None:
        self._model: Optional[ClapModel] = None
        self._processor: Optional[ClapProcessor] = None
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
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    # Shared categories for all S/M/L audio demo datasets.
    # All three sizes use all 50 ESC-50 categories; only the underlying
    # medias differ (disjoint slices of each category's 40 ESC-50 medias).
    _DEMO_CATEGORIES = [
        # Animals
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
        # Natural soundscapes
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
        # Human, non-speech
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
        # Interior / domestic
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
        # Exterior / urban
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

    # Categories for GTZAN Music Genre (10 genres, 100 medias each = 1000 total).
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

    # Categories for Google Speech Commands v2 (35 keywords).
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

    # Categories for UrbanSound8K (10 classes).
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

    @property
    def demo_datasets(self) -> list:
        cats = self._DEMO_CATEGORIES
        folder = DATA_DIR / "ESC-50-master" / "audio"
        return [
            DemoDataset(
                id="esc50_s",
                label="ESC-50 (S)",
                description="Real-world environmental recordings — animals, nature, cities, homes, and people.",
                categories=cats,
                required_folder=folder,
                slice_start=0,
                slice_end=7,
                download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="esc50_m",
                label="ESC-50 (M)",
                description="Real-world environmental recordings — animals, nature, cities, homes, and people.",
                categories=cats,
                required_folder=folder,
                slice_start=7,
                slice_end=20,
                download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="esc50_l",
                label="ESC-50 (L)",
                description="Real-world environmental recordings — animals, nature, cities, homes, and people.",
                categories=cats,
                required_folder=folder,
                slice_start=20,
                slice_end=40,
                download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="gtzan_a",
                label="GTZAN Music Genre (A)",
                description="30-second music excerpts, one per genre.",
                categories=self._GTZAN_CATEGORIES,
                source="gtzan",
                slice_start=0,
                slice_end=100,
                download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="speech_commands_v2_a",
                label="Speech Commands v2 (A)",
                description="One-second keyword utterances from crowd-sourced speakers.",
                categories=self._SPEECH_COMMANDS_CATEGORIES,
                source="speech_commands_v2",
                slice_start=0,
                slice_end=3000,
                download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="urbansound8k_a",
                label="UrbanSound8K (A)",
                description="Real urban field recordings, pre-segmented into labeled sounds.",
                categories=self._URBANSOUND8K_CATEGORIES,
                source="urbansound8k",
                slice_start=0,
                slice_end=873,
                download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def load_demo_source(self, source, categories, slice_start, slice_end, clips, on_progress=None):
        import hashlib  # noqa: PLC0415

        if on_progress is None:
            from vtsearch.utils import update_progress

            on_progress = update_progress

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
        if getattr(self, "_model", None) is None:
            on_progress("loading", "Loading audio embedding model…", 0, 0)
            self.load_models()

        clip_id = 1
        total = len(audio_files)
        on_progress("embedding", f"Starting embedding for {total} audio files...", 0, total)
        demo_origin: dict = {"importer": "demo", "params": {}}

        for i, (audio_path, meta) in enumerate(audio_files):
            on_progress(
                "embedding",
                f"Embedding {meta['category']}: {audio_path.name} ({i + 1}/{total})",
                i + 1,
                total,
            )
            embedding = self.embed_media(audio_path)
            if embedding is None:
                continue

            with open(audio_path, "rb") as f:
                wav_bytes = f.read()

            media_fields = self.load_media_data(audio_path)
            clips[clip_id] = {
                "id": clip_id,
                "type": self.type_id,
                "duration": media_fields["duration"],
                "file_size": len(wav_bytes),
                "md5": hashlib.md5(wav_bytes).hexdigest(),
                "embedding": embedding,
                "media_bytes": wav_bytes,
                "filename": audio_path.name,
                "category": meta["category"],
                "origin": demo_origin,
                "origin_name": audio_path.name,
            }
            clip_id += 1

        return str(audio_dir.absolute())

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    @property
    def description_wrappers(self) -> list[str]:
        return [
            "the sound of {text}",
            "a recording of {text}",
            "{text}",
            "audio of {text}",
            "the noise of {text}",
        ]

    def load_models(self) -> None:
        if self._model is not None:
            return
        import gc

        from transformers import ClapModel, ClapProcessor  # noqa: PLC0415

        gc.collect()
        cache_dir = str(MODELS_CACHE_DIR)
        self._on_progress("loading", "Loading CLAP model weights…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._model = ClapModel.from_pretrained(
                CLAP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False
            )
        self._on_progress("loading", "Loading CLAP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID, cache_dir=cache_dir, token=False)

        # Warmup: import librosa (heavy — pulls in numba, scipy, etc.) and
        # run a single dummy forward pass so that the first real embed_media
        # call runs at the same speed as every subsequent one.  Without this
        # the first embedding stalls inside the progress-bar loop and skews
        # the ETA estimate for all remaining items.
        #
        # Three explicit steps are reported (1/3, 2/3, 3/3) so the frontend
        # can show a determinate progress bar for the warmup phase.  The
        # status remains "loading" so the frontend knows not to include this
        # phase in its ETA calculation for the subsequent "embedding" phase.
        self._on_progress("loading", "Warming up audio pipeline: importing libraries…", 1, 3)
        import librosa  # noqa: F401, PLC0415 — lazy warmup import; pulls in numba, scipy, etc.
        import torch  # noqa: PLC0415

        self._on_progress("loading", "Warming up audio pipeline: preprocessing…", 2, 3)
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        inputs = self._processor(
            audio=dummy_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,
            truncation=True,
        )
        self._on_progress("loading", "Warming up audio pipeline: running model…", 3, 3)
        with torch.no_grad():
            outputs = self._model.audio_model(**inputs)
            self._model.audio_projection(outputs.pooler_output)

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import librosa  # noqa: PLC0415
            import torch  # noqa: PLC0415

            audio_data, _sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            inputs = self._processor(
                audio=audio_data,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding="max_length",
                max_length=480000,
                truncation=True,
            )
            with torch.no_grad():
                outputs = self._model.audio_model(**inputs)
                embedding = self._model.audio_projection(outputs.pooler_output).detach().cpu().numpy()
            return embedding[0]
        except Exception as e:
            print(f"Error embedding {file_path}: {e}")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._processor(text=[text], return_tensors="pt")
            with torch.no_grad():
                outputs = self._model.text_model(**inputs)
                text_vec = self._model.text_projection(outputs.pooler_output).detach().cpu().numpy()[0]
            return text_vec
        except Exception as e:
            print(f"Error embedding text query for audio: {e}")
            return None

    # internal helpers used by loader.py's get_clap_model() bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor

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
