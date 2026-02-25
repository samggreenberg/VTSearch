"""Audio media type — CLAP embeddings, WAV/MP3/FLAC/OGG/M4A files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import CLAP_MODEL_ID, DATA_DIR, MODELS_CACHE_DIR, SAMPLE_RATE

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
    """Handles audio clips using the CLAP model (laion/clap-htsat-unfused).

    * Embeds audio files via CLAP's audio encoder + projection head.
    * Embeds text queries via CLAP's text encoder + projection head, so
      queries land in the same 512-dimensional space as audio embeddings.
    * Serves clips as ``audio/wav`` streams.
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
    # clips differ (disjoint slices of each category's 40 ESC-50 clips).
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

    # Categories for GTZAN Music Genre (10 genres, 100 clips each = 1000 total).
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
                description=(
                    "~350 clips across 50 sound categories — animals, nature,"
                    " urban, domestic, and human sounds from the ESC-50 collection."
                ),
                categories=cats,
                required_folder=folder,
                slice_start=0,
                slice_end=7,
            ),
            DemoDataset(
                id="esc50_m",
                label="ESC-50 (M)",
                description=(
                    "~650 clips across 50 sound categories — animals, nature,"
                    " urban, domestic, and human sounds from the ESC-50 collection."
                ),
                categories=cats,
                required_folder=folder,
                slice_start=7,
                slice_end=20,
            ),
            DemoDataset(
                id="esc50_l",
                label="ESC-50 (L)",
                description=(
                    "~1000 clips across 50 sound categories — animals, nature,"
                    " urban, domestic, and human sounds from the ESC-50 collection."
                ),
                categories=cats,
                required_folder=folder,
                slice_start=20,
                slice_end=40,
            ),
            DemoDataset(
                id="gtzan_a",
                label="GTZAN Music Genre (A)",
                description=(
                    "~1000 audio clips across 10 music genres — blues, classical,"
                    " country, disco, hip-hop, jazz, metal, pop, reggae, and rock."
                ),
                categories=self._GTZAN_CATEGORIES,
                source="gtzan",
                slice_start=0,
                slice_end=100,
            ),
            DemoDataset(
                id="speech_commands_v2_a",
                label="Speech Commands v2 (A)",
                description=(
                    "~105,000 one-second utterances of 35 keywords — digits,"
                    " directions, and common words from the Google Speech Commands"
                    " v2 dataset."
                ),
                categories=self._SPEECH_COMMANDS_CATEGORIES,
                source="speech_commands_v2",
                slice_start=0,
                slice_end=3000,
            ),
            DemoDataset(
                id="urbansound8k_a",
                label="UrbanSound8K (A)",
                description=(
                    "~8,732 urban sound clips across 10 classes — air conditioner,"
                    " car horn, children playing, dog bark, drilling, engine, gun shot,"
                    " jackhammer, siren, and street music."
                ),
                categories=self._URBANSOUND8K_CATEGORIES,
                source="urbansound8k",
                slice_start=0,
                slice_end=873,
            ),
        ]

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
            self._model = ClapModel.from_pretrained(CLAP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False)
        self._on_progress("loading", "Loading CLAP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID, cache_dir=cache_dir, token=False)

        # Warmup: import librosa (heavy — pulls in numba, scipy, etc.) and
        # run a single dummy forward pass so that the first real embed_media
        # call runs at the same speed as every subsequent one.  Without this
        # the first embedding stalls inside the progress-bar loop and skews
        # the ETA estimate for all remaining items.
        self._on_progress("loading", "Warming up audio pipeline…", 0, 0)
        import librosa  # noqa: PLC0415
        import torch  # noqa: PLC0415

        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        inputs = self._processor(
            audio=dummy_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,
            truncation=True,
        )
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

    def load_clip_data(self, file_path: Path) -> dict:
        import librosa  # noqa: PLC0415

        with open(file_path, "rb") as f:
            clip_bytes = f.read()
        try:
            audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            duration = len(audio_data) / sr
        except Exception:
            duration = 0.0
        return {"clip_bytes": clip_bytes, "duration": duration}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def clip_response(self, clip: dict) -> MediaResponse:
        data = self._resolve_clip_bytes(clip)
        if data is None:
            return MediaResponse(data=b"", mimetype="audio/wav", download_name=f"clip_{clip['id']}.wav")
        return MediaResponse(
            data=data,
            mimetype="audio/wav",
            download_name=f"clip_{clip['id']}.wav",
        )
