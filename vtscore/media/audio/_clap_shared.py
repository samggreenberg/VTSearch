"""Shared base class for the CLAP audio embedder variants.

VTSearch ships three CLAP audio embedders that share an identical
architecture (LAION CLAP: a 512-dim audio encoder + a text encoder) and
differ only in which pretrained checkpoint they load:

- ``clap``          - ``laion/clap-htsat-unfused`` (the default baseline).
- ``clap_general``  - ``laion/larger_clap_general`` (broader audio mix).
- ``clap_music``    - ``laion/larger_clap_music_and_speech`` (music/speech).

All three share this base; subclasses set :attr:`name`,
:attr:`display_name`, :attr:`model_id`, and (for the baseline) override
:attr:`is_default`.  The underscore-prefixed filename keeps it out of the
auto-discovery scan in :mod:`vtscore.media` (only ``embedder*.py`` files
are imported as plugins) - the variant modules import from here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vtscore.config import CLAP_SAMPLE_RATE

# CLAP embeds a single 10 s window (480000 samples at 48 kHz). The feature
# extractor only knows how to *truncate* longer audio via "rand_trunc" (a
# random crop) or "fusion" (4-channel, fusion checkpoints only) - and the
# unfused HTSAT checkpoints we load support neither cleanly. A random crop
# would also make re-embedding the same file non-deterministic, which breaks
# the origin -> embedding rederivation contract. So we truncate to exactly
# CLAP_MAX_SAMPLES ourselves (deterministic first window); the feature
# extractor then takes its equal-length branch and never crops.
CLAP_MAX_SAMPLES = 480000
from vtscore.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    hf_token,
    intercept_tqdm_progress,
    load_pretrained_local_first,
    timed_progress,
)


class _ClapBase(MediaEmbedder):
    """Backbone loader + audio / text forward passes for a CLAP checkpoint.

    Subclasses set :attr:`name`, :attr:`display_name`, and
    :attr:`model_id`.  :attr:`label` (a short human label like ``"CLAP
    Music"``) is derived from :attr:`display_name` for progress / log
    lines but may be overridden.
    """

    def __init__(self) -> None:
        super().__init__()
        # Typed ``Any``: transformers stubs miss several ``ClapProcessor.__call__``
        # kwargs we pass at runtime; runtime ``None`` checks guard the calls.
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # Identity - subclasses override name / display_name / model_id
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """Hugging Face repo id of the checkpoint this variant loads."""
        raise NotImplementedError

    @property
    def label(self) -> str:
        """Short human label for progress / log lines (e.g. ``"CLAP Music"``)."""
        raise NotImplementedError

    @property
    def media_type_id(self) -> str:
        return "audio"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 4):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers…", 2, 4):
            from transformers import ClapModel, ClapProcessor  # noqa: PLC0415

        with timed_progress(self._on_progress, "loading", "Importing librosa…", 3, 4):
            import librosa  # noqa: PLC0415

        with timed_progress(self._on_progress, "loading", "Importing soundfile…", 4, 4):
            import soundfile  # noqa: F401, PLC0415

        cache_dir = embedder_load_setup(self._on_progress, f"Loading {self.label} model weights…")
        with intercept_tqdm_progress(self._on_progress):
            self._model = load_pretrained_local_first(
                ClapModel.from_pretrained,
                self.model_id,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=hf_token(),
                on_progress=self._on_progress,
            )
        # Materialize any tensors left on the ``meta`` device.
        self._model = self._model.to("cpu")
        self._on_progress("loading", f"Loading {self.label} processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                ClapProcessor.from_pretrained, self.model_id, cache_dir=cache_dir, token=hf_token()
            )

        # Warmup: trigger the numba JIT for audio resampling, and run a
        # single dummy forward pass so that the first real embed_media call
        # runs at the same speed as every subsequent one.
        #
        # Trigger librosa/soxr resampling JIT by loading a tiny WAV at a
        # different sample rate.  Without this, the first embed_media()
        # call stalls for 10-30 s while numba compiles resampling kernels,
        # making the embedding progress bar appear frozen.
        self._on_progress("loading", f"Warming up {self.label} pipeline: resampling JIT…", 1, 3)
        import io  # noqa: PLC0415

        import soundfile as sf  # noqa: PLC0415

        _warmup_sr = 16000  # intentionally different from CLAP_SAMPLE_RATE
        _warmup_buf = io.BytesIO()
        sf.write(_warmup_buf, np.zeros(_warmup_sr, dtype=np.float32), _warmup_sr, format="WAV")
        _warmup_buf.seek(0)
        librosa.load(_warmup_buf, sr=CLAP_SAMPLE_RATE, mono=True)

        self._on_progress("loading", f"Warming up {self.label} pipeline: preprocessing…", 2, 3)
        dummy_audio = np.zeros(CLAP_SAMPLE_RATE, dtype=np.float32)
        inputs = self._processor(
            audio=dummy_audio,
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=CLAP_MAX_SAMPLES,
            truncation="rand_trunc",
        )
        self._on_progress("loading", f"Warming up {self.label} pipeline: running model…", 3, 3)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.audio_model(**inputs)
            self._model.audio_projection(outputs.pooler_output)

    # ------------------------------------------------------------------
    # Embedding
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

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        # Prefer in-memory bytes (set by clip re-embed) over a disk path so
        # the caller does not need to round-trip through a tempfile.
        audio_bytes = media.get("media_bytes")
        file_path: Optional[Path] = None
        if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
            audio_bytes = None
            path_str = media.get("media_path")
            if not path_str:
                return None
            file_path = Path(path_str)
        source_repr = file_path if file_path is not None else "<bytes>"
        try:
            import io  # noqa: PLC0415
            import librosa  # noqa: PLC0415
            import torch  # noqa: PLC0415

            if audio_bytes is not None:
                source: io.BytesIO | Path = io.BytesIO(bytes(audio_bytes))
            else:
                assert file_path is not None  # narrowed by the path_str check above
                source = file_path
            audio_data, _sr = librosa.load(source, sr=CLAP_SAMPLE_RATE, mono=True)
            # Deterministic truncation to a single 10 s window: drop everything
            # past CLAP_MAX_SAMPLES so the feature extractor never reaches its
            # random-crop ("rand_trunc") path. The explicit truncation string is
            # a valid defensive value; it is not actually exercised here.
            if audio_data.shape[0] > CLAP_MAX_SAMPLES:
                audio_data = audio_data[:CLAP_MAX_SAMPLES]
            inputs = self._processor(
                audio=audio_data,
                sampling_rate=CLAP_SAMPLE_RATE,
                return_tensors="pt",
                padding="max_length",
                max_length=CLAP_MAX_SAMPLES,
                truncation="rand_trunc",
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.audio_model(**inputs)
                embedding = self._model.audio_projection(outputs.pooler_output).detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", source_repr)
            return None

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._processor(text=[text], return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.text_model(**inputs)
                text_vec = self._model.text_projection(outputs.pooler_output).detach().cpu().numpy()[0]
            return text_vec
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for audio (%s)", self.label)
            return None

    # Internal helper used by loader.py bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor
