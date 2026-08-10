"""Audio embedder - BEATs (self-supervised audio-event encoder).

BEATs ("Audio Pre-Training with Acoustic Tokenizers", Chen et al. 2022) is
Microsoft's iterative self-supervised audio encoder and the long-standing
state of the art on AudioSet tagging.  It is the strongest *audio-only*
representation VTSearch ships: like :mod:`~vtscore.media.audio.embedder_ast`
it has no paired text tower, so :attr:`supports_text` is ``False`` and the UI
hides text-search affordances for datasets embedded with it - but where AST is
a supervised AudioSet classifier whose features are shaped by its 527-label
head, BEATs' features come from masked-prediction pre-training over acoustic
tokens, which transfers noticeably better to categories the label set never
covered.

Reach for it over ``clap`` when the search is driven by voted examples rather
than a text query, and over ``ast`` whenever quality matters more than the
extra ~90M-parameter forward pass.

There is no ``transformers`` implementation, so the architecture is vendored
in :mod:`vtscore.media.audio._beats_model` and the released checkpoint is
overlaid onto it.  Both the code and the weights are MIT-licensed, so there is
no :attr:`license_notice`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vtscore.config import (
    BEATS_CHECKPOINT_FILE,
    BEATS_CHECKPOINT_REPO,
    BEATS_FBANK_MEAN,
    BEATS_FBANK_STD,
    BEATS_MAX_SAMPLES,
    BEATS_MIN_SAMPLES,
    BEATS_SAMPLE_RATE,
)
from vtscore.media.embedder import (
    IMPORT_MODULE_ESTIMATES,
    MediaEmbedder,
    embedder_load_setup,
    hf_token,
    intercept_tqdm_progress,
    load_pretrained_local_first,
    timed_progress,
    to_compute_device,
)


class AudioBEATsEmbedder(MediaEmbedder):
    """Embeds audio files using the BEATs iter3+ AudioSet-2M encoder.

    * Audio files → 768-dimensional vectors, mean-pooled over the encoder's
      patch tokens.
    * No text encoder; :meth:`embed_text` returns ``None``.
    """

    def __init__(self) -> None:
        super().__init__()
        # Typed ``Any``: the vendored ``BEATs`` module has no stubs, and the
        # runtime ``None`` checks below guard every call.
        self._model: Any = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "beats"

    @property
    def display_name(self) -> str:
        return "BEATs (audio events)"

    @property
    def model_id(self) -> str:
        return BEATS_CHECKPOINT_REPO

    @property
    def media_type_id(self) -> str:
        return "audio"

    @property
    def supports_text(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(
            self._on_progress, "loading", "Importing torch…", est_modules=IMPORT_MODULE_ESTIMATES["torch"]
        ):
            import torch  # noqa: PLC0415

        with timed_progress(
            self._on_progress, "loading", "Importing soundfile…", est_modules=IMPORT_MODULE_ESTIMATES["soundfile"]
        ):
            import soundfile  # noqa: F401, PLC0415

        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        from vtscore.media.audio._beats_model import BEATs  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading BEATs checkpoint…")
        with intercept_tqdm_progress(self._on_progress):
            checkpoint_path = load_pretrained_local_first(
                hf_hub_download,
                repo_id=BEATS_CHECKPOINT_REPO,
                filename=BEATS_CHECKPOINT_FILE,
                cache_dir=cache_dir,
                token=hf_token(),
            )

        # The released ``.pt`` carries the architecture config alongside the
        # weights, so the encoder is built from the checkpoint's own ``cfg``
        # rather than from constants that could drift away from it.
        self._on_progress("loading", "Building BEATs encoder…", 0, 0)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = BEATs(checkpoint["cfg"])
        model.load_state_dict(checkpoint["model"], strict=True)
        self._model = to_compute_device(model)
        self._model.eval()

        # Warmup: resampling JIT + one dummy forward, so the first real
        # embed_media call runs at steady-state speed instead of appearing to
        # stall the progress bar.
        self._on_progress("loading", "Warming up BEATs: resampling JIT…", 1, 2)
        import io  # noqa: PLC0415

        import soundfile as sf  # noqa: PLC0415

        from vtscore.media.audio.decode import decode_audio  # noqa: PLC0415

        _warmup_sr = 22050  # intentionally different from BEATS_SAMPLE_RATE
        _warmup_buf = io.BytesIO()
        sf.write(_warmup_buf, np.zeros(_warmup_sr, dtype=np.float32), _warmup_sr, format="WAV")
        _warmup_buf.seek(0)
        decode_audio(_warmup_buf, sr=BEATS_SAMPLE_RATE, mono=True)

        self._on_progress("loading", "Warming up BEATs: running model…", 2, 2)
        dummy = torch.zeros(BEATS_SAMPLE_RATE, device=next(self._model.parameters()).device)
        with torch.no_grad():
            self._model.extract_features(dummy, BEATS_FBANK_MEAN, BEATS_FBANK_STD)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        # Prefer in-memory bytes (set by clip re-embed) over a disk path so the
        # caller does not need to round-trip through a tempfile.
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
            import torch  # noqa: PLC0415

            from vtscore.media.audio.decode import decode_audio  # noqa: PLC0415

            if audio_bytes is not None:
                source: bytes | Path = bytes(audio_bytes)
            else:
                assert file_path is not None  # narrowed by the path_str check above
                source = file_path
            audio_data, _sr = decode_audio(source, sr=BEATS_SAMPLE_RATE, mono=True)
            # Deterministic leading window, mirroring the CLAP embedders: a
            # random crop would break the origin -> embedding rederivation
            # contract that the no-persisted-vectors design relies on.
            if audio_data.shape[0] > BEATS_MAX_SAMPLES:
                audio_data = audio_data[:BEATS_MAX_SAMPLES]
            # A clip shorter than one 16-frame patch row yields zero tokens, so
            # pad up to a floor rather than failing on very short media.
            if audio_data.shape[0] < BEATS_MIN_SAMPLES:
                audio_data = np.pad(audio_data, (0, BEATS_MIN_SAMPLES - audio_data.shape[0]))
            device = next(self._model.parameters()).device
            waveform = torch.from_numpy(np.ascontiguousarray(audio_data, dtype=np.float32)).to(device)
            with torch.no_grad():
                features = self._model.extract_features(waveform, BEATS_FBANK_MEAN, BEATS_FBANK_STD)
                embedding = features.mean(dim=1).detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", source_repr)
            return None


EMBEDDER = AudioBEATsEmbedder()
