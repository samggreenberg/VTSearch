"""Audio embedder - ParaSpeechCLAP (speech ↔ text style search).

ParaSpeechCLAP is a dual-encoder CLAP for *speech style*: it maps a spoken clip
and a rich textual style description into a shared 768-dim space, so you can
text-search a speech collection by voice/emotion/speaking-style ("a deep,
raspy voice", "an enthusiastic, fast-paced delivery", "a whispered, anxious
style").  This is the gap the other speech-capable audio embedders leave open:
``ast`` and ``whisper_encoder`` have no paired text tower
(:attr:`supports_text` is ``False``), so they cannot be driven by a text query.

The model is reconstructed from the released ``combined`` checkpoint
(``ajd12342/paraspeechclap-combined``, MIT) via the vendored architecture in
:mod:`vtscore.media.audio._paraspeechclap_model`: a WavLM-Large speech encoder
(``microsoft/wavlm-large``, MIT) + a Granite text encoder
(``ibm-granite/granite-embedding-278m-multilingual``, Apache-2.0) + two small
projection heads.  All three licences are permissive, so there is no
:attr:`license_notice`.

Heavier than the CLAP embedders (~600M params across the two towers), so it is
opt-in, not the default.  Clips are capped at
:data:`~vtscore.config.PARASPEECHCLAP_MAX_SAMPLES` to bound CPU memory; speech
style is a global property, so the leading window is representative.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vtscore.config import (
    PARASPEECHCLAP_CHECKPOINT_FILE,
    PARASPEECHCLAP_CHECKPOINT_REPO,
    PARASPEECHCLAP_EMBED_DIM,
    PARASPEECHCLAP_MAX_SAMPLES,
    PARASPEECHCLAP_SAMPLE_RATE,
    PARASPEECHCLAP_SPEECH_MODEL_ID,
    PARASPEECHCLAP_TEXT_MODEL_ID,
)
from vtscore.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    load_pretrained_local_first,
    timed_progress,
)


class AudioParaSpeechClapEmbedder(MediaEmbedder):
    """Embeds speech and text-style queries into a shared 768-dim space.

    * Speech clips → 768-dim vector via WavLM + audio projection head.
    * Text style descriptions → 768-dim vector via Granite + text projection head.
    """

    def __init__(self) -> None:
        super().__init__()
        # Typed ``Any``: the vendored ``CLAP`` module and transformers
        # processors have no stubs; runtime ``None`` checks guard every call.
        self._model: Any = None
        self._feature_extractor: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "paraspeechclap"

    @property
    def display_name(self) -> str:
        return "ParaSpeechCLAP (speech style)"

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
            from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor  # noqa: PLC0415

        with timed_progress(self._on_progress, "loading", "Importing librosa…", 3, 4):
            import librosa  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing soundfile…", 4, 4):
            import soundfile  # noqa: F401, PLC0415

        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        from vtscore.media.audio._paraspeechclap_model import CLAP  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading ParaSpeechCLAP encoders…")

        # Build the two towers.  ``load_pretrained_local_first`` is threaded
        # through the vendored builder so the WavLM/Granite downloads honour
        # the offline-first + transient-error-retry policy.
        with intercept_tqdm_progress(self._on_progress):
            model = CLAP(
                speech_name=PARASPEECHCLAP_SPEECH_MODEL_ID,
                text_name=PARASPEECHCLAP_TEXT_MODEL_ID,
                embedding_dim=PARASPEECHCLAP_EMBED_DIM,
                loader=lambda fn, *a, **k: load_pretrained_local_first(fn, *a, cache_dir=cache_dir, **k),
            )

        # Overlay the fine-tuned ParaSpeechCLAP weights (encoders + projection
        # heads) on top of the freshly-initialised towers.  ``strict=False``
        # tolerates harmless buffer mismatches (e.g. ``position_ids``).
        self._on_progress("loading", "Loading ParaSpeechCLAP checkpoint…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            checkpoint_path = load_pretrained_local_first(
                hf_hub_download,
                repo_id=PARASPEECHCLAP_CHECKPOINT_REPO,
                filename=PARASPEECHCLAP_CHECKPOINT_FILE,
                cache_dir=cache_dir,
            )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            logging.getLogger(__name__).debug(
                "ParaSpeechCLAP load_state_dict: %d missing, %d unexpected keys",
                len(result.missing_keys),
                len(result.unexpected_keys),
            )
        self._model = model.to("cpu")
        self._model.eval()

        self._on_progress("loading", "Loading ParaSpeechCLAP processors…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._feature_extractor = load_pretrained_local_first(
                Wav2Vec2FeatureExtractor.from_pretrained, PARASPEECHCLAP_SPEECH_MODEL_ID, cache_dir=cache_dir
            )
            self._tokenizer = load_pretrained_local_first(
                AutoTokenizer.from_pretrained, PARASPEECHCLAP_TEXT_MODEL_ID, cache_dir=cache_dir
            )

        # Warmup: run one dummy forward per tower so the first real embed call
        # runs at steady-state speed and the progress bar does not appear to stall.
        self._on_progress("loading", "Warming up ParaSpeechCLAP: speech tower…", 1, 2)
        dummy_audio = np.zeros(PARASPEECHCLAP_SAMPLE_RATE, dtype=np.float32)
        audio_inputs = self._feature_extractor(
            dummy_audio, sampling_rate=PARASPEECHCLAP_SAMPLE_RATE, return_tensors="pt", padding="do_not_pad"
        )
        with torch.no_grad():
            self._model.get_audio_embedding(audio_inputs.input_values, normalize=False)
        self._on_progress("loading", "Warming up ParaSpeechCLAP: text tower…", 2, 2)
        text_inputs = self._tokenizer(["warmup"], padding="longest", truncation=True, return_tensors="pt")
        with torch.no_grad():
            self._model.get_text_embedding(dict(text_inputs), normalize=False)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @property
    def description_wrappers(self) -> list[str]:
        # ParaSpeechCLAP was trained on style-description sentences; the first
        # wrapper matches the upstream classification template most closely.
        return [
            "A person is speaking in a {text} style.",
            "A person speaks in a {text} tone.",
            "A person with a {text} voice is speaking.",
            "{text}",
        ]

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._feature_extractor is None:
            return None
        # Prefer in-memory bytes (set by clip re-embed) over a disk path.
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
            import librosa  # noqa: PLC0415
            import torch  # noqa: PLC0415

            if audio_bytes is not None:
                source: io.BytesIO | Path = io.BytesIO(bytes(audio_bytes))
            else:
                assert file_path is not None  # narrowed by the path_str check above
                source = file_path
            audio_data, _sr = librosa.load(source, sr=PARASPEECHCLAP_SAMPLE_RATE, mono=True)
            if len(audio_data) > PARASPEECHCLAP_MAX_SAMPLES:
                audio_data = audio_data[:PARASPEECHCLAP_MAX_SAMPLES]
            inputs = self._feature_extractor(
                audio_data, sampling_rate=PARASPEECHCLAP_SAMPLE_RATE, return_tensors="pt", padding="do_not_pad"
            )
            device = next(self._model.parameters()).device
            input_values = inputs.input_values.to(device)
            with torch.no_grad():
                emb = self._model.get_audio_embedding(input_values, normalize=False).detach().cpu().numpy()
            return emb[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", source_repr)
            return None

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._tokenizer is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._tokenizer([text], padding="longest", truncation=True, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self._model.get_text_embedding(inputs, normalize=False).detach().cpu().numpy()[0]
            return emb
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for ParaSpeechCLAP")
            return None


EMBEDDER = AudioParaSpeechClapEmbedder()
