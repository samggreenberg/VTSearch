"""Render audio as a spectrogram image."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

from vtsearch.converters.base import MediaConverter
from vtsearch.plugins import PluginField


class Audio2ImageMediaConverter(MediaConverter):
    """Render an audio file as a mel-spectrogram (or CQT) PNG image.

    Unlocks image embedders (SigLIP, DINOv3, etc.) over audio data — useful
    for cross-model ensembling and for visually spotting recurring spectral
    patterns.

    User-configurable parameters
    ----------------------------
    ``spectrogram_type``
        ``"mel"`` (default) or ``"cqt"``.
    ``n_mels``
        Number of mel bands for the mel-spectrogram.  Defaults to ``128``.
        Ignored for CQT.
    ``time_window_s``
        Maximum number of seconds of audio to render.  Longer recordings
        are truncated to the first window so the rendering cost stays
        bounded.  ``0`` or empty means "render the whole file".  Defaults
        to ``30``.
    ``colormap``
        Any matplotlib colormap name (e.g. ``"magma"``, ``"viridis"``,
        ``"inferno"``, ``"gray"``).  Defaults to ``"magma"``.
    """

    display_name = "Audio → Image (spectrogram)"
    converter_description = "Render audio as a mel-spectrogram or CQT image"
    fields = [
        PluginField(
            key="spectrogram_type",
            label="Spectrogram type",
            field_type="select",
            description="mel = mel-frequency power spectrogram; cqt = constant-Q transform.",
            options=["mel", "cqt"],
            default="mel",
            required=False,
        ),
        PluginField(
            key="n_mels",
            label="Mel bands",
            field_type="number",
            description="Number of mel bands (mel only).",
            default="128",
            required=False,
            integer=True,
            min_value=8,
            max_value=512,
            step=1,
        ),
        PluginField(
            key="time_window_s",
            label="Window (seconds)",
            field_type="number",
            description="Render at most this many seconds from the start. 0 = whole file.",
            default="30",
            required=False,
            min_value=0,
            step=1,
        ),
        PluginField(
            key="colormap",
            label="Colormap",
            field_type="select",
            description=(
                "Matplotlib colormap.  The dropdown lists common choices; any "
                "matplotlib colormap name is accepted when passed via CLI or JSON."
            ),
            options=[
                "magma",
                "viridis",
                "inferno",
                "plasma",
                "cividis",
                "gray",
                "hot",
                "cool",
                "jet",
                "coolwarm",
                "twilight",
                "hsv",
            ],
            default="magma",
            required=False,
        ),
    ]

    @property
    def source_type(self) -> str:
        return "audio"

    @property
    def target_type(self) -> str:
        return "image"

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        spec_type = str(self.get_param(params, "spectrogram_type") or "mel").lower()
        if spec_type not in ("mel", "cqt"):
            spec_type = "mel"

        try:
            n_mels = int(self.get_param(params, "n_mels") or 128)
        except (TypeError, ValueError):
            n_mels = 128
        if n_mels < 8:
            n_mels = 8

        try:
            time_window_s = float(self.get_param(params, "time_window_s") or 0)
        except (TypeError, ValueError):
            time_window_s = 0.0
        if time_window_s < 0:
            time_window_s = 0.0

        colormap = str(self.get_param(params, "colormap") or "magma")

        media_bytes = media.get("media_bytes")
        media_path = media.get("media_path")
        filename = media.get("filename", "audio.wav")
        stem = Path(filename).stem

        if media_bytes is None and media_path:
            path = Path(media_path)
            if path.exists():
                media_bytes = path.read_bytes()

        if not media_bytes:
            return []

        try:
            import librosa  # noqa: PLC0415
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            print("Audio2ImageMediaConverter requires librosa, matplotlib, and Pillow")
            return []

        try:
            duration = time_window_s if time_window_s > 0 else None
            with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".wav", delete=False) as tmp:
                tmp.write(media_bytes)
                tmp_path = tmp.name
            try:
                audio_data, sr = librosa.load(tmp_path, sr=None, mono=True, duration=duration)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Audio2ImageMediaConverter: failed to load {filename}: {e}")
            return []

        if audio_data is None or len(audio_data) == 0:
            return []

        try:
            if spec_type == "cqt":
                cqt = np.abs(librosa.cqt(y=audio_data, sr=sr))
                spec_db = librosa.amplitude_to_db(cqt, ref=np.max)
                y_axis = "cqt_note"
            else:
                mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels)
                spec_db = librosa.power_to_db(mel, ref=np.max)
                y_axis = "mel"
        except Exception as e:
            print(f"Audio2ImageMediaConverter: failed to compute {spec_type} for {filename}: {e}")
            return []

        try:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            librosa.display.specshow(
                spec_db,
                sr=sr,
                x_axis="time",
                y_axis=y_axis,
                ax=ax,
                cmap=colormap,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout(pad=0)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            png_bytes = buf.getvalue()
        except Exception as e:
            print(f"Audio2ImageMediaConverter: failed to render {filename}: {e}")
            return []

        try:
            with Image.open(io.BytesIO(png_bytes)) as img:
                width, height = img.width, img.height
        except Exception:
            width, height = None, None

        suffix = "spec_cqt" if spec_type == "cqt" else "spec_mel"
        return [
            {
                "filename": f"{stem}_{suffix}.png",
                "media_bytes": png_bytes,
                "duration": 0,
                "width": width,
                "height": height,
            }
        ]


CONVERTER = Audio2ImageMediaConverter()
