"""Render audio as a spectrogram image."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

from vtscore.converters.base import MediaConverter
from vtscore.plugins import PluginField


def _resolve_media_bytes(media: dict[str, Any]) -> bytes | None:
    """Read raw bytes from ``media_bytes`` or, failing that, ``media_path``."""
    media_bytes = media.get("media_bytes")
    if media_bytes is not None:
        return media_bytes
    media_path = media.get("media_path")
    if media_path:
        path = Path(media_path)
        if path.exists():
            return path.read_bytes()
    return None


def _load_audio_array(media_bytes: bytes, filename: str, duration: float | None, librosa) -> tuple[Any, int] | None:
    """Decode WAV bytes through librosa, optionally truncated to *duration* seconds."""
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".wav", delete=False) as tmp:
            tmp.write(media_bytes)
            tmp_path = tmp.name
        try:
            audio_data, sr = librosa.load(tmp_path, sr=None, mono=True, duration=duration)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"Audio2ImageMediaConverter: failed to load {filename}: {e}")
        return None

    if audio_data is None or len(audio_data) == 0:
        return None
    return audio_data, sr


def _compute_spectrogram(
    spec_type: str, audio_data, sr: int, n_mels: int, filename: str, librosa, np
) -> tuple[Any, str] | None:
    """Compute the mel or CQT spectrogram and return ``(spec_db, y_axis)``."""
    try:
        if spec_type == "cqt":
            cqt = np.abs(librosa.cqt(y=audio_data, sr=sr))
            return librosa.amplitude_to_db(cqt, ref=np.max), "cqt_note"
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels)
        return librosa.power_to_db(mel, ref=np.max), "mel"
    except Exception as e:
        print(f"Audio2ImageMediaConverter: failed to compute {spec_type} for {filename}: {e}")
        return None


def _render_spectrogram_png(spec_db, sr: int, y_axis: str, colormap: str, filename: str, librosa, plt) -> bytes | None:
    """Render a spectrogram array to a tight-bbox PNG using matplotlib."""
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
        return buf.getvalue()
    except Exception as e:
        print(f"Audio2ImageMediaConverter: failed to render {filename}: {e}")
        return None


def _get_png_dimensions(png_bytes: bytes, Image) -> tuple[int | None, int | None]:
    """Return ``(width, height)`` from PNG bytes, or ``(None, None)`` on failure."""
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            return img.width, img.height
    except Exception:
        return None, None


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
        Matplotlib colormap chosen from a curated drop-down (``"magma"``,
        ``"viridis"``, ``"inferno"``, ``"plasma"``, ``"cividis"``,
        ``"turbo"``, ``"jet"``, ``"hot"``, ``"cool"``, ``"coolwarm"``,
        ``"gray"``, ``"bone"``, ``"copper"``, ``"twilight"``).  Defaults
        to ``"magma"``.
    """

    display_name = "Audio → Image (spectrogram)"
    description = "Render audio as a mel-spectrogram or CQT image"
    summary_template = (
        "Render each audio file as a {spectrogram_type} spectrogram "
        "(first {time_window_s}s, {colormap})."
    )
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
            min="8",
            max="512",
            step="1",
        ),
        PluginField(
            key="time_window_s",
            label="Window (seconds)",
            field_type="number",
            description="Render at most this many seconds from the start. 0 = whole file.",
            default="30",
            required=False,
            min="0",
            step="0.5",
        ),
        PluginField(
            key="colormap",
            label="Colormap",
            field_type="select",
            description="Matplotlib colormap.",
            options=[
                "magma",
                "viridis",
                "inferno",
                "plasma",
                "cividis",
                "turbo",
                "jet",
                "hot",
                "cool",
                "coolwarm",
                "gray",
                "bone",
                "copper",
                "twilight",
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

    def _coerce_params(self, params: dict[str, Any] | None) -> tuple[str, int, float, str]:
        """Coerce raw params into ``(spec_type, n_mels, time_window_s, colormap)``."""
        spec_type = str(self.get_param(params, "spectrogram_type") or "mel").lower()
        if spec_type not in ("mel", "cqt"):
            spec_type = "mel"

        try:
            n_mels = int(self.get_param(params, "n_mels") or 128)
        except (TypeError, ValueError):
            n_mels = 128
        # Defensive clamp to the declared PluginField range — the upstream
        # plugin schema enforces this for any API-supplied params, but direct
        # callers of convert() (tests, ad-hoc scripts) skip that path.
        n_mels = max(8, min(512, n_mels))

        try:
            time_window_s = float(self.get_param(params, "time_window_s") or 0)
        except (TypeError, ValueError):
            time_window_s = 0.0
        time_window_s = max(0.0, time_window_s)

        colormap = str(self.get_param(params, "colormap") or "magma")
        return spec_type, n_mels, time_window_s, colormap

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        spec_type, n_mels, time_window_s, colormap = self._coerce_params(params)

        filename = media.get("filename", "audio.wav")
        stem = Path(filename).stem
        media_bytes = _resolve_media_bytes(media)
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

        duration = time_window_s if time_window_s > 0 else None
        audio = _load_audio_array(media_bytes, filename, duration, librosa)
        if audio is None:
            return []
        audio_data, sr = audio

        spec = _compute_spectrogram(spec_type, audio_data, sr, n_mels, filename, librosa, np)
        if spec is None:
            return []
        spec_db, y_axis = spec

        png_bytes = _render_spectrogram_png(spec_db, sr, y_axis, colormap, filename, librosa, plt)
        if png_bytes is None:
            return []

        width, height = _get_png_dimensions(png_bytes, Image)
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
