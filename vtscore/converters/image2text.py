"""OCR an image to extract any embedded text."""

from __future__ import annotations

import io
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


def _run_paddleocr(media_bytes: bytes, filename: str, language: str) -> list | None:
    """Decode image bytes and run PaddleOCR, returning the raw per-region results."""
    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        print("Image2TextMediaConverter requires Pillow and numpy")
        return None

    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("Image2TextMediaConverter requires PaddleOCR: pip install paddleocr paddlepaddle")
        return None

    try:
        image = Image.open(io.BytesIO(media_bytes)).convert("RGB")
    except Exception as e:
        print(f"Image2TextMediaConverter: failed to open {filename}: {e}")
        return None

    try:
        model = PaddleOCR(use_angle_cls=True, lang=language, show_log=False)
        return model.ocr(np.array(image), cls=True)
    except Exception as e:
        print(f"Image2TextMediaConverter: PaddleOCR failed on {filename}: {e}")
        return None


def _extract_text_lines(results: list, threshold: float) -> list[str]:
    """Flatten PaddleOCR per-region output into a list of confident text lines."""
    lines: list[str] = []
    if not results:
        return lines
    for line_group in results:
        if not line_group:
            continue
        for line in line_group:
            try:
                _polygon, (text, conf) = line
            except (TypeError, ValueError):
                continue
            if conf is None or conf < threshold:
                continue
            text = (text or "").strip()
            if text:
                lines.append(text)
    return lines


class Image2TextMediaConverter(MediaConverter):
    """Run OCR over an image and emit the detected text as a single text media.

    Uses PaddleOCR (the same backend as the built-in OCR extractor processor),
    but flattens its per-region output into a single newline-joined string so
    the result composes cleanly with text embedders (E5, BGE, etc.) — letting
    you treat scanned pages, screenshots, infographics, and comics as text.

    User-configurable parameters
    ----------------------------
    ``language``
        PaddleOCR language code chosen from a curated drop-down (English,
        Chinese, French, German, Japanese, Korean, Russian, Arabic, …).
        Defaults to ``"en"``.
    ``threshold``
        Minimum per-region confidence in ``[0, 1]``.  Lower-confidence
        regions are dropped.  Defaults to ``"0.5"``.
    """

    display_name = "Image → Text (OCR)"
    description = "Extract text from images via OCR"
    fields = [
        PluginField(
            key="language",
            label="OCR language",
            field_type="select",
            description="PaddleOCR language code.",
            options=[
                "en",
                "ch",
                "chinese_cht",
                "fr",
                "german",
                "japan",
                "korean",
                "ru",
                "arabic",
                "cyrillic",
                "devanagari",
                "latin",
                "es",
                "pt",
                "it",
                "ta",
                "te",
            ],
            default="en",
            required=False,
        ),
        PluginField(
            key="threshold",
            label="Confidence threshold",
            field_type="number",
            description="Drop detected regions whose confidence is below this (0–1).",
            default="0.5",
            required=False,
            min="0",
            max="1",
            step="0.05",
        ),
    ]

    @property
    def source_type(self) -> str:
        return "image"

    @property
    def target_type(self) -> str:
        return "text"

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        language = str(self.get_param(params, "language") or "en")
        try:
            threshold = float(self.get_param(params, "threshold") or 0.5)
        except (TypeError, ValueError):
            threshold = 0.5

        filename = media.get("filename", "image.png")
        stem = Path(filename).stem
        media_bytes = _resolve_media_bytes(media)
        if not media_bytes:
            return []

        results = _run_paddleocr(media_bytes, filename, language)
        if results is None:
            return []

        lines = _extract_text_lines(results, threshold)
        if not lines:
            return []

        full_text = "\n".join(lines)
        return [
            {
                "filename": f"{stem}.txt",
                "media_string": full_text,
                "duration": 0,
                "word_count": len(full_text.split()),
                "character_count": len(full_text),
            }
        ]


CONVERTER = Image2TextMediaConverter()
