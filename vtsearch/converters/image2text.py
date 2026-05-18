"""OCR an image to extract any embedded text."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from vtsearch.converters.base import MediaConverter
from vtsearch.plugins import PluginField


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
    converter_description = "Extract text from images via OCR"
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

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:  # noqa: C901
        language = str(self.get_param(params, "language") or "en")
        try:
            threshold = float(self.get_param(params, "threshold") or 0.5)
        except (TypeError, ValueError):
            threshold = 0.5

        media_bytes = media.get("media_bytes")
        media_path = media.get("media_path")
        filename = media.get("filename", "image.png")
        stem = Path(filename).stem

        if media_bytes is None and media_path:
            path = Path(media_path)
            if path.exists():
                media_bytes = path.read_bytes()

        if not media_bytes:
            return []

        try:
            import numpy as np  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            print("Image2TextMediaConverter requires Pillow and numpy")
            return []

        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        except ImportError:
            print("Image2TextMediaConverter requires PaddleOCR: pip install paddleocr paddlepaddle")
            return []

        try:
            image = Image.open(io.BytesIO(media_bytes)).convert("RGB")
        except Exception as e:
            print(f"Image2TextMediaConverter: failed to open {filename}: {e}")
            return []

        try:
            model = PaddleOCR(use_angle_cls=True, lang=language, show_log=False)
            results = model.ocr(np.array(image), cls=True)
        except Exception as e:
            print(f"Image2TextMediaConverter: PaddleOCR failed on {filename}: {e}")
            return []

        lines: list[str] = []
        if results:
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
