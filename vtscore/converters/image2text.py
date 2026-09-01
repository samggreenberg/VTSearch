"""OCR an image to extract any embedded text."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vtscore.converters.base import MediaConverter, resolve_media_bytes
from vtscore.plugins import PluginField

logger = logging.getLogger(__name__)


def _run_paddleocr(media_bytes: bytes, filename: str, language: str) -> list | None:
    """Decode image bytes and run PaddleOCR, returning the raw per-region results."""
    try:
        import numpy as np  # noqa: PLC0415

        from vtscore.media.image.decode import decode_bounded_rgb  # noqa: PLC0415
    except ImportError:
        logger.warning("image2text requires Pillow and numpy; producing no text")
        return None

    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError:
        logger.warning("image2text requires PaddleOCR: pip install paddleocr paddlepaddle")
        return None

    try:
        # Bounded decode: OCR reads a transcript, not pixel coordinates, so a
        # huge scan can be capped without changing what comes back.
        image, _scale = decode_bounded_rgb(media_bytes)
    except Exception:
        logger.error("Failed to open image %s", filename, exc_info=True)
        return None

    try:
        model = _make_paddleocr(PaddleOCR, language)
        return model.ocr(np.array(image), cls=True)
    except Exception:
        logger.error("PaddleOCR failed on %s", filename, exc_info=True)
        return None


def _make_paddleocr(paddle_ocr_cls: Any, language: str) -> Any:
    """Construct a PaddleOCR engine, requesting GPU when one is resolved.

    Routes through :func:`vtscore.config.resolve_device` so the engine honours
    ``VTSEARCH_DEVICE`` and the CUDA smoke-test fallback.  ``use_gpu`` only
    actually offloads when ``paddlepaddle-gpu`` is installed; with the CPU build
    it is a harmless no-op.  The kwarg was removed in PaddleOCR 3.x, so an
    unsupported-argument error falls back to the plain (CPU/default) constructor
    rather than breaking OCR entirely.
    """
    from vtscore.config import resolve_device  # noqa: PLC0415

    kwargs = {"use_angle_cls": True, "lang": language, "show_log": False}
    if resolve_device().startswith("cuda"):
        try:
            return paddle_ocr_cls(**kwargs, use_gpu=True)
        except (TypeError, ValueError):
            pass  # PaddleOCR build without a use_gpu kwarg; fall back to default.
    return paddle_ocr_cls(**kwargs)


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
    the result composes cleanly with text embedders (E5, BGE, etc.) - letting
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
    summary_template = "Run PaddleOCR ({language}) on each image; drop regions below confidence {threshold}."
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
        media_bytes = resolve_media_bytes(media)
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
