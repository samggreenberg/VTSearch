"""Image clippers — tile, crop, face-detect, or pass-through image media."""

from __future__ import annotations

import io
import math
from typing import Any, Optional

from vtscore.media.clipper import MediaClipper


class ImageDefaultClipper(MediaClipper):
    """Returns the image media unchanged."""

    @property
    def name(self) -> str:
        return "image_default"

    @property
    def display_name(self) -> str:
        return "None"

    @property
    def media_type(self) -> str:
        return "image"

    @property
    def description(self) -> str:
        return "Import each image as-is, without splitting."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [media]


class ImageTilingClipper(MediaClipper):
    """Tile an image into equidistant square crops.

    The tile size is the shorter dimension of the image (so an already-square
    image is returned unchanged).  Tiles are placed equidistantly along the
    longer axis so that the first tile is flush with one edge and the last
    tile is flush with the other.

    For example an 8.5 x 11 image (w < h) would yield two 8.5 x 8.5 tiles
    spaced to cover the full height.
    """

    @property
    def name(self) -> str:
        return "image_tiling"

    @property
    def media_type(self) -> str:
        return "image"

    @property
    def description(self) -> str:
        return "Tile each image into equidistant square crops along the longer axis."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        from PIL import Image  # noqa: PLC0415

        width = media.get("width")
        height = media.get("height")
        media_bytes = media.get("media_bytes")

        if width is None or height is None or media_bytes is None:
            return [media]

        tile_size = min(width, height)

        # Already square — return unchanged.
        if width == tile_size and height == tile_size:
            return [media]

        img = Image.open(io.BytesIO(media_bytes))

        if width >= height:
            # Landscape: tile along the x-axis.
            long_axis = width
            n_tiles = max(1, math.ceil(long_axis / tile_size))
            if n_tiles == 1:
                offsets = [0]
            else:
                offsets = [round(i * (long_axis - tile_size) / (n_tiles - 1)) for i in range(n_tiles)]
            boxes = [(x, 0, x + tile_size, tile_size) for x in offsets]
        else:
            # Portrait: tile along the y-axis.
            long_axis = height
            n_tiles = max(1, math.ceil(long_axis / tile_size))
            if n_tiles == 1:
                offsets = [0]
            else:
                offsets = [round(i * (long_axis - tile_size) / (n_tiles - 1)) for i in range(n_tiles)]
            boxes = [(0, y, tile_size, y + tile_size) for y in offsets]

        results: list[dict[str, Any]] = []
        fmt = img.format or "PNG"
        for idx, box in enumerate(boxes):
            cropped = img.crop(box)
            buf = io.BytesIO()
            cropped.save(buf, format=fmt)
            crop_bytes = buf.getvalue()
            tile = dict(media)
            tile["media_bytes"] = crop_bytes
            tile["width"] = tile_size
            tile["height"] = tile_size
            tile["file_size"] = len(crop_bytes)
            tile["clip_index"] = idx
            tile["clip_box"] = list(box)
            results.append(tile)
        return results


class ImageBboxClipper(MediaClipper):
    """Crop an image to a single user-specified bounding box.

    Unlike :class:`ImageTilingClipper`, which auto-tiles an image into many
    equally-spaced square crops, ``ImageBboxClipper`` returns exactly one
    crop bounded by the ``[x1, y1, x2, y2]`` box the caller provides
    (in pixel coordinates of the original image).  The intended use is
    user-driven cropping — e.g. picking a sub-region of an example image
    to drive a similarity search or a training example.

    The returned media dict carries the same ``clip_box`` / ``width`` /
    ``height`` fields as a tiling clip, so downstream code (embedding,
    learning, label export) treats the cropped result as a first-class clip.
    """

    def __init__(self, box: list[int] | tuple[int, int, int, int]) -> None:
        if box is None or len(box) != 4:
            raise ValueError("box must be a 4-tuple [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (int(v) for v in box)
        if x1 < 0 or y1 < 0:
            raise ValueError("box coordinates must be non-negative")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box must have positive width and height")
        self._box = (x1, y1, x2, y2)

    @property
    def name(self) -> str:
        return "image_bbox"

    @property
    def media_type(self) -> str:
        return "image"

    @property
    def description(self) -> str:
        return "Crop the image to a single user-specified bounding box."

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self._box

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        from PIL import Image  # noqa: PLC0415

        media_bytes = media.get("media_bytes")
        if media_bytes is None:
            return [media]

        img = Image.open(io.BytesIO(media_bytes))
        img_w, img_h = img.size

        x1 = max(0, min(self._box[0], img_w))
        y1 = max(0, min(self._box[1], img_h))
        x2 = max(x1, min(self._box[2], img_w))
        y2 = max(y1, min(self._box[3], img_h))
        if x2 <= x1 or y2 <= y1:
            return [media]

        cropped = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        fmt = img.format or "PNG"
        cropped.save(buf, format=fmt)
        crop_bytes = buf.getvalue()

        clip = dict(media)
        clip["media_bytes"] = crop_bytes
        clip["width"] = x2 - x1
        clip["height"] = y2 - y1
        clip["file_size"] = len(crop_bytes)
        clip["clip_index"] = 0
        clip["clip_box"] = [x1, y1, x2, y2]
        return [clip]

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "box",
                "label": "Bounding box [x1, y1, x2, y2]",
                "description": "Pixel coordinates of the crop in the original image.",
                "type": "string",
                "default": list(self._box),
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "ImageBboxClipper":
        box = params.get("box", self._box)
        return ImageBboxClipper(box)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["box"] = list(self._box)
        return d


class ImageObjectClipper(MediaClipper):
    """Crop one clip per detected object using a lightweight YOLO/RT-DETR.

    Each output clip is a crop around one detection's bounding box, with an
    optional padding margin to give downstream embedders a bit of context.
    Detections can be filtered by class and confidence, and capped at
    ``max_detections`` (sorted by confidence, highest first) to prevent a
    single image from exploding into hundreds of clips.

    Images with no surviving detections fall through unchanged so the load
    pipeline keeps them as the original (single-element) media; the
    ``MediaClipper.clip`` contract requires at least one element.

    The default ``model_id`` of ``"yolo11n.pt"`` matches
    :class:`~vtscore.media.image.extractor.ImageClassExtractor` so a single
    YOLO weight serves both detection-as-metadata and detection-as-clips.
    """

    def __init__(
        self,
        threshold: float = 0.25,
        class_filter: str = "",
        max_detections: int = 20,
        padding: float = 0.0,
        model_id: str = "yolo11n.pt",
    ) -> None:
        self._threshold = float(threshold)
        self._class_filter = str(class_filter or "").strip()
        self._max_detections = max(1, int(max_detections))
        self._padding = max(0.0, float(padding))
        self._model_id = str(model_id)
        self._model: Optional[Any] = None

    @property
    def name(self) -> str:
        return "image_object"

    @property
    def media_type(self) -> str:
        return "image"

    @property
    def description(self) -> str:
        return "Detect objects with YOLO/RT-DETR and crop one clip per detection."

    @property
    def summary_template(self) -> str:
        return (
            "Detect objects with {model_id} (confidence ≥ {threshold}, max "
            "{max_detections} per image) and crop one clip per detection."
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def class_filter(self) -> str:
        return self._class_filter

    @property
    def max_detections(self) -> int:
        return self._max_detections

    @property
    def padding(self) -> float:
        return self._padding

    @property
    def model_id(self) -> str:
        return self._model_id

    def _class_whitelist(self) -> set[str] | None:
        if not self._class_filter:
            return None
        return {c.strip() for c in self._class_filter.split(",") if c.strip()} or None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage] # noqa: PLC0415

        self._model = YOLO(self._model_id)

    def _collect_hits(
        self,
        results: Any,
        img_w: int,
        img_h: int,
    ) -> list[tuple[float, tuple[int, int, int, int]]]:
        whitelist = self._class_whitelist()
        hits: list[tuple[float, tuple[int, int, int, int]]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {}) or {}
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < self._threshold:
                    continue
                cls_id = int(boxes.cls[i])
                label = names.get(cls_id, str(cls_id))
                if whitelist is not None and label not in whitelist:
                    continue
                box = self._pad_and_clamp(boxes.xyxy[i].tolist(), img_w, img_h)
                if box is None:
                    continue
                hits.append((conf, box))
        return hits

    def _pad_and_clamp(self, raw: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = (float(v) for v in raw)
        if self._padding > 0:
            bw = x2 - x1
            bh = y2 - y1
            x1 -= bw * self._padding
            x2 += bw * self._padding
            y1 -= bh * self._padding
            y2 += bh * self._padding
        ix1 = max(0, int(math.floor(x1)))
        iy1 = max(0, int(math.floor(y1)))
        ix2 = min(img_w, int(math.ceil(x2)))
        iy2 = min(img_h, int(math.ceil(y2)))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return (ix1, iy1, ix2, iy2)

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        from PIL import Image  # noqa: PLC0415

        media_bytes = media.get("media_bytes")
        if media_bytes is None:
            return [media]

        img = Image.open(io.BytesIO(media_bytes)).convert("RGB")
        fmt = Image.open(io.BytesIO(media_bytes)).format or "PNG"
        img_w, img_h = img.size

        self._load_model()
        assert self._model is not None
        results = self._model(img, verbose=False)

        hits = self._collect_hits(results, img_w, img_h)
        hits.sort(key=lambda h: h[0], reverse=True)
        hits = hits[: self._max_detections]

        if not hits:
            return [media]

        clips: list[dict[str, Any]] = []
        for idx, (_conf, (cx1, cy1, cx2, cy2)) in enumerate(hits):
            cropped = img.crop((cx1, cy1, cx2, cy2))
            buf = io.BytesIO()
            cropped.save(buf, format=fmt)
            crop_bytes = buf.getvalue()
            clip = dict(media)
            clip["media_bytes"] = crop_bytes
            clip["width"] = cx2 - cx1
            clip["height"] = cy2 - cy1
            clip["file_size"] = len(crop_bytes)
            clip["clip_index"] = idx
            clip["clip_box"] = [cx1, cy1, cx2, cy2]
            clips.append(clip)
        return clips

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "threshold",
                "label": "Confidence threshold",
                "description": "Minimum YOLO confidence (0–1) for a detection to become a clip.",
                "type": "number",
                "default": self._threshold,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
            {
                "key": "class_filter",
                "label": "Class filter",
                "description": "Comma-separated YOLO class names to keep (e.g. 'person,car'). Empty = all classes.",
                "type": "string",
                "default": self._class_filter,
            },
            {
                "key": "max_detections",
                "label": "Max clips per image",
                "description": "Cap on clips per image; highest-confidence detections win.",
                "type": "number",
                "default": self._max_detections,
                "min": 1,
                "max": 200,
                "step": 1,
            },
            {
                "key": "padding",
                "label": "Box padding",
                "description": "Fractional margin added around each box (0.0 = tight, 0.1 = 10% wider).",
                "type": "number",
                "default": self._padding,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
            {
                "key": "model_id",
                "label": "YOLO model",
                "description": "ultralytics weight file (e.g. yolo11n.pt, yolo11s.pt, yolo11l.pt).",
                "type": "string",
                "default": self._model_id,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "ImageObjectClipper":
        return ImageObjectClipper(
            threshold=float(params.get("threshold", self._threshold)),
            class_filter=str(params.get("class_filter", self._class_filter)),
            max_detections=int(params.get("max_detections", self._max_detections)),
            padding=float(params.get("padding", self._padding)),
            model_id=str(params.get("model_id", self._model_id)),
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["threshold"] = self._threshold
        d["class_filter"] = self._class_filter
        d["max_detections"] = self._max_detections
        d["padding"] = self._padding
        d["model_id"] = self._model_id
        return d


class ImageFaceClipper(MediaClipper):
    """Detect faces in an image and emit one square-ish crop per face.

    Uses MediaPipe Face Detection to locate every face above the
    configured confidence threshold, then crops each face with optional
    padding so the embedder receives some context around the face.
    Detections smaller than ``min_size`` pixels (on either axis after
    padding) are dropped — these are usually noisy background hits.

    Images with **no** detected faces yield zero clips and are dropped
    from the dataset; that is the intended semantic for a face-only
    dataset. Pair this clipper with the ``face`` embedder for face-
    identity search, or with SigLIP/CLIP for generic appearance queries
    on the detected faces.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model_selection: int = 1,
        padding: float = 0.25,
        min_size: int = 32,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if model_selection not in (0, 1):
            raise ValueError("model_selection must be 0 (short-range) or 1 (full-range)")
        if padding < 0:
            raise ValueError("padding must be non-negative")
        if min_size < 1:
            raise ValueError("min_size must be a positive integer")
        self._threshold = float(threshold)
        self._model_selection = int(model_selection)
        self._padding = float(padding)
        self._min_size = int(min_size)
        self._detector: Optional[Any] = None

    @property
    def name(self) -> str:
        return "image_face"

    @property
    def media_type(self) -> str:
        return "image"

    @property
    def display_name(self) -> str:
        return "Face crops"

    @property
    def description(self) -> str:
        return (
            "Detect faces with MediaPipe and emit one crop per detected face; "
            "images with no detected faces are skipped."
        )

    @property
    def summary_template(self) -> str:
        return (
            "Detect faces with MediaPipe (confidence ≥ {threshold}, min size {min_size}px) and crop one clip per face."
        )

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "threshold",
                "label": "Min detection confidence",
                "description": "Skip detections below this confidence (0–1).",
                "type": "number",
                "default": self._threshold,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
            {
                "key": "padding",
                "label": "Crop padding (fraction of face size)",
                "description": "Expand each crop by this fraction on every side so the embedder gets some context.",
                "type": "number",
                "default": self._padding,
                "min": 0.0,
                "max": 2.0,
                "step": 0.05,
            },
            {
                "key": "min_size",
                "label": "Minimum face size (pixels)",
                "description": "Drop detections whose padded crop is smaller than this on either axis.",
                "type": "number",
                "default": self._min_size,
                "min": 1,
                "step": 1,
            },
            {
                "key": "model_selection",
                "label": "Detector range",
                "description": "0 = short-range (within ~2m), 1 = full-range (within ~5m).",
                "type": "number",
                "default": self._model_selection,
                "min": 0,
                "max": 1,
                "step": 1,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "ImageFaceClipper":
        return ImageFaceClipper(
            threshold=float(params.get("threshold", self._threshold)),
            model_selection=int(params.get("model_selection", self._model_selection)),
            padding=float(params.get("padding", self._padding)),
            min_size=int(params.get("min_size", self._min_size)),
        )

    def _load_detector(self) -> None:
        if self._detector is not None:
            return
        # MediaPipe is an opt-in dependency (declared in pyproject's DEP001
        # ignore-list); face_localizer.py uses the same pattern.
        import mediapipe as mp  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        self._detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=self._threshold,
            model_selection=self._model_selection,
        )

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: C901
        from PIL import Image  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        media_bytes = media.get("media_bytes")
        if media_bytes is None:
            return []
        try:
            img = Image.open(io.BytesIO(media_bytes)).convert("RGB")
        except Exception:
            return []
        img_w, img_h = img.size
        fmt = img.format or "PNG"

        try:
            self._load_detector()
            results = self._detector.process(np.array(img))  # type: ignore[union-attr]
        except Exception:
            return []
        detections = getattr(results, "detections", None) or []

        boxes: list[tuple[int, int, int, int, float]] = []
        for det in detections:
            try:
                conf = float(det.score[0])
            except (AttributeError, IndexError, TypeError):
                continue
            if conf < self._threshold:
                continue
            rel = det.location_data.relative_bounding_box
            fx1 = rel.xmin * img_w
            fy1 = rel.ymin * img_h
            fx2 = (rel.xmin + rel.width) * img_w
            fy2 = (rel.ymin + rel.height) * img_h
            pad_x = (fx2 - fx1) * self._padding
            pad_y = (fy2 - fy1) * self._padding
            x1 = max(0, int(round(fx1 - pad_x)))
            y1 = max(0, int(round(fy1 - pad_y)))
            x2 = min(img_w, int(round(fx2 + pad_x)))
            y2 = min(img_h, int(round(fy2 + pad_y)))
            if x2 - x1 < self._min_size or y2 - y1 < self._min_size:
                continue
            boxes.append((x1, y1, x2, y2, conf))

        # Highest-confidence face first so clip_index=0 is the "primary" face.
        boxes.sort(key=lambda b: b[4], reverse=True)

        results_list: list[dict[str, Any]] = []
        for idx, (x1, y1, x2, y2, _conf) in enumerate(boxes):
            cropped = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            cropped.save(buf, format=fmt)
            crop_bytes = buf.getvalue()
            clip = dict(media)
            clip["media_bytes"] = crop_bytes
            clip["width"] = x2 - x1
            clip["height"] = y2 - y1
            clip["file_size"] = len(crop_bytes)
            clip["clip_index"] = idx
            clip["clip_box"] = [x1, y1, x2, y2]
            # Drop parent-inherited md5 + embedding so the load pipeline's
            # fixup recomputes them from the cropped bytes even in the
            # single-face case (where is_real_clip is False).
            clip.pop("embedding", None)
            clip.pop("md5", None)
            results_list.append(clip)
        return results_list

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["threshold"] = self._threshold
        d["padding"] = self._padding
        d["min_size"] = self._min_size
        d["model_selection"] = self._model_selection
        return d
