"""Image clippers — tile or pass-through image media."""

from __future__ import annotations

import io
import math
from typing import Any

from vtsearch.media.clipper import MediaClipper


class ImageDefaultClipper(MediaClipper):
    """Returns the image media unchanged."""

    @property
    def name(self) -> str:
        return "image_default"

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
