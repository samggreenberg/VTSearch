"""Image clippers — tile or pass-through image media."""

from __future__ import annotations

import io
import math
from typing import Any

from vtsearch.media.base import MediaClipper


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
