"""Synthetic image generation — smileys and colored shapes.

Each image is rendered with PIL and saved as PNG. The dataset cycles through
two "ideas" so embedders have something semantic to distinguish:

- **Smileys**: a circular face on a colored background, with one of four
  emotions (happy / sad / neutral / angry), random face/skin colour, size,
  and position.
- **Shapes**: 1–5 coloured shapes (circle / square / triangle) on a plain
  background, each at a random position and size.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

CANVAS_SIZE = 256

_BG_COLORS = [
    (240, 240, 240),
    (30, 30, 40),
    (200, 220, 255),
    (255, 230, 200),
    (210, 255, 210),
    (255, 210, 230),
    (255, 255, 200),
]

_FACE_COLORS = [
    (255, 220, 100),
    (255, 200, 150),
    (200, 230, 255),
    (180, 255, 180),
    (255, 180, 220),
    (220, 200, 255),
]

_SHAPE_COLORS = [
    (220, 50, 50),
    (50, 180, 50),
    (50, 80, 220),
    (220, 180, 30),
    (160, 60, 200),
    (240, 120, 40),
    (40, 200, 200),
]

_EMOTIONS = ["happy", "sad", "neutral", "angry"]
_SHAPE_KINDS = ["circle", "square", "triangle"]


def _draw_smiley(rng: np.random.Generator, draw, size: int) -> tuple[str, dict]:
    """Render one smiley face. Returns (idea, label dict)."""
    bg = _BG_COLORS[int(rng.integers(0, len(_BG_COLORS)))]
    face_color = _FACE_COLORS[int(rng.integers(0, len(_FACE_COLORS)))]
    emotion = _EMOTIONS[int(rng.integers(0, len(_EMOTIONS)))]

    radius = int(rng.integers(size // 5, size // 3))
    cx = int(rng.integers(radius + 10, size - radius - 10))
    cy = int(rng.integers(radius + 10, size - radius - 10))

    draw.rectangle([0, 0, size, size], fill=bg)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=face_color, outline=(20, 20, 20), width=2)

    eye_dx = radius // 2
    eye_dy = radius // 3
    eye_r = max(2, radius // 8)
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        ey = cy - eye_dy
        draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=(20, 20, 20))

    if emotion == "angry":
        for sign in (-1, 1):
            ex = cx + sign * eye_dx
            ey = cy - eye_dy
            draw.line(
                [ex - eye_r * 2, ey - eye_r * 2, ex + eye_r * 2, ey - eye_r],
                fill=(20, 20, 20),
                width=3,
            )

    mouth_w = int(radius * 0.9)
    mouth_h = int(radius * 0.6)
    mx0, mx1 = cx - mouth_w // 2, cx + mouth_w // 2
    my = cy + radius // 3
    if emotion == "happy":
        draw.arc([mx0, my - mouth_h // 2, mx1, my + mouth_h // 2], start=0, end=180, fill=(20, 20, 20), width=3)
    elif emotion == "sad":
        draw.arc([mx0, my - mouth_h // 4, mx1, my + mouth_h], start=180, end=360, fill=(20, 20, 20), width=3)
    elif emotion == "angry":
        draw.line([mx0, my, mx1, my], fill=(20, 20, 20), width=3)
    else:
        draw.line([mx0, my, mx1, my], fill=(20, 20, 20), width=3)

    return "smiley", {"emotion": emotion, "face_color": face_color}


def _draw_shapes(rng: np.random.Generator, draw, size: int) -> tuple[str, dict]:
    """Render 1–5 colored shapes on a plain background."""
    bg = _BG_COLORS[int(rng.integers(0, len(_BG_COLORS)))]
    draw.rectangle([0, 0, size, size], fill=bg)

    n = int(rng.integers(1, 6))
    kinds: list[str] = []
    for _ in range(n):
        kind = _SHAPE_KINDS[int(rng.integers(0, len(_SHAPE_KINDS)))]
        color = _SHAPE_COLORS[int(rng.integers(0, len(_SHAPE_COLORS)))]
        radius = int(rng.integers(size // 12, size // 5))
        cx = int(rng.integers(radius + 5, size - radius - 5))
        cy = int(rng.integers(radius + 5, size - radius - 5))
        if kind == "circle":
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
        elif kind == "square":
            draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
        else:
            pts = [
                (cx, cy - radius),
                (cx - radius, cy + radius),
                (cx + radius, cy + radius),
            ]
            angle = float(rng.uniform(0, 2 * math.pi))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            pts = [(cx + (x - cx) * cos_a - (y - cy) * sin_a, cy + (x - cx) * sin_a + (y - cy) * cos_a) for x, y in pts]
            draw.polygon(pts, fill=color)
        kinds.append(kind)

    return "shapes", {"count": n, "kinds": kinds}


_GENERATORS = [_draw_smiley, _draw_shapes]


def generate_image_dataset(output_dir: Path, count: int, seed: int = 42) -> list[Path]:
    """Generate ``count`` synthetic images into ``output_dir``.

    Existing files matching the deterministic naming scheme are kept (the
    importer caches its output dir across reloads). Returns the list of
    image paths in generation order.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    width = max(4, len(str(count)))
    for i in range(count):
        gen = _GENERATORS[i % len(_GENERATORS)]
        idea = "smiley" if gen is _draw_smiley else "shapes"
        path = output_dir / f"{idea}_{i:0{width}d}.png"
        paths.append(path)
        if path.exists():
            continue
        rng = np.random.default_rng(seed + i)
        img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        gen(rng, draw, CANVAS_SIZE)
        img.save(path, format="PNG")
    return paths
