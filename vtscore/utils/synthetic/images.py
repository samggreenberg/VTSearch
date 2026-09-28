"""Synthetic image generation - a small world of cartoon smiley faces.

Each picture is drawn with PIL (at twice its size, then scaled down, which is
how shapes that PIL draws without anti-aliasing get smooth edges) and saved as
a PNG. The set mixes three kinds of picture, so there is something to learn
beyond "which of two classes is this":

- **face** - one round cartoon face on a background. Seven face colours
  (yellow the commonest, orange the nearest miss) and seven expressions: three
  smiling (smile / grin / wink) and four not (frown / flat / surprised /
  angry).
- **shapes** - one to five circles, squares, triangles and stars in the same
  colours; a lone yellow disc is a face with nothing on it.
- **scene** - three to six small faces and shapes scattered over one
  background, where what you are after is only a *part* of the picture.

Backgrounds are plain, polka-dotted, striped, checked or a gradient, and can
take any of the colours - yellow included, so a red smiley on yellow polka
dots turns up as well as a yellow smiley on white.

That mix is what makes "the yellow smiley faces" a useful thing to hunt for
(the user guide's worked example does exactly that): a text query finds the
yellow faces easily and is less sure which of them are smiling, and a yellow
disc, an orange smiley or a yellow background all pull on it.

:func:`describe_image_dataset` returns what every picture holds without
drawing anything, so a caller can know the ground truth of a generated set.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vtscore.concurrency.progress import ProgressCallback

#: Side of every generated PNG, in pixels.
CANVAS_SIZE = 512

#: Pictures are drawn this many times larger and then scaled down to
#: :data:`CANVAS_SIZE`, which anti-aliases every edge.
_SUPERSAMPLE = 2

#: The kind of each picture, repeating: six faces, two shape piles and two
#: scenes in every ten. A fixed cycle rather than a draw, so even a handful of
#: pictures holds every kind.
_KIND_CYCLE = ("face", "face", "shapes", "face", "scene", "face", "face", "shapes", "face", "scene")

#: Expressions that count as a smile.
SMILING_EXPRESSIONS = frozenset({"smile", "grin", "wink"})

#: Fill colours for faces and shapes. Each drawing nudges its own a little
#: (:func:`_tint`), so a pile holds a spread of yellows rather than one swatch.
_COLORS: dict[str, tuple[int, int, int]] = {
    "yellow": (255, 210, 40),
    "orange": (255, 140, 30),
    "red": (225, 55, 55),
    "green": (75, 180, 75),
    "blue": (60, 125, 225),
    "pink": (255, 130, 180),
    "purple": (150, 95, 210),
}

#: Background-only colours: pale and neutral, so most faces sit on something
#: quiet and the saturated backgrounds stay the exception.
_BACKGROUND_ONLY: dict[str, tuple[int, int, int]] = {
    "white": (248, 248, 245),
    "cream": (255, 243, 215),
    "sky": (195, 225, 255),
    "mint": (205, 242, 218),
    "blush": (255, 222, 228),
    "grey": (205, 208, 214),
    "navy": (35, 42, 78),
}
_ALL_COLORS = {**_COLORS, **_BACKGROUND_ONLY}

_FACE_COLOR_WEIGHTS = {
    "yellow": 0.40,
    "orange": 0.14,
    "red": 0.10,
    "green": 0.09,
    "blue": 0.09,
    "pink": 0.09,
    "purple": 0.09,
}
_EXPRESSION_WEIGHTS = {
    "smile": 0.30,
    "grin": 0.15,
    "wink": 0.08,
    "frown": 0.17,
    "flat": 0.12,
    "surprised": 0.10,
    "angry": 0.08,
}
_SHAPE_WEIGHTS = {"circle": 1.0, "square": 1.0, "triangle": 1.0, "star": 1.0}
_SHAPE_COLOR_WEIGHTS = {"yellow": 2.0, "orange": 1.0, "red": 1.0, "green": 1.0, "blue": 1.0, "pink": 1.0, "purple": 1.0}
_BACKGROUND_STYLE_WEIGHTS = {"plain": 0.35, "dots": 0.20, "stripes": 0.15, "checks": 0.10, "gradient": 0.20}
#: The colour a background is mostly made of.
_BACKGROUND_BASE_WEIGHTS = {
    "white": 3.0,
    "cream": 2.0,
    "sky": 2.0,
    "mint": 2.0,
    "blush": 1.5,
    "grey": 1.0,
    "navy": 1.0,
    "yellow": 1.0,
    "orange": 0.5,
    "red": 0.5,
    "green": 0.5,
    "blue": 0.5,
    "pink": 0.5,
    "purple": 0.5,
}
#: The second colour of a patterned background: its dots, stripes, checks or
#: the far end of its gradient. Yellow is weighted up on purpose.
_BACKGROUND_ACCENT_WEIGHTS = {
    "yellow": 3.0,
    "white": 1.5,
    "orange": 1.0,
    "red": 1.0,
    "green": 1.0,
    "blue": 1.0,
    "pink": 1.0,
    "purple": 1.0,
}

#: Colour of eyes, mouths and brows.
_INK = (45, 32, 25)
#: Colour of a grin's tongue.
_TONGUE = (235, 95, 105)


# ---------------------------------------------------------------------------
# Planning: what each picture holds, as plain JSON-able data
# ---------------------------------------------------------------------------


def _pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights)
    p = np.array([weights[n] for n in names], dtype=float)
    return names[int(rng.choice(len(names), p=p / p.sum()))]


def _tint(rng: np.random.Generator, rgb: tuple[int, int, int], spread: int = 14) -> list[int]:
    """*rgb* nudged by up to *spread* per channel."""
    return [int(np.clip(c + int(rng.integers(-spread, spread + 1)), 0, 255)) for c in rgb]


def _box(cx: float, cy: float, r: float) -> list[float]:
    """The square round a circle, as canvas fractions ``[x0, y0, x1, y1]``."""
    return [round(max(0.0, cx - r), 4), round(max(0.0, cy - r), 4), round(min(1.0, cx + r), 4), round(min(1.0, cy + r), 4)]


def _plan_background(rng: np.random.Generator, avoid: str = "") -> dict[str, Any]:
    """A background whose base colour is not *avoid* (the lone subject's)."""
    style = _pick(rng, _BACKGROUND_STYLE_WEIGHTS)
    base = _pick(rng, {k: v for k, v in _BACKGROUND_BASE_WEIGHTS.items() if k != avoid})
    colors = [base]
    background: dict[str, Any] = {"style": style, "colors": colors}
    if style != "plain":
        colors.append(_pick(rng, {k: v for k, v in _BACKGROUND_ACCENT_WEIGHTS.items() if k != base}))
    if style == "dots":
        background["spacing"] = round(float(rng.uniform(0.09, 0.18)), 4)
        background["dot"] = round(float(rng.uniform(0.18, 0.3)), 4)
    elif style == "stripes":
        background["spacing"] = round(float(rng.uniform(0.07, 0.15)), 4)
        background["angle"] = int(rng.choice([0, 45, 90, 135]))
    elif style == "checks":
        background["spacing"] = round(float(rng.uniform(0.08, 0.16)), 4)
    return background


def _plan_face(rng: np.random.Generator, cx: float, cy: float, r: float) -> dict[str, Any]:
    color = _pick(rng, _FACE_COLOR_WEIGHTS)
    expression = _pick(rng, _EXPRESSION_WEIGHTS)
    smiling = expression in SMILING_EXPRESSIONS
    return {
        "shape": "face",
        "color": color,
        "expression": expression,
        "smiling": smiling,
        "box": _box(cx, cy, r),
        "rgb": _tint(rng, _COLORS[color]),
        "cheeks": bool(smiling and rng.random() < 0.3),
        "rotation": round(float(rng.uniform(-14, 14)), 2),
        "center": [round(cx, 4), round(cy, 4)],
        "radius": round(r, 4),
    }


def _plan_shape(rng: np.random.Generator, cx: float, cy: float, r: float) -> dict[str, Any]:
    shape = _pick(rng, _SHAPE_WEIGHTS)
    color = _pick(rng, _SHAPE_COLOR_WEIGHTS)
    return {
        "shape": shape,
        "color": color,
        "box": _box(cx, cy, r),
        "rgb": _tint(rng, _COLORS[color]),
        "rotation": round(float(rng.uniform(0, 360)), 2) if shape != "circle" else 0.0,
        "center": [round(cx, 4), round(cy, 4)],
        "radius": round(r, 4),
    }


def _place(rng: np.random.Generator, n: int, r_lo: float, r_hi: float) -> list[tuple[float, float, float]]:
    """Up to *n* non-overlapping circles ``(cx, cy, r)`` inside the canvas."""
    placed: list[tuple[float, float, float]] = []
    for _ in range(n):
        for _attempt in range(200):
            r = float(rng.uniform(r_lo, r_hi))
            cx = float(rng.uniform(r + 0.03, 1 - r - 0.03))
            cy = float(rng.uniform(r + 0.03, 1 - r - 0.03))
            if all(math.hypot(cx - x, cy - y) > r + q + 0.02 for x, y, q in placed):
                placed.append((cx, cy, r))
                break
    return placed


def _plan(index: int, seed: int, width: int) -> dict[str, Any]:
    """Everything needed to draw picture *index* of the set seeded *seed*."""
    # Seed on the pair, not on ``seed + index``: that would make picture 1 of
    # seed 1 the same as picture 0 of seed 2, and two "different" sets would
    # share nearly every picture.
    rng = np.random.default_rng([seed, index])
    kind = _KIND_CYCLE[index % len(_KIND_CYCLE)]
    objects: list[dict[str, Any]]
    if kind == "face":
        r = float(rng.uniform(0.26, 0.4))
        cx = float(rng.uniform(r + 0.04, 1 - r - 0.04))
        cy = float(rng.uniform(r + 0.04, 1 - r - 0.04))
        objects = [_plan_face(rng, cx, cy, r)]
        background = _plan_background(rng, avoid=objects[0]["color"])
    elif kind == "shapes":
        n = int(rng.integers(1, 6))
        r_lo, r_hi = (0.24, 0.38) if n == 1 else (0.09, 0.22)
        objects = [_plan_shape(rng, cx, cy, r) for cx, cy, r in _place(rng, n, r_lo, r_hi)]
        background = _plan_background(rng, avoid=objects[0]["color"] if n == 1 else "")
    else:
        n = int(rng.integers(3, 7))
        objects = [
            (_plan_face if rng.random() < 0.55 else _plan_shape)(rng, cx, cy, r)
            for cx, cy, r in _place(rng, n, 0.09, 0.21)
        ]
        background = _plan_background(rng)
    return {
        "filename": f"{kind}_{index:0{width}d}.png",
        "kind": kind,
        "background": background,
        "objects": objects,
    }


def describe_image_dataset(count: int, seed: int = 42) -> list[dict[str, Any]]:
    """What :func:`generate_image_dataset` draws for ``(count, seed)``, undrawn.

    One dict per picture, in generation order. The stable keys:

    - ``filename`` - the file :func:`generate_image_dataset` writes.
    - ``kind`` - ``"face"``, ``"shapes"`` or ``"scene"``.
    - ``background`` - ``{"style": ..., "colors": [...]}``: one of ``plain``,
      ``dots``, ``stripes``, ``checks`` or ``gradient``, and its colour names
      (the base colour first).
    - ``objects`` - what is drawn on it, each with a ``shape`` (``face``,
      ``circle``, ``square``, ``triangle`` or ``star``), a ``color`` name and a
      ``box`` (``[x0, y0, x1, y1]`` as fractions of the picture). A face also
      has an ``expression`` and ``smiling`` (whether the expression is one of
      :data:`SMILING_EXPRESSIONS`).

    The other keys are drawing parameters and may change.
    """
    width = max(4, len(str(count)))
    return [_plan(i, seed, width) for i in range(count)]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _shade(rgb: list[int] | tuple[int, ...], factor: float) -> tuple[int, int, int]:
    r, g, b = (int(c * factor) for c in rgb[:3])
    return (r, g, b)


def _mix(a: list[int] | tuple[int, ...], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    r, g, bb = (int(round(x * (1 - t) + y * t)) for x, y in zip(a[:3], b))
    return (r, g, bb)


def _draw_background(background: dict[str, Any], side: int):
    from PIL import Image, ImageDraw  # noqa: PLC0415

    colors = [_ALL_COLORS[name] for name in background["colors"]]
    style = background["style"]
    if style == "gradient":
        # One column, stretched: every row of a vertical gradient is one colour.
        t = np.linspace(0.0, 1.0, side)[:, None]
        column = np.asarray(colors[0], dtype=float) * (1 - t) + np.asarray(colors[1], dtype=float) * t
        strip = Image.fromarray(column.round().astype(np.uint8)[:, None, :], "RGB")
        return strip.resize((side, side), Image.Resampling.NEAREST)
    img = Image.new("RGB", (side, side), colors[0])
    draw = ImageDraw.Draw(img)
    if style == "dots":
        step = background["spacing"] * side
        dot = background["dot"] * step
        for row in range(int(side / step) + 2):
            offset = step / 2 if row % 2 else 0.0
            for col in range(int(side / step) + 2):
                cx, cy = col * step + offset, row * step
                draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=colors[1])
    elif style == "checks":
        cell = background["spacing"] * side
        cells = int(side / cell) + 1
        for row in range(cells):
            for col in range(row % 2, cells, 2):
                draw.rectangle([col * cell, row * cell, (col + 1) * cell, (row + 1) * cell], fill=colors[1])
    elif style == "stripes":
        # Every other band of width *w* across the stripes' normal, each band a
        # parallelogram long enough to cross the whole canvas.
        w = background["spacing"] * side / 2
        angle = math.radians(background["angle"])
        nx, ny = math.cos(angle), math.sin(angle)
        tx, ty = -ny, nx
        reach = 2 * side
        extent = [x * nx + y * ny for x in (0, side) for y in (0, side)]
        for k in range(int(math.floor(min(extent) / w)), int(math.ceil(max(extent) / w)) + 1):
            if k % 2 == 0:
                continue
            u0, u1 = k * w, (k + 1) * w
            draw.polygon(
                [
                    (u0 * nx - reach * tx, u0 * ny - reach * ty),
                    (u0 * nx + reach * tx, u0 * ny + reach * ty),
                    (u1 * nx + reach * tx, u1 * ny + reach * ty),
                    (u1 * nx - reach * tx, u1 * ny - reach * ty),
                ],
                fill=colors[1],
            )
    return img


def _draw_face(draw, c: float, r: float, obj: dict[str, Any]) -> None:
    rgb = tuple(obj["rgb"])
    line = max(2, int(r * 0.07))
    draw.ellipse([c - r, c - r, c + r, c + r], fill=rgb, outline=_shade(rgb, 0.55), width=line)
    if obj["cheeks"]:
        blush = _mix(rgb, (255, 90, 110), 0.45)
        for sign in (-1, 1):
            x, y = c + sign * 0.55 * r, c + 0.2 * r
            draw.ellipse([x - 0.12 * r, y - 0.09 * r, x + 0.12 * r, y + 0.09 * r], fill=blush)

    expression = obj["expression"]
    eye_y = c - 0.18 * r
    for sign in (-1, 1):
        x = c + sign * 0.34 * r
        if expression == "wink" and sign == 1:
            draw.arc([x - 0.11 * r, eye_y - 0.06 * r, x + 0.11 * r, eye_y + 0.14 * r], 200, 340, fill=_INK, width=line)
        else:
            draw.ellipse([x - 0.075 * r, eye_y - 0.12 * r, x + 0.075 * r, eye_y + 0.12 * r], fill=_INK)
        if expression == "angry":
            draw.line([(c + sign * 0.52 * r, c - 0.46 * r), (c + sign * 0.16 * r, c - 0.3 * r)], fill=_INK, width=line)

    mouth = int(line * 1.2)
    if expression in ("smile", "wink"):
        draw.arc([c - 0.5 * r, c - 0.15 * r, c + 0.5 * r, c + 0.55 * r], 20, 160, fill=_INK, width=mouth)
    elif expression == "grin":
        draw.chord([c - 0.52 * r, c - 0.02 * r, c + 0.52 * r, c + 0.62 * r], 0, 180, fill=_INK)
        draw.ellipse([c - 0.2 * r, c + 0.33 * r, c + 0.2 * r, c + 0.57 * r], fill=_TONGUE)
    elif expression == "frown":
        draw.arc([c - 0.42 * r, c + 0.28 * r, c + 0.42 * r, c + 0.78 * r], 200, 340, fill=_INK, width=mouth)
    elif expression == "angry":
        draw.arc([c - 0.36 * r, c + 0.34 * r, c + 0.36 * r, c + 0.7 * r], 205, 335, fill=_INK, width=mouth)
    elif expression == "surprised":
        draw.ellipse([c - 0.13 * r, c + 0.24 * r, c + 0.13 * r, c + 0.6 * r], fill=_INK)
    else:
        draw.line([(c - 0.33 * r, c + 0.42 * r), (c + 0.33 * r, c + 0.42 * r)], fill=_INK, width=mouth)


def _draw_shape(draw, c: float, r: float, obj: dict[str, Any]) -> None:
    rgb = tuple(obj["rgb"])
    outline = _shade(rgb, 0.55)
    line = max(2, int(r * 0.07))
    shape = obj["shape"]
    if shape == "circle":
        draw.ellipse([c - r, c - r, c + r, c + r], fill=rgb, outline=outline, width=line)
        return
    if shape == "square":
        points = [(c + 0.82 * r * sx, c + 0.82 * r * sy) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    elif shape == "triangle":
        points = [(c + r * math.cos(math.radians(a)), c + r * math.sin(math.radians(a))) for a in (-90, 30, 150)]
    else:
        points = [
            (c + rad * math.cos(math.radians(-90 + 36 * k)), c + rad * math.sin(math.radians(-90 + 36 * k)))
            for k, rad in enumerate([r, 0.45 * r] * 5)
        ]
    draw.polygon(points, fill=rgb, outline=outline, width=line)


def _draw_object(img, obj: dict[str, Any], side: int) -> None:
    """Draw *obj* on its own transparent tile, turn it, and lay it on *img*."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    r = obj["radius"] * side
    half = int(math.ceil(r * 1.1)) + 2
    tile = Image.new("RGBA", (2 * half, 2 * half), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    (_draw_face if obj["shape"] == "face" else _draw_shape)(draw, float(half), r, obj)
    if obj["rotation"]:
        # Bilinear is plenty: the whole picture is scaled down afterwards.
        tile = tile.rotate(obj["rotation"], resample=Image.Resampling.BILINEAR, center=(half, half))
    cx, cy = obj["center"]
    img.paste(tile, (int(round(cx * side)) - half, int(round(cy * side)) - half), tile)


def _render(plan: dict[str, Any]):
    side = CANVAS_SIZE * _SUPERSAMPLE
    img = _draw_background(plan["background"], side)
    for obj in plan["objects"]:
        _draw_object(img, obj, side)
    # A box-filter reduction is the classic supersampling resolve, and far
    # cheaper than a general resize.
    return img.reduce(_SUPERSAMPLE)


def generate_image_dataset(
    output_dir: Path,
    count: int,
    seed: int = 42,
    on_progress: Optional[ProgressCallback] = None,
) -> list[Path]:
    """Generate ``count`` synthetic images into ``output_dir``.

    Existing files matching the deterministic naming scheme are kept (the
    importer caches its output dir across reloads). Returns the list of
    image paths in generation order. :func:`describe_image_dataset` says what
    each one shows.

    If *on_progress* is supplied, it is called as
    ``on_progress(status, message, current, total)`` once per file (and
    once at the start and end) so the caller can drive a progress bar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if on_progress is not None:
        on_progress("downloading", f"Generating {count} synthetic images…", 0, count)
    for i, plan in enumerate(describe_image_dataset(count, seed)):
        path = output_dir / plan["filename"]
        paths.append(path)
        cached = path.exists()
        if on_progress is not None:
            verb = "Reusing cached" if cached else "Rendering"
            on_progress(
                "downloading",
                f"{verb} synthetic image {i + 1}/{count} ({plan['kind']})…",
                i,
                count,
            )
        if cached:
            continue
        _render(plan).save(path, format="PNG")
    if on_progress is not None:
        on_progress("downloading", f"Generated {count} synthetic images", count, count)
    return paths
