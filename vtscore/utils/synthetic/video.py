"""Synthetic video generation - bouncing ball, walking smiley, rotating shape, scrolling text.

Each generator renders frames with PIL and encodes them to mp4 via
``imageio_ffmpeg`` (the bundled static ffmpeg binary already required by the
video media type).  Variety across four "ideas" gives X-CLIP and similar
video embedders both appearance and motion features to cluster on.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ProgressCallback = Callable[[str, str, int, int], None]

CANVAS_SIZE = 224
FPS = 12
DURATION_SEC = 2.0
NUM_FRAMES = int(FPS * DURATION_SEC)

_BG_COLORS = [
    (240, 240, 240),
    (30, 30, 40),
    (200, 220, 255),
    (255, 230, 200),
    (210, 255, 210),
]


def _bouncing_ball_frame(rng: np.random.Generator, t: float, params: dict, draw, size: int) -> None:
    radius = params["radius"]
    color = params["color"]
    bg = params["bg"]
    speed_x = params["speed_x"]
    speed_y = params["speed_y"]
    # Reflect off walls.
    period_x = 2.0 * (size - 2 * radius) / max(1.0, abs(speed_x))
    period_y = 2.0 * (size - 2 * radius) / max(1.0, abs(speed_y))
    fx = (t * abs(speed_x)) % (2 * (size - 2 * radius))
    fy = (t * abs(speed_y)) % (2 * (size - 2 * radius))
    x = radius + (fx if fx < (size - 2 * radius) else 2 * (size - 2 * radius) - fx)
    y = radius + (fy if fy < (size - 2 * radius) else 2 * (size - 2 * radius) - fy)
    draw.rectangle([0, 0, size, size], fill=bg)
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    _ = period_x, period_y  # keep names for clarity


def _walking_smiley_frame(rng: np.random.Generator, t: float, params: dict, draw, size: int) -> None:
    radius = params["radius"]
    bg = params["bg"]
    face = params["face"]
    speed = params["speed"]
    bob_amp = params["bob_amp"]
    fx = (t * speed) % (size + 2 * radius)
    cx = int(fx - radius)
    cy = int(size / 2 + bob_amp * math.sin(2 * math.pi * t * 2.0))
    draw.rectangle([0, 0, size, size], fill=bg)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=face, outline=(20, 20, 20), width=2)
    eye_dx = radius // 2
    eye_dy = radius // 3
    eye_r = max(2, radius // 8)
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        ey = cy - eye_dy
        draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=(20, 20, 20))
    mouth_w = int(radius * 0.8)
    mx0, mx1 = cx - mouth_w // 2, cx + mouth_w // 2
    my = cy + radius // 3
    draw.arc([mx0, my - mouth_w // 2, mx1, my + mouth_w // 2], start=0, end=180, fill=(20, 20, 20), width=3)


def _rotating_shape_frame(rng: np.random.Generator, t: float, params: dict, draw, size: int) -> None:
    bg = params["bg"]
    color = params["color"]
    radius = params["radius"]
    sides = params["sides"]
    rps = params["rps"]
    cx = size // 2
    cy = size // 2
    angle = 2 * math.pi * rps * t
    draw.rectangle([0, 0, size, size], fill=bg)
    pts = []
    for i in range(sides):
        a = angle + i * 2 * math.pi / sides
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    draw.polygon(pts, fill=color, outline=(20, 20, 20))


def _scrolling_text_frame(rng: np.random.Generator, t: float, params: dict, draw, size: int) -> None:
    bg = params["bg"]
    fg = params["fg"]
    text = params["text"]
    speed = params["speed"]
    draw.rectangle([0, 0, size, size], fill=bg)
    # Render each character as a coloured rectangle so we don't depend on a
    # specific TTF font being available in the environment.
    char_w = size // 10
    char_h = size // 6
    n_chars = len(text)
    total_w = n_chars * char_w + size
    x_off = int((-t * speed) % total_w) - size
    y = size // 2 - char_h // 2
    for i, ch in enumerate(text):
        x = x_off + i * char_w
        if x + char_w < 0 or x > size:
            continue
        # Per-character colour from ASCII so different texts look different.
        c = (
            (ord(ch) * 53) % 255,
            (ord(ch) * 97) % 255,
            (ord(ch) * 151) % 255,
        )
        draw.rectangle([x + 2, y, x + char_w - 2, y + char_h], fill=c, outline=fg)


def _make_bouncing_ball_params(rng: np.random.Generator) -> tuple[str, dict]:
    return "ball", {
        "radius": int(rng.integers(15, 35)),
        "color": (int(rng.integers(50, 255)), int(rng.integers(50, 255)), int(rng.integers(50, 255))),
        "bg": _BG_COLORS[int(rng.integers(0, len(_BG_COLORS)))],
        "speed_x": float(rng.uniform(60, 180)),
        "speed_y": float(rng.uniform(60, 180)),
    }


def _make_walking_smiley_params(rng: np.random.Generator) -> tuple[str, dict]:
    return "walker", {
        "radius": int(rng.integers(20, 40)),
        "bg": _BG_COLORS[int(rng.integers(0, len(_BG_COLORS)))],
        "face": (int(rng.integers(180, 255)), int(rng.integers(180, 255)), int(rng.integers(80, 200))),
        "speed": float(rng.uniform(80, 200)),
        "bob_amp": float(rng.uniform(5, 25)),
    }


def _make_rotating_shape_params(rng: np.random.Generator) -> tuple[str, dict]:
    return "rotator", {
        "bg": _BG_COLORS[int(rng.integers(0, len(_BG_COLORS)))],
        "color": (int(rng.integers(50, 255)), int(rng.integers(50, 255)), int(rng.integers(50, 255))),
        "radius": int(rng.integers(40, 80)),
        "sides": int(rng.integers(3, 8)),
        "rps": float(rng.uniform(0.5, 2.0)),
    }


def _make_scrolling_text_params(rng: np.random.Generator) -> tuple[str, dict]:
    samples = ["HELLO", "VTSEARCH", "DEMO", "SYNTHETIC", "MEDIA", "TEST"]
    return "marquee", {
        "bg": _BG_COLORS[int(rng.integers(0, len(_BG_COLORS)))],
        "fg": (20, 20, 20),
        "text": samples[int(rng.integers(0, len(samples)))],
        "speed": float(rng.uniform(80, 220)),
    }


_GENERATORS = [
    (_make_bouncing_ball_params, _bouncing_ball_frame),
    (_make_walking_smiley_params, _walking_smiley_frame),
    (_make_rotating_shape_params, _rotating_shape_frame),
    (_make_scrolling_text_params, _scrolling_text_frame),
]


def _encode_frames(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Encode RGB uint8 frames to mp4 via the bundled ffmpeg."""
    import imageio_ffmpeg  # noqa: PLC0415

    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        quality=6,
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(frame.tobytes())
    finally:
        writer.close()


def generate_video_dataset(
    output_dir: Path,
    count: int,
    seed: int = 42,
    on_progress: Optional[ProgressCallback] = None,
) -> list[Path]:
    """Generate ``count`` short synthetic mp4s into ``output_dir``.

    Cycles through four ideas (bouncing ball, walking smiley, rotating
    shape, scrolling marquee) so the dataset has both motion and
    appearance variety. Existing files are kept across reloads.

    If *on_progress* is supplied, it is called as
    ``on_progress(status, message, current, total)`` once per file (and
    once at the start) so the caller can drive a progress bar. The
    ``status`` is always ``"downloading"`` so it maps to the dataset
    pipeline's "fetching files" step.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    width = max(4, len(str(count)))
    if on_progress is not None:
        on_progress("downloading", f"Generating {count} synthetic videos…", 0, count)
    for i in range(count):
        make_params, render = _GENERATORS[i % len(_GENERATORS)]
        idea, _ = make_params(np.random.default_rng(seed + i))
        path = output_dir / f"{idea}_{i:0{width}d}.mp4"
        paths.append(path)
        cached = path.exists()
        if on_progress is not None:
            verb = "Reusing cached" if cached else "Rendering"
            on_progress(
                "downloading",
                f"{verb} synthetic video {i + 1}/{count} ({idea})…",
                i,
                count,
            )
        if cached:
            continue
        rng = np.random.default_rng(seed + i)
        _name, params = make_params(rng)
        frames: list[np.ndarray] = []
        for f_idx in range(NUM_FRAMES):
            t = f_idx / FPS
            img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            render(rng, t, params, draw, CANVAS_SIZE)
            frames.append(np.asarray(img, dtype=np.uint8))
        _encode_frames(path, frames, FPS)
    if on_progress is not None:
        on_progress("downloading", f"Generated {count} synthetic videos", count, count)
    return paths
