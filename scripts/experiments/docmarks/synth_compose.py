"""Layer 3 — paste real mark artwork onto real scanned pages at known geometry.

Why bother, when layers 1 and 2 are real?  Because the real strata cannot be
*swept*.  The 2026-07-13 study's most actionable findings were all about where a
threshold sits — the ~32 px floor below which nothing verifies, the inlier floor
near 24 for SuperPoint+LightGlue, the 2.2x advantage of a canonical-artwork
query over a crop of an in-scene instance — and a found dataset only samples the
sizes its own scanner and layout happened to produce.  Here size, rotation,
count and placement are inputs, so a sweep can locate a cliff instead of
straddling it.

What it costs: pasted marks are not printed marks.  The prior report already
flagged this as a limitation of its ``synth`` corpus, and nothing here fixes it
— the degradation pipeline narrows the gap and does not close it.  So the rule
that comes with this module is: **synthetic numbers quantify a mechanism, real
numbers size the effect.**  A finding that appears only in ``synth`` is a
hypothesis about the pipeline, not a claim about documents.

Backgrounds are held out.  A page used as a synthetic canvas is removed from
every other stratum by ``build_corpus.py``, so a synthetic class can never be
scored against a page that also carries a real mark of some other class.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from sources._common import Mark, Page


@dataclass(frozen=True)
class Placement:
    """Where one mark went, in page pixels.  This *is* the ground truth."""

    class_id: str
    box: tuple[int, int, int, int]
    scale_px: int
    rotation_deg: float


def _rand_log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def degrade_mark(mark: Any, rng: random.Random, *, ink_jitter: float = 0.18) -> Any:
    """Make a clean logo crop look like ink that went through a scanner.

    Three effects, in the order the physical process applies them: the ink is
    laid down unevenly (alpha jitter + a slight blur at the edge), the paper
    absorbs it (a small dilation, so thin strokes thicken), and the scanner
    quantises it (contrast loss).  Each is small; together they are the
    difference between a mark that SIFT finds trivially and one it has to work
    for.
    """
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    rgba = mark.convert("RGBA")

    # Uneven ink coverage: scale alpha by a smooth random field.
    arr = np.array(rgba).astype(np.float32)
    h, w = arr.shape[:2]
    coarse = rng.random()
    field = np.random.default_rng(int(coarse * 2**32)).random((max(2, h // 16), max(2, w // 16)))
    field = np.array(Image.fromarray((field * 255).astype("uint8")).resize((w, h), Image.BILINEAR)) / 255.0
    arr[..., 3] *= 1.0 - ink_jitter * field
    rgba = Image.fromarray(arr.clip(0, 255).astype("uint8"), mode="RGBA")

    # Ink bleed: widen the alpha slightly, then soften the edge.
    if rng.random() < 0.7:
        alpha = rgba.getchannel("A").filter(ImageFilter.MaxFilter(3))
        rgba.putalpha(alpha)
    rgba = rgba.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))

    # Scanner contrast loss.
    return ImageEnhance.Contrast(rgba).enhance(rng.uniform(0.75, 1.0))


def paste_mark(
    page: Any,
    mark: Any,
    *,
    target_px: int,
    rotation_deg: float,
    position: tuple[float, float],
) -> tuple[int, int, int, int]:
    """Composite *mark* onto *page* in place.  Returns the tight box it occupies.

    The box is computed from the rotated mark's **alpha bounding box**, not from
    the paste rectangle: a rotated logo's paste rectangle includes transparent
    corners, and a ground-truth box that is 30% empty would quietly make every
    query crop worse than it needed to be.
    """
    scale = target_px / max(mark.width, mark.height)
    sized = mark.resize(
        (max(1, int(round(mark.width * scale))), max(1, int(round(mark.height * scale)))),
        _resample_bicubic(),
    )
    rotated = sized.rotate(rotation_deg, expand=True, resample=_resample_bicubic())

    max_x = max(1, page.width - rotated.width)
    max_y = max(1, page.height - rotated.height)
    x = int(round(position[0] * max_x))
    y = int(round(position[1] * max_y))

    page.alpha_composite(rotated, (x, y))

    bbox = rotated.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if bbox is None:
        return (x, y, rotated.width, rotated.height)
    bx0, by0, bx1, by1 = bbox
    return (x + bx0, y + by0, bx1 - bx0, by1 - by0)


def _resample_bicubic() -> int:
    from PIL import Image

    return Image.BICUBIC


def compose_page(
    background_path: Path,
    artwork: Sequence[tuple[str, Any]],
    out_path: Path,
    rng: random.Random,
    *,
    size_px: tuple[int, int],
    rotation_deg: tuple[float, float],
    jpeg_quality: Optional[int] = None,
) -> list[Placement]:
    """Compose one synthetic page and write it.  Returns the placements.

    *artwork* is a list of ``(class_id, RGBA image)``.  Marks are placed in
    non-overlapping quadrant-ish positions so that two pasted marks on one page
    do not occlude each other, which would make the ground-truth box a lie.
    """
    from PIL import Image

    page = Image.open(background_path).convert("RGBA")
    placements: list[Placement] = []

    slots = [(0.15, 0.10), (0.72, 0.12), (0.18, 0.70), (0.70, 0.74), (0.45, 0.42)]
    rng.shuffle(slots)

    for i, (class_id, art) in enumerate(artwork):
        if i >= len(slots):
            break
        target = int(round(_rand_log_uniform(rng, size_px[0], size_px[1])))
        angle = rng.uniform(*rotation_deg)
        jitter = (slots[i][0] + rng.uniform(-0.06, 0.06), slots[i][1] + rng.uniform(-0.06, 0.06))
        position = (min(max(jitter[0], 0.0), 1.0), min(max(jitter[1], 0.0), 1.0))
        box = paste_mark(
            page,
            degrade_mark(art, rng),
            target_px=target,
            rotation_deg=angle,
            position=position,
        )
        placements.append(Placement(class_id=class_id, box=box, scale_px=target, rotation_deg=angle))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = page.convert("RGB")
    if jpeg_quality is not None:
        rgb.save(out_path, quality=jpeg_quality)
    else:
        rgb.save(out_path)
    return placements


def build_synthetic_pages(
    backgrounds: Sequence[Path],
    pool: dict[str, Path],
    out_images: Path,
    *,
    instances_per_class: int,
    size_px: tuple[int, int],
    rotation_deg: tuple[float, float],
    seed: int,
    marks_per_page: int = 1,
) -> list[Page]:
    """Generate the whole synthetic stratum.

    Each class gets *instances_per_class* placements spread across distinct
    background pages.  Backgrounds are consumed round-robin so no single scan
    carries a disproportionate share of the ground truth — a corpus where one
    page holds 40 marks would let a matcher "win" by recognising that page.
    """
    from PIL import Image

    if not backgrounds:
        raise ValueError("synthetic composition needs at least one background page")
    if not pool:
        raise ValueError("synthetic composition needs a non-empty artwork pool")

    from sources.artwork import to_rgba_mark

    rng = random.Random(seed)
    art_cache = {name: to_rgba_mark(Image.open(path)) for name, path in sorted(pool.items())}

    jobs: list[tuple[str, Any]] = []
    for name in sorted(art_cache):
        jobs.extend((f"synth/{name}", art_cache[name]) for _ in range(instances_per_class))
    rng.shuffle(jobs)

    pages: list[Page] = []
    bg_index = 0
    for start in range(0, len(jobs), marks_per_page):
        batch = jobs[start : start + marks_per_page]
        background = backgrounds[bg_index % len(backgrounds)]
        bg_index += 1
        page_id = f"synth/{bg_index:06d}"
        out_path = out_images / f"{bg_index:06d}.png"
        placements = compose_page(
            background,
            batch,
            out_path,
            rng,
            size_px=size_px,
            rotation_deg=rotation_deg,
        )
        with Image.open(out_path) as im:
            width, height = im.size
        pages.append(
            Page(
                page_id=page_id,
                source="synth",
                path=str(out_path),
                width=width,
                height=height,
                marks=[
                    Mark(kind="logo", box=p.box, class_id=p.class_id, provenance="synthetic") for p in placements
                ],
                meta={
                    "background": str(background),
                    "placements": [
                        {"class_id": p.class_id, "scale_px": p.scale_px, "rotation_deg": round(p.rotation_deg, 2)}
                        for p in placements
                    ],
                },
            )
        )
    return pages
