#!/usr/bin/env python
"""Section 7's method slides: how VTSearch finds one mark on a pile of scanned pages.

    python slides/figs/src/make-sift-figs.py

Writes six full-bleed figures and their build stages, one per fragment:

* `figs/logo-keypoints*.webp`: one crop of the Philip Morris crest, its SIFT
  keypoints, and one keypoint's 128-number descriptor drawn as the 4 x 4 grid
  of direction histograms it actually is.
* `figs/logo-match*.webp`: that crop checked against a letter carrying the crest
  and a letter that does not. First the ratio-tested matches, then the ones a
  single shift-rotate-scale explains.
* `figs/logo-budget.webp`: one page's keypoints at 1,024 and at 8,192, and how
  few of the 1,024 the crest gets.
* `figs/logo-stages*.webp`: the two-stage shape, a shortlist and then a check,
  drawn with real pages and the real top of the ranking.
* `figs/logo-tiles*.webp`: the crest's share of the page's keypoints, against its
  share of the best tile's.
* `figs/logo-votes*.webp`: a Good vote's box becoming a second template, and a
  letter the crop alone misses that the new template finds.

**Nothing here is drawn from a guess.** The pages are Tobacco800 scans, the same
source FullMarks' crest class was anchored on, fetched by `fullmarks_media` into
the gitignored `data/fullmarks-sources/`. Every keypoint, match, inlier and
count comes from the shipped matcher: `SiftMatcher.detect_and_describe` at the
`sift_vlad_doc` budget, `ratio_test_matches` for the correspondences, and
`SiftMatcher.verify` for the counts. The figures redraw what those return,
and `_fit` re-runs the same RANSAC call only to learn *which* matches are the
inliers, which `verify` does not report; it asserts its count equals
`verify`'s, so a drawing can never disagree with the matcher.

**The numbers the notes quote are pinned in `EXPECT`.** The presenter notes are
hand-written, so a matcher change that moves a count has to fail here rather
than leave a note describing a figure that no longer says it.

**Why these pages.** A full scan of Tobacco800's 1,290 pages against the crop
(the query's own page first, at 265 inliers) ranks `MATCH_PAGE` second and the
three `RANKED` pages after it. `MISS_PAGE` is the non-match with the most
ratio-test survivors that the gate still turns away. `BUDGET_PAGE` is the page
where the budget changes the verdict most plainly. `VOTED_PAGE` and
`FOUND_PAGE` carry the *PM* monogram crest, which the FullMarks datasheet
records as the same mark as the query's globe crest by owner ruling
(`scripts/experiments/fullmarks/DATASHEET.md`), so a method that ranks only
globe copies high is missing positives. The crop alone fails `FOUND_PAGE`;
`VOTED_PAGE`'s boxed crest finds it. None of this needs rerunning to redraw the
figures; the choice is recorded here so it can be re-checked.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fullmarks_media as dm  # noqa: E402
from slide_figure import FULL_BLEED, save  # noqa: E402

from vtscore.config import MAX_STRUCTURAL_DETECT_PIXELS  # noqa: E402
from vtscore.media.structural import (  # noqa: E402
    DEFAULT_MAX_FEATURES,
    DOCUMENT_MAX_FEATURES,
    StructuralFeatures,
    SiftMatcher,
    ratio_test_matches,
)
from vtscore.media.structural_tiles import (  # noqa: E402
    MIN_TILE_KP,
    TILE_LAYERS,
    load_tile_projection,
    tile_vectors,
    tile_windows,
)
from vtscore.training.structural_similarity import filter_features_to_box  # noqa: E402
from vtscore.training.structural_stage1 import example_queries  # noqa: E402

OUT = Path(__file__).resolve().parent.parent

INK = "#14181f"
SOFT = "#5b6472"
RULE = "#d8dee6"
FAINT = "#eceff3"  # a grid under a `RULE` grid, there but not read first
BLUE = "#0b5fa5"  # the shipped thing: here, the matcher's own keypoints
RED = "#b91c1c"  # the negative side: matches no fit explains
GREEN = "#0d8a5f"  # the positive side: the mark, and matches that agree on it

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 18,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)

#: Exactly the slide, at 100 slide pixels to the inch: every figure is drawn on
#: one full-figure axes whose data units are inches, and saved `tight=False`,
#: so a coordinate below is a slide position divided by 100. At this size a
#: point renders at 1.39 slide pixels, so the 20px floor is 14.4pt.
CANVAS = (12.8, 7.2)
#: Output pixels per inch. Pages are resampled to exactly this before they are
#: drawn, so matplotlib never has to downsample a 2,560px scan itself.
DPI = 200

LABEL_PT = 18
COUNT_PT = 22

# --------------------------------------------------------------------------
# What is drawn, and what the notes say it shows
# --------------------------------------------------------------------------

QUERY_CLASS = "tobacco800/logo_aah97e00-page02_1_0"
MATCH_PAGE = "xxr58e00"
MISS_PAGE = "bbn00d00-first"
BUDGET_PAGE = "wry97e00"
RANKED = ("wry97e00", "uiw28e00", "zqs18e00-page01_1")
VOTED_PAGE = "anr29e00-page02_1"
FOUND_PAGE = "gko55e00-page02_1"

#: Counts the presenter notes quote. `main` fails if the matcher returns
#: anything else, because then the notes are describing a different figure.
EXPECT = {
    "query_keypoints": 1090,
    "match_tentative": 210,
    "match_inliers": 141,
    "miss_tentative": 23,
    "miss_inliers": 5,
    "budget_on_crest_1024": 7,
    "budget_inliers_1024": 4,
    "budget_on_crest_8192": 460,
    "budget_inliers_8192": 101,
    "tile_page_keypoints": 7685,
    "tile_on_crest": 442,
    "tile_best_keypoints": 350,
    "tile_best_crest": 350,
    "ranked": (101, 91, 91),
    "found_by_query": 6,
    "found_by_vote": 32,
}

#: The cold-start gate: fewer than this many agreeing keypoints is "no".
GATE = 8


@dataclass(frozen=True)
class Page:
    image: Image.Image  # the scan, greyscale
    features: StructuralFeatures  # at `DOCUMENT_MAX_FEATURES` unless stated
    #: Scan pixels per detection pixel. The matcher detects at no more than
    #: `MAX_STRUCTURAL_DETECT_PIXELS`, so a full page's keypoint *sizes* are in
    #: a downsampled image's pixels while its positions are normalised; a crop
    #: under the cap is detected as it is, at 1.0.
    detect_scale: float = 1.0

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def keypoints_px(self) -> np.ndarray:
        """`(M, 2)` keypoint positions in this scan's own pixels."""
        w, h = self.size
        kp = self.features.keypoints_f32()
        return np.column_stack([kp[:, 0] * w, kp[:, 1] * h])


@dataclass(frozen=True)
class Fit:
    template_px: np.ndarray  # (T, 2) tentative matches' template ends, template pixels
    page_px: np.ndarray  # (T, 2) the page ends, page pixels
    inlier: np.ndarray  # (T,) bool
    model: np.ndarray | None  # 2x3 similarity, template-normalised -> page-normalised

    @property
    def inliers(self) -> int:
        return int(self.inlier.sum())


def _matcher() -> SiftMatcher:
    return SiftMatcher()


@cache
def query() -> Page:
    crop = dm.mark_crop(QUERY_CLASS, pad=0.0).convert("L")
    feats = _matcher().detect_and_describe(np.asarray(crop), max_features=DOCUMENT_MAX_FEATURES)
    return Page(crop, feats)


@cache
def page(stem: str, budget: int = DOCUMENT_MAX_FEATURES) -> Page:
    _, images = dm._tobacco800_marks()
    with Image.open(images[stem]) as raw:
        image = raw.convert("L")
    feats = _matcher().detect_and_describe(np.asarray(image), max_features=budget)
    return Page(image, feats, _detect_scale(image))


def _detect_scale(image: Image.Image) -> float:
    pixels = image.width * image.height
    return max(1.0, math.sqrt(pixels / MAX_STRUCTURAL_DETECT_PIXELS))


def logo_box(stem: str) -> tuple[int, int, int, int]:
    """The page's Tobacco800 logo box, `(x, y, w, h)` in scan pixels."""
    marks, _ = dm._tobacco800_marks()
    boxes = [m.box for m in marks.get(stem.lower(), []) if m.kind == "logo"]
    if len(boxes) != 1:
        raise SystemExit(f"{stem}: expected one logo box, found {boxes}")
    return boxes[0]


def on_box(p: Page, box: tuple[int, int, int, int]) -> int:
    x, y, w, h = box
    pts = p.keypoints_px()
    return int(((pts[:, 0] >= x) & (pts[:, 0] <= x + w) & (pts[:, 1] >= y) & (pts[:, 1] <= y + h)).sum())


def template_from_vote(stem: str) -> Page:
    """What a Good vote with a drawn box becomes: that page's keypoints inside the box.

    Exactly `build_templates`' rule (`filter_features_to_box`), keypoints still
    normalised to the whole page as the app keeps them, so the fit below is the
    fit the app makes. The figure draws the template by cropping the page to
    the box (`draw_scan(region=...)`), which leaves the coordinates alone.
    """
    p = page(stem)
    w, h = p.size
    x, y, bw, bh = logo_box(stem)
    feats = filter_features_to_box(p.features, (x / w, y / h, (x + bw) / w, (y + bh) / h))
    return Page(p.image, feats, p.detect_scale)


def fit(template: Page, target: Page) -> Fit:
    """The shipped matcher's correspondences and inliers for one pair, for drawing.

    `ratio_test_matches` is the matcher's own correspondence step. The RANSAC
    call repeats `SiftMatcher._fit_similarity`'s exactly, because the inlier
    mask is not part of what `verify` returns; the assertion is what keeps the
    two in step.
    """
    import cv2

    from vtscore.media import structural as st

    stats = _matcher().verify(template.features, target.features)
    ((t_idx, c_idx),) = ratio_test_matches(
        template.features.descriptors_f32(), [target.features.descriptors_f32()], ratio=st._LOWE_RATIO
    )
    t_kp, c_kp = template.features.keypoints_f32(), target.features.keypoints_f32()
    src = np.ascontiguousarray(t_kp[t_idx, :2], dtype=np.float32)
    dst = np.ascontiguousarray(c_kp[c_idx, :2], dtype=np.float32)
    model, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=st._RANSAC_REPROJ_THRESHOLD,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    inlier = np.zeros(len(t_idx), dtype=bool) if mask is None else mask.ravel().astype(bool)
    if len(t_idx) != stats.tentative_count or int(inlier.sum()) != stats.inlier_count:
        raise SystemExit(
            f"fit: drew {len(t_idx)} matches / {int(inlier.sum())} inliers, but the matcher reports "
            f"{stats.tentative_count} / {stats.inlier_count}. `fit` has drifted from `_fit_similarity`."
        )
    tw, th = template.size
    pw, ph = target.size
    return Fit(
        template_px=src * np.array([tw, th]),
        page_px=dst * np.array([pw, ph]),
        inlier=inlier,
        model=model if stats.model_ok else None,
    )


def _in_window(kp: np.ndarray, window: tuple[float, float, float, float]) -> np.ndarray:
    """Which normalised keypoints a tile holds: `raw_tiles`' own test, half-open on the far edges."""
    x0, y0, x1, y1 = window
    return (kp[:, 0] >= x0) & (kp[:, 0] < x1) & (kp[:, 1] >= y0) & (kp[:, 1] < y1)


def best_tile(p: Page, q: Page) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """The tile the app scores the page by, as `(x0, y0, x1, y1)` normalised, and which keypoints it holds.

    Nothing here re-implements Stage 1. The page's tiles are `tile_vectors`
    over its compacted features, which is what the embedding stage stores and
    tiles from; the query is `example_queries`, the app's own crop query; both
    go through the cached `tile_projection_v1`, so a checkout without it stops
    here with the command that rebuilds it. The tiles are cut from every layer
    in `TILE_LAYERS`, so the winner may be a coarse tile or a fine one.

    `tile_vectors` does not say which keypoints a tile holds, so the windows
    are enumerated again, in its order, to learn that; the count check is what
    keeps the enumeration and `raw_tiles` in step.
    """
    stored = p.features.compact()
    tiles = tile_vectors(stored, load_tile_projection())
    queries = example_queries([q.features])
    if queries is None:
        raise SystemExit("tiles: the query crop gave no Stage-1 query")
    best = int(np.argmax((tiles.vectors.astype(np.float32) @ queries.T).max(axis=1)))
    kp = stored.keypoints_f32()
    windows = [w for width, height in TILE_LAYERS for w in tile_windows(width, height)]
    kept = [(w, inside) for w in windows if (inside := _in_window(kp, w)).sum() >= MIN_TILE_KP]
    if len(kept) != tiles.count:
        raise SystemExit(
            f"tiles: enumerated {len(kept)} tiles, but `tile_vectors` made {tiles.count}. "
            "`best_tile` has drifted from `raw_tiles`."
        )
    return kept[best]


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def _canvas() -> tuple[Figure, Axes]:
    fig = plt.figure(figsize=CANVAS)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


@dataclass(frozen=True)
class Placed:
    """A scan (or part of one) drawn at `(x0, y0)` inches, `scale` inches per scan pixel."""

    x0: float
    y0: float
    scale: float
    region: tuple[int, int, int, int]  # (left, top, right, bottom) in scan pixels

    @property
    def width(self) -> float:
        return (self.region[2] - self.region[0]) * self.scale

    @property
    def height(self) -> float:
        return (self.region[3] - self.region[1]) * self.scale

    def xy(self, px: ArrayLike) -> np.ndarray:
        """Scan pixels (y down) to canvas inches (y up)."""
        px = np.asarray(px, dtype=float).reshape(-1, 2)
        x = self.x0 + (px[:, 0] - self.region[0]) * self.scale
        y = self.y0 + self.height - (px[:, 1] - self.region[1]) * self.scale
        return np.column_stack([x, y])

    def contains(self, px: ArrayLike) -> np.ndarray:
        return inside(px, self.region)


def inside(px: ArrayLike, region: tuple[int, int, int, int]) -> np.ndarray:
    """Which of the scan-pixel points `px` fall in `region` (left, top, right, bottom)."""
    px = np.asarray(px, dtype=float).reshape(-1, 2)
    left, top, right, bottom = region
    return (px[:, 0] >= left) & (px[:, 0] <= right) & (px[:, 1] >= top) & (px[:, 1] <= bottom)


def draw_scan(
    ax: Axes,
    image: Image.Image,
    x0: float,
    y0: float,
    *,
    width: float | None = None,
    height: float | None = None,
    region: tuple[int, int, int, int] | None = None,
    fade: float = 0.0,
    frame: bool = True,
) -> Placed:
    """Draw `region` of `image` with its bottom-left at `(x0, y0)`, sized by width or height.

    `fade` lightens the ink toward white, so coloured marks drawn over a page
    read as the thing being pointed at rather than as more of the page.
    """
    region = region or (0, 0, image.width, image.height)
    rw, rh = region[2] - region[0], region[3] - region[1]
    scale = width / rw if width is not None else height / rh  # type: ignore[operator]
    placed = Placed(x0, y0, scale, region)
    out_px = (max(1, round(placed.width * DPI)), max(1, round(placed.height * DPI)))
    pixels = np.asarray(image.crop(region).resize(out_px, Image.Resampling.LANCZOS), dtype=float)
    pixels = 255 - (255 - pixels) * (1 - fade)
    ax.imshow(
        pixels,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=(x0, x0 + placed.width, y0, y0 + placed.height),
        interpolation="none",
        zorder=1,
    )
    if frame:
        ax.add_patch(Rectangle((x0, y0), placed.width, placed.height, fill=False, ec=RULE, lw=1.2, zorder=2))
    return placed


def draw_keypoints(ax: Axes, placed: Placed, p: Page, *, color: str = BLUE, lw: float = 0.7) -> None:
    """Each keypoint as SIFT sees it: a circle its own size, and a tick for its direction."""
    kp = p.features.keypoints_f32()
    pts = p.keypoints_px()
    keep = placed.contains(pts)
    xy = placed.xy(pts[keep])
    diam = kp[keep, 2] * p.detect_scale * placed.scale
    ax.add_collection(
        EllipseCollection(
            diam,
            diam,
            np.zeros_like(diam),
            units="xy",
            offsets=xy,
            offset_transform=ax.transData,
            facecolors="none",
            edgecolors=color,
            linewidths=lw,
            alpha=0.85,
            zorder=3,
        )
    )
    ang = kp[keep, 3]
    tip = xy + np.column_stack([np.cos(ang), -np.sin(ang)]) * (diam / 2)[:, None]
    ax.add_collection(
        LineCollection(list(np.stack([xy, tip], axis=1)), colors=color, linewidths=lw, alpha=0.85, zorder=3)
    )


def draw_dots(ax: Axes, placed: Placed, pts: np.ndarray, *, color: str, size: float = 1.6, alpha: float = 0.9) -> None:
    keep = placed.contains(pts)
    xy = placed.xy(pts[keep])
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color, lw=0, alpha=alpha, zorder=3)


def draw_box(ax: Axes, placed: Placed, box: tuple[int, int, int, int], *, color: str, lw: float = 2.0, ls="-"):
    x, y, w, h = box
    ((bx, by),) = placed.xy([[x, y + h]])
    ax.add_patch(Rectangle((bx, by), w * placed.scale, h * placed.scale, fill=False, ec=color, lw=lw, ls=ls, zorder=4))


def draw_matches(
    ax: Axes,
    left: Placed,
    right: Placed,
    f: Fit,
    which: str,
    *,
    lw: float = 0.8,
) -> None:
    """Lines from template to page: every match (`all`), or split by the fit (`split`)."""
    a = left.xy(f.template_px)
    b = right.xy(f.page_px)
    segs = np.stack([a, b], axis=1)
    if which == "all":
        ax.add_collection(LineCollection(list(segs), colors=SOFT, linewidths=lw, alpha=0.55, zorder=5))
        return
    if which in ("split", "outliers"):
        ax.add_collection(LineCollection(list(segs[~f.inlier]), colors=RED, linewidths=lw, alpha=0.55, zorder=5))
    if which in ("split", "inliers"):
        ax.add_collection(LineCollection(list(segs[f.inlier]), colors=GREEN, linewidths=lw, alpha=0.9, zorder=6))


def draw_fit_outline(ax: Axes, template: Page, target: Page, right: Placed, f: Fit) -> None:
    """The template's outline carried onto the page by the fitted similarity."""
    if f.model is None:
        return
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)  # template-normalised
    mapped = corners @ f.model[:, :2].T + f.model[:, 2]
    pw, ph = target.size
    ax.add_patch(Polygon(right.xy(mapped * np.array([pw, ph])), closed=True, fill=False, ec=GREEN, lw=2.6, zorder=7))


def text(ax: Axes, x: float, y: float, s: str, *, size: float = LABEL_PT, **kw) -> None:
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    kw.setdefault("color", INK)
    ax.text(x, y, s, fontsize=size, zorder=8, **kw)


def _save(fig: Figure, name: str) -> None:
    """Scans on a slide are photographs, so they go out as WebP, through `save`'s checks."""
    png = f"{name}.png"
    with contextlib.redirect_stdout(io.StringIO()):
        save(fig, OUT, png, column=FULL_BLEED, tight=False)
    with Image.open(OUT / png) as image:
        image.convert("RGB").save(OUT / f"{name}.webp", quality=88, method=6)
    (OUT / png).unlink()
    print(f"wrote figs/{name}.webp")


def build(name: str, draw, stages: int) -> None:
    """Save `draw(ax, k)` for k = 1..stages: builds 1..stages-1, and the final slide."""
    for k in range(1, stages + 1):
        fig, ax = _canvas()
        draw(ax, k)
        _save(fig, name if k == stages else f"{name}.build{k}")


# --------------------------------------------------------------------------
# 1. Points of interest: a crop, its keypoints, one keypoint's 128 numbers
# --------------------------------------------------------------------------

#: The keypoint the zoom describes, pinned by where it sits on the crop (crop
#: pixels) rather than by index. It is on the rim of the globe, where the
#: grid of the globe meets the ring round it, and its described square fits
#: inside the crop, so the zoom shows what SIFT actually read rather than
#: padding. The nearest keypoint to this point is the one drawn.
ZOOM_AT = (161.0, 85.0)


def _zoom_keypoint(q: Page) -> int:
    pts = q.keypoints_px()
    return int(np.argmin(np.hypot(pts[:, 0] - ZOOM_AT[0], pts[:, 1] - ZOOM_AT[1])))


def _described_square(q: Page, i: int) -> tuple[np.ndarray, np.ndarray]:
    """The patch SIFT describes for keypoint `i`, turned to its direction, and its corners on the crop.

    The descriptor covers 4 x 4 cells, each 1.5 x the keypoint's size on a
    side (OpenCV: three times the keypoint's sigma, which is half its size), so
    the square is 6 x size across. Rotating by the keypoint's own angle is what
    makes the descriptor the same when the mark turns: its first column always
    faces the keypoint's direction, which the zoom draws pointing right.
    """
    import cv2

    kp = q.features.keypoints_f32()[i]
    ((x, y),) = q.keypoints_px()[i : i + 1]
    side = int(round(6 * kp[2] * q.detect_scale))
    half = side / 2
    rot = cv2.getRotationMatrix2D((float(x), float(y)), float(np.rad2deg(kp[3])), 1.0)
    rot[0, 2] += half - x
    rot[1, 2] += half - y
    patch = cv2.warpAffine(np.asarray(q.image), rot, (side, side), flags=cv2.INTER_CUBIC, borderValue=255)
    inverse = cv2.invertAffineTransform(rot)
    corners = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=float)
    return patch, corners @ inverse[:, :2].T + inverse[:, 2]


def keypoints_fig() -> None:
    q = query()
    assert q.features.count == EXPECT["query_keypoints"], q.features.count
    i = _zoom_keypoint(q)
    patch, corners = _described_square(q, i)
    desc = q.features.descriptors_f32()[i].reshape(4, 4, 8)  # [row, col, direction]

    def draw(ax: Axes, stage: int) -> None:
        crop = draw_scan(ax, q.image, 0.7, 1.6, width=6.3, fade=0.45 if stage >= 2 else 0.0)
        text(ax, 0.7, 1.15, "the query: one crop of the crest")
        if stage >= 2:
            draw_keypoints(ax, crop, q, lw=0.9)
            text(ax, 0.7, 0.7, f"{q.features.count:,} keypoints, each a place, a size and a direction")
        if stage < 3:
            return
        on_crop = crop.xy(corners)
        ax.add_patch(Polygon(on_crop, closed=True, fill=False, ec=INK, lw=2.4, zorder=9))
        px0, py0, side = 7.9, 1.6, 4.0
        zoom = draw_scan(ax, Image.fromarray(patch), px0, py0, width=side, fade=0.72, frame=False)
        cell = side / 4
        for k in range(5):
            ax.plot([px0, px0 + side], [py0 + k * cell] * 2, color=SOFT, lw=1.2, zorder=4)
            ax.plot([px0 + k * cell] * 2, [py0, py0 + side], color=SOFT, lw=1.2, zorder=4)
        # Patch corners 0 (top-left) and 3 (bottom-left) to the zoom's left edge:
        # the two connectors that can never cross each other or the zoom.
        for corner, target in ((0, (px0, py0 + side)), (3, (px0, py0))):
            ax.plot([on_crop[corner, 0], target[0]], [on_crop[corner, 1], target[1]], color=SOFT, lw=1.0, zorder=2)
        longest = desc.max()
        reach = 0.46 * cell
        for row in range(4):
            for col in range(4):
                cx = px0 + (col + 0.5) * cell
                cy = py0 + side - (row + 0.5) * cell
                for o in range(8):
                    length = reach * desc[row, col, o] / longest
                    if length < 0.02:
                        continue
                    ang = np.deg2rad(45 * o)
                    # A head on a stub is all head: draw short bins as plain strokes.
                    head = "-|>,head_length=3,head_width=2" if length > 0.15 else "-"
                    ax.add_patch(
                        FancyArrowPatch(
                            (cx, cy),
                            (cx + length * np.cos(ang), cy + length * np.sin(ang)),
                            arrowstyle=head,
                            color=BLUE,
                            lw=2.0,
                            shrinkA=0,
                            shrinkB=0,
                            zorder=6,
                        )
                    )
        text(ax, px0 + side / 2, 1.15, "4 × 4 cells × 8 directions", ha="center")
        text(ax, px0 + side / 2, 0.7, "= 128 numbers", ha="center")
        del zoom

    build("logo-keypoints", draw, 3)


# --------------------------------------------------------------------------
# 2. Common ground: matches, then the ones one shift-turn-scale explains
# --------------------------------------------------------------------------


def match_fig() -> None:
    q = query()
    hit, miss = page(MATCH_PAGE), page(MISS_PAGE)
    f_hit, f_miss = fit(q, hit), fit(q, miss)
    assert (len(f_hit.inlier), f_hit.inliers) == (EXPECT["match_tentative"], EXPECT["match_inliers"])
    assert (len(f_miss.inlier), f_miss.inliers) == (EXPECT["miss_tentative"], EXPECT["miss_inliers"])

    def draw(ax: Axes, stage: int) -> None:
        # Whole pages, because a wrong match lands anywhere: the five the miss
        # page "agrees" on are in its body text, not its letterhead.
        left = draw_scan(ax, hit.image, 0.7, 1.05, height=4.4, fade=0.35)
        crop = draw_scan(ax, q.image, 5.0, 2.55, width=2.8)
        right = draw_scan(ax, miss.image, 8.75, 1.05, height=4.4, fade=0.35)
        text(ax, 5.0 + 1.4, 2.2, "the query", ha="center")
        if stage >= 2:
            draw_matches(ax, crop, left, f_hit, "all")
            draw_matches(ax, crop, right, f_miss, "all")
            text(ax, 0.7, 0.7, f"{len(f_hit.inlier)} matches", size=COUNT_PT, fontweight="bold")
            text(ax, 8.75, 0.7, f"{len(f_miss.inlier)} matches", size=COUNT_PT, fontweight="bold")
        if stage >= 3:
            draw_matches(ax, crop, left, f_hit, "split")
            draw_matches(ax, crop, right, f_miss, "split")
            draw_fit_outline(ax, q, hit, left, f_hit)
            text(ax, 0.7, 0.28, f"{f_hit.inliers} agree", size=COUNT_PT, fontweight="bold", color=GREEN)
            text(ax, 8.75, 0.28, f"{f_miss.inliers} agree", size=COUNT_PT, fontweight="bold", color=GREEN)

    build("logo-match", draw, 3)


# --------------------------------------------------------------------------
# 3. Text eats the budget: one page's keypoints at 1,024 and at 8,192
# --------------------------------------------------------------------------

#: The crest and a margin round it, in `BUDGET_PAGE`'s pixels: what each inset shows.
BUDGET_INSET = (1040, 15, 1570, 300)


def budget_fig() -> None:
    q = query()
    lean, full = page(BUDGET_PAGE, DEFAULT_MAX_FEATURES), page(BUDGET_PAGE)
    box = logo_box(BUDGET_PAGE)
    counts = {b: (on_box(p, box), _matcher().verify(q.features, p.features)) for b, p in ((1024, lean), (8192, full))}
    for b, (on, st) in counts.items():
        got = (on, st.inlier_count if st.model_ok else 0)
        want = (EXPECT[f"budget_on_crest_{b}"], EXPECT[f"budget_inliers_{b}"])
        assert got == want, (b, got, want)

    fig, ax = _canvas()
    for x0, p, b in ((0.7, lean, 1024), (6.55, full, 8192)):
        placed = draw_scan(ax, p.image, x0, 1.05, height=4.4, fade=0.7)
        draw_dots(ax, placed, p.keypoints_px(), color=BLUE, size=3.5)
        draw_box(ax, placed, box, color=INK, lw=1.4)
        inset = draw_scan(ax, p.image, x0 + placed.width + 0.2, 3.1, width=2.3, region=BUDGET_INSET, fade=0.6)
        draw_dots(ax, inset, p.keypoints_px(), color=BLUE, size=9)
        draw_box(ax, inset, box, color=INK, lw=1.4)
        # The inset is the box on the page, drawn larger: tie the two together.
        ((bx, by),) = placed.xy([[box[0] + box[2], box[1]]])
        ax.plot([bx, inset.x0], [by, inset.y0 + inset.height], color=SOFT, lw=1.0, zorder=2)
        on, st = counts[b]
        text(ax, x0, 0.7, f"{b:,} keypoints", size=COUNT_PT, fontweight="bold")
        text(ax, x0, 0.28, f"{on:,} on the crest, {st.inlier_count} agree")
    _save(fig, "logo-budget")


# --------------------------------------------------------------------------
# 4. Short list, long look: the two-stage shape, with real pages
# --------------------------------------------------------------------------

#: Pages drawn in the "every page" stack, back to front. Any Tobacco800 pages
#: would do; these include the miss page and two crest letters, because the
#: haystack holds both.
HAYSTACK = ("bbn00d00-first", "cmw44e00", "xxr58e00", "fcb93e00", "rin95e00")


def _stack(ax: Axes, stems, x0: float, y0: float, height: float, step: float) -> Placed:
    placed = None
    for k, stem in enumerate(stems):
        placed = draw_scan(ax, page(stem).image, x0 + k * step, y0 - k * step, height=height, fade=0.2)
        ax.add_patch(
            Rectangle((placed.x0, placed.y0), placed.width, placed.height, fill=False, ec=SOFT, lw=1.2, zorder=2)
        )
    assert placed is not None
    return placed


def _arrow(ax: Axes, x0: float, x1: float, y: float) -> None:
    ax.add_patch(
        FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>,head_length=10,head_width=6", color=INK, lw=2.2, zorder=6)
    )


def stages_fig() -> None:
    q = query()
    ranked = (MATCH_PAGE, *RANKED)
    counts = [fit(q, page(s)).inliers for s in ranked]
    assert counts[0] == EXPECT["match_inliers"] and tuple(counts[1:]) == EXPECT["ranked"], counts

    def draw(ax: Axes, stage: int) -> None:
        # Every page: the stack's front page sits at the bottom right.
        front = _stack(ax, HAYSTACK, 0.7, 2.45, 2.35, 0.12)
        cx = 0.7 + 4 * 0.12 + front.width / 2
        text(ax, cx, 1.35, "every page", ha="center", size=COUNT_PT, fontweight="bold")
        # Not "one vector each": a document page is cut into tiles, and region
        # voting tiles photos too, so a page may carry many (#4445).
        text(ax, cx, 0.92, "made into vectors", ha="center")
        if stage >= 2:
            _arrow(ax, 3.25, 5.75, 3.4)
            text(ax, 4.5, 3.9, "Stage 1", ha="center", size=COUNT_PT, fontweight="bold")
            text(ax, 4.5, 2.9, "compare vectors,", ha="center")
            text(ax, 4.5, 2.5, "all pages at once", ha="center")
            short = _stack(ax, ranked[:3], 6.0, 2.6, 1.95, 0.12)
            sx = 6.0 + 2 * 0.12 + short.width / 2
            text(ax, sx, 1.35, "a shortlist", ha="center", size=COUNT_PT, fontweight="bold")
            text(ax, sx, 0.92, "the top few", ha="center")
        if stage >= 3:
            _arrow(ax, 7.95, 10.3, 3.4)
            text(ax, 9.125, 3.9, "Stage 2", ha="center", size=COUNT_PT, fontweight="bold")
            text(ax, 9.125, 2.9, "SIFT + RANSAC,", ha="center")
            text(ax, 9.125, 2.5, "page by page", ha="center")
            for k, (stem, n) in enumerate(zip(ranked, counts, strict=True)):
                # One aspect for every crest, so the column reads as a list
                # rather than four differently-sized pictures.
                x, y, w, h = logo_box(stem)
                cx, cy, half_h = x + w / 2, y + h / 2, 0.62 * h
                half_w = 0.875 * 2 * half_h
                region = (int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h))
                y0 = 4.6 - k * 0.95
                crest = draw_scan(ax, page(stem).image, 10.55, y0, height=0.78, region=region)
                text(ax, 10.55 + crest.width + 0.12, y0 + 0.39, f"{n}", size=COUNT_PT, fontweight="bold", color=GREEN)
            text(ax, 11.3, 1.35, "ranked", ha="center", size=COUNT_PT, fontweight="bold")
            text(ax, 11.3, 0.92, "by how many agree", ha="center")

    build("logo-stages", draw, 3)


# --------------------------------------------------------------------------
# 5. Drowned out: the crest's share of a page vector, and of a tile's
# --------------------------------------------------------------------------


def tiles_fig() -> None:
    p = page(MATCH_PAGE)
    box = logo_box(MATCH_PAGE)
    pts = p.keypoints_px()
    x, y, w, h = box
    on = (pts[:, 0] >= x) & (pts[:, 0] <= x + w) & (pts[:, 1] >= y) & (pts[:, 1] <= y + h)
    (tx0, ty0, tx1, ty1), in_tile = best_tile(p, query())
    pw, ph = p.size
    got = {
        "tile_page_keypoints": p.features.count,
        "tile_on_crest": int(on.sum()),
        "tile_best_keypoints": int(in_tile.sum()),
        "tile_best_crest": int((in_tile & on).sum()),
    }
    want = {k: EXPECT[k] for k in got}
    if got != want:
        raise SystemExit(
            f"tiles: drew {got}, but EXPECT pins {want}. Pin the new counts, and update the "
            "shares slides/fragments/logo-tiles.md quotes from them."
        )
    n_tile, n_crest = got["tile_best_keypoints"], got["tile_best_crest"]

    def draw(ax: Axes, stage: int) -> None:
        left = draw_scan(ax, p.image, 0.7, 1.05, height=4.4, fade=0.6)
        draw_dots(ax, left, pts[~on], color=SOFT, size=1.3)
        draw_dots(ax, left, pts[on], color=GREEN, size=1.6)
        share = on.sum() / len(pts)
        text(ax, 0.7, 0.7, "one vector for the page", size=COUNT_PT, fontweight="bold")
        text(ax, 0.7, 0.28, f"the crest: {int(on.sum())} of {len(pts):,} keypoints, {share:.0%}")
        if stage < 2:
            return
        right = draw_scan(ax, p.image, 7.0, 1.05, height=4.4, fade=0.6)
        # Each layer's tiles start on a grid of half a tile, and the fine layer's
        # grid holds the coarse one's, so the coarse lines are drawn over it, heavier.
        # The fine grid is faint as well as thin: at `RULE` it made the page graph paper.
        for k, (tw, th) in enumerate(TILE_LAYERS):
            color, lw, z = (RULE, 1.2, 2.0) if k == 0 else (FAINT, 0.5, 1.5)
            for gx in np.arange(0, 1 + 1e-9, tw / 2):
                ax.plot(*right.xy([[gx * pw, 0], [gx * pw, ph]]).T, color=color, lw=lw, zorder=z)
            for gy in np.arange(0, 1 + 1e-9, th / 2):
                ax.plot(*right.xy([[0, gy * ph], [pw, gy * ph]]).T, color=color, lw=lw, zorder=z)
        # Only what the winning tile's vector sums over.
        draw_dots(ax, right, pts[in_tile & ~on], color=SOFT, size=1.3)
        draw_dots(ax, right, pts[in_tile & on], color=GREEN, size=1.6)
        tile_px = (round(tx0 * pw), round(ty0 * ph), round((tx1 - tx0) * pw), round((ty1 - ty0) * ph))
        draw_box(ax, right, tile_px, color=GREEN, lw=2.6)
        text(ax, 7.0, 0.7, "one vector per tile", size=COUNT_PT, fontweight="bold")
        if n_crest == n_tile:
            text(ax, 7.0, 0.28, f"the crest: all {n_tile} of its tile's keypoints")
        else:
            text(ax, 7.0, 0.28, f"the crest: {n_crest} of its tile's {n_tile}, {n_crest / n_tile:.0%}")

    build("logo-tiles", draw, 2)


# --------------------------------------------------------------------------
# 6. Show, don't tell: a Good vote's box becomes a second template
# --------------------------------------------------------------------------

#: The top of `FOUND_PAGE`, where its crest is: the only part the figure shows,
#: which holds every inlier either template finds.
FOUND_REGION = (560, 0, 2200, 760)


def votes_fig() -> None:
    q = query()
    voted = template_from_vote(VOTED_PAGE)
    found = page(FOUND_PAGE)
    f_query, f_vote = fit(q, found), fit(voted, found)
    by_query = f_query.inliers if f_query.model is not None else 0
    by_vote = f_vote.inliers if f_vote.model is not None else 0
    assert (by_query, by_vote) == (EXPECT["found_by_query"], EXPECT["found_by_vote"]), (by_query, by_vote)
    for f in (f_query, f_vote):
        if not inside(f.page_px[f.inlier], FOUND_REGION).all():
            raise SystemExit("votes: an inlier lands outside the region drawn")

    def draw(ax: Axes, stage: int) -> None:
        cand = draw_scan(ax, found.image, 6.7, 2.0, width=5.7, region=FOUND_REGION, fade=0.3)
        text(ax, 6.7 + cand.width / 2, 1.6, "a U.S.A. letter: the other crest", ha="center")
        top = draw_scan(ax, q.image, 2.9, 3.7, width=2.5)
        text(ax, 2.9, 3.4, "the query")
        draw_matches(ax, top, cand, f_query, "inliers", lw=1.1)
        text(ax, 5.55, 5.15, f"{by_query} agree", size=COUNT_PT, fontweight="bold", color=GREEN)
        text(ax, 7.0, 5.15, f"under {GATE}: a miss", color=SOFT)
        if stage >= 2:
            vp = page(VOTED_PAGE)
            thumb = draw_scan(ax, vp.image, 0.7, 0.55, height=2.6, fade=0.2)
            draw_box(ax, thumb, logo_box(VOTED_PAGE), color=GREEN, lw=2.2)
            x, y, w, h = logo_box(VOTED_PAGE)
            tpl = draw_scan(ax, voted.image, 2.9, 1.0, width=2.5, region=(x, y, x + w, y + h))
            text(ax, 2.9, 0.7, "a Good vote, boxed")
            ((bx, by),) = thumb.xy([[x + w, y]])
            ax.plot([bx, tpl.x0], [by, tpl.y0 + tpl.height], color=SOFT, lw=1.0, zorder=2)
            ((bx, by),) = thumb.xy([[x + w, y + h]])
            ax.plot([bx, tpl.x0], [by, tpl.y0], color=SOFT, lw=1.0, zorder=2)
        if stage >= 3:
            draw_matches(ax, tpl, cand, f_vote, "inliers", lw=1.1)
            text(ax, 5.55, 1.15, f"{by_vote} agree", size=COUNT_PT, fontweight="bold", color=GREEN)
            text(
                ax,
                6.7 + cand.width / 2,
                1.15,
                f"its score: the best template, {max(by_query, by_vote)}",
                ha="center",
                fontweight="bold",
            )

    build("logo-votes", draw, 3)


def main() -> None:
    keypoints_fig()
    match_fig()
    budget_fig()
    stages_fig()
    tiles_fig()
    votes_fig()


if __name__ == "__main__":
    main()
