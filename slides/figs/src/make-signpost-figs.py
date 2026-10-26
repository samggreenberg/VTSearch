#!/usr/bin/env python
"""What's in a Name (#5097): how a region of the Browse map gets its signpost, drawn from the Vision pile.

    python slides/figs/src/make-signpost-figs.py <container.pkl>
    python slides/figs/src/make-signpost-figs.py <container.pkl> --where

`shoot-ui-figs.mjs`'s `browse` group runs it on the container the app saved `photos-all` into: first
with `--where`, which prints the sign of the region below as JSON, so the zoomed frames close in on it;
then plain, once the shots are taken, to draw the figure. Run by hand it takes any dataset container
whose Browse map was built at import (Build Browse map now), because that build saves each photo's tags
beside the map.

The region is the one holding the most of the pile's book photos, among the regions of Toponymy's
finest layer. The deck hunts books, but no sign on this map says book: in COCO a book is mostly a shelf
in somebody's living room, so that is where the books landed, and what the region is called.

One figure, a four-page build (`vision-names`): on the left four of the region's book photos, the ones
nearest its sign; on the right the map, framed on the coarsest region holding it:

- a: the map, every photo a dot, and the four photos with the five tags SigLIP gave each, out of six
  hundred words (`signpost_texts.ZeroShotTagProvider`).
- b: the region picked out on the map, and the words of its name picked out in its photos' tags.
- c: the name, lettered at the region's sign as the app letters it, among the names of the regions
  nearest it: each is the phrase its own photos keep using and the rest of the map does not.
- d: the region of the coarsest layer holding it, and that region's name, hedged with a `~` as the app
  hedges its two coarsest bands (`sign-layout.ts`): at that size a phrase is a guess.

Nothing the slide shows is decided here. The map is the app's layout, the tags the app's, and the names
the signposts the app saved with them. Only which photos each region holds is worked out again,
because the app saves a name and where it goes, not what it covers: Toponymy's clusterer runs again on
the same vectors under the layout's own seed (`signpost_build._clusterable_vectors`), and every region
it finds has to land on one of the app's own signs, whose anchor is its members' medoid, or the script
refuses to draw. That needs the app's float rounding too, so the script pins OpenBLAS's kernels as
`app.py` does before numpy loads; under the machine's own kernels UMAP lays the clusterable space out a
little differently and the regions come out different.
"""

from __future__ import annotations

import io
import json
import os
import platform
import sys
from pathlib import Path

# The app's OpenBLAS kernels (`app.py`, #4481), set before numpy loads; see the module docstring.
if platform.machine() in ("x86_64", "AMD64"):
    os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
    if os.environ["OPENBLAS_CORETYPE"].lower() == "haswell":
        os.environ.setdefault("OPENBLAS_L2_SIZE", "2048")

# Ensure the repo root is importable no matter the cwd: ``python
# slides/figs/src/x.py`` only puts the script's own dir on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, TITLE_NOTCH_PX, save  # noqa: E402

from vtscore.datasets.container import read_container, read_projection, read_region_labels  # noqa: E402
from vtscore.projection.labels import medoid  # noqa: E402
from vtscore.projection.signpost_build import _base_min_cluster_size, _clusterable_vectors  # noqa: E402

OUT = Path(__file__).resolve().parent.parent

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 17,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)

#: The canvas: the whole slide, at 100 slide pixels to the inch, saved uncropped so every page of the
#: build shares one frame and the layout below is in slide pixels.
FIG_W, FIG_H = 12.8, 7.2
W, H = 1280, 720
#: Slide pixels per point on this canvas.
PX_PER_PT = W / (FIG_W * 72)

#: The photo column, under the title notch: one row per photo, the thumbnail and then its tags.
COLUMN_X, COLUMN_TOP, COLUMN_BOTTOM = 60.0, 196.0, 690.0
THUMB = 104.0
TAGS_X = COLUMN_X + THUMB + 18.0
TAGS_RIGHT = 590.0
PHOTOS = 4

#: The map, right of the column, the whole height of the slide.
MAP_BOX = (630.0, 40.0, 1240.0, 690.0)

#: Type, in points: 15pt renders at 21 slide pixels, over the 20px floor. The coarse region's sign is
#: larger, as the app draws a sign the view has zoomed in past.
TAG_PT = 15.0
SIGN_PT = 15.0
COARSE_PT = 19.0
#: How many of the nearest regions' names stage c letters beside the region's own.
NEIGHBOURS = 5

#: Ink. Emphasis is darkness and weight, never hue (`slides/STYLE.md`), and that holds on the map too:
#: the region is the dark dots, the coarse region round it the mid ones, everything else the light.
DOT_REST = "#c8ced8"
DOT_COARSE = "#8f99a8"
DOT_REGION = INK
#: The app's sign: heavy lettering on a pale pill, which lets the map show through a little.
PILL = "#eef1f5e0"

#: The app hedges a clustered sign this many bands coarser than the finest (`SIGN_APPROXIMATE_RANK`
#: in `sign-layout.ts`), and the band step its levels sit at (`signpost_build._LEVEL_STEP`).
APPROXIMATE_RANK = 2
LEVEL_STEP = 1.8


def _load(pkl: Path) -> dict:
    """Everything the figure draws, from the container: layout, tags, signs, thumbnails, vectors."""
    data, meta = read_container(pkl)
    found = read_projection(pkl, "square") or read_projection(pkl, "hex")
    if found is None:
        raise SystemExit(f"{pkl.name}: no Browse map saved in it; build one (Build Browse map now) and re-run")
    proj = found[0]
    if proj.random_state is None:
        raise SystemExit(
            f"{pkl.name}: its map was laid out unseeded, so its regions cannot be found again; serve it from "
            "an app started with VTSEARCH_PROJECTION_SEED set, as shoot-ui-figs.mjs starts one"
        )
    signs = read_region_labels(pkl)
    if signs is None or not signs[0].labels:
        raise SystemExit(f"{pkl.name}: no signposts saved with its map")
    medias = data["medias"]
    ids = [int(i) for i in proj.ids]
    texts = [medias[i].get("signpost_text") or "" for i in ids]
    if sum(bool(t) for t in texts) < 0.9 * len(ids):
        raise SystemExit(
            f"{pkl.name}: its photos carry no saved tags; import it with Build Browse map now, which saves "
            "them with the map"
        )
    embedder = meta.get("embedder") or medias[ids[0]].get("embedder")
    return {
        "ids": ids,
        "coords": np.asarray(proj.coords, dtype=np.float32),
        "seed": proj.random_state,
        "texts": texts,
        "names": [str(medias[i].get("filename") or "") for i in ids],
        "vectors": np.stack([np.asarray(medias[i]["embeddings"][embedder], dtype=np.float32) for i in ids]),
        "signs": list(signs[0].labels),
        "thumbs": [medias[i].get("thumbnail_bytes") for i in ids],
    }


def _regions(d: dict) -> list[dict]:
    """Every region of every Toponymy layer, finest first, each matched to the app's sign for it.

    The clusterer is the one `signpost_build._fit_topic_layers` hands Toponymy, on the clusterable
    vectors the app fit it on, under the same seed; naming plays no part in which photos a region
    holds. A region is matched to a sign by level and anchor, and a region that matches none, or a sign
    no region matches, stops the script: the photos it would draw would not be the ones the app named.
    """
    from toponymy.clustering import ToponymyClusterer  # noqa: PLC0415

    clusterable = _clusterable_vectors(d["vectors"], random_state=d["seed"])
    clusterer = ToponymyClusterer(
        min_clusters=4, base_min_cluster_size=_base_min_cluster_size(len(d["ids"])), verbose=False
    )
    layers, _tree = clusterer.fit_predict(clusterable, d["vectors"])
    signs = {(round(s.level, 3), round(s.x, 4), round(s.y, 4)): s for s in d["signs"]}
    regions = []
    for i, layer in enumerate(layers):
        level = round((len(layers) - 1 - i) * LEVEL_STEP, 3)
        assignment = np.asarray(layer.cluster_labels)
        for topic in range(int(assignment.max()) + 1):
            members = np.nonzero(assignment == topic)[0]
            if members.size == 0:
                continue
            ax, ay = medoid(d["coords"][members])
            sign = signs.pop((level, round(float(ax), 4), round(float(ay), 4)), None)
            if sign is None:
                raise SystemExit(
                    f"a layer-{i} region of {members.size} photos lands on none of the app's signs: the "
                    "clusters did not reproduce, so the figure would name photos the app did not"
                )
            regions.append({"layer": i, "level": level, "members": members, "sign": sign})
    if signs:
        raise SystemExit(f"{len(signs)} of the app's signs match no region found again")
    return regions


def _pick(regions: list[dict], d: dict) -> tuple[dict, list[dict], dict]:
    """The region the figure is about, the finest regions nearest it, and the coarsest region holding it.

    The region is the finest one holding the most book photos (the corpus files a photo under `book/`
    when COCO's largest box in it is a book), the largest finest one on a pile with none.
    """
    books = np.array([name.startswith("book/") for name in d["names"]])
    finest = [r for r in regions if r["layer"] == 0]
    region = max(finest, key=lambda r: (int(books[r["members"]].sum()), r["members"].size))
    here = np.array([region["sign"].x, region["sign"].y])
    nearby = sorted(
        (r for r in finest if r is not region),
        key=lambda r: float(np.hypot(r["sign"].x - here[0], r["sign"].y - here[1])),
    )
    top = max(r["layer"] for r in regions)
    coarse = max(
        (r for r in regions if r["layer"] == top),
        key=lambda r: np.intersect1d(r["members"], region["members"]).size,
    )
    return region, nearby, coarse


def _square_thumb(raw: bytes | None) -> np.ndarray:
    """A thumbnail cropped square about its centre, as the app's tiles crop theirs."""
    if not raw:
        return np.full((8, 8, 3), 230, dtype=np.uint8)
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        side = min(im.size)
        left, top = (im.width - side) // 2, (im.height - side) // 2
        return np.asarray(im.crop((left, top, left + side, top + side)).resize((256, 256), Image.Resampling.LANCZOS))


def _frame(region: dict, nearby: list[dict], coarse: dict):
    """Projection space to slide pixels, centred on the region's sign and as wide as its neighbourhood.

    Wide enough to hold the nearest `NEIGHBOURS` signs and the coarse region's, with a margin for their
    lettering; at the whole pile's scale the region is a smudge and its neighbours' names pile up.
    """
    here = np.array([region["sign"].x, region["sign"].y])
    anchors = [(r["sign"].x, r["sign"].y) for r in nearby[:NEIGHBOURS]] + [(coarse["sign"].x, coarse["sign"].y)]
    reach = max(float(np.abs(np.array(a) - here).max()) for a in anchors) * 1.4
    x0, y0, x1, y1 = MAP_BOX
    scale = min(x1 - x0, y1 - y0) / (2 * reach)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return lambda px, py: (cx + (np.asarray(px) - here[0]) * scale, cy + (np.asarray(py) - here[1]) * scale)


def _words(text: str) -> set[str]:
    return {w.lower() for w in text.replace(",", " ").split() if w}


def _draw_tags(ax, x: float, y: float, text: str, keep: set[str] | None) -> None:
    """One photo's tags, wrapped to the column; with *keep*, its name's words dark and bold, the rest soft.

    A tag never breaks across lines, so the comma-separated list reads as five things.
    """
    renderer = ax.figure.canvas.get_renderer()

    def width(s: str, bold: bool = False) -> float:
        t = ax.text(0, 0, s, fontsize=TAG_PT, fontweight="bold" if bold else "normal")
        w = t.get_window_extent(renderer).width
        t.remove()
        return w

    space = width("a a") - width("aa")
    line = TAG_PT * PX_PER_PT * 1.3
    tags = [t.strip() for t in text.split(",") if t.strip()]
    cx, cy = x, y
    for n, tag in enumerate(tags):
        piece = tag + ("," if n < len(tags) - 1 else "")
        if cx > x and cx + width(piece, bold=True) > TAGS_RIGHT:
            cx, cy = x, cy + line
        for word in piece.split():
            strong = keep is not None and word.rstrip(",").lower() in keep
            t = ax.text(
                cx,
                cy,
                word,
                fontsize=TAG_PT,
                fontweight="bold" if strong else "normal",
                color=INK if (keep is None or strong) else SOFT,
                va="top",
                ha="left",
            )
            cx += t.get_window_extent(renderer).width + space


def _sign(ax, x: float, y: float, text: str, hedged: bool, pt: float):
    """A signpost as the app letters one: bold type on a pale pill; a hedged one `~`, italic."""
    return ax.text(
        x,
        y,
        ("~ " + text) if hedged else text,
        fontsize=pt,
        fontweight="bold",
        fontstyle="italic" if hedged else "normal",
        color=INK,
        ha="center",
        va="center",
        zorder=8,
        bbox={"boxstyle": "round,pad=0.3,rounding_size=0.8", "fc": PILL, "ec": "none"},
    )


def _plan_signs(ax, to_px, region: dict, nearby: list[dict], coarse: dict, finest_level: float) -> list[tuple]:
    """Where the last page's signs go, decided once, so that no page of the build moves one.

    The region's own sign first, at its anchor; then the coarse region's, larger, as the app draws a sign
    the view is zoomed in past, and moved straight up or down off anything it would cover; then the
    nearest neighbours that fit without covering either, as the app de-clutters, the nearer keeping its
    place. Returns `(region, hedged, pt, x, y, role)` per sign.
    """
    renderer = ax.figure.canvas.get_renderer()
    x0, y0, x1, y1 = MAP_BOX
    taken: list[tuple[float, float, float, float]] = []

    def box(text: str, hedged: bool, pt: float, x: float, y: float) -> tuple[float, float, float, float]:
        t = _sign(ax, x, y, text, hedged, pt)
        e = t.get_window_extent(renderer)
        t.remove()
        pad = 0.3 * pt * PX_PER_PT
        return e.x0 - pad, e.x1 + pad, H - e.y1 - pad, H - e.y0 + pad

    def fits(b: tuple[float, float, float, float]) -> bool:
        left, right, top, bottom = b
        inside = left >= x0 - 10 and right <= W - 20 and top >= y0 and bottom <= y1
        return inside and all(
            right + 8 < lo or left - 8 > hi or bottom + 4 < up or top - 4 > down for lo, hi, up, down in taken
        )

    plan = []
    rx, ry = (float(v) for v in to_px(region["sign"].x, region["sign"].y))
    taken.append(box(region["sign"].text, False, SIGN_PT, rx, ry))
    plan.append((region, False, SIGN_PT, rx, ry, "region"))

    hedged = round((finest_level - coarse["level"]) / LEVEL_STEP) >= APPROXIMATE_RANK
    cx, cy = (float(v) for v in to_px(coarse["sign"].x, coarse["sign"].y))
    step = COARSE_PT * PX_PER_PT * 1.6
    for dy in [0.0] + [s * k * step for k in range(1, 12) for s in (-1, 1)]:
        b = box(coarse["sign"].text, hedged, COARSE_PT, cx, cy + dy)
        if fits(b):
            taken.append(b)
            plan.append((coarse, hedged, COARSE_PT, cx, cy + dy, "coarse"))
            break

    near = 0
    for r in nearby:
        if near >= NEIGHBOURS:
            break
        nx, ny = (float(v) for v in to_px(r["sign"].x, r["sign"].y))
        b = box(r["sign"].text, False, SIGN_PT, nx, ny)
        if fits(b):
            taken.append(b)
            plan.append((r, False, SIGN_PT, nx, ny, "near"))
            near += 1
    return plan


def draw(stage: int, d: dict, region: dict, nearby: list[dict], coarse: dict, finest_level: float):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.canvas.draw()

    to_px = _frame(region, nearby, coarse)
    xs, ys = to_px(d["coords"][:, 0], d["coords"][:, 1])
    x0, y0, x1, y1 = MAP_BOX
    framed = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    shade = np.zeros(len(xs), dtype=int)
    if stage >= 4:
        shade[coarse["members"]] = 1
    if stage >= 2:
        shade[region["members"]] = 2
    for level, colour, size in ((0, DOT_REST, 14), (1, DOT_COARSE, 14), (2, DOT_REGION, 20)):
        on = (shade == level) & framed
        ax.scatter(xs[on], ys[on], s=size, c=colour, linewidths=0, zorder=2 + level)

    keep = _words(region["sign"].text) if stage >= 2 else None
    here = np.array([region["sign"].x, region["sign"].y])
    nearest = sorted(
        region["members"].tolist(),
        key=lambda m: (not d["names"][m].startswith("book/"), float(np.hypot(*(d["coords"][m] - here)))),
    )
    row = (COLUMN_BOTTOM - COLUMN_TOP) / PHOTOS
    for k, member in enumerate(nearest[:PHOTOS]):
        top = COLUMN_TOP + k * row
        ax.imshow(
            _square_thumb(d["thumbs"][member]),
            extent=(COLUMN_X, COLUMN_X + THUMB, top + THUMB, top),
            interpolation="lanczos",
            zorder=3,
        )
        _draw_tags(ax, TAGS_X, top + 2, d["texts"][member], keep)

    shown = {"region", "near"} if stage == 3 else {"region", "near", "coarse"} if stage >= 4 else set()
    for r, hedged, pt, x, y, role in _plan_signs(ax, to_px, region, nearby, coarse, finest_level):
        if role in shown:
            _sign(ax, x, y, r["sign"].text, hedged, pt)
    return fig


def main() -> None:
    args = sys.argv[1:]
    where = "--where" in args
    paths = [a for a in args if a != "--where"]
    if len(paths) != 1:
        raise SystemExit("usage: make-signpost-figs.py <dataset container .pkl> [--where]")
    d = _load(Path(paths[0]))
    regions = _regions(d)
    region, nearby, coarse = _pick(regions, d)
    if where:
        sign = region["sign"]
        print(json.dumps({"text": sign.text, "x": sign.x, "y": sign.y, "level": sign.level}))
        return
    books = sum(d["names"][m].startswith("book/") for m in region["members"].tolist())
    print(
        f"region {region['sign'].text!r}: {region['members'].size} photos, {books} of them books; "
        f"coarsest {coarse['sign'].text!r} ({coarse['members'].size} photos)"
    )
    finest_level = max(r["level"] for r in regions)
    assert TITLE_NOTCH_PX[0] + TITLE_NOTCH_PX[2] <= MAP_BOX[0] and TITLE_NOTCH_PX[1] + TITLE_NOTCH_PX[3] <= COLUMN_TOP
    for stage, name in (
        (1, "vision-names.build1.webp"),
        (2, "vision-names.build2.webp"),
        (3, "vision-names.build3.webp"),
        (4, "vision-names.webp"),
    ):
        save(draw(stage, d, region, nearby, coarse, finest_level), OUT, name, FULL_BLEED, tight=False, photo=True)


if __name__ == "__main__":
    main()
