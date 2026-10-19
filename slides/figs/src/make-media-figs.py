#!/usr/bin/env python
"""What VTSearch searches, how a file becomes something it can search, and how a photo becomes a face.

    python slides/figs/src/make-media-figs.py

Four line-art schematics, drawn in Embed-time Stories' vocabulary (`make-intro-figs.py`: hollow block
arrows with their word inside, a wire cube of hollow dots, line-art media) so the room reads them the
way it read that slide:

* `figs/media-types*.webp`, **Mixed Media** (section X): the six media types, each with the model
  that turns it into vectors and, where it has one, a phrase a user could type. Pages: a, images
  alone, as section I met them; b, audio, video and text, which take a typed phrase too; c, documents
  and faces, which do not.
* `figs/media-convert*.webp`, **Shape Shifters** (section X): the same six, and the converters
  between them. Pages: a, a document's two; b, video's and audio's; c, an image's.
* `figs/media-clean*.webp`, **Cut and Dried** (section X): what an import can do to each file after
  that, one row per kind: cut it into pieces, then tidy each piece. Pages: a, the files; b, cut;
  c, cleaned.
* `figs/face-find*.webp`, **Face Off** (section XII, after the FHIBE card): a photo, the face the
  detector finds in it with its five points, the crop turned upright by them, and FaceNet's space,
  where one person's crops land together. Pages a to d, one per step.

**Nothing on them is a guess.** `_check_registries` asserts what the figures draw against the
shipped registries, so a new converter or a changed default fails here rather than leaving a slide
that no longer describes the app: the six types; each one's default embedder and whether it takes a
typed phrase (`MediaEmbedder.supports_text`); the eight converters, drawn as exactly the arrows
`list_converters()` returns; and the clippers and cleaners Cut and Dried names. Face Off's five
points are the converter's own `FACE_TEMPLATE` turned by `FACE_TILT`, and `_check_face` asserts that
the converter's `fit_similarity` reads that turn back off them, which is the angle a `roll` crop is
turned by.

Face Off is drawn rather than photographed. FHIBE's photos live on the GRID, and the card's two
pages already show ten of the twenty its terms allow (`make-data-cards.FHIBE_PHOTO_LIMIT`), every
one under the card's own face box.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SRC))
from slide_figure import FULL_BLEED, LABEL_GAP_PT, save  # noqa: E402

from vtscore.converters import list_converters  # noqa: E402
from vtscore.converters.image2face import FACE_TEMPLATE, fit_similarity  # noqa: E402
from vtscore.media import all_cleaners, all_clippers, all_embedders, all_types  # noqa: E402

OUT = SRC.parent


def _intro():
    """Embed-time Stories' generator, imported by path, as `make-atlas-figs.py` does."""
    path = SRC / "make-intro-figs.py"
    spec = importlib.util.spec_from_file_location("intro_figs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["intro_figs"] = module
    spec.loader.exec_module(module)
    return module


INTRO = _intro()
INK, SOFT, GREEN, WIRE = INTRO.INK, INTRO.SOFT, INTRO.GREEN, INTRO.WIRE
CANVAS, UNIT_PT = INTRO.CANVAS, INTRO.UNIT_PT
#: Type sizes: Embed-time Stories' word in an arrow, name under an object and typed phrase.
ARROW_PT, NAME_PT, QUERY_PT = INTRO.FLOW_ARROW_PT, INTRO.FLOW_NAME_PT, 18.0
#: A line of detail under a name: an embedder, a converter's word, a note.
DETAIL_PT = 17.0
#: The deck's object and label gaps, in this canvas's units.
GAP = INTRO.FLOW_GAP
LABEL_GAP = LABEL_GAP_PT / UNIT_PT
#: The FHIBE card's box blue (`data_card.CUT`): "Blue box is the face" on that card, so the
#: detector's box and points are drawn in it here.
BOX_BLUE = "#2b6cb0"
#: Line weights: an object's frame, and the drawing inside it.
FRAME_LW, INNER_LW = 1.6, 1.4

plt.rcParams.update({"font.family": ["DejaVu Sans"], "savefig.dpi": 200})


def _canvas() -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=tuple(c * UNIT_PT / 72 for c in CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


def _text_width(text: str, size: float, bold: bool = False) -> float:
    """A string's width in canvas units, measured off its glyphs."""
    prop = FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return TextPath((0, 0), text, size=size, prop=prop).get_extents().width / UNIT_PT


# ──────────────────────────────────────────────────────────────────────────── the media, drawn

#: An icon's frame: landscape for the media a screen shows wide, portrait for a page.
ICON_W, ICON_H = 1.9, 1.4
PAGE_W, PAGE_H = 1.15, 1.5


def _frame(ax: plt.Axes, cx: float, cy: float, w: float, h: float, ls: str = "-") -> tuple[float, float]:
    """A white card centred on (cx, cy); returns its bottom-left corner."""
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor=INK, lw=FRAME_LW, ls=ls, zorder=3))
    return x0, y0


def _image(ax: plt.Axes, cx: float, cy: float, w: float = ICON_W, h: float = ICON_H) -> None:
    """Embed-time Stories' photograph: two hills and a sun."""
    x0, y0 = _frame(ax, cx, cy, w, h)
    inset = 0.12
    left, right, base, top = x0 + inset, x0 + w - inset, y0 + inset, y0 + h - inset
    span, rise = right - left, top - base
    for pts, face in (
        ([(left, base), (left + 0.72 * span, base), (left + 0.34 * span, base + 0.62 * rise)], "none"),
        ([(left + 0.44 * span, base), (right, base), (right - 0.22 * span, base + 0.46 * rise)], "white"),
    ):
        ax.add_patch(Polygon(pts, closed=True, facecolor=face, edgecolor=SOFT, lw=INNER_LW, zorder=4))
    ax.add_patch(plt.Circle((left + 0.78 * span, base + 0.76 * rise), 0.12 * h, fill=False, ec=SOFT, lw=INNER_LW,
                            zorder=4))  # fmt: skip


def _wave(t: np.ndarray, quiet: list[tuple[float, float]]) -> np.ndarray:
    """A made-up recording's waveform at times *t* in [0, 1], flat inside each *quiet* span."""
    envelope = 0.55 + 0.45 * np.sin(7.3 * t + 0.6) * np.sin(2.1 * t + 1.0)
    signal = envelope * np.sin(2 * np.pi * 15 * t) * (0.65 + 0.35 * np.sin(2 * np.pi * 37 * t + 0.4))
    for a, b in quiet:
        signal[(t >= a) & (t <= b)] = 0.0
    return signal


def _audio(ax: plt.Axes, cx: float, cy: float, w: float = ICON_W, h: float = ICON_H,
           quiet: list[tuple[float, float]] | None = None, span: tuple[float, float] = (0.0, 1.0)) -> None:  # fmt: skip
    """A waveform in a card: the stretch *span* of `_wave`'s recording, with its *quiet* spans flat."""
    x0, y0 = _frame(ax, cx, cy, w, h)
    inset = 0.12
    t = np.linspace(*span, 400)
    y = _wave(t, quiet or [])
    xs = np.linspace(x0 + inset, x0 + w - inset, len(t))
    ax.plot(xs, cy + y * (h / 2 - inset), color=SOFT, lw=INNER_LW, zorder=4, solid_joinstyle="round")


def _video(ax: plt.Axes, cx: float, cy: float, w: float = ICON_W, h: float = ICON_H, bars: bool = False) -> None:
    """A film frame: sprocket holes along both edges and a play mark; with *bars*, letterbox bars inside."""
    x0, y0 = _frame(ax, cx, cy, w, h)
    hole_w, hole_h, strip = 0.11, 0.085, 0.2
    n = max(4, int(round(w / 0.32)))
    for k in range(n):
        hx = x0 + (k + 0.5) * w / n - hole_w / 2
        for hy in (y0 + (strip - hole_h) / 2, y0 + h - strip + (strip - hole_h) / 2):
            ax.add_patch(Rectangle((hx, hy), hole_w, hole_h, facecolor="none", edgecolor=SOFT, lw=1.0, zorder=4))
    for yy in (y0 + strip, y0 + h - strip):
        ax.plot([x0, x0 + w], [yy, yy], color=SOFT, lw=INNER_LW, zorder=4)
    inner = h - 2 * strip
    if bars:
        bar = 0.2 * inner
        for by in (y0 + strip, y0 + h - strip - bar):
            ax.add_patch(Rectangle((x0, by), w, bar, facecolor=SOFT, edgecolor="none", zorder=4))
    tri = 0.22 * inner if bars else 0.3 * inner
    ax.add_patch(Polygon([(cx - tri * 0.8, cy - tri), (cx - tri * 0.8, cy + tri), (cx + tri, cy)], closed=True,
                         facecolor="none", edgecolor=SOFT, lw=INNER_LW, zorder=5))  # fmt: skip


def _lines(ax: plt.Axes, x0: float, y_top: float, width: float, count: int, step: float, last: float = 0.6) -> None:
    """A paragraph: *count* lines of text, the last one *last* as long."""
    for k in range(count):
        yy = y_top - k * step
        length = width * (last if k == count - 1 else 1.0)
        ax.plot([x0, x0 + length], [yy, yy], color=SOFT, lw=2.4, zorder=4, solid_capstyle="round")


def _text(ax: plt.Axes, cx: float, cy: float, w: float = ICON_W, h: float = ICON_H, count: int = 4) -> None:
    """A card of plain text."""
    x0, y0 = _frame(ax, cx, cy, w, h)
    inset = 0.2
    _lines(ax, x0 + inset, y0 + h - inset, w - 2 * inset, count, (h - 2 * inset) / (count - 1))


#: Markup, as a text card draws it: small, monospaced, quiet.
TAG = dict(fontsize=12, color=SOFT, family="DejaVu Sans Mono", va="center", zorder=4)


def _paragraph(ax: plt.Axes, cx: float, cy: float, w: float, h: float, tags: bool) -> None:
    """One paragraph in a card; with *tags*, wrapped in markup, each tag on a line of its own."""
    x0, y0 = _frame(ax, cx, cy, w, h)
    inset, step = 0.17, (h - 0.34) / 2
    top = y0 + h - inset
    if tags:
        ax.text(x0 + inset, top, "<p>", ha="left", **TAG)
        ax.text(x0 + inset, top - 2 * step, "</p>", ha="left", **TAG)
        _lines(ax, x0 + inset, top - step, w - 2 * inset, 1, step, last=1.0)
    else:
        _lines(ax, x0 + inset, top, w - 2 * inset, 3, step)


def _paragraphs(ax: plt.Axes, cx: float, cy: float, w: float, h: float) -> None:
    """A whole file of them: three paragraphs, a line each, the markup either side of it."""
    x0, y0 = _frame(ax, cx, cy, w, h)
    inset, step = 0.17, (h - 0.34) / 2
    for k in range(3):
        yy = y0 + h - inset - k * step
        ax.text(x0 + inset, yy, "<p>", ha="left", **TAG)
        ax.text(x0 + w - inset, yy, "</p>", ha="right", **TAG)
        ax.plot([x0 + inset + 0.6, x0 + w - inset - 0.75], [yy, yy], color=SOFT, lw=2.4, zorder=4,
                solid_capstyle="round")  # fmt: skip


def _document(ax: plt.Axes, cx: float, cy: float, w: float = PAGE_W, h: float = PAGE_H, ls: str = "-") -> None:
    """A page: folded corner, a letterhead mark and lines of text. Dashed *ls*: nothing embeds it as it is."""
    x0, y0 = cx - w / 2, cy - h / 2
    fold = 0.28
    outline = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h - fold), (x0 + w - fold, y0 + h), (x0, y0 + h)]
    ax.add_patch(Polygon(outline, closed=True, facecolor="white", edgecolor=INK, lw=FRAME_LW, ls=ls, zorder=3))
    ax.plot([x0 + w - fold, x0 + w - fold, x0 + w], [y0 + h, y0 + h - fold, y0 + h - fold], color=INK,
            lw=FRAME_LW * 0.8, ls=ls, zorder=4)  # fmt: skip
    ax.add_patch(plt.Circle((x0 + 0.27, y0 + h - 0.27), 0.12, fill=False, ec=SOFT, lw=INNER_LW, zorder=4))
    _lines(ax, x0 + 0.17, y0 + h - 0.62, w - 0.34, 4, 0.22)


# ──────────────────────────────────────────────────────────────────────────── a face, from the template


#: How far Face Off's photo tilts the head, in degrees counter-clockwise: enough that a crop turned
#: upright is plainly turned. The FHIBE photo on the card's zoom tilts by about this much.
FACE_TILT = 24.0


def _face_points(cx: float, cy: float, side: float, angle: float) -> np.ndarray:
    """`FACE_TEMPLATE`'s five points for a face box of *side* centred on (cx, cy), turned *angle*°.

    The template is in a box's units with y down; the canvas has y up, so a point's height is flipped
    about the box's middle before it is turned.
    """
    t = np.radians(angle)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    local = (np.array(FACE_TEMPLATE) - 0.5) * np.array([1.0, -1.0]) * side
    return local @ rot.T + np.array([cx, cy])


def _to_box(cx: float, cy: float, side: float, angle: float, u: float, v: float) -> np.ndarray:
    """A point given in the face box's units (y down), on the canvas."""
    t = np.radians(angle)
    lx, ly = (u - 0.5) * side, (0.5 - v) * side
    return np.array([cx + lx * math.cos(t) - ly * math.sin(t), cy + lx * math.sin(t) + ly * math.cos(t)])


def _face(ax: plt.Axes, cx: float, cy: float, side: float, angle: float, z: float = 4) -> None:
    """A line-art face whose eyes, nose and mouth sit on `FACE_TEMPLATE`'s points.

    The head is an ellipse a little taller than the box and centred above it, as a face box crops
    eyebrows-to-chin; the mouth is a smile through the two mouth corners.
    """
    head = _to_box(cx, cy, side, angle, 0.5, 0.46)
    ax.add_patch(Ellipse(head, 1.04 * side, 1.34 * side, angle=angle, facecolor="white", edgecolor=SOFT,
                         lw=INNER_LW + 0.3, zorder=z))  # fmt: skip
    pts = _face_points(cx, cy, side, angle)
    for eye in pts[:2]:
        ax.add_patch(plt.Circle(eye, 0.055 * side, facecolor=SOFT, edgecolor="none", zorder=z + 0.1))
    bridge = _to_box(cx, cy, side, angle, 0.5, 0.43)
    ax.plot([bridge[0], pts[2][0]], [bridge[1], pts[2][1]], color=SOFT, lw=INNER_LW, zorder=z + 0.1,
            solid_capstyle="round")  # fmt: skip
    dip = _to_box(cx, cy, side, angle, 0.515, 0.87)
    mouth = MplPath([pts[3], dip, pts[4]], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax.add_patch(PathPatch(mouth, facecolor="none", edgecolor=SOFT, lw=INNER_LW, zorder=z + 0.1, capstyle="round"))


def _face_icon(ax: plt.Axes, cx: float, cy: float, w: float = 1.4, h: float = ICON_H) -> None:
    """Faces, as a card: one upright face, the way a crop holds it."""
    _frame(ax, cx, cy, w, h)
    _face(ax, cx, cy - 0.06 * h, 0.62 * h, 0.0)


def _check_face() -> None:
    """The converter reads Face Off's turn back off the drawn points: the angle a `roll` crop turns by.

    `fit_similarity` takes the template onto pixel coordinates, whose y runs down, so the canvas
    points are flipped into that frame first; a counter-clockwise tilt on the canvas then reads as
    a clockwise turn there, and `atan2(b, a)` returns it negated.
    """
    pts = _face_points(0.0, 0.0, 1.0, FACE_TILT) * np.array([1.0, -1.0])
    a, b, _tx, _ty = fit_similarity(FACE_TEMPLATE, pts)
    turn = -math.degrees(math.atan2(b, a))
    if abs(turn - FACE_TILT) > 1e-6:
        raise SystemExit(f"Face Off: the converter reads a turn of {turn:.3f}° off the drawn face, not {FACE_TILT}°")


# ──────────────────────────────────────────────────────────────────────────── what the registries say

#: Every media type, where Shape Shifters puts it, what both media slides call it, how it is drawn,
#: its default embedder's name as the slide prints it, and what the slide prints for a typed
#: phrase: a phrase in quotes for a type whose default embedder takes one, else a note. Two rows of
#: three, sources on the left and the image hub in the middle, so the converters' arrows are short.
MEDIA = {
    "video": dict(at=(0, 1), name="Video", embedder="X-CLIP", phrase='"a goal"'),
    "image": dict(at=(1, 1), name="Images", embedder="SigLIP", phrase='"book"'),
    "face": dict(at=(2, 1), name="Faces", embedder="FaceNet", note="start from a photo"),
    "audio": dict(at=(0, 0), name="Audio", embedder="CLAP", phrase='"dog barking"'),
    "text": dict(at=(1, 0), name="Text", embedder="E5", phrase='"a complaint"'),
    "document": dict(at=(2, 0), name="Documents", embedder=None, note="convert first"),
}
#: The registry name of each default embedder the slide prints by its family name.
EMBEDDER_IDS = {"X-CLIP": "xclip", "SigLIP": "siglip", "FaceNet": "face", "CLAP": "clap_general", "E5": "e5"}

#: Every converter as an arrow on Shape Shifters: (source, target, the word along it), in the
#: order the build reveals them, a source at a time.
CONVERTERS = (
    ("document", "image", "pages"),
    ("document", "text", "text"),
    ("video", "image", "frames"),
    ("video", "audio", "sound"),
    ("audio", "text", "speech"),
    ("audio", "image", "spectrogram"),
    ("image", "face", "faces"),
    ("image", "text", "OCR"),
)
#: Which page of Shape Shifters each source's arrows arrive on.
CONVERT_PAGE = {"document": 1, "video": 2, "audio": 2, "image": 3}

#: Cut and Dried's rows: a kind of media, the clipper that cuts it and the word for its pieces, and
#: the cleaner that tidies each piece and what it does, as the slide prints them.
CLEAN_ROWS = (
    ("audio", "sound_tiling", "clips", "audio_silence_trim", "trim silence"),
    ("video", "video_scene", "scenes", "video_letterbox_crop", "crop bars"),
    ("text", "text_paragraph", "paragraphs", "text_markup_strip", "strip markup"),
)


def _default_embedders(types: set[str]) -> dict:
    """Each type's default embedder: the one it marks, or its only one (as the docs' media-types table reads it)."""
    defaults = {}
    for type_id in types:
        of_type = [e for e in all_embedders() if e.media_type_id == type_id]
        marked = [e for e in of_type if e.is_default]
        if marked or len(of_type) == 1:
            defaults[type_id] = (marked or of_type)[0]
    return defaults


def _check_embedders(types: set[str]) -> None:
    """Mixed Media's second and third lines: each type's model, and whether it takes a typed phrase."""
    defaults = _default_embedders(types)
    for type_id, spec in MEDIA.items():
        e = defaults.get(type_id)
        if spec["embedder"] is None:
            if e is not None:
                raise SystemExit(f"media slides: {type_id} now embeds with {e.name}; the slide says convert first")
            continue
        if e is None or e.name != EMBEDDER_IDS[spec["embedder"]]:
            got = getattr(e, "name", None)
            raise SystemExit(f"media slides: {type_id}'s embedder is {got}, not {spec['embedder']}")
        if e.supports_text != ("phrase" in spec):
            raise SystemExit(f"media slides: {type_id}'s embedder supports_text={e.supports_text}; the slide disagrees")


def _check_registries() -> None:
    """What the media slides draw is what the registries hold (module docstring)."""
    types = {t.type_id for t in all_types()}
    if types != set(MEDIA):
        raise SystemExit(f"media slides: the app's media types are {sorted(types)}, the slides draw {sorted(MEDIA)}")
    _check_embedders(types)
    shipped = sorted((c.source_type, c.target_type) for c in list_converters())
    drawn = sorted((s, t) for s, t, _w in CONVERTERS)
    if shipped != drawn:
        raise SystemExit(f"Shape Shifters: the shipped converters are {shipped}, the slide draws {drawn}")
    clippers, cleaners = {c.name for c in all_clippers()}, {c.name for c in all_cleaners()}
    for _type, clipper, _w, cleaner, _d in CLEAN_ROWS:
        if clipper not in clippers or cleaner not in cleaners:
            raise SystemExit(f"Cut and Dried: {clipper} or {cleaner} is no longer registered")


def _draw_icon(ax: plt.Axes, type_id: str, cx: float, cy: float, **kw) -> tuple[float, float]:
    """Draw *type_id*'s icon centred on (cx, cy); returns its half width and half height."""
    if type_id == "document":
        _document(ax, cx, cy, **kw)
        return PAGE_W / 2, PAGE_H / 2
    if type_id == "face":
        _face_icon(ax, cx, cy)
        return 0.7, ICON_H / 2
    {"image": _image, "audio": _audio, "video": _video, "text": _text}[type_id](ax, cx, cy, **kw)
    return ICON_W / 2, ICON_H / 2


def _thin_arrow(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float], z: float = 3) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, lw=2.0, color=INK, zorder=z,
                                 shrinkA=0, shrinkB=0))  # fmt: skip


# ──────────────────────────────────────────────────────────────────────────── Mixed Media

#: The tiles: two columns of three, each an icon and, right of it, three lines: the name, the
#: embedder, and a phrase or a note. Images first, as section I met them; the two that take no
#: phrase last, so each page fills the grid from the top.
TILE_X = (0.9, 8.4)
TILE_Y = (5.55, 3.35, 1.15)
TILE_AT = {"image": (0, 0), "video": (1, 0), "audio": (0, 1), "text": (1, 1), "document": (0, 2), "face": (1, 2)}
#: Where a tile's text column starts, right of its icon, and its three lines' offsets from the
#: tile's centre.
TEXT_X = ICON_W + 0.35
LINE_DY = (0.52, -0.02, -0.52)
#: Which page of Mixed Media each type arrives on.
TYPES_PAGE = {"image": 1, "audio": 2, "video": 2, "text": 2, "document": 3, "face": 3}


def _media_types(stage: int) -> Figure:
    fig, ax = _canvas()
    for type_id, spec in MEDIA.items():
        if TYPES_PAGE[type_id] > stage:
            continue
        col, row = TILE_AT[type_id]
        x0, cy = TILE_X[col], TILE_Y[row]
        _draw_icon(ax, type_id, x0 + ICON_W / 2, cy, **({"ls": "--"} if type_id == "document" else {}))
        tx = x0 + TEXT_X
        ax.text(tx, cy + LINE_DY[0], spec["name"], ha="left", va="center", fontsize=NAME_PT, fontweight="bold",
                color=INK)  # fmt: skip
        ax.text(tx, cy + LINE_DY[1], spec["embedder"] or "no model of its own", ha="left", va="center",
                fontsize=DETAIL_PT, color=INK if spec["embedder"] else SOFT)  # fmt: skip
        if "phrase" in spec:
            ax.text(tx, cy + LINE_DY[2], spec["phrase"], ha="left", va="center", fontsize=QUERY_PT, fontweight="bold",
                    color=GREEN)  # fmt: skip
        else:
            ax.text(tx, cy + LINE_DY[2], spec["note"], ha="left", va="center", fontsize=DETAIL_PT, color=SOFT,
                    style="italic")  # fmt: skip
    return fig


# ──────────────────────────────────────────────────────────────────────────── Shape Shifters

#: The graph's node centres: three columns, two rows, so the image hub's five arrows are its four
#: neighbours' straight lines and two diagonals that cross nothing.
NODE_X = (2.45, 8.0, 13.55)
NODE_Y = {1: 4.95, 0: 1.95}


def _node(type_id: str) -> tuple[float, float]:
    col, row = MEDIA[type_id]["at"]
    return NODE_X[col], NODE_Y[row]


def _half(type_id: str) -> tuple[float, float]:
    if type_id == "document":
        return PAGE_W / 2, PAGE_H / 2
    if type_id == "face":
        return 0.7, ICON_H / 2
    return ICON_W / 2, ICON_H / 2


def _edge_point(type_id: str, toward: tuple[float, float]) -> np.ndarray:
    """Where a line from *type_id*'s centre toward *toward* leaves its icon, one object gap out."""
    c = np.array(_node(type_id))
    d = np.array(toward) - c
    hw, hh = _half(type_id)
    scale = min(hw / abs(d[0]) if d[0] else math.inf, hh / abs(d[1]) if d[1] else math.inf)
    unit = d / np.linalg.norm(d)
    return c + d * scale + unit * GAP


def _media_convert(stage: int) -> Figure:
    fig, ax = _canvas()
    for type_id, spec in MEDIA.items():
        cx, cy = _node(type_id)
        _draw_icon(ax, type_id, cx, cy, **({"ls": "--"} if type_id == "document" else {}))
        # Names on the outside of the two rows, so the band between them is the arrows' alone, and
        # level along a row whatever the icon's height.
        above = spec["at"][1] == 1
        hh = max(_half(t)[1] for t, other in MEDIA.items() if other["at"][1] == spec["at"][1])
        ax.text(cx, cy + hh + LABEL_GAP if above else cy - hh - LABEL_GAP, spec["name"], ha="center",
                va="bottom" if above else "top", fontsize=NAME_PT, color=INK)  # fmt: skip
    for source, target, word in CONVERTERS:
        if CONVERT_PAGE[source] > stage:
            continue
        a, b = _edge_point(source, _node(target)), _edge_point(target, _node(source))
        _thin_arrow(ax, tuple(a), tuple(b))
        d = b - a
        angle = math.degrees(math.atan2(d[1], d[0]))
        if angle > 90 or angle < -90:
            angle += 180
        if abs(abs(angle) - 90) < 1e-6:
            # A vertical arrow: its word beside it, level, on the side away from the diagonals.
            mid = (a + b) / 2
            side = -1 if _node(source)[0] < 5 else 1
            ax.text(mid[0] + side * LABEL_GAP * 2, mid[1], word, ha="right" if side < 0 else "left", va="center",
                    fontsize=ARROW_PT, color=INK)  # fmt: skip
        else:
            # Along the arrow, on its upper side, a label gap off the line.
            normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            if normal[1] < 0:
                normal = -normal
            mid = (a + b) / 2 + normal * (LABEL_GAP + 0.04)
            ax.text(mid[0], mid[1], word, ha="center", va="bottom", rotation=angle, rotation_mode="anchor",
                    fontsize=ARROW_PT, color=INK)  # fmt: skip
    return fig


# ──────────────────────────────────────────────────────────────────────────── Cut and Dried

#: The rows' centres. Each row is a file, its pieces and the tidied pieces, laid out left to right
#: from `CLEAN_LEFT` with each block arrow as long as `CUT_LEN` and `CLEAN_LEN`, which hold their words.
CLEAN_Y = (5.3, 3.3, 1.3)
CLEAN_LEFT = 0.75
#: A file's card is wider than an icon (it holds a whole recording); a piece's is smaller.
FILE_W, PIECE_W, PIECE_H = 2.4, 1.1, 1.0
PIECE_STEP = 1.28
CUT_LEN, CLEAN_LEN = 1.5, 1.9
FILE_X = CLEAN_LEFT + FILE_W / 2
CUT_FROM = FILE_X + FILE_W / 2 + GAP
PIECES_X = CUT_FROM + CUT_LEN + GAP + PIECE_W / 2 + PIECE_STEP
CLEAN_FROM = PIECES_X + PIECE_STEP + PIECE_W / 2 + GAP
TIDY_X = CLEAN_FROM + CLEAN_LEN + GAP + PIECE_W / 2 + PIECE_STEP
#: The quiet stretches of the made-up recording, as fractions of it: a lead-in, one pause, a tail.
QUIET = [(0.0, 0.07), (0.47, 0.55), (0.93, 1.0)]


def _pieces(ax: plt.Axes, type_id: str, cx: float, cy: float, tidy: bool) -> None:
    """Three pieces of one file, side by side; *tidy*, each after its cleaner."""
    for k in range(3):
        px = cx + (k - 1) * PIECE_STEP
        if type_id == "audio":
            # The recording in thirds: the first and last carry its quiet head and tail, and the pause
            # sits inside the middle one, which trimming leaves alone.
            span = (k / 3, (k + 1) / 3)
            if tidy:
                lo = max(span[0], QUIET[0][1]) if k == 0 else span[0]
                hi = min(span[1], QUIET[-1][0]) if k == 2 else span[1]
                span = (lo, hi)
            _audio(ax, px, cy, PIECE_W, PIECE_H, quiet=QUIET, span=span)
        elif type_id == "video":
            _video(ax, px, cy, PIECE_W, PIECE_H, bars=not tidy)
        else:
            _paragraph(ax, px, cy, PIECE_W, PIECE_H, tags=not tidy)


def _media_clean(stage: int) -> Figure:
    fig, ax = _canvas()
    for (type_id, _clipper, pieces, _cleaner, does), cy in zip(CLEAN_ROWS, CLEAN_Y, strict=True):
        if type_id == "audio":
            _audio(ax, FILE_X, cy, FILE_W, PIECE_H, quiet=QUIET)
        elif type_id == "video":
            _video(ax, FILE_X, cy, FILE_W, PIECE_H, bars=True)
        else:
            _paragraphs(ax, FILE_X, cy, FILE_W, PIECE_H)
        ax.text(FILE_X, cy - PIECE_H / 2 - LABEL_GAP, MEDIA[type_id]["name"], ha="center", va="top",
                fontsize=DETAIL_PT + 2, color=INK)  # fmt: skip
        if stage >= 2:
            INTRO._block_arrow(ax, (CUT_FROM, cy), (CUT_FROM + CUT_LEN, cy), "cut")
            _pieces(ax, type_id, PIECES_X, cy, tidy=False)
            ax.text(PIECES_X, cy - PIECE_H / 2 - LABEL_GAP, pieces, ha="center", va="top", fontsize=DETAIL_PT,
                    color=SOFT)  # fmt: skip
        if stage >= 3:
            INTRO._block_arrow(ax, (CLEAN_FROM, cy), (CLEAN_FROM + CLEAN_LEN, cy), "clean")
            _pieces(ax, type_id, TIDY_X, cy, tidy=True)
            ax.text(TIDY_X, cy - PIECE_H / 2 - LABEL_GAP, does, ha="center", va="top", fontsize=DETAIL_PT,
                    color=SOFT)  # fmt: skip
    return fig


# ──────────────────────────────────────────────────────────────────────────── Face Off

#: The photo: a portrait card with a person in it, head tilted by `FACE_TILT`.
PHOTO_XY, PHOTO_W, PHOTO_H = (0.85, 1.95), 2.9, 3.9
#: The face box's side in the photo, and where its centre sits.
FACE_SIDE = 1.25
FACE_XY = (2.38, 4.4)
#: The crop: a square holding the face upright, at the photo's middle height.
CROP_SIDE = 2.0
CROP_X = 6.05
#: The cube: a smaller one than Embed-time Stories', since three objects share the row.
CUBE_X, CUBE_Y, CUBE_SIDE, CUBE_DEPTH = 10.85, 1.6, 3.9, 1.05
#: The other faces in the cube, and one person's: her crop, filled, and her other photos, as rings.
CLOUD_ITEMS, CLOUD_APART, CLOUD_R = 44, 0.44, 0.13
HER_CENTRE = np.array([0.7, 0.68, 0.45])
HER_OTHERS = np.array([[0.63, 0.78, 0.40], [0.79, 0.74, 0.52], [0.66, 0.58, 0.55], [0.82, 0.60, 0.38]])


def _cube_point(p: np.ndarray) -> np.ndarray:
    return np.array([CUBE_X + p[0] * CUBE_SIDE + p[2] * CUBE_DEPTH, CUBE_Y + p[1] * CUBE_SIDE + p[2] * CUBE_DEPTH])


def _cloud() -> np.ndarray:
    """Everyone else's faces, spread through the cube and clear of her photos (best-candidate, seeded)."""
    rng = np.random.default_rng(31)
    hers = [_cube_point(p) for p in (HER_CENTRE, *HER_OTHERS)]
    pts: list[np.ndarray] = []
    flat: list[np.ndarray] = []
    for _ in range(CLOUD_ITEMS):
        best, best_gap = None, -1.0
        for _ in range(600):
            p = rng.uniform(0.07, 0.93, size=3)
            q = _cube_point(p)
            if any(np.hypot(*(q - r)) < CLOUD_APART for r in flat + hers):
                continue
            if np.linalg.norm(p - HER_CENTRE) < 0.3:
                continue
            gap = min((float(np.linalg.norm(p - o)) for o in pts), default=math.inf)
            if gap > best_gap:
                best, best_gap = p, gap
        if best is None:
            break
        pts.append(best)
        flat.append(_cube_point(best))
    return np.array(pts)


def _wire_cube(ax: plt.Axes) -> None:
    """Embed-time Stories' wire box at this figure's size: hidden edges under the dots, the rest over."""
    d, side, x0, y0 = CUBE_DEPTH, CUBE_SIDE, CUBE_X, CUBE_Y
    front = [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)]
    back = [(x + d, y + d) for x, y in front]
    hidden = {(0, 1), (0, 3)}
    edges = []
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        edges.append((front[a], front[b], False))
        edges.append((back[a], back[b], (a, b) in hidden or (b, a) in hidden))
    for corner in range(4):
        edges.append((front[corner], back[corner], corner == 0))
    for (ax_, ay), (bx, by), behind in edges:
        ax.plot([ax_, bx], [ay, by], color=WIRE, lw=1.4, solid_capstyle="projecting", zorder=2 if behind else 7.5)


def _photo(ax: plt.Axes, boxed: bool) -> None:
    """The photo: a person, shoulders level and head tilted; *boxed*, the detector's box and points."""
    x0, y0 = PHOTO_XY
    ax.add_patch(Rectangle((x0, y0), PHOTO_W, PHOTO_H, facecolor="white", edgecolor=INK, lw=FRAME_LW, zorder=3))
    cx = x0 + PHOTO_W / 2
    # Shoulders and neck, level: the person tilts their head, not their body.
    body = MplPath(
        [(x0 + 0.15, y0), (x0 + 0.2, y0 + 1.25), (cx - 0.25, y0 + 1.55), (cx - 0.22, y0 + 1.8),
         (cx + 0.22, y0 + 1.8), (cx + 0.25, y0 + 1.55), (x0 + PHOTO_W - 0.2, y0 + 1.25), (x0 + PHOTO_W - 0.15, y0)],
        [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.MOVETO, MplPath.LINETO,
         MplPath.CURVE3, MplPath.CURVE3],
    )  # fmt: skip
    ax.add_patch(PathPatch(body, facecolor="none", edgecolor=SOFT, lw=INNER_LW + 0.3, zorder=4))
    _face(ax, *FACE_XY, FACE_SIDE, FACE_TILT, z=5)
    if boxed:
        # The detector's box is upright whatever the head does: the bounds of the turned face box.
        corners = np.array([_to_box(*FACE_XY, FACE_SIDE, FACE_TILT, u, v) for u in (0, 1) for v in (0, 1)])
        lo, hi = corners.min(0), corners.max(0)
        ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec=BOX_BLUE, lw=2.4, zorder=6))
        for p in _face_points(*FACE_XY, FACE_SIDE, FACE_TILT):
            ax.add_patch(plt.Circle(p, 0.065, facecolor=BOX_BLUE, edgecolor="white", lw=1.0, zorder=7))


def _face_find(stage: int) -> Figure:
    fig, ax = _canvas()
    x0, y0 = PHOTO_XY
    mid_y = y0 + PHOTO_H / 2

    def name(x: float, top: float, text: str) -> None:
        ax.text(x, top - LABEL_GAP * 2, text, ha="center", va="top", fontsize=NAME_PT, color=INK)

    _photo(ax, boxed=stage >= 2)
    name(x0 + PHOTO_W / 2, y0, "Photo")
    if stage >= 3:
        INTRO._block_arrow(ax, (x0 + PHOTO_W + GAP, mid_y), (CROP_X - GAP, mid_y), "crop")
        cy0 = mid_y - CROP_SIDE / 2
        ax.add_patch(Rectangle((CROP_X, cy0), CROP_SIDE, CROP_SIDE, facecolor="white", edgecolor=INK, lw=FRAME_LW,
                               zorder=3))  # fmt: skip
        fc = (CROP_X + CROP_SIDE / 2, mid_y - 0.05)
        _face(ax, *fc, 0.72 * CROP_SIDE, 0.0)
        for p in _face_points(*fc, 0.72 * CROP_SIDE, 0.0):
            ax.add_patch(plt.Circle(p, 0.065, facecolor=BOX_BLUE, edgecolor="white", lw=1.0, zorder=7))
        name(CROP_X + CROP_SIDE / 2, cy0, "Turned upright")
    if stage >= 4:
        INTRO._block_arrow(ax, (CROP_X + CROP_SIDE + GAP, mid_y), (CUBE_X - GAP, mid_y), "embed")
        # The model's name over its arrow, a label gap above the arrowhead's reach.
        ax.text((CROP_X + CROP_SIDE + CUBE_X) / 2, mid_y + INTRO.FLOW_HEAD_W / 2 + LABEL_GAP, "FaceNet",
                ha="center", va="bottom", fontsize=NAME_PT, color=INK)  # fmt: skip
        _wire_cube(ax)
        for p in _cloud():
            q = _cube_point(p)
            ax.add_patch(plt.Circle(q, CLOUD_R * (1 - 0.3 * p[2]), facecolor="white", edgecolor=INK, lw=1.3,
                                    zorder=5 + 2 * (1 - p[2])))  # fmt: skip
        for p in HER_OTHERS:
            q = _cube_point(p)
            ax.add_patch(plt.Circle(q, CLOUD_R * 1.15, facecolor="white", edgecolor=GREEN, lw=2.6,
                                    zorder=5 + 2 * (1 - p[2])))  # fmt: skip
        ax.add_patch(plt.Circle(_cube_point(HER_CENTRE), CLOUD_R * 1.2, facecolor=GREEN, edgecolor=GREEN, zorder=8))
        name(CUBE_X + (CUBE_SIDE + CUBE_DEPTH) / 2, CUBE_Y, "512-d")
    return fig


def _save_stages(build, stem: str, stages: int) -> None:
    for stage in range(1, stages):
        save(build(stage), OUT, f"{stem}.build{stage}.webp", column=FULL_BLEED, tight=False)
    save(build(stages), OUT, f"{stem}.webp", column=FULL_BLEED, tight=False)


def main() -> int:
    _check_registries()
    _check_face()
    _save_stages(_media_types, "media-types", 3)
    _save_stages(_media_convert, "media-convert", 3)
    _save_stages(_media_clean, "media-clean", 3)
    _save_stages(_face_find, "face-find", 4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
