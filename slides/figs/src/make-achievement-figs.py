#!/usr/bin/env python
"""The achievement slides, after Conclusion (#4967): the tiers, the ten entries, and how an unlock is told.

Run from the repo root:

    python slides/figs/src/make-achievement-figs.py

Three figures, one per slide:

- `achievement-tiers` (**Keeping Score**): one entry's badge drawn five ways, locked and then at each of the
  four tiers, with the threshold that reaches each and the points it adds to the panel's Total Score.
  Pages: a, the badges; b, the points.
- `achievement-entries` (**Trophy Case**): all ten entries, each with its badge, what it counts and its four
  thresholds. Pages: a, the five that count how much you have done; b, the five that count how often
  and how widely.
- `achievement-unlock` (**Seen and Heard**): a timeline of one unlock. The toast fires once, at the vote that
  crosses the threshold, and not again after a restart; the trophy's dot stays lit until the panel is
  opened. Pages: a, the count crossing the threshold; b, the toast; c, the dot.

Nothing the slides say about the system is typed in twice. The entries, their names and thresholds come from
`vtsearch/achievements_catalog.py`; the badges are rasterised from the app's own icon SVGs
(`icon-svgs-extended.ts`), the badge component's decoration template and its tier colours; the
points come from the panel's own Total Score tooltip; the toast's wording from the component that
pops it. Only the short line saying what each entry counts is written here (`COUNTS`), because the
catalog's descriptions are sentences sized for the panel, not for a slide, and the script refuses to
run while any catalog entry lacks one. So adding an achievement or moving a threshold is a re-run of
this script. The backdrop is the exception: `SURFACE`, `MUTED`, `LOCKED_OPACITY` and `ACCENT` are
copies of the theme tokens each one names, because a theme block is not worth parsing for a backdrop.

The badges sit on the dark theme's surface colour, as they do in the app's default theme: two of the
four tier colours are near-white by design (they are deliberately not re-themed), and on a white slide
Silver and Platinum would vanish.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure the repo root is importable no matter the cwd: ``python
# slides/figs/src/x.py`` only puts the script's own dir on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, TITLE_NOTCH_PX, save  # noqa: E402

from vtsearch.achievements_catalog import ACHIEVEMENTS, TIER_NAMES  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
FRONTEND = _REPO_ROOT / "frontend" / "src"
ICON_FILES = [
    FRONTEND / "app/components/icon/icon-svgs.ts",
    FRONTEND / "app/components/icon/icon-svgs-extended.ts",
]
BADGE_TEMPLATE = FRONTEND / "app/components/achievement-badge/achievement-badge.component.html"
TAB_TEMPLATE = FRONTEND / "app/components/achievements-tab/achievements-tab.component.html"
UNLOCK_HOST = FRONTEND / "app/components/achievement-unlock-host/achievement-unlock-host.component.ts"
VARIABLES = FRONTEND / "scss/_variables.scss"

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

#: The canvas: the whole slide, at 100 slide pixels to the inch, saved uncropped so every page of a
#: build shares one frame and the layout below is in slide pixels.
FIG_W, FIG_H = 12.8, 7.2
W, H = 1280, 720

#: What each entry counts, in a line that fits a slide. Keyed by catalog id; `_counts` refuses a
#: catalog entry with no line here, and a line for an entry the catalog no longer has.
COUNTS: dict[str, str] = {
    "datasets_loaded": "loads of datasets you imported",
    "votes_cast": "Good and Bad votes cast",
    "detectors_trained": "detectors you have voted on",
    "detectors_imported": "detectors given imported labels",
    "find_media": "items scored by Find and Test",
    "days_active": "days on which you voted",
    "media_types_touched": "media types you have voted on",
    "vote_streak": "votes in a row, 10 minutes apart at most",
    "hours_voted": "hours of the day you have voted in",
    "docs_read": "docs whose code phrase you found",
}

#: The entry `achievement-tiers` and `achievement-unlock` use as their example: the one everybody unlocks first.
EXAMPLE = "votes_cast"
#: The unit its thresholds count, for the tiers slide's captions.
EXAMPLE_UNIT = "votes"

#: The dark theme's surface colour (`--bg-surface`), which the badges sit on in the app's default theme.
SURFACE = "#1a1d27"
#: The dark theme's muted text (`--text-muted`), and the dim a locked badge is drawn at (`--opacity-dim`).
MUTED = "#888888"
LOCKED_OPACITY = 0.7
#: The light theme's accent (`--accent`), the colour of the trophy's notification dot on a light page.
ACCENT = "#5a68d8"
#: A light rule, for the lanes of the timeline.
RULE = "#d5d9e0"


# ---------------------------------------------------------------------------
# What the app draws, read from the app
# ---------------------------------------------------------------------------


def _icons() -> dict[str, str]:
    """Every icon the app's `vt-icon` knows, as `{name: '<svg …>…</svg>'}`."""
    found: dict[str, str] = {}
    pattern = re.compile(r"^\s*'?([\w-]+)'?:\s*\n?\s*'(<svg[^']*</svg>)'", re.MULTILINE)
    for path in ICON_FILES:
        for name, svg in pattern.findall(path.read_text(encoding="utf-8")):
            found[name] = svg
    return found


def _tier_colours() -> list[str]:
    """`--tier-bronze` … `--tier-platinum` from the app's stylesheet, in tier order."""
    text = VARIABLES.read_text(encoding="utf-8")
    colours = []
    for tier in TIER_NAMES:
        match = re.search(rf"--tier-{tier.lower()}:\s*(#[0-9a-fA-F]{{6}})", text)
        if match is None:
            raise SystemExit(f"no --tier-{tier.lower()} in {VARIABLES}")
        colours.append(match.group(1))
    return colours


#: The badge component's decoration blocks, keyed by the condition the template guards each with,
#: mapped to the lowest tier index that shows it (`achievement-badge.component.ts`: the outer ring
#: from Bronze, the inner ring from Silver, laurels from Gold, sparkles at Platinum; the dashed ring
#: only when locked, which the template spells `tierIdx() === -1`).
DECORATIONS = {
    "tierIdx() === -1": None,
    "showOuterRing": 0,
    "showDoubleRing": 1,
    "showLaurel": 2,
    "showSparkles": 3,
}


def _decorations() -> dict[str, str]:
    """The badge template's SVG shapes, `{condition: '<circle …/>…'}`, for the conditions above."""
    text = BADGE_TEMPLATE.read_text(encoding="utf-8")
    svg = text[text.index("<svg") : text.index("</svg>")]
    blocks = dict(re.findall(r"@if \(([^)]*\)?(?: === -1)?)\) \{(.*?)\n\s*\}", svg, re.DOTALL))
    if set(blocks) != set(DECORATIONS):
        raise SystemExit(
            f"{BADGE_TEMPLATE.name}: expected decoration blocks {sorted(DECORATIONS)}, found {sorted(blocks)}"
        )
    return {cond: body.strip() for cond, body in blocks.items()}


def _points() -> list[int]:
    """Points per tier, from the Total Score tooltip the panel itself carries."""
    text = TAB_TEMPLATE.read_text(encoding="utf-8")
    match = re.search(r'class="achievements-total" title="([^"]+)"', text)
    if match is None:
        raise SystemExit(f"{TAB_TEMPLATE.name}: no Total Score tooltip")
    by_tier = {tier: int(n) for n, tier in re.findall(r"(\d+)pts? (\w+)", match.group(1))}
    if list(by_tier) != list(TIER_NAMES):
        raise SystemExit(f"{TAB_TEMPLATE.name}: Total Score tooltip names {list(by_tier)}, not {list(TIER_NAMES)}")
    return [by_tier[tier] for tier in TIER_NAMES]


def _toast_lines(entry: dict, tier_idx: int) -> tuple[str, str]:
    """The unlock toast's two lines, worded as `achievement-unlock-host` words them."""
    text = UNLOCK_HOST.read_text(encoding="utf-8")
    message = re.search(r"message: `([^`]*)`", text)
    detail = re.search(r"detail: `([^`]*)`", text)
    if message is None or detail is None:
        raise SystemExit(f"{UNLOCK_HOST.name}: cannot find the toast's message and detail")
    fields = {
        "p.tier_name": TIER_NAMES[tier_idx],
        "p.name": entry["name"],
        "p.threshold.toLocaleString()": f"{entry['tiers'][tier_idx]:,}",
    }

    def fill(template: str) -> str:
        return re.sub(r"\$\{([^}]*)\}", lambda m: fields[m.group(1)], template)

    return fill(message.group(1)), fill(detail.group(1))


def _counts() -> dict[str, str]:
    ids = [a["id"] for a in ACHIEVEMENTS]
    missing = [i for i in ids if i not in COUNTS]
    stale = [i for i in COUNTS if i not in ids]
    if missing or stale:
        raise SystemExit(f"COUNTS is out of step with the catalog: missing {missing}, stale {stale}")
    return COUNTS


ICONS = _icons()
TIER_COLOURS = _tier_colours()
DECORATION_SHAPES = _decorations()
POINTS = _points()
_BADGE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def badge_image(icon: str, tier_idx: int, px: int = 480) -> np.ndarray:
    """One badge as the app draws it, on the dark surface, as an RGBA array `px` square.

    The composition is the component's: a 100-unit square of decorations, the icon at 55% of the
    badge (`innerIconSize`) centred in it, everything stroked in the tier's colour (`currentColor`),
    and a locked badge in the muted text colour at the dimmed opacity. The disc behind it reaches four
    units past that square, because Platinum's corner sparkles do too.
    """
    key = (icon, tier_idx)
    if key in _BADGE_CACHE:
        return _BADGE_CACHE[key]
    if icon not in ICONS:
        raise SystemExit(f"icon {icon!r} is not in the app's icon set")
    colour = MUTED if tier_idx < 0 else TIER_COLOURS[tier_idx]
    shapes = "".join(
        body
        for cond, body in DECORATION_SHAPES.items()
        if (DECORATIONS[cond] is None and tier_idx < 0)
        or (DECORATIONS[cond] is not None and tier_idx >= DECORATIONS[cond])
    )
    # The sparkles are filled with `currentColor` and carry no stroke, which MuPDF draws as a hairline
    # rather than as nothing; say it the way it means.
    shapes = shapes.replace('stroke-width="0"', 'stroke="none"').replace('"currentColor"', f'"{colour}"')
    inner = re.sub(r"^<svg[^>]*>|</svg>$", "", ICONS[icon])
    icon_attrs = re.search(r"^<svg([^>]*)>", ICONS[icon]).group(1)
    stroke_width = re.search(r'stroke-width="([\d.]+)"', icon_attrs).group(1)
    opacity = LOCKED_OPACITY if tier_idx < 0 else 1.0
    scale = 55.0 / 24.0
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-4 -4 108 108" width="{px}" height="{px}">'
        f'<circle cx="50" cy="50" r="54" fill="{SURFACE}"/>'
        f'<g opacity="{opacity}">'
        f'<g fill="none" stroke="{colour}" stroke-linecap="round" stroke-linejoin="round">{shapes}</g>'
        f'<g transform="translate(22.5 22.5) scale({scale:.6f})" fill="none" stroke="{colour}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">{inner}</g>'
        "</g></svg>"
    )
    doc = pymupdf.open(stream=svg.encode("utf-8"), filetype="svg")
    pix = doc[0].get_pixmap(alpha=True)
    image = np.asarray(Image.frombytes("RGBA", (pix.width, pix.height), pix.samples))
    _BADGE_CACHE[key] = image
    return image


# ---------------------------------------------------------------------------
# Drawing helpers, in slide pixels (origin top-left)
# ---------------------------------------------------------------------------


def _canvas() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return fig, ax


def _badge(ax: plt.Axes, icon: str, tier_idx: int, cx: float, cy: float, d: float) -> None:
    image = badge_image(icon, tier_idx)
    ax.imshow(image, extent=(cx - d / 2, cx + d / 2, cy + d / 2, cy - d / 2), interpolation="lanczos", zorder=3)


def _width_px(text: matplotlib.text.Text) -> float:
    """How wide a drawn label is, in slide pixels (the canvas is 100 slide pixels to the inch)."""
    fig = text.figure
    bbox = text.get_window_extent(fig.canvas.get_renderer())
    return bbox.width / fig.dpi * (W / FIG_W)


def _pill(ax: plt.Axes, x: float, cy: float, text: str, colour: str, size: float) -> float:
    """A tier's threshold on the dark surface, left edge at `x`; returns the right edge."""
    pad_x, h = 9.0, 30.0
    label = ax.text(x + pad_x, cy, text, fontsize=size, color=colour, ha="left", va="center", weight="bold", zorder=4)
    w = _width_px(label) + 2 * pad_x
    ax.add_patch(
        FancyBboxPatch(
            (x, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=7",
            facecolor=SURFACE,
            edgecolor="none",
            zorder=3,
        )
    )
    return x + w


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")  # codespell:ignore nd
    return f"{n:,}{suffix}"


def _short(n: int) -> str:
    """A threshold as a slide says it: 100, 1k, 20k, 2M."""
    for unit, size in (("M", 1_000_000), ("k", 1_000)):
        if n >= size and n % size == 0:
            return f"{n // size}{unit}"
    return f"{n:,}"


def _check_notch_clear(left: float, top: float) -> None:
    """Assert the layout leaves the title notch alone (save() re-checks the pixels)."""
    nx, ny, nw, nh = TITLE_NOTCH_PX
    if left < nx + nw and top < ny + nh:
        raise SystemExit(f"layout reaches into the title notch at ({left:.0f}, {top:.0f})")


# ---------------------------------------------------------------------------
# Keeping Score
# ---------------------------------------------------------------------------


def tiers_fig() -> None:
    entry = next(a for a in ACHIEVEMENTS if a["id"] == EXAMPLE)
    tiers = entry["tiers"]
    centres = [200 + 220 * i for i in range(5)]
    cy, d = 330.0, 172.0
    _check_notch_clear(centres[0] - d / 2, cy - d / 2)

    def draw(stage: int) -> plt.Figure:
        fig, ax = _canvas()
        for i, cx in enumerate(centres):
            tier_idx = i - 1
            _badge(ax, entry["icon"], tier_idx, cx, cy, d)
            name = "Locked" if tier_idx < 0 else TIER_NAMES[tier_idx]
            reach = f"under {tiers[0]:,}" if tier_idx < 0 else f"{tiers[tier_idx]:,}"
            ax.text(cx, cy + d / 2 + 34, name, ha="center", va="center", fontsize=21, weight="bold")
            ax.text(cx, cy + d / 2 + 72, f"{reach} {EXAMPLE_UNIT}", ha="center", va="center", fontsize=16, color=SOFT)
            if stage >= 2:
                points = 0 if tier_idx < 0 else POINTS[tier_idx]
                ax.text(
                    cx,
                    cy + d / 2 + 132,
                    f"{points} point" + ("" if points == 1 else "s"),
                    ha="center",
                    va="center",
                    fontsize=21,
                )
        if stage >= 2:
            top = len(ACHIEVEMENTS) * POINTS[-1]
            ax.text(
                W / 2,
                cy + d / 2 + 210,
                f"Total Score: every entry's points, added up. All {len(ACHIEVEMENTS)} at Platinum is {top}.",
                ha="center",
                va="center",
                fontsize=17,
                color=SOFT,
            )
        return fig

    save(draw(1), OUT, "achievement-tiers.build1.webp", FULL_BLEED, tight=False)
    save(draw(2), OUT, "achievement-tiers.webp", FULL_BLEED, tight=False)


# ---------------------------------------------------------------------------
# Trophy Case
# ---------------------------------------------------------------------------


def entries_fig() -> None:
    counts = _counts()
    if len(ACHIEVEMENTS) % 2:
        raise SystemExit("Trophy Case lays the entries out in two equal columns; the catalog has an odd count")
    rows = len(ACHIEVEMENTS) // 2
    top, bottom = 205.0, 705.0
    pitch = (bottom - top) / rows
    d = 82.0
    columns = [(105.0, 168.0), (705.0, 768.0)]  # (badge centre x, text left x)
    _check_notch_clear(columns[0][0] - d / 2, top)

    def draw(stage: int) -> plt.Figure:
        fig, ax = _canvas()
        for k, entry in enumerate(ACHIEVEMENTS):
            col, row = divmod(k, rows)
            if col >= stage:
                continue
            bx, tx = columns[col]
            cy = top + pitch * (row + 0.5)
            # The panel shows a badge at the tier you have reached; the case shows each one complete.
            _badge(ax, entry["icon"], len(TIER_NAMES) - 1, bx, cy, d)
            ax.text(tx, cy - 31, entry["name"], ha="left", va="center", fontsize=17, weight="bold")
            ax.text(tx, cy - 3, counts[entry["id"]], ha="left", va="center", fontsize=15, color=SOFT)
            x = tx
            for tier_idx, threshold in enumerate(entry["tiers"]):
                x = _pill(ax, x, cy + 30, _short(threshold), TIER_COLOURS[tier_idx], 15) + 8
        return fig

    save(draw(1), OUT, "achievement-entries.build1.webp", FULL_BLEED, tight=False)
    save(draw(2), OUT, "achievement-entries.webp", FULL_BLEED, tight=False)


# ---------------------------------------------------------------------------
# Seen and Heard
# ---------------------------------------------------------------------------


#: The timeline's geometry, in slide pixels: the time axis runs `T_X0`…`T_X1`, each lane is a `(top,
#: bottom)` band, and the lane names are right-aligned at `T_LABEL_X`.
T_X0, T_X1 = 400.0, 1190.0
LANES = {"votes": (205.0, 380.0), "toast": (405.0, 490.0), "dot": (512.0, 572.0)}
LANE_NAMES = {"votes": "Votes", "toast": "Toast", "dot": "Trophy dot"}
T_AXIS_Y = 600.0
T_LABEL_X = 370.0

#: A running vote count over three sittings, as fractions of the Bronze threshold at fractions of the
#: time axis: the first sitting ends short of the threshold, the second crosses it, and the app
#: restarts in the long pause before the third, well after the toast has gone (its box must not reach
#: the restart, or the drawing says it was still up). The panel is opened during the third.
T_TIMES = np.array([0.00, 0.15, 0.20, 0.33, 0.78, 1.00])
T_VOTES = np.array([0.40, 0.70, 0.70, 1.30, 1.30, 2.00])
T_RESTART, T_OPEN = 0.72, 0.93
#: The narrowest gap the moments' names under the axis may leave between them, in slide pixels.
T_LABEL_GAP = 24.0
T_VMAX = 2.2


def _tx(u: float) -> float:
    return T_X0 + (T_X1 - T_X0) * u


def _vy(v: float) -> float:
    """Height of a vote count `v` (in Bronze thresholds) in the votes lane."""
    lo, hi = LANES["votes"]
    return hi - (hi - lo) * v / T_VMAX


def _t_cross() -> float:
    return float(np.interp(1.0, T_VOTES[2:4], T_TIMES[2:4]))


def _lanes(ax: plt.Axes, shown: list[str]) -> None:
    for lane in shown:
        lo, hi = LANES[lane]
        ax.text(T_LABEL_X, (lo + hi) / 2, LANE_NAMES[lane], ha="right", va="center", fontsize=19, weight="bold")
        if lane != "votes":
            # The lane's own baseline: where nothing is happening, it is all there is.
            ax.plot([T_X0, T_X1], [(lo + hi) / 2] * 2, color=RULE, lw=2.0, zorder=1)
    _check_notch_clear(T_LABEL_X - 120, LANES["votes"][0])


def _moments(ax: plt.Axes, moments: list[tuple[float, str]]) -> None:
    """The time axis, and a dashed line up through the lanes at each moment, named under the axis."""
    ax.plot([T_X0, T_X1], [T_AXIS_Y, T_AXIS_Y], color=SOFT, lw=1.6, solid_capstyle="butt")
    spans = []
    for u, label in moments:
        ax.plot([_tx(u)] * 2, [LANES["votes"][0] - 5, T_AXIS_Y], color=RULE, lw=1.6, ls=(0, (4, 4)), zorder=1)
        name = ax.text(_tx(u), T_AXIS_Y + 28, label, ha="center", va="center", fontsize=16, color=SOFT)
        half = _width_px(name) / 2
        spans.append((_tx(u) - half, _tx(u) + half, label))
    spans.sort()
    for (_, right, a), (left, _, b) in zip(spans, spans[1:], strict=False):
        if left - right < T_LABEL_GAP:
            raise SystemExit(f"achievement-unlock: {a!r} and {b!r} crowd each other under the axis")


def _votes(ax: plt.Axes, bronze: int) -> None:
    """The count, and the Bronze threshold it crosses, marked where it does.

    The threshold is named under its right end, where the count is above it and nothing else is but
    a moment's dashed line, which the name's white ground covers.
    """
    ax.plot([T_X0, T_X1], [_vy(1.0)] * 2, color=TIER_COLOURS[0], lw=2.4, ls=(0, (7, 5)), zorder=2)
    name = ax.text(T_X1, _vy(1.0) + 18, f"{TIER_NAMES[0]}: {bronze:,}", ha="right", va="center", fontsize=16)
    name.set(bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=4)
    ax.plot([_tx(u) for u in T_TIMES], [_vy(v) for v in T_VOTES], color=INK, lw=3.0, zorder=3)
    crossing = Circle((_tx(_t_cross()), _vy(1.0)), 8, facecolor=TIER_COLOURS[0], edgecolor=INK, lw=1.5, zorder=4)
    ax.add_patch(crossing)


def _toast(ax: plt.Axes, message: str, detail: str) -> None:
    """The unlock toast, at the vote that crossed the threshold, as wide as its own words."""
    lo, hi = LANES["toast"]
    bh = hi - lo
    bx = _tx(_t_cross()) - 18
    pad_left, pad_right = 24.0, 18.0
    lines = [
        ax.text(bx + pad_left, lo + bh * 0.34, message, ha="left", va="center", fontsize=15, weight="bold", zorder=4),
        ax.text(bx + pad_left, lo + bh * 0.70, detail, ha="left", va="center", fontsize=15, color=SOFT, zorder=4),
    ]
    bw = pad_left + max(_width_px(line) for line in lines) + pad_right
    if bx + bw >= _tx(T_RESTART) - 20:
        raise SystemExit("achievement-unlock: the toast's box reaches the restart; move the restart later")
    box = FancyBboxPatch((bx, lo), bw, bh, boxstyle="round,pad=0,rounding_size=10", facecolor="white")
    box.set(edgecolor=RULE, lw=1.6, zorder=3)
    ax.add_patch(box)
    stripe = FancyBboxPatch((bx, lo), 9, bh, boxstyle="round,pad=0,rounding_size=4", facecolor=TIER_COLOURS[0])
    stripe.set(edgecolor="none", zorder=4)
    ax.add_patch(stripe)


def _dot(ax: plt.Axes) -> None:
    """The trophy's dot, lit from the crossing until the panel is opened."""
    lo, hi = LANES["dot"]
    cy = (lo + hi) / 2
    x_lit = _tx(_t_cross())
    band = FancyBboxPatch((x_lit, cy - 9), _tx(T_OPEN) - x_lit, 18, boxstyle="round,pad=0,rounding_size=9")
    band.set(facecolor=ACCENT, edgecolor="none", zorder=3)
    ax.add_patch(band)


def unlock_fig() -> None:
    entry = next(a for a in ACHIEVEMENTS if a["id"] == EXAMPLE)
    bronze = entry["tiers"][0]
    message, detail = _toast_lines(entry, 0)
    stages = [
        (["votes"], [(_t_cross(), f"{_ordinal(bronze)} vote")]),
        (["votes", "toast"], [(T_RESTART, "app restarts")]),
        (["votes", "toast", "dot"], [(T_OPEN, "panel opened")]),
    ]
    names = ["achievement-unlock.build1.webp", "achievement-unlock.build2.webp", "achievement-unlock.webp"]
    moments: list[tuple[float, str]] = []
    for (shown, added), name in zip(stages, names, strict=True):
        moments += added
        fig, ax = _canvas()
        _lanes(ax, shown)
        _moments(ax, moments)
        _votes(ax, bronze)
        if "toast" in shown:
            _toast(ax, message, detail)
        if "dot" in shown:
            _dot(ax)
        save(fig, OUT, name, FULL_BLEED, tight=False)


if __name__ == "__main__":
    tiers_fig()
    entries_fig()
    unlock_fig()
