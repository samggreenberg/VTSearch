"""Recover the plotted series from a committed matplotlib SVG.

The overview-benchmark figures in `docs/experiments/2026-08-12-overview-bench/figures/`
are the only surviving copy of their own data: they were drawn from per-step
cell CSVs under a GRID scratch directory that is not in the repo and will not
outlive the study. Re-tailoring one of those figures for a slide therefore
cannot mean "re-run the generator with bigger fonts" — there is nothing to
re-run it against.

It can mean this instead. Matplotlib's SVG backend writes every line as a
`<path>` in display coordinates, and every tick as a labelled pixel position,
so the data → pixel transform is recoverable from the ticks and invertible on
the paths. What comes back is the *plotted* series — post-`path.simplify`, so
collinear runs are thinned — which is exactly what a redrawn figure needs and
noticeably less than what an analysis would need. Do not treat what comes back
as a data source for new statistics; it is a figure, read back.

`slides/figs/src/make-bench-figs.py` is the consumer. The SVGs are committed to
this repo, so they are a durable enough source; nothing is cached alongside
them.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

SVG = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}"

#: `M 1 2 L 3 4` and friends — matplotlib emits absolute moveto/lineto only.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
#: `10^{2}` inside the `<!-- $\mathdefault{...}$ -->` comment on a log tick.
_POWER = re.compile(r"(\d+)\^\{(-?\d+)\}")


def _style(element: ET.Element) -> dict[str, str]:
    raw = element.get("style") or ""
    out = {}
    for part in raw.split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def _points(path: ET.Element) -> list[tuple[float, float]]:
    numbers = [float(n) for n in _NUMBER.findall(path.get("d") or "")]
    return list(zip(numbers[0::2], numbers[1::2]))


@dataclass
class Axis:
    """The data ↔ pixel mapping for one axis, recovered from its tick labels."""

    log: bool
    scale: float
    offset: float

    def to_data(self, pixel: float) -> float:
        value = self.scale * pixel + self.offset
        return 10.0**value if self.log else value


def _fit(pairs: list[tuple[float, float]], log: bool) -> tuple[Axis, float]:
    """Least-squares pixel → data fit, with its worst residual as a check."""
    xs = [p for p, _ in pairs]
    ys = [math.log10(v) if log else v for _, v in pairs]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    scale = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var
    axis = Axis(log=log, scale=scale, offset=mean_y - scale * mean_x)
    spread = max(ys) - min(ys) or 1.0
    worst = max(abs((scale * x + axis.offset) - y) for x, y in zip(xs, ys)) / spread
    return axis, worst


def _tick_value(group: ET.Element) -> float | None:
    """The numeric value of one tick label, plain (`25`) or math (`10^{2}`)."""
    for child in group.iter():
        if child.tag is ET.Comment:
            power = _POWER.search(child.text or "")
            if power:
                return float(power.group(1)) ** float(power.group(2))
    text = "".join(t for t in group.itertext()).strip().replace("−", "-")
    try:
        return float(text)
    except ValueError:
        return None


def _ticks(axis_group: ET.Element, prefix: str, coordinate: int) -> list[tuple[float, float | None]]:
    """(pixel, value) for every tick, with value None when the label is absent.

    A shared axis draws its ticks in each subplot but labels them only once, so
    an unlabelled run of ticks is normal and is resolved by `_borrow`.
    """
    out: list[tuple[float, float | None]] = []
    for tick in axis_group:
        if not (tick.get("id") or "").startswith(prefix):
            continue
        mark = tick.find(f".//{SVG}use")
        if mark is None:
            continue
        out.append((float(mark.get(("x", "y")[coordinate], "0")), _tick_value(tick)))
    return out


def _axis_from(ticks: list[tuple[float, float | None]], prefix: str) -> Axis:
    pairs = [(pixel, value) for pixel, value in ticks if value is not None]
    if len(pairs) < 2:
        raise ValueError(f"{prefix}: need two labelled ticks, found {len(pairs)}")

    linear, linear_error = _fit(pairs, log=False)
    if all(value > 0 for _, value in pairs):
        logarithmic, log_error = _fit(pairs, log=True)
        if log_error < linear_error:
            return logarithmic
    if linear_error > 1e-3:
        raise ValueError(f"{prefix}: ticks are not on a linear or log scale")
    return linear


def _borrow(ticks: list[tuple[float, float | None]], known: dict[tuple[int, ...], Axis]) -> Axis | None:
    """The axis of another subplot that draws its ticks at the same pixels.

    That is what `sharex=True` produces: identical tick positions, labels on
    the outer subplot only.
    """
    return known.get(tuple(round(pixel) for pixel, _ in ticks))


@dataclass
class Panel:
    """One subplot: its title, its axis mapping, and what was drawn in it."""

    title: str
    x: Axis
    y: Axis
    #: label → list of (x, y) in data coordinates, in draw order.
    lines: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    #: label → colour, as written in the SVG.
    colors: dict[str, str] = field(default_factory=dict)
    #: label → the path's full SVG style, which is how a thin faded trace is
    #: told apart from the median drawn over it in the same colour.
    styles: dict[str, dict[str, str]] = field(default_factory=dict)
    #: colour → scatter points, for markers drawn as `<use>` of a shared glyph.
    markers: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


def _legend_labels(axes: ET.Element) -> dict[str, str]:
    """colour → legend label, so unlabelled data paths can be named by hue."""
    named: dict[str, str] = {}
    legend = None
    for group in axes:
        if (group.get("id") or "").startswith("legend_"):
            legend = group
    if legend is None:
        return named

    swatches = [
        _style(path).get("stroke", "")
        for entry in legend
        if (entry.get("id") or "").startswith("line2d_")
        for path in entry.findall(f"{SVG}path")
    ]
    texts = [
        "".join(t for t in entry.itertext()).strip() for entry in legend if (entry.get("id") or "").startswith("text_")
    ]
    for color, label in zip(swatches, texts):
        if color and color != "none":
            named.setdefault(color, label)
    return named


def read_panels(path: str) -> list[Panel]:
    """Every subplot in one figure, with its lines back in data coordinates."""
    # Comments are kept because a log tick's value survives only in the
    # `<!-- $\mathdefault{10^{2}}$ -->` matplotlib writes above it. The input is
    # a figure committed to this repo, not untrusted XML.
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))  # noqa: S314
    root = ET.parse(path, parser=parser).getroot()  # noqa: S314

    all_axes = [g for g in root.iter(f"{SVG}g") if (g.get("id") or "").startswith("axes_")]
    ticks_of = {
        id(axes): [
            _ticks(group, prefix, coordinate)
            for group, (prefix, coordinate) in zip(
                [g for g in axes if (g.get("id") or "").startswith("matplotlib.axis_")],
                (("xtick_", 0), ("ytick_", 1)),
            )
        ]
        for axes in all_axes
    }
    glyphs = {
        glyph.get("id", ""): _style(glyph).get("stroke", "") or _style(glyph).get("fill", "")
        for glyph in root.iter(f"{SVG}path")
        if glyph.get("id")
    }
    known: dict[tuple[int, ...], Axis] = {}
    for ticks in (t for pair in ticks_of.values() for t in pair):
        try:
            known[tuple(round(pixel) for pixel, _ in ticks)] = _axis_from(ticks, "tick")
        except ValueError:
            continue

    panels: list[Panel] = []
    for axes in all_axes:
        children = list(axes)
        x_ticks, y_ticks = ticks_of[id(axes)]
        x_axis = _borrow(x_ticks, known) or _axis_from(x_ticks, "xtick_")
        y_axis = _borrow(y_ticks, known) or _axis_from(y_ticks, "ytick_")

        by_color = _legend_labels(axes)
        panel = Panel(title=_panel_title(axes), x=x_axis, y=y_axis)
        for index, group in enumerate(children):
            if not (group.get("id") or "").startswith("line2d_"):
                continue
            for element in group.findall(f"{SVG}path"):
                style = _style(element)
                color = style.get("stroke", "")
                label = by_color.get(color) or f"{color or 'unnamed'}#{index}"
                while label in panel.lines:
                    label += "'"
                panel.lines[label] = [(x_axis.to_data(px), y_axis.to_data(py)) for px, py in _points(element)]
                panel.colors[label] = color
                panel.styles[label] = style
            _read_markers(group, panel, (x_axis, y_axis), glyphs)
        panels.append(panel)
    return panels


def _read_markers(group: ET.Element, panel: Panel, axes: tuple[Axis, Axis], glyphs: dict[str, str]) -> None:
    """Scatter points: one `<use>` per point, of a glyph defined once per figure."""
    x_axis, y_axis = axes
    for use in group.iter(f"{SVG}use"):
        href = (use.get(f"{XLINK}href") or use.get("href") or "").lstrip("#")
        color = glyphs.get(href, "")
        point = (x_axis.to_data(float(use.get("x", "0"))), y_axis.to_data(float(use.get("y", "0"))))
        panel.markers.setdefault(color, []).append(point)


def _panel_title(axes: ET.Element) -> str:
    """The last free `text_` group in an axes is matplotlib's subplot title."""
    titles = [
        "".join(t for t in group.itertext()).strip() for group in axes if (group.get("id") or "").startswith("text_")
    ]
    return titles[-1] if titles else ""
