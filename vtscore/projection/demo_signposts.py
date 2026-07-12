"""Ground-truth region signposts derived from a dataset's category annotations.

The "cheating" counterpart to the real Toponymy pipeline (see
``docs/plans/vtsbrowse-toponymy.md``): instead of clustering + contrastive
keyphrases + an LLM, this reads each media's **hierarchical ``category``** —
a ``/``-separated path such as ``"Europe/France/Île-de-France/Paris"`` — and
letters the map straight from those ground-truth labels.  It exists so the
signpost *display* (zoom-band fading, multi-level hand-off, de-cluttering)
can be exercised end-to-end without running the heavy naming pipeline, and so
demo datasets that ship a path-encoded taxonomy light up the moment they're
browsed.

The mapping is deliberately simple and layout-driven:

* each distinct path *prefix* is one named region (``"Europe"`` at depth 0,
  ``"Europe/France"`` at depth 1, …);
* a region's **anchor** is the medoid of its members' points in the frozen
  2-D layout (an actual point, robust to cluster shape);
* a region's **level** spreads its taxonomy depth across the pyramid's zoom
  range, so continents show when zoomed out and cities when zoomed in — the
  same ``level`` axis the canvas fades on.

Output is a :class:`~vtscore.projection.labels.RegionLabelSet` of derived text
+ 2-D anchors + scalar scores; no vectors, matching the No-Persisted-Vectors
rule.  Library tier: numpy only, no Flask, no media registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vtscore.projection.labels import RegionLabel, RegionLabelSet, make_label_set

if TYPE_CHECKING:
    from vtscore.projection.umap_projection import Projection

#: Separator between hierarchy segments in a media's ``category`` string.
PATH_SEPARATOR = "/"

#: Minimum members a region needs before it earns a sign — a lone outlier
#: shouldn't letter the map.
_MIN_MEMBERS = 2

#: Never emit more than this many signs total, however deep the taxonomy — a
#: backstop against a pathological category space, not a limit any real demo
#: hits.  The canvas de-clutters what's visible; this only bounds the payload.
_MAX_SIGNS = 600

#: Zoom-level gap between adjacent hierarchy depths.  A sign's ``level`` is a
#: *zoom band*, not a tile LOD: the canvas shows the current tier crisp with its
#: parent one step coarser still faintly visible (the canvas fades over roughly
#: ±1.5–2.5 levels).  ~1.8 means each tier hands off to the next after about a
#: 3.5× zoom, so all four tiers span a comfortable ~5.4 zoom levels — never the
#: dozen tiling levels a dense layout can carry (which would bury city names at
#: 2000× zoom).  Tuned for a Toponymy-like "a couple of layers legible at once"
#: feel; independent of dataset size.
_LEVEL_STEP = 1.8


def clean_category_path(category: Any) -> list[str]:
    """Split a ``category`` string into cleaned hierarchy segments.

    Splits on :data:`PATH_SEPARATOR`, trims whitespace, and drops empty
    segments.  A leading **single-character** segment is dropped too: it's the
    alphabetical index bucket some sources prepend (Places365's
    ``"/a/arena/hockey"`` → ``["arena", "hockey"]``), never a real region name.
    Returns ``[]`` for a missing/blank/non-string category.
    """
    if not isinstance(category, str):
        return []
    parts = [seg.strip() for seg in category.split(PATH_SEPARATOR)]
    parts = [seg for seg in parts if seg]
    if parts and len(parts[0]) == 1:
        parts = parts[1:]
    return parts


def has_hierarchical_categories(medias: dict[int, dict[str, Any]], *, min_fraction: float = 0.5) -> bool:
    """Whether enough medias carry a multi-segment ``category`` path to letter.

    A cheap pure-Python probe (no numpy, no projection) the request path uses to
    decide whether ground-truth signposts are even applicable before importing
    the builder.  ``True`` when at least *min_fraction* of medias have a
    category that cleans to two or more segments — i.e. an actual hierarchy, not
    a flat tag set.
    """
    if not medias:
        return False
    deep = sum(1 for m in medias.values() if len(clean_category_path(m.get("category"))) >= 2)
    return deep >= max(1, int(min_fraction * len(medias)))


def _level_for_depth(depth: int) -> float:
    """Map a taxonomy *depth* (0 = coarsest) to a signpost zoom level.

    A fixed :data:`_LEVEL_STEP` per tier: depth 0 (continents) sits at level 0,
    visible at the whole-projection fit, and each finer tier is one step deeper
    in.  Deliberately *not* scaled to the pyramid's tiling depth — sign levels
    are zoom bands, not LODs (see :data:`_LEVEL_STEP`).
    """
    return depth * _LEVEL_STEP


def build_category_signposts(
    projection: "Projection",
    medias: dict[int, dict[str, Any]],
) -> RegionLabelSet:
    """Build a hierarchical :class:`RegionLabelSet` from category annotations.

    For every distinct path prefix across the medias' cleaned ``category``
    paths (see :func:`clean_category_path`), emit one sign: text = the prefix's
    last segment, anchor = the medoid of that region's members in
    *projection*'s 2-D layout, level = its depth mapped to a signpost zoom band
    (see :func:`_level_for_depth`).  Regions with fewer than
    :data:`_MIN_MEMBERS` members are skipped; the whole set is capped at
    :data:`_MAX_SIGNS` (coarsest, largest regions kept first).

    The set is pinned to ``projection.projection_id`` so the serving layer only
    ever hands it back over the layout it was measured on.  Returns an empty
    (but still id-pinned) set when no media carries a usable path — callers can
    cache that as "no signs for this layout" without recomputing.
    """
    empty = make_label_set(projection.projection_id, [])
    n = projection.coords.shape[0]
    if n == 0 or not medias:
        return empty

    coords = np.asarray(projection.coords, dtype=np.float32)
    index_of = {int(mid): i for i, mid in enumerate(projection.ids)}

    # region prefix (tuple of segments) -> list of row indices into `coords`
    members: dict[tuple[str, ...], list[int]] = {}
    for mid, media in medias.items():
        row = index_of.get(int(mid))
        if row is None:
            continue
        path = clean_category_path(media.get("category"))
        for depth in range(len(path)):
            prefix = tuple(path[: depth + 1])
            members.setdefault(prefix, []).append(row)

    if not members:
        return empty

    labels: list[RegionLabel] = []
    for prefix, rows in members.items():
        if len(rows) < _MIN_MEMBERS:
            continue
        depth = len(prefix) - 1
        pts = coords[rows]
        anchor = _medoid(pts)
        labels.append(
            RegionLabel(
                level=_level_for_depth(depth),
                x=float(anchor[0]),
                y=float(anchor[1]),
                text=prefix[-1],
                # Larger, coarser regions win the de-clutter tiebreak; keeping
                # member count as the score makes that ordering explicit.
                score=float(len(rows)),
                source="ground-truth",
            )
        )

    # Coarsest-and-largest first, so the density cap (if ever hit) drops the
    # least important fine signs rather than a continent.
    labels.sort(key=lambda lab: (lab.level, -lab.score))
    if len(labels) > _MAX_SIGNS:
        labels = labels[:_MAX_SIGNS]

    return make_label_set(projection.projection_id, labels)


def _medoid(pts: np.ndarray) -> np.ndarray:
    """Return the member point nearest the centroid of *pts* (an ``(m, 2)`` array).

    A medoid — an actual member, not the raw mean — so the anchor sits on the
    cluster even when it's crescent-shaped or split, where the centroid could
    land in empty space between lobes.
    """
    if pts.shape[0] == 1:
        return pts[0]
    centroid = pts.mean(axis=0)
    d2 = np.einsum("ij,ij->i", pts - centroid, pts - centroid)
    return pts[int(np.argmin(d2))]


__all__ = [
    "build_category_signposts",
    "clean_category_path",
    "has_hierarchical_categories",
    "PATH_SEPARATOR",
]
