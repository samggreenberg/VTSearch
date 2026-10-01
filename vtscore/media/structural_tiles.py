"""Tiled VLAD: the structural Stage 1 for document pages (#3928).

A page-level VLAD averages every keypoint on a page, and on a document page the
text wins, so a stamp or logo is invisible to it (FullMarks: AP 0.003 at 50,000
pages).  Tiling the page lets a mark compete with its neighbourhood instead:
one VLAD per overlapping window, projected by a whitened PCA to
:data:`TILE_DIM` dimensions, and a page scores its **best tile**.  Measured on
FullMarks v5.0 (``docs/plans/structural-tiled-stage1.md``): the top 1,000 pages
it hands Stage 2 keep 84% of verifying every page, and 97% once Good votes add
their boxes as queries.

This module is the one definition of the tiling, the projection and the
segment-max score.  The app and the FullMarks experiment scripts both import it.

**The projection is a calculated asset, so it lives in the models cache, not
the repo** (owner, 2026-09-30).  :func:`fit_projection` plus
``scripts/experiments/fullmarks/fit_tile_projection.py`` regenerate it
deterministically, and :func:`load_tile_projection` fails loudly, naming the
file and the rebuild command, when it is absent.  Nothing ever falls back to
page VLAD in silence.

Library-tier: numpy only, plus the structural primitives.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from vtscore.media.structural import StructuralFeatures, aggregate_vlad, load_vlad_codebook

#: The layout #3928 measured as the better of two: 0.25 x 0.18 of a page, stride
#: half a tile.  A FullMarks mark is ~0.1-0.3 of a page's width.
TILE_W, TILE_H = 0.25, 0.18
#: A tile with fewer keypoints than this has nothing to aggregate.
MIN_TILE_KP = 20
#: The tile layers a page is cut into, as ``(width, height)``; a page scores its best
#: tile across all of them. The 0.25 x 0.18 layer is #3928's. The 0.125 x 0.09 layer
#: was added by #4415 for small marks in text-dense tiles: +0.02 AP overall at 50,000
#: pages, and the weakest classes +0.10-0.20, for ~3x the tiles (178 a page).
TILE_LAYERS: tuple[tuple[float, float], ...] = ((TILE_W, TILE_H), (TILE_W / 2, TILE_H / 2))
#: The stored width.  512 beat 256 at 50,000 pages (#3928, 2026-09-18: 0.73 against
#: 0.70 after SIFT at K = 1,000) for 46 KiB a page.
TILE_DIM = 512
#: The cached projection's name.  A refit that changes the numbers takes the next
#: version, so a cached file never changes meaning.  v1 was fit on one tile layer;
#: v2 on both (#4415).
PROJECTION_NAME = "tile_projection_v2"
#: Pages sampled to fit the projection: ~20,000 tiles, well over the 8,192
#: dimensions being reduced.
FIT_SAMPLE_PAGES = 400
#: The command that rebuilds the cached projection, quoted in every missing-asset error.
REBUILD_COMMAND = "python scripts/experiments/fullmarks/fit_tile_projection.py --tier s"


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def tile_windows(width: float = TILE_W, height: float = TILE_H) -> list[tuple[float, float, float, float]]:
    """Overlapping windows covering the unit square, stride half a tile.

    The last window starts on the stride grid, so at 0.25 x 0.18 the bottom edge
    stops at 0.99.  #3928's numbers are measured under this geometry, and a test
    pins it.
    """
    xs = np.arange(0.0, max(1e-9, 1.0 - width) + 1e-9, width / 2)
    ys = np.arange(0.0, max(1e-9, 1.0 - height) + 1e-9, height / 2)
    return [(float(x), float(y), float(x + width), float(y + height)) for y in ys for x in xs]


def tile_rows(
    keypoints: np.ndarray,
    descriptors: np.ndarray,
    codebook: np.ndarray,
    aggregate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    width: float = TILE_W,
    height: float = TILE_H,
    min_kp: int = MIN_TILE_KP,
) -> tuple[np.ndarray, np.ndarray]:
    """``(vectors, boxes)`` for the tiles of one page that hold enough keypoints.

    ``aggregate`` is injected so the geometry can be tested without the SIFT
    stack.  A page whose tiles are all too sparse still gets one row, a VLAD of
    every descriptor it has, because a page with no row could never be
    retrieved.
    """
    rows: list[np.ndarray] = []
    boxes: list[tuple[float, float, float, float]] = []
    if keypoints.shape[0]:
        x, y = keypoints[:, 0], keypoints[:, 1]
        for x0, y0, x1, y1 in tile_windows(width, height):
            inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            if int(inside.sum()) >= min_kp:
                rows.append(aggregate(descriptors[inside], codebook))
                boxes.append((x0, y0, x1, y1))
    if not rows:
        rows.append(aggregate(descriptors, codebook))
        boxes.append((0.0, 0.0, 1.0, 1.0))
    return np.asarray(rows, dtype=np.float32), np.asarray(boxes, dtype=np.float32)


def raw_tiles(features: StructuralFeatures) -> tuple[np.ndarray, np.ndarray]:
    """A page's unprojected 8,192-d tile VLADs and their boxes, from its stored features.

    Equal to :func:`tile_rows` with :func:`~vtscore.media.structural.aggregate_vlad`
    (a test pins it), but vectorised so it is cheap enough to run at load time.
    ``tile_rows`` re-assigns every tile's descriptors to the codebook, so a
    keypoint in four overlapping tiles is assigned four times. That costs ~100 ms
    a page, over an hour at 50,000 pages. Here each descriptor is assigned once,
    and each codeword's residuals are summed into every tile with one
    membership matmul.
    """
    from vtscore.media.structural import rootsift  # noqa: PLC0415

    kp = features.keypoints_f32()
    desc = features.descriptors_f32()
    codebook = load_vlad_codebook()
    k, d = codebook.shape
    if kp.shape[0] == 0 or desc.size == 0:
        return (
            np.zeros((1, k * d), dtype=np.float32),
            np.asarray([(0.0, 0.0, 1.0, 1.0)], dtype=np.float32),
        )
    x, y = kp[:, 0], kp[:, 1]
    windows = [w for width, height in TILE_LAYERS for w in tile_windows(width, height)]
    member = np.stack([(x >= x0) & (x < x1) & (y >= y0) & (y < y1) for x0, y0, x1, y1 in windows])
    keep = member.sum(axis=1) >= MIN_TILE_KP
    if keep.any():
        member = member[keep]
        boxes = np.asarray([w for w, kept in zip(windows, keep) if kept], dtype=np.float32)
    else:
        # As tile_rows: a page too sparse to tile gets one row over every keypoint.
        member = np.ones((1, kp.shape[0]), dtype=bool)
        boxes = np.asarray([(0.0, 0.0, 1.0, 1.0)], dtype=np.float32)

    desc_r = rootsift(desc)
    cb_r = rootsift(np.asarray(codebook, dtype=np.float32))
    assign = np.argmin((cb_r**2).sum(axis=1)[None, :] - 2.0 * (desc_r @ cb_r.T), axis=1)
    weights = member.astype(np.float32)
    vlad = np.zeros((member.shape[0], k, d), dtype=np.float32)
    for j in range(k):
        idx = np.flatnonzero(assign == j)
        if idx.size:
            vlad[:, j, :] = weights[:, idx] @ (desc_r[idx] - cb_r[j])
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    flat = vlad.reshape(vlad.shape[0], -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    return (flat / np.where(norm > 0, norm, 1.0)).astype(np.float32), boxes


def box_vlad(features: StructuralFeatures, box: tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """The unprojected VLAD of the keypoints inside *box* (normalised ``x0, y0, x1, y1``).

    This is a Good vote's Stage-1 query: the mark it boxed, described the way
    a tile is.  ``None`` when the box holds no keypoint.
    """
    kp = features.keypoints_f32()
    if kp.shape[0] == 0:
        return None
    x0, x1 = sorted((float(box[0]), float(box[2])))
    y0, y1 = sorted((float(box[1]), float(box[3])))
    inside = (kp[:, 0] >= x0) & (kp[:, 0] <= x1) & (kp[:, 1] >= y0) & (kp[:, 1] <= y1)
    if not inside.any():
        return None
    return aggregate_vlad(features.descriptors_f32()[inside], load_vlad_codebook())


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------


def normalise(rows: np.ndarray) -> np.ndarray:
    """L2-normalise float32 rows, so a stored dot product is a cosine."""
    rows = np.asarray(rows, dtype=np.float32)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norm, 1e-12)


@dataclass
class Projection:
    """PCA with whitening, fitted once at the widest width and sliced for the rest.

    Whitening scales each component by its own standard deviation, so the first
    ``d`` rows of a wider fit *are* the ``d``-wide fit.  Rows are L2-normalised
    after projection, so a dot product is a cosine.
    """

    mean: np.ndarray
    components: np.ndarray  # (dim, 8192), already divided by each component's sigma

    @property
    def dim(self) -> int:
        return int(self.components.shape[0])

    def slice(self, dim: int) -> "Projection":
        if dim > self.dim:
            raise ValueError(f"cannot widen a {self.dim}-dim projection to {dim}")
        return Projection(self.mean, self.components[:dim])

    def apply(self, rows: np.ndarray) -> np.ndarray:
        out = (np.asarray(rows, dtype=np.float32) - self.mean) @ self.components.T
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-12)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, components=self.components)

    @staticmethod
    def load(path: Path) -> "Projection":
        with np.load(path) as z:
            return Projection(z["mean"].astype(np.float32), z["components"].astype(np.float32))


def fit_projection(sample: np.ndarray, dim: int) -> Projection:
    """Fit a whitened PCA of width ``dim`` on tile vectors ``sample``."""
    sample = np.asarray(sample, dtype=np.float32)
    if dim > min(sample.shape):
        raise ValueError(f"dim {dim} exceeds the sample's rank ({min(sample.shape)})")
    mean = sample.mean(axis=0)
    centred = sample - mean
    # Economy SVD: rows are far fewer than 8,192 columns for any sane sample.
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    sigma = np.maximum(s[:dim] / np.sqrt(max(1, centred.shape[0] - 1)), 1e-6)
    return Projection(mean, (vt[:dim] / sigma[:, None]).astype(np.float32))


def fit_sample_ids(page_ids: Sequence[str], n: int = FIT_SAMPLE_PAGES) -> list[str]:
    """The pages a fit samples: every ``len/n``-th id in sorted order, so a refit is exact."""
    ids = sorted(page_ids)
    return ids[:: max(1, len(ids) // n)][:n]


def sample_digest(page_ids: Sequence[str]) -> str:
    """A short hash of a fit's page sample, recorded in its provenance."""
    return hashlib.sha256("\n".join(page_ids).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# The cached asset
# --------------------------------------------------------------------------


class TileProjectionMissing(RuntimeError):
    """The cached tile projection is absent or unreadable; tiled Stage 1 cannot run."""


def projection_path(models_dir: Optional[Path] = None) -> Path:
    """Where the cached projection lives: ``<models dir>/structural/<PROJECTION_NAME>.npz``."""
    if models_dir is None:
        from vtscore.config.paths import MODELS_CACHE_DIR  # noqa: PLC0415

        models_dir = MODELS_CACHE_DIR
    return Path(models_dir) / "structural" / f"{PROJECTION_NAME}.npz"


def write_projection(projection: Projection, path: Path, provenance: dict[str, Any]) -> None:
    """Write the projection and its provenance record (``<stem>.json``) beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    projection.save(tmp)
    tmp.replace(path)
    meta = {"name": PROJECTION_NAME, "dim": projection.dim, **provenance}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


@functools.lru_cache(maxsize=4)
def _load_cached(path: str) -> Projection:
    return Projection.load(Path(path))


def load_tile_projection(models_dir: Optional[Path] = None, dim: int = TILE_DIM) -> Projection:
    """The cached projection at width *dim*, or :class:`TileProjectionMissing` naming the fix."""
    path = projection_path(models_dir)
    if not path.is_file():
        raise TileProjectionMissing(
            f"tiled Stage 1 needs the cached projection {path}, which does not exist. "
            f"Rebuild it with `{REBUILD_COMMAND}` (writes it under $VTSEARCH_MODELS_DIR)."
        )
    try:
        projection = _load_cached(str(path))
    except Exception as exc:  # noqa: BLE001 - any unreadable file is the same failure to the caller
        raise TileProjectionMissing(f"cannot read the cached tile projection {path}: {exc}") from exc
    if projection.dim < dim:
        raise TileProjectionMissing(f"{path} is {projection.dim}-wide; tiled Stage 1 needs {dim}")
    return projection.slice(dim)


# --------------------------------------------------------------------------
# A page's stored tiles, and scoring
# --------------------------------------------------------------------------


@dataclass
class TileVectors:
    """One page's projected tiles: ``vectors`` ``(T, dim)`` fp16 unit rows, ``boxes`` ``(T, 4)``."""

    vectors: np.ndarray
    boxes: np.ndarray

    @property
    def count(self) -> int:
        return int(self.vectors.shape[0])


def tile_vectors(features: StructuralFeatures, projection: Projection) -> TileVectors:
    """A page's :class:`TileVectors`, derived from its stored local features."""
    rows, boxes = raw_tiles(features)
    return TileVectors(projection.apply(rows).astype(np.float16), boxes.astype(np.float16))


def project_query(projection: Projection, raw: np.ndarray) -> np.ndarray:
    """A raw 8,192-d query VLAD (or a stack of them) projected like the tiles."""
    rows = np.atleast_2d(np.asarray(raw, dtype=np.float32))
    return projection.apply(rows)


def starts_from_counts(counts: Sequence[int] | np.ndarray) -> np.ndarray:
    """Row index where each page's tiles begin."""
    counts = np.asarray(counts, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


def page_scores(tiles: np.ndarray, starts: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Max over each page's tiles of the dot with ``query``, one score per page.

    ``query`` is one vector ``(dim,)`` or a stack ``(Q, dim)``; a stack scores each
    page by its best (query, tile) pair, which is how several queries (a crop plus
    each Good's box) combine.  ``starts`` holds each page's first row, so the
    segment max is one ``reduceat`` over a single matmul.
    """
    q = np.asarray(query, dtype=np.float32)
    flat = np.asarray(tiles, dtype=np.float32) @ (q.T if q.ndim == 2 else q)
    if flat.ndim == 2:
        flat = flat.max(axis=1)
    return np.maximum.reduceat(flat, starts)


def page_scores_per_query(tiles: np.ndarray, starts: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Per-page best tile for each query separately: ``(pages, Q)`` for ``queries`` ``(dim, Q)``.

    The batch form experiments use to score many classes in one matmul; the app
    wants :func:`page_scores`, which takes the max across queries.
    """
    flat = np.asarray(tiles, dtype=np.float32) @ np.asarray(queries, dtype=np.float32)
    if flat.ndim == 1:
        return np.maximum.reduceat(flat, starts)
    return np.maximum.reduceat(flat, starts, axis=0)
