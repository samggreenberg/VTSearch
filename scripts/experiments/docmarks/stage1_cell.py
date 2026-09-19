"""A cached tiled-VLAD Stage 1 for DocMarks (#3928).

#3928's probe measured the shape of the answer -- max-over-tiles VLAD ranks the
roster at AP 0.40 on tier ``s`` against 0.029 for one vector per page, and the
top 1,000 it hands SIFT keep 91% of exhaustive verification -- but it recomputed
every tile from the page images on each run.  That is what makes it a probe and
not a stage: tier ``m`` cost 39 minutes of 40 CPUs to answer 23 queries, and
tier ``l`` cannot be searched at all.

**The obstacle is size, not time.**  A tile VLAD is 64 x 128 = 8,192 dimensions;
~50 tiles survive per page; at fp16 that is 0.82 MB per page, so storing them is
41 GB at tier ``m`` and 164 GB at tier ``l`` -- as large as the local features
Stage 2 needs, and more than the filesystem can spare.

So the cell stores **projected** tiles.  PCA with whitening, fitted on a sample
of tiles and applied to all of them, is the standard compression for VLAD
descriptors, and slicing a wider projection's components gives the narrower one
exactly, so one fit serves every width.  What the projection costs in retrieval
is *measured* -- see ``eval_stage1_cell.py`` and the #3928 report -- never
assumed: a narrower cell that keeps the shortlist's recall is the result, and a
narrower cell that loses it is the finding.

    python stage1_cell.py build --tier s --dims 0,512,256,128 --out <dir>
    python stage1_cell.py search --cell <dir>/d256 --class-id spods/logo_00003_0

A cell directory holds ``meta.json``, ``projection.npz`` (absent for a raw
cell), and ``shard-*.npz`` with one page's tiles contiguous, so a page's score
is a segment max over a single matmul (``numpy.maximum.reduceat``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: The layout #3928 measured as the better of two: 0.25 x 0.18 of a page, stride
#: half a tile.  A DocMarks mark is ~0.1-0.3 of a page's width.
TILE_W, TILE_H = 0.25, 0.18
#: A tile with fewer keypoints than this has nothing to aggregate.
MIN_TILE_KP = 20
#: Pages sampled to fit the projection.  400 pages is ~20,000 tiles, well over
#: the 8,192 dimensions being reduced, and costs ~1% of the build.
SAMPLE_PAGES = 400
#: Widths built by default: 0 means the raw 8,192-dimensional tile, kept so the
#: compression can be priced against something.
DEFAULT_DIMS = (0, 512, 256, 128)
#: Pages per shard file.
SHARD_PAGES = 2000


def tile_windows(width: float = TILE_W, height: float = TILE_H) -> list[tuple[float, float, float, float]]:
    """Overlapping windows covering the unit square, stride half a tile."""
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
    stack.  A page whose tiles are all too sparse still gets one row -- a VLAD of
    every descriptor it has -- because a page with no row cannot be retrieved at
    all, and the probe scored it that way.
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


@dataclass
class Projection:
    """PCA with whitening, fitted once at the widest width and sliced for the rest.

    Whitening scales each component by its own standard deviation, so the first
    ``d`` rows of a wider fit *are* the ``d``-wide fit -- which is why one pass
    over the corpus can write every width.  Rows are L2-normalised after
    projection, so a dot product is a cosine.
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
    # economy SVD: rows are far fewer than 8,192 columns for any sane sample
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    sigma = np.maximum(s[:dim] / np.sqrt(max(1, centred.shape[0] - 1)), 1e-6)
    return Projection(mean, (vt[:dim] / sigma[:, None]).astype(np.float32))


def normalise(rows: np.ndarray) -> np.ndarray:
    """L2-normalise float32 rows, so a stored dot product is a cosine."""
    rows = np.asarray(rows, dtype=np.float32)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norm, 1e-12)


def page_scores(tiles: np.ndarray, starts: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Max over each page's tiles of the dot with ``query``, one score per page.

    ``starts`` holds each page's first row, so the segment max is one
    ``reduceat`` over a single matmul rather than a Python loop per page.
    """
    flat = np.asarray(tiles, dtype=np.float32) @ np.asarray(query, dtype=np.float32)
    if flat.ndim == 1:
        return np.maximum.reduceat(flat, starts)
    return np.maximum.reduceat(flat, starts, axis=0)


def starts_from_counts(counts: np.ndarray) -> np.ndarray:
    """Row index where each page's tiles begin."""
    counts = np.asarray(counts, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


def top_k(page_ids: Sequence[str], scores: np.ndarray, k: int) -> list[tuple[str, float]]:
    """The ``k`` best-scoring pages, ties broken by page id so a run is repeatable."""
    k = min(k, len(page_ids))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    order = sorted(idx, key=lambda i: (-float(scores[i]), page_ids[i]))
    return [(page_ids[i], float(scores[i])) for i in order]


class Cell:
    """A built cell, read back for search."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.meta: dict[str, Any] = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        proj = self.path / "projection.npz"
        self.projection: Optional[Projection] = Projection.load(proj) if proj.exists() else None
        if not self.meta.get("complete", False):
            raise ValueError(
                f"{self.path} is a PARTIAL cell: its build did not finish, so a search over it would "
                "silently miss pages.  Rebuild it, or pass the shards you trust explicitly."
            )
        ids: list[str] = []
        counts: list[np.ndarray] = []
        tiles: list[np.ndarray] = []
        for shard in sorted(self.path.glob("shard-*.npz")):
            with np.load(shard, allow_pickle=False) as z:
                ids.extend(str(p) for p in z["page_ids"])
                counts.append(z["counts"].astype(np.int64))
                tiles.append(z["tiles"])
        if not ids:
            raise ValueError(f"no shards in {self.path}")
        if len(ids) != int(self.meta["pages"]):
            raise ValueError(
                f"{self.path} holds {len(ids)} pages but its meta claims {self.meta['pages']}: "
                "a shard is missing or was written twice"
            )
        self.page_ids: list[str] = ids
        self.counts: np.ndarray = np.concatenate(counts)
        self.tiles: np.ndarray = np.concatenate(tiles)
        self.starts: np.ndarray = starts_from_counts(self.counts)

    @property
    def dim(self) -> int:
        return int(self.tiles.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.tiles.nbytes)

    def prepare_query(self, vlad: np.ndarray) -> np.ndarray:
        """Project and normalise a raw query VLAD the way the stored tiles were."""
        rows = np.atleast_2d(np.asarray(vlad, dtype=np.float32))
        rows = self.projection.apply(rows) if self.projection is not None else normalise(rows)
        return rows.T if rows.shape[0] > 1 else rows[0]

    def scores(self, query: np.ndarray) -> np.ndarray:
        return page_scores(self.tiles, self.starts, query)

    def search(self, vlad: np.ndarray, k: int = 1000) -> list[tuple[str, float]]:
        return top_k(self.page_ids, self.scores(self.prepare_query(vlad)), k)


# ---------------------------------------------------------------------------
# build (GRID side)
# ---------------------------------------------------------------------------

_BUDGET = 0
_PROJECTIONS: dict[int, Projection] = {}


def _init(budget: int, projections: dict[int, Projection]) -> None:
    global _BUDGET, _PROJECTIONS
    _BUDGET, _PROJECTIONS = budget, projections


def _page_tiles(item: tuple[str, str]) -> tuple[str, dict[int, np.ndarray], np.ndarray]:
    from eval_splg_rank import _gray  # noqa: PLC0415
    from vtscore.media.structural import SiftMatcher, aggregate_vlad, load_vlad_codebook  # noqa: PLC0415

    page_id, path = item
    feats = SiftMatcher().detect_and_describe(_gray(path), max_features=_BUDGET)
    rows, boxes = tile_rows(feats.keypoints_f32(), feats.descriptors_f32(), load_vlad_codebook(), aggregate_vlad)
    out = {
        dim: (normalise(rows) if proj is None else proj.apply(rows)).astype(np.float16)
        for dim, proj in _PROJECTIONS.items()
    }
    return page_id, out, boxes.astype(np.float16)


def build(
    pages: Sequence[Any],
    out: Path,
    dims: Sequence[int],
    budget: int,
    workers: int,
    sample_pages: int = SAMPLE_PAGES,
    log: Callable[[str], None] = print,
    projection: Optional[Path] = None,
) -> dict[int, Path]:
    """Build one cell per requested width, in a single pass over the pages.

    ``projection`` reuses a fit from an earlier run rather than refitting, which
    is what lets a tier be built in shards that stay comparable: two shards
    projected by two different fits are not the same cell.
    """
    from multiprocessing import get_context  # noqa: PLC0415

    # Every worker projects a (tiles x 8,192) matrix, which is big enough for
    # OpenBLAS to go multi-threaded -- and with one pool worker per core that is
    # cores^2 threads fighting over cores.  Measured on tier m: 1.05 pages/s
    # against 20 for the probe, whose per-page matmul was too small to trigger
    # it.  The launcher pins OMP_NUM_THREADS=1; warn if it did not.
    if os.environ.get("OMP_NUM_THREADS") != "1":
        log("  WARNING: OMP_NUM_THREADS is not 1; BLAS threads will oversubscribe the pool (see #3928)")

    items = [(p.page_id, p.path) for p in pages]
    widest = max(d for d in dims if d > 0) if any(d > 0 for d in dims) else 0

    projections: dict[int, Optional[Projection]] = {d: None for d in dims if d == 0}
    if widest:
        if projection is not None:
            fitted = Projection.load(projection)
            log(f"  reusing the projection at {projection} ({fitted.dim} dims)")
            if fitted.dim < widest:
                raise ValueError(f"saved projection is {fitted.dim}-wide, too narrow for {widest}")
        else:
            sample = _raw_sample(items[:: max(1, len(items) // sample_pages)][:sample_pages], budget, workers, log)
            log(f"  fitting PCA {sample.shape[0]} tiles x {sample.shape[1]} -> {widest}")
            fitted = fit_projection(sample, widest)
        for d in dims:
            if d > 0:
                projections[d] = fitted.slice(d)

    dirs = {}
    for d in dims:
        cell = out / (f"d{d}" if d else "raw")
        cell.mkdir(parents=True, exist_ok=True)
        if projections.get(d) is not None:
            projections[d].save(cell / "projection.npz")
        dirs[d] = cell

    buffers: dict[int, dict[str, list]] = {d: {"ids": [], "counts": [], "tiles": [], "boxes": []} for d in dims}
    shard_no, written, t0 = 0, 0, time.time()
    with get_context("fork").Pool(workers, initializer=_init, initargs=(budget, projections)) as pool:
        for i, (page_id, per_dim, boxes) in enumerate(pool.imap(_page_tiles, items, chunksize=8)):
            for d in dims:
                buf = buffers[d]
                buf["ids"].append(page_id)
                buf["counts"].append(per_dim[d].shape[0])
                buf["tiles"].append(per_dim[d])
                buf["boxes"].append(boxes)
            if len(buffers[dims[0]]["ids"]) >= SHARD_PAGES:
                _flush(buffers, dirs, dims, shard_no)
                shard_no += 1
            written = i + 1
            if written % 2000 == 0:
                log(f"  tiled {written}/{len(items)} pages in {time.time() - t0:.0f}s")
    if buffers[dims[0]]["ids"]:
        _flush(buffers, dirs, dims, shard_no)

    for d in dims:
        size = sum(f.stat().st_size for f in dirs[d].glob("shard-*.npz"))
        (dirs[d] / "meta.json").write_text(
            json.dumps(
                {
                    "dim": d or 8192,
                    "projected": bool(d),
                    "budget": budget,
                    "tile": [TILE_W, TILE_H],
                    "min_tile_kp": MIN_TILE_KP,
                    "pages": written,
                    "complete": written == len(items),
                    "expected_pages": len(items),
                    "bytes": size,
                    "bytes_per_page": round(size / max(1, written), 1),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        log(f"  cell d={d or 8192}: {size / 2**30:.2f} GiB, {size / max(1, written) / 1024:.1f} KiB/page")
    return dirs


def _raw_sample(items: Sequence[tuple[str, str]], budget: int, workers: int, log: Callable[[str], None]) -> np.ndarray:
    """Raw (unprojected) tile vectors from a sample of pages, for the PCA fit."""
    from multiprocessing import get_context  # noqa: PLC0415

    log(f"  sampling {len(items)} pages for the projection")
    rows: list[np.ndarray] = []
    with get_context("fork").Pool(workers, initializer=_init, initargs=(budget, {0: None})) as pool:
        for _pid, per_dim, _boxes in pool.imap_unordered(_page_tiles, list(items), chunksize=4):
            rows.append(per_dim[0].astype(np.float32))
    return np.concatenate(rows)


def _flush(buffers: dict[int, dict[str, list]], dirs: dict[int, Path], dims: Sequence[int], shard_no: int) -> None:
    for d in dims:
        buf = buffers[d]
        np.savez(
            dirs[d] / f"shard-{shard_no:04d}.npz",
            page_ids=np.array(buf["ids"], dtype=object).astype("U"),
            counts=np.array(buf["counts"], dtype=np.int32),
            tiles=np.concatenate(buf["tiles"]),
            boxes=np.concatenate(buf["boxes"]),
        )
        for key in buf:
            buf[key].clear()


def query_vlad(crop_path: str, budget: int) -> np.ndarray:
    """The raw VLAD of a query crop, before any projection."""
    from eval_splg_rank import _gray  # noqa: PLC0415
    from vtscore.media.structural import SiftMatcher, aggregate_vlad, load_vlad_codebook  # noqa: PLC0415

    feats = SiftMatcher().detect_and_describe(_gray(crop_path), max_features=budget)
    return aggregate_vlad(feats.descriptors_f32(), load_vlad_codebook()).astype(np.float32)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import docmarks_config as cfg  # noqa: PLC0415
    import embed_corpus  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--corpus", type=Path, default=cfg.OUT)
    b.add_argument("--tier", default="s")
    b.add_argument("--budget", type=int, default=8192)
    b.add_argument("--dims", default=",".join(str(d) for d in DEFAULT_DIMS), help="0 keeps the raw 8,192 dims")
    b.add_argument("--sample-pages", type=int, default=SAMPLE_PAGES)
    b.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--projection", type=Path, help="reuse this projection.npz instead of fitting one")

    s = sub.add_parser("search")
    s.add_argument("--corpus", type=Path, default=cfg.OUT)
    s.add_argument("--cell", type=Path, required=True)
    s.add_argument("--class-id", required=True)
    s.add_argument("--k", type=int, default=1000)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        if args.out.resolve().is_relative_to(args.corpus.resolve()):
            ap.error("--out is inside the corpus; a cell never writes there")
        dims = [int(d) for d in args.dims.split(",")]
        pages = embed_corpus.pages_for_tier(args.corpus, args.tier)
        print(f"tier {args.tier}: {len(pages)} pages, dims {dims}", flush=True)
        build(pages, args.out, dims, args.budget, args.workers, args.sample_pages, projection=args.projection)
        return 0

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    cell = Cell(args.cell)
    t0 = time.time()
    hits = cell.search(query_vlad(classes[args.class_id]["query_crop"], cell.meta["budget"]), args.k)
    print(f"{len(hits)} hits in {time.time() - t0:.2f}s from {cell.meta['pages']} pages")
    for page_id, score in hits[:20]:
        print(f"  {score:.3f}  {page_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
