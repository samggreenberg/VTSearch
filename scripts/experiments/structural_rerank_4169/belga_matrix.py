"""BelgaLogos template matrix for #4169, in ``template_matrix.py``'s format.

#4169 asks whether the structural re-rank should score by the match-statistic MLP or by
inliers.  FullMarks (documents) answered it; this builds the same matrix on BelgaLogos, a
10k-image press-photo corpus with instance-level boxes, so ``vote_curve.py`` can replay
the same arms on photos.  OpenLogo, the corpus the MLP was built on, is gone from the
GRID.

Protocol (BelgaLogos' own):

* **classes:** every logo with a curated ``qset1`` query and at least
  :data:`MIN_POSITIVES` images carrying a clearly visible (``state 1``) instance;
* **query crop:** the logo's first ``qset1`` query box; its image leaves the pool;
* **positives:** images with at least one ``state 1`` instance;
* **ignored:** images whose only instances are ``state 0`` (partly visible), as the
  benchmark does -- they are neither positive nor negative;
* **templates:** the query crop, then each positive's own features inside its largest
  ``state 1`` box (``filter_features_to_box``, the production RegionYes template).

Features are the shipped ``sift_vlad`` photo configuration: ``DEFAULT_MAX_FEATURES``
(1,024) keypoints, detected under ``MAX_STRUCTURAL_DETECT_PIXELS``.

    python belga_matrix.py --out <dir> [--workers 32]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

FULLMARKS = Path(__file__).resolve().parents[1] / "fullmarks"
sys.path.insert(0, str(FULLMARKS))

import eval_sift_rank as esr  # noqa: E402
import template_matrix as tm  # noqa: E402

BELGA = Path("/expscratch/ccarpenter/logomatch/data/belgalogos")
MIN_POSITIVES = 10
#: ``qset1`` query stems whose spelling differs from ``qset3``'s logo names.
QUERY_ALIASES = {
    "Addidas": "Adidas",
    "Addidas-text": "Adidas-text",
    "President": "US_President",
    "Eleclerc": "ELeclerc",
}

Box = tuple[float, float, float, float]


def read_instances(root: Path) -> dict[str, dict[str, list[tuple[int, Box]]]]:
    """``{logo: {image: [(state, (x0, y0, x1, y1)), ...]}}`` from ``qset3``."""
    out: dict[str, dict[str, list[tuple[int, Box]]]] = {}
    for line in (root / "qset3_internal_and_local.gt").read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if len(f) != 9:
            continue
        _, logo, image, _, state, *xyxy = f
        box = tuple(float(v) for v in xyxy)
        out.setdefault(logo, {}).setdefault(image, []).append((int(state), box))  # type: ignore[arg-type]
    return out


def read_queries(root: Path) -> dict[str, tuple[str, Box]]:
    """Each logo's first ``qset1`` query: ``{logo: (image, box)}``."""
    out: dict[str, tuple[str, Box]] = {}
    for line in (root / "qset1_internal.qry").read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) != 6:
            continue
        name, image, *xyxy = f
        stem = name.rstrip("0123456789")
        logo = QUERY_ALIASES.get(stem, stem)
        out.setdefault(logo, (image, tuple(float(v) for v in xyxy)))  # type: ignore[arg-type]
    return out


def normalised(box: Box, size: tuple[int, int]) -> Box:
    w, h = size
    x0, y0, x1, y1 = box
    return (max(0.0, x0 / w), max(0.0, y0 / h), min(1.0, x1 / w), min(1.0, y1 / h))


def main(argv: Optional[Sequence[str]] = None) -> int:
    from PIL import Image  # noqa: PLC0415

    from vtscore.media.structural import DEFAULT_MAX_FEATURES  # noqa: PLC0415
    from vtscore.training.structural_similarity import filter_features_to_box  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=BELGA)
    ap.add_argument("--budget", type=int, default=DEFAULT_MAX_FEATURES)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--classes", default="", help="comma-separated subset")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    instances = read_instances(args.root)
    queries = read_queries(args.root)
    classes = sorted(
        logo
        for logo, (_, _) in queries.items()
        if sum(any(s == 1 for s, _ in inst) for inst in instances.get(logo, {}).values()) >= MIN_POSITIVES
    )
    if args.classes:
        classes = [c for c in classes if c in set(args.classes.split(","))]
    print(f"=== {len(classes)} classes: {', '.join(classes)}", flush=True)

    images = sorted(p.name for p in (args.root / "images").glob("*.jpg"))
    t0 = time.time()
    vlads: dict[str, np.ndarray] = {}
    with get_context("fork").Pool(args.workers, initializer=esr._init_budget, initargs=(args.budget,)) as pool:
        for i, (image, feats, _n, vlad) in enumerate(
            pool.imap_unordered(esr._extract, [(im, str(args.root / "images" / im)) for im in images], chunksize=16)
        ):
            esr._FEATURES[image] = feats
            vlads[image] = vlad
            if (i + 1) % 1000 == 0:
                print(f"  extracted {i + 1}/{len(images)} in {time.time() - t0:.0f}s", flush=True)
    esr._init_budget(args.budget)
    sizes = {}

    def size_of(image: str) -> tuple[int, int]:
        if image not in sizes:
            with Image.open(args.root / "images" / image) as im:
                sizes[image] = im.size
        return sizes[image]

    query_vlad: dict[str, np.ndarray] = {}
    crops = {}
    for logo in classes:
        image, box = queries[logo]
        gray = esr._gray(str(args.root / "images" / image))
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        crops[logo] = esr._matcher().detect_and_describe(gray[y0:y1, x0:x1], max_features=args.budget)
        query_vlad[tm.slug(logo)] = esr._vlad(crops[logo])
    ids = sorted(vlads)
    np.savez(
        args.out / "vectors-0.npz",
        page_ids=np.array(ids),
        page_vlad=np.stack([vlads[p] for p in ids]).astype(np.float32),
        **{f"qvlad__{k}": v for k, v in query_vlad.items()},
    )

    for logo in classes:
        dest = args.out / f"{tm.slug(logo)}.npz"
        if dest.exists():
            continue
        t1 = time.time()
        q_image, _ = queries[logo]
        inst = instances[logo]
        positives = {im for im, rows in inst.items() if any(s == 1 for s, _ in rows)}
        ignored = {im for im in inst if im not in positives}
        pool_ids = [im for im in images if im != q_image and im not in ignored]
        template_ids = [f"query:{q_image}"]
        templates = [crops[logo].compact()]
        for im in sorted(positives - {q_image}):
            box = max((b for s, b in inst[im] if s == 1), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            template_ids.append(im)
            templates.append(filter_features_to_box(esr._FEATURES[im], normalised(box, size_of(im))))
        stats, inliers, tentative = tm.verify_all(templates, pool_ids, args.workers)
        np.savez_compressed(
            dest,
            template_ids=np.array(template_ids),
            template_sizes=np.array([t.count for t in templates]),
            pool_ids=np.array(pool_ids),
            stats=stats,
            inliers=inliers,
            tentative=tentative,
            positives=np.array([p in positives for p in pool_ids]),
            unboxed=np.array([], dtype=str),
        )
        print(
            f"  {logo}: {len(templates)} templates x {len(pool_ids)} images, "
            f"{len(positives - {q_image})} positive, in {time.time() - t1:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
