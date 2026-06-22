"""Build the shipped VLAD visual vocabulary for the structural embedder.

The structural (SIFT/VLAD) embedder aggregates each image's SIFT descriptors
into a fixed-D VLAD vector against a **fixed, shipped codebook** (a pre-computed
k-means vocabulary).  Per the design (``docs/plans/structural-embedder.md``) the
codebook is a *code asset*, not per-dataset state - it ships with the repo and is
loaded like model weights.

Two modes:

* **No ``--images`` (default).** Generate a deterministic, seeded placeholder
  codebook so the pipeline is end-to-end functional and reproducible without a
  corpus download.  This is what is checked in for v1; the spike replaces it
  with a real corpus-fit codebook once the corpus + size are pinned.

* **``--images DIR``.** Extract SIFT descriptors from a corpus of images, sample
  them, fit k-means, and write the resulting centroids.  This is the real build
  the spike runs.

Usage::

    python scripts/build_vlad_codebook.py                      # seeded placeholder
    python scripts/build_vlad_codebook.py --images /data/corpus --centroids 64

Output: ``vtscore/media/assets/vlad_codebook_v1.npy`` (shape ``(K, 128)`` fp32).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vtscore.media.structural import DEFAULT_VLAD_CENTROIDS, SIFT_DESCRIPTOR_DIM, rootsift

_ASSET_PATH = Path(__file__).resolve().parent.parent / "vtscore" / "media" / "assets" / "vlad_codebook_v1.npy"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _seeded_codebook(k: int, seed: int) -> np.ndarray:
    """Deterministic placeholder codebook in the SIFT descriptor range.

    Centroids are drawn to resemble (non-negative, ``[0, 255]``) SIFT
    descriptors so VLAD assignment is meaningful, then rootSIFT-normalised into
    the same space :func:`vtscore.media.structural.aggregate_vlad` uses.  Random
    centroids give a valid (if suboptimal) vocabulary - good enough to prove the
    plumbing; the corpus fit below replaces them for production.
    """
    rng = np.random.default_rng(seed)
    raw = rng.random((k, SIFT_DESCRIPTOR_DIM), dtype=np.float64).astype(np.float32) * 255.0
    return rootsift(raw).astype(np.float32)


def _collect_descriptors(image_dir: Path, max_per_image: int, max_total: int) -> np.ndarray:
    """Extract SIFT descriptors from every image under *image_dir*."""
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    sift = cv2.SIFT_create(nfeatures=max_per_image)
    chunks: list[np.ndarray] = []
    total = 0
    files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    for path in files:
        if total >= max_total:
            break
        try:
            with Image.open(path) as img:
                gray = np.asarray(img.convert("L"), dtype=np.uint8)
        except Exception:  # noqa: BLE001 - skip unreadable files
            continue
        _, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            continue
        chunks.append(np.asarray(desc, dtype=np.float32))
        total += len(desc)
    if not chunks:
        raise SystemExit(f"No SIFT descriptors extracted from {image_dir}")
    alld = np.concatenate(chunks, axis=0)
    if alld.shape[0] > max_total:
        rng = np.random.default_rng(0)
        idx = rng.choice(alld.shape[0], size=max_total, replace=False)
        alld = alld[idx]
    return alld


def _fit_codebook(descriptors: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fit a k-means codebook on rootSIFT-normalised descriptors."""
    from sklearn.cluster import KMeans  # noqa: PLC0415

    feats = rootsift(descriptors)
    km = KMeans(n_clusters=k, random_state=seed, n_init=4)
    km.fit(feats)
    return km.cluster_centers_.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=None, help="Corpus directory to fit a real codebook from.")
    parser.add_argument("--centroids", type=int, default=DEFAULT_VLAD_CENTROIDS, help="Codebook size K.")
    parser.add_argument("--seed", type=int, default=20260619, help="Determinism seed.")
    parser.add_argument("--max-per-image", type=int, default=512, help="SIFT keypoint cap per corpus image.")
    parser.add_argument("--max-total", type=int, default=1_000_000, help="Max descriptors sampled for k-means.")
    parser.add_argument("--out", type=Path, default=_ASSET_PATH, help="Output .npy path.")
    args = parser.parse_args()

    if args.images is not None:
        print(f"Extracting SIFT descriptors from {args.images} ...")
        descriptors = _collect_descriptors(args.images, args.max_per_image, args.max_total)
        print(f"Fitting {args.centroids}-centroid k-means on {descriptors.shape[0]} descriptors ...")
        codebook = _fit_codebook(descriptors, args.centroids, args.seed)
    else:
        print(f"Generating seeded placeholder codebook (K={args.centroids}, seed={args.seed}) ...")
        codebook = _seeded_codebook(args.centroids, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, codebook)
    print(f"Wrote {codebook.shape} {codebook.dtype} codebook -> {args.out}")


if __name__ == "__main__":
    main()
