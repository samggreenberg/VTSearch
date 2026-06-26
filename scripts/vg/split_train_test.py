#!/usr/bin/env python3
"""Split a flat directory of images into train/ and test/ subdirectories.

Shuffles the images (seeded, reproducible), splits by ``--train-frac``, and
copies (or moves) each into ``<out-dir>/train`` and ``<out-dir>/test``.

Usage::

    python scripts/vg/split_train_test.py data/vg/vg_hat_100_max
    python scripts/vg/split_train_test.py data/vg/vg_hat_100_max \\
        --out-dir data/vg/hat_split --train-frac 0.8 --seed 0
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="directory of images to split")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output root holding train/ and test/ (default: <input_dir>_split)",
    )
    ap.add_argument("--train-frac", type=float, default=0.8, help="fraction of images for train (default: 0.8)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the shuffle (default: 0)")
    ap.add_argument("--move", action="store_true", help="move files instead of copying")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"not a directory: {args.input_dir}")
    if not 0.0 < args.train_frac < 1.0:
        raise SystemExit(f"--train-frac must be in (0, 1), got {args.train_frac}")

    images = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"no images found in {args.input_dir}")

    random.Random(args.seed).shuffle(images)
    n_train = round(len(images) * args.train_frac)
    splits = {"train": images[:n_train], "test": images[n_train:]}

    out_dir = args.out_dir or args.input_dir.parent / f"{args.input_dir.name}_split"
    op = shutil.move if args.move else shutil.copy2
    verb = "moved" if args.move else "copied"

    for split, files in splits.items():
        dest = out_dir / split
        dest.mkdir(parents=True, exist_ok=True)
        for src in files:
            op(str(src), str(dest / src.name))
        print(f"  {split}: {len(files)} images -> {dest}", flush=True)

    print(f"\n{verb} {len(images)} images ({n_train} train / {len(images) - n_train} test) into {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
