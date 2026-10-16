#!/usr/bin/env python
"""What ``image2face``'s ``sideways`` search costs per photo (#4788).

Times MTCNN on a fixed sample of FHIBE's stored copies: the plain pass, and the plain pass plus the
two quarter-turned copies ``sideways`` adds, through the converter's own ``_detect_faces``. Run it
once with the GPU hidden (``CUDA_VISIBLE_DEVICES=``, the app on a CPU host) and once on a GPU.

    python sideways_cost_4788.py --long-side 1024 --n 200 [--threads 4]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pile"))

import pile_config as pc  # noqa: E402
from pilebuild.loaders import fhibe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from vtscore.converters.image2face import CONVERTER  # noqa: PLC0415

    torch.set_num_threads(args.threads)
    photos = fhibe.read_photos(fhibe.data_dir())[:: max(1, len(fhibe.read_photos(fhibe.data_dir())) // args.n)][
        : args.n
    ]
    copies = pc.fhibe_derived_dir(args.long_side)
    images = []
    for p in photos:
        with Image.open(io.BytesIO((copies / f"{p.uid}.jpg").read_bytes())) as im:
            images.append(im.convert("RGB"))
    params = CONVERTER._resolve_params(None)
    CONVERTER._detect_faces(images[0], params.threshold, params.padding, params.min_size)  # warm-up
    device = "cuda" if torch.cuda.is_available() else f"cpu ({args.threads} threads)"
    for sideways in (False, True):
        t0 = time.perf_counter()
        for im in images:
            CONVERTER._detect_faces(im, params.threshold, params.padding, params.min_size, sideways)
        dt = (time.perf_counter() - t0) / len(images)
        print(f"{args.long_side} px, {device}, sideways={sideways}: {dt * 1000:.0f} ms a photo over {len(images)}")


if __name__ == "__main__":
    main()
