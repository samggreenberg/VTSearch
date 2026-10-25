"""#4711: what a raw-patch tree would cost the app against max_patch, by array shape.

Ingest is the per-image tree build. A retrain is scoring N images: the chunked fp16 -> fp32 forward, sigmoid and
segment max, through the eval styles' own helpers. Storage is rows per image x 768 x fp16. Timings depend on shapes,
not values, so the grids are synthetic (a low-frequency field plus noise, so merges look like an image's).

    python region_styles_4711_bench.py THREADS     # gridenv shell on a compute node, ~48 GB
"""

import sys
import time

import numpy as np
import torch

from vtscore.eval.patch_styles import _forward_sigmoid_chunked, _segment_max, build_patch_hac_tree

D = 768
CORPUS = 24_507  # coco_better as #5191 rebuilt it


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _grid(rng: np.random.Generator) -> np.ndarray:
    base = rng.standard_normal((4, D))
    weights = rng.random((14, 14, 4))
    return _unit(weights @ base + 0.5 * rng.standard_normal((14, 14, D))).astype(np.float32)


def main() -> None:
    threads = int(sys.argv[1])
    torch.set_num_threads(threads)
    rng = np.random.default_rng(0)
    grids = [_grid(rng) for _ in range(60)]
    cls = _unit(rng.standard_normal(D)).astype(np.float32)
    for name, kw in [("hac", {}), ("pca_hac", {"pca_dims": 32})]:
        build_patch_hac_tree(grids[0], cls, **kw)
        start = time.perf_counter()
        nodes = [len(build_patch_hac_tree(g, cls, **kw)) for g in grids[1:]]
        ms = (time.perf_counter() - start) / len(nodes) * 1e3
        print(
            f"[{threads}t] ingest build {name}: {ms:.0f} ms/image, {nodes[0]} nodes -> "
            f"{ms * CORPUS / 6e4:.1f} min for {CORPUS:,} images, {ms * 1e5 / 3.6e6:.1f} h per 100k",
            flush=True,
        )
    model = torch.nn.Sequential(torch.nn.Linear(D, 1)).eval()
    for n in (11_600, CORPUS):
        best = {}
        for rows in (197, 391):
            matrix = (rng.standard_normal((n * rows, D), dtype=np.float32) * 0.036).astype(np.float16)
            seg = np.arange(n, dtype=np.int64) * rows
            _segment_max(_forward_sigmoid_chunked(model, matrix[: 50 * rows]), seg[:50])
            times = []
            for _ in range(3):
                start = time.perf_counter()
                _segment_max(_forward_sigmoid_chunked(model, matrix), seg)
                times.append(time.perf_counter() - start)
            best[rows] = min(times)
            del matrix
            print(f"[{threads}t] score N={n} rows={rows}: {best[rows]:.2f} s ({n * rows * D * 2 / 1e9:.1f} GB fp16)")
        print(f"[{threads}t] score N={n} ratio 391/197 = {best[391] / best[197]:.2f}x", flush=True)
    print(f"storage per image: 197 rows {197 * D * 2 / 1e6:.2f} MB vs 391 rows {391 * D * 2 / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
