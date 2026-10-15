"""#4934: each cached page's keypoint rows ranked by SIFT response (strongest first).

The feature cache stores no response, and its rows are not in response order (OpenCV's nfeatures
cap only partitions them). Today's OpenCV re-detects the cached keypoints in the cached row order
(checked on 40 tier-s pages), so each page is re-detected (detect only, as SiftMatcher does) and
its responses are attached to the cached rows: by position when the rows match in order, else by
exact (x, y, size, angle) key. A row that cannot be matched ranks last.

    python strength_order.py <repo> <tier> <out.npz> [workers]
"""

from __future__ import annotations

import sys
import time
from multiprocessing import get_context

import numpy as np

repo, tier, out = sys.argv[1], sys.argv[2], sys.argv[3]
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
sys.path.insert(0, f"{repo}/scripts/experiments/fullmarks")
sys.path.insert(0, repo)
import cv2  # noqa: E402
import embed_corpus  # noqa: E402
import fullmarks_config as cfg  # noqa: E402
from eval_splg_rank import _gray  # noqa: E402
from vtscore.media.structural import DOCUMENT_MAX_FEATURES, SiftMatcher, cap_detect_resolution  # noqa: E402

CACHE = "/expscratch/sgreenberg/fullmarks/features"
pages = {p.page_id: p for p in embed_corpus.pages_for_tier(cfg.OUT, tier)}
ids = sorted(pages)
z = np.load(f"{CACHE}/features-{tier}.npz")
cached_ids = [str(p) for p in z["page_ids"]]
assert set(ids) <= set(cached_ids), "the tier's own cache file must cover it"
starts = np.append(z["starts"], len(z["page_ids"]) and z["keypoints"].shape[0])
KP = z["keypoints"]  # (N, 4) fp16, shared with the forked workers
ROW = {p: i for i, p in enumerate(cached_ids)}
MATCHER = SiftMatcher()


def one(pid: str) -> tuple[str, np.ndarray, int, bool]:
    i = ROW[pid]
    ck = KP[starts[i] : starts[i + 1]]
    gray = cap_detect_resolution(np.asarray(_gray(pages[pid].path)), MATCHER._max_detect_pixels)
    h, w = gray.shape
    kps = cv2.SIFT_create(nfeatures=DOCUMENT_MAX_FEATURES).detect(gray, None)
    fresh = (
        np.array(
            [(k.pt[0] / max(w, 1), k.pt[1] / max(h, 1), k.size, np.deg2rad(k.angle)) for k in kps], dtype=np.float32
        )
        .astype(np.float16)
        .reshape(-1, 4)
    )
    resp = np.array([k.response for k in kps], dtype=np.float64)
    in_order = fresh.shape == ck.shape and np.array_equal(fresh.view(np.uint16), np.asarray(ck).view(np.uint16))
    row_resp = np.full(ck.shape[0], -np.inf)
    if in_order:
        row_resp[:] = resp
        matched = ck.shape[0]
    else:
        slots: dict[bytes, list[int]] = {}
        for j, r in enumerate(np.asarray(ck)):
            slots.setdefault(r.tobytes(), []).append(j)
        matched = 0
        for j, r in enumerate(fresh):
            lst = slots.get(r.tobytes())
            if lst:
                row_resp[lst.pop(0)] = resp[j]
                matched += 1
    order = np.argsort(-row_resp, kind="stable").astype(np.int16)
    return pid, order, matched, in_order


if __name__ == "__main__":
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        res = pool.map(one, ids, chunksize=16)
    got = {pid: (o, m, ok) for pid, o, m, ok in res}
    counts = np.array([starts[ROW[p] + 1] - starts[ROW[p]] for p in ids], dtype=np.int64)
    np.savez(
        out,
        page_ids=np.array(ids),
        starts=np.concatenate([[0], np.cumsum(counts)[:-1]]),
        order=np.concatenate([got[p][0] for p in ids]),
    )
    matched = np.array([got[p][1] for p in ids]) / np.maximum(counts, 1)
    in_order = np.mean([got[p][2] for p in ids])
    print(
        f"tier {tier}: {len(ids)} pages in {time.time() - t0:.0f}s; rows in cached order {in_order:.3f} of pages; "
        f"rows matched median {np.median(matched):.4f}, min {matched.min():.4f}, pages < 99% matched {(matched < 0.99).sum()}; "
        f"cv2 {cv2.__version__}; wrote {out}"
    )
