"""What a detection cap costs per page: time, keypoints and peak RSS (#3908).

``eval_sift_rank.py`` answers whether a cap changes *retrieval*; this answers
what each cap costs to run, which is the other half of pinning the value.  Cost
is measured per cap in its own child process, because peak RSS is a
high-water mark that never comes back down within a process -- measuring several
caps in one would report the largest for all of them.  The child is forked, so
it inherits the parent's high-water mark; the parent's own RSS is recorded and
subtracted, and both numbers are reported.

FullMarks is the corpus that makes this worth measuring: its pages run from
0.13 MP to 63 MP (median 2.3), so the shipped 2 MP cap binds on 87% of tier `s`
and the tail is 30x the median.

    python bench_detect_cap.py --out bench.json [--sample 150] [--budget 8192]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Caps to price, in pixels; 0 means detect at native size.
CAPS: tuple[int, ...] = (0, 4_000_000, 2_000_000, 1_000_000, 500_000)
#: Pages whose size puts them in the tail, reported individually.
N_LARGEST = 10


def cap_label(cap: int) -> str:
    """``0`` is the uncapped arm; everything else reads in megapixels."""
    return "uncapped" if cap <= 0 else f"{cap / 1_000_000:g}MP"


def stratified_sample(sizes: Sequence[tuple[float, str]], n: int) -> list[str]:
    """*n* page ids spread evenly over the megapixel distribution.

    A uniform random sample of FullMarks is a sample of 2 MP pages; the cost
    question is about the spread, so walk the sorted order at a fixed stride and
    keep the tail (which a stride can skip) by construction.
    """
    ordered = sorted(sizes)
    if n >= len(ordered):
        return [pid for _, pid in ordered]
    keep = {len(ordered) - 1 - i for i in range(min(N_LARGEST, len(ordered)))}
    stride = len(ordered) / float(max(1, n - len(keep)))
    keep |= {min(len(ordered) - 1, int(i * stride)) for i in range(n - len(keep))}
    return [ordered[i][1] for i in sorted(keep)]


def _measure(args: tuple[int, int, list[tuple[str, str, float]]]) -> dict[str, Any]:
    cap, budget, items = args
    os.environ["VTSEARCH_MAX_STRUCTURAL_DETECT_PIXELS"] = str(cap)
    from vtscore.media.structural import SiftMatcher  # noqa: PLC0415

    from eval_splg_rank import _gray  # noqa: PLC0415

    matcher = SiftMatcher(max_detect_pixels=cap)
    rows = []
    for page_id, path, mp in items:
        gray = _gray(path)
        t0 = time.perf_counter()
        feats = matcher.detect_and_describe(gray, max_features=budget)
        rows.append(
            {
                "page_id": page_id,
                "mp": round(mp, 2),
                "detect_ms": round((time.perf_counter() - t0) * 1000, 1),
                "keypoints": int(feats.count),
            }
        )
    # ru_maxrss is KiB on Linux.  A forked child starts at the parent's
    # high-water mark, so the caller subtracts its own to get the marginal cost.
    return {
        "cap": cap,
        "label": cap_label(cap),
        "budget": budget,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "rows": rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import fullmarks_config as cfg  # noqa: PLC0415
    import embed_corpus  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--caps", default=" ".join(str(c) for c in CAPS), help="space-separated pixel caps; 0 = native")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(args.corpus, args.tier)}
    sizes = [(p.width * p.height / 1e6, pid) for pid, p in pages.items()]
    chosen = stratified_sample(sizes, args.sample)
    items = [(pid, pages[pid].path, pages[pid].width * pages[pid].height / 1e6) for pid in chosen]
    caps = [int(c) for c in args.caps.split()]
    print(f"=== {len(items)} pages, budget {args.budget}, caps {[cap_label(c) for c in caps]}", flush=True)

    results = []
    # Fork, not spawn: a spawned child re-imports from a bare interpreter and
    # loses this script's sys.path.  One child per cap still isolates the
    # high-water mark from the previous cap's.
    ctx = get_context("fork")
    base_rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    for cap in caps:
        t0 = time.time()
        with ctx.Pool(1) as pool:
            res = pool.map(_measure, [(cap, args.budget, items)])[0]
        res["wall_s"] = round(time.time() - t0, 1)
        res["parent_rss_mb"] = base_rss
        res["peak_rss_delta_mb"] = round(res["peak_rss_mb"] - base_rss, 1)
        results.append(res)
        ms = sorted(r["detect_ms"] for r in res["rows"])
        kp = sorted(r["keypoints"] for r in res["rows"])
        mid = len(ms) // 2
        print(
            f"  {res['label']:>8}: detect median {ms[mid]:7.1f} ms  p90 {ms[int(0.9 * len(ms))]:7.1f} ms  "
            f"keypoints median {kp[mid]:5d}  peak RSS {res['peak_rss_mb']:7.1f} MB "
            f"(+{res['peak_rss_delta_mb']:.1f} over parent)",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"budget": args.budget, "parent_rss_mb": base_rss, "arms": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
