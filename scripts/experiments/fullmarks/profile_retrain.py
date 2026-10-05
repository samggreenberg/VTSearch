"""Where a document retrain's time goes after a Good click (#4469).

Replays ``sota_documents.py``'s closed loop for a few classes and times every retrain
(``maybe_structural_rerank``), split into phases by wrapping the functions a Good step calls:

* ``stage1``: ``vote_queries`` + ``tiled_stage1`` (the tiled VLAD scoring);
* ``ratio_test``: ``ratio_test_matches`` (descriptor prep, the batched distance, top-2), with the
  GPU synchronised before the clock stops;
* ``ransac``: ``SiftMatcher._fit_similarity``, per pair, on the CPU;
* ``other``: the rest of the retrain.

It also writes a ``cProfile`` of the Good steps, the top functions by cumulative time. Tier
``s`` is enough: the shortlist is 2,000 pages there as on tier ``m``, and Stage 1 is
milliseconds on the GPU at either size.

    python profile_retrain.py --tier s --classes a,b,c --max-v 12 \\
        --matrix <votes-4162>/matrix-s --feature-cache /expscratch/$USER/fullmarks/features --out <dir>
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import io
import json
import pstats
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import embed_corpus  # noqa: E402
import fullmarks_config as cfg  # noqa: E402
import template_matrix as tm  # noqa: E402
from app_replay_tiled import _extract  # noqa: E402
from sota_documents import _cached_tiles, in_test_half, load_or_extract  # noqa: E402

PHASES = ("stage1", "ratio_test", "ransac")


def _sync() -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


class Timers:
    """Accumulates wall time per phase while a retrain runs."""

    def __init__(self) -> None:
        self.acc: dict[str, float] = defaultdict(float)

    def wrap(self, owner: Any, name: str, phase: str, sync: bool = False) -> None:
        fn = getattr(owner, name)

        def timed(*a: Any, **k: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                if sync:
                    _sync()
                self.acc[phase] += time.perf_counter() - t0

        setattr(owner, name, timed)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--max-v", type=int, default=12)
    ap.add_argument("--feature-cache", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import vtscore.media.structural as structural  # noqa: PLC0415
    import vtscore.training.structural_stage1 as s1  # noqa: PLC0415
    from vtscore.media.structural_tiles import load_tile_projection, tile_vectors  # noqa: PLC0415
    from vtscore.state.core import DetectorContext  # noqa: PLC0415
    from vtscore.training import structural_similarity as ss  # noqa: PLC0415

    timers = Timers()
    timers.wrap(s1, "vote_queries", "stage1")
    timers.wrap(s1, "tiled_stage1", "stage1", sync=True)
    # Stacking the tile matrix happens once per page set in the app (a dataset load), but once per
    # class here, since each class has its own pool: timed apart so it is not read as per-click cost.
    timers.wrap(s1, "_tile_matrix", "matrix")
    # structural_similarity imported these names from structural_stage1; time those bindings too.
    for name in ("vote_queries", "tiled_stage1"):
        if hasattr(ss, name):
            timers.wrap(ss, name, "stage1", sync=True)
    timers.wrap(structural, "ratio_test_matches", "ratio_test", sync=True)
    timers.wrap(structural.SiftMatcher, "_fit_similarity", "ransac")

    args.out.mkdir(parents=True, exist_ok=True)
    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(args.corpus, args.tier)}
    ids = sorted(pages)
    feats = load_or_extract(ids, {p: pages[p].path for p in ids}, args.feature_cache, args.tier, args.workers)
    import vtscore.media.structural_tiles as st  # noqa: PLC0415

    tiles = _cached_tiles(ids, args.feature_cache, args.tier, st) if args.feature_cache else {}
    if len(tiles) < len(ids):  # no cache covering the tier: tile here (slow at 50k pages)
        projection = load_tile_projection()
        tiles = {p: tile_vectors(feats[p], projection) for p in ids}
    snap_all = {p: {"embedder": "sift_vlad_doc", "local_features": feats[p], "tile_vectors": tiles[p]} for p in ids}
    host = {"node": socket.gethostname()}
    try:
        import torch  # noqa: PLC0415

        host["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:  # noqa: BLE001
        host["gpu"] = "none"
    print(json.dumps(host), flush=True)

    rows: list[dict[str, Any]] = []
    prof = cProfile.Profile()
    for cid in [c for c in args.classes.split(",") if c]:
        z = np.load(args.matrix / f"{cid.replace('/', '__', 1)}.npz")
        pool_ids = [str(p) for p in z["pool_ids"]]
        positive = z["positives"].astype(bool)
        test = np.array([in_test_half(p) for p in pool_ids])
        snap = {p: snap_all[p] for p in pool_ids}
        crop = _extract(classes[cid]["query_crop"])
        placeholder = [{"id": p, "score": 0.0} for p in pool_ids]
        det_ctx = DetectorContext(detector_id=f"prof-{cid}", media_type="image")
        goods: dict[str, None] = {}
        bads: dict[str, None] = {}
        boxes: dict[str, Any] = {}
        last = None
        col = {p: i for i, p in enumerate(pool_ids)}
        for v in range(args.max_v + 1):
            timers.acc.clear()
            profiling = last == "good"
            if profiling:
                prof.enable()
            t0 = time.perf_counter()
            if goods:
                ranked, line = ss.maybe_structural_rerank(placeholder, 0.5, snap, goods, boxes, det_ctx, bad_votes=bads)
            else:
                ranked, line = ss.maybe_structural_rerank_example(placeholder, 0.5, snap, crop)
            _sync()
            total = time.perf_counter() - t0
            if profiling:
                prof.disable()
            phases = {p: timers.acc.get(p, 0.0) for p in PHASES}
            rows.append(
                {
                    "class_id": cid,
                    "v": v,
                    "after": last or "start",
                    "goods": len(goods),
                    "total_s": round(total, 4),
                    # The once-per-page-set matrix build, inside stage1; per_click_s leaves it out.
                    "matrix_s": round(timers.acc.get("matrix", 0.0), 4),
                    "per_click_s": round(total - timers.acc.get("matrix", 0.0), 4),
                    **{f"{p}_s": round(t, 4) for p, t in phases.items()},
                    "other_s": round(total - sum(phases.values()), 4),
                    # The ranking, scores and line, to show a speed change leaves the results alone.
                    "result_sha1": hashlib.sha1(
                        json.dumps([[e["id"], e["score"]] for e in ranked] + [line]).encode()
                    ).hexdigest()[:16],
                }
            )
            labelled = set(goods) | set(bads)
            nxt = next((e["id"] for e in ranked if not test[col[e["id"]]] and e["id"] not in labelled), None)
            if nxt is None:
                break
            if positive[col[nxt]]:
                goods[nxt] = None
                box = tm.largest_box(pages[nxt], cid)  # the mark's box, as sota_documents.py gives a Good
                if box is not None:
                    boxes[nxt] = box
                last = "good"
            else:
                bads[nxt] = None
                last = "bad"
        print(f"  {cid}: {len(goods)} Goods in {args.max_v} clicks", flush=True)

    with (args.out / "steps.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"host: {host['node']}, GPU: {host['gpu']}", ""]
    for after in ("start", "good", "bad"):
        sel = [r for r in rows if r["after"] == after]
        if not sel:
            continue
        keys = ("total", "per_click", "matrix", *PHASES, "other")
        cells = " | ".join(f"{k}: {np.median([r[f'{k}_s'] for r in sel]):.2f}" for k in keys)
        lines.append(f"after {after} ({len(sel)} steps), medians (s): {cells}")
        lines.append(f"  per-click p90: {np.percentile([r['per_click_s'] for r in sel], 90):.2f} s")
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(30)
    (args.out / "profile_good_steps.txt").write_text(buf.getvalue(), encoding="utf-8")
    text = "\n".join(lines) + "\n"
    (args.out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
