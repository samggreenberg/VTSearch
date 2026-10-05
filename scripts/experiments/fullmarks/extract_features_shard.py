"""Extract one shard of a tier's missing document features, for an array job (#4488).

Pages already held by any ``features-*.npz`` under the cache are skipped. The rest of the tier is
split into ``--shards`` and this job extracts shard ``--shard``, writing
``features-<tier>-shard-<i>of<n>.npz``. ``sota_documents.py`` then reads all the files together.

    python extract_features_shard.py --tier l --shard 0 --shards 30 --feature-cache <dir> [--workers 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import embed_corpus  # noqa: E402
import fullmarks_config as cfg  # noqa: E402
from app_replay_tiled import _extract  # noqa: E402
from sota_documents import save_features  # noqa: E402


def held_ids(cache: Path) -> set[str]:
    import numpy as np  # noqa: PLC0415

    held: set[str] = set()
    for f in cache.glob("features-*.npz"):
        # Shard files are written while the array runs; counting them would change every task's split.
        if ".tmp" in f.name or "-shard-" in f.name:
            continue
        with np.load(f, allow_pickle=False) as z:
            held |= {str(p) for p in z["page_ids"]}
    return held


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--feature-cache", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args(argv)
    out = args.feature_cache / f"features-{args.tier}-shard-{args.shard}of{args.shards}.npz"
    if out.exists():
        print(f"{out.name} exists", flush=True)
        return 0
    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(args.corpus, args.tier)}
    held = held_ids(args.feature_cache)
    todo = sorted(p for p in pages if p not in held)
    mine = todo[args.shard :: args.shards]
    t0 = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        feats = dict(zip(mine, pool.map(_extract, [pages[p].path for p in mine], chunksize=8)))
    save_features(feats, mine, out)
    print(
        f"shard {args.shard}/{args.shards}: {len(mine)} of {len(todo)} missing pages in {time.time() - t0:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
