"""Where do the positives sit in tiled Stage 1's ranking as the collection grows? (#4493)

Stage 2 verifies only Stage 1's top ``TILED_TOP_K`` (2,000) pages; a positive below that scores 0
and no click reaches it. At 200k pages 9.7-13% of test positives sat there at click 50 (#4488).
Whether a larger or scaling K would recover them depends on where they rank: just past 2,000, or
far down.

Replays the recorded sessions of ``sota_documents.py`` runs (``clicks.csv``): at each state v in
``--states``, the Goods of clicks 1..v (boxed as the harness boxes them), or the class's query crop
at v = 0, give the queries; the app's ``tiled_stage1`` ranks the class pool; every test-half
positive's Stage-1 rank is recorded. The summary is recall within the top K, positive-weighted over
classes and replicates, per state.

    python stage1_reach_documents.py --tier l --matrix <pools> --feature-cache <dir> \\
        --runs <rep1 dir>[+<part>],<rep2 dir>[+<part>] --out <dir>

The second replicate is the swapped-halves run, as in the grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import embed_corpus  # noqa: E402
import fullmarks_config as cfg  # noqa: E402
import template_matrix as tm  # noqa: E402
from app_replay_tiled import _extract  # noqa: E402
from sota_documents import _cached_tiles, in_test_half, load_or_extract  # noqa: E402

from vtscore.training import structural_stage1 as s1  # noqa: E402

KS = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000)


def session_goods(parts: Sequence[Path]) -> dict[str, list[tuple[int, str]]]:
    """Each class's Good votes as ``(click, page)`` in click order, from a replicate's (possibly resumed) run dirs."""
    goods: dict[str, list[tuple[int, str]]] = {}
    for part in parts:
        with (part / "clicks.csv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["label"] == "good":
                    goods.setdefault(r["class_id"], []).append((int(r["click"]), r["page_id"]))
    return {c: sorted(v) for c, v in goods.items()}


def session_classes(parts: Sequence[Path]) -> set[str]:
    out: set[str] = set()
    for part in parts:
        with (part / "clicks.csv").open(encoding="utf-8") as fh:
            out |= {r["class_id"] for r in csv.DictReader(fh)}
    return out


def summarise(rows: Sequence[dict[str, Any]], states: Sequence[int]) -> list[str]:
    lines = [
        "| click | test positives | " + " | ".join(f"top {k:,}" for k in KS) + " | beyond 64,000 |",
        "|---:|---:|" + "---:|" * (len(KS) + 1),
    ]
    for v in states:
        ranks = np.array([r["rank"] for r in rows if r["v"] == v])
        if not len(ranks):
            continue
        cells = [f"{np.mean(ranks < k):.1%}" for k in KS]
        lines.append(f"| {v} | {len(ranks)} | " + " | ".join(cells) + f" | {np.mean(ranks >= KS[-1]):.1%} |")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--feature-cache", type=Path, required=True)
    ap.add_argument("--runs", required=True, help="<rep1 dir>[+<part>],<rep2 dir>[+<part>]; rep2 is swapped")
    ap.add_argument("--states", default="0,10,50")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import vtscore.media.structural_tiles as st  # noqa: PLC0415

    states = [int(x) for x in args.states.split(",")]
    reps = [[Path(p) for p in rep.split("+")] for rep in args.runs.split(",")]
    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(args.corpus, args.tier)}
    ids = sorted(pages)
    t0 = time.time()
    tiles = _cached_tiles(ids, args.feature_cache, args.tier, st)
    if len(tiles) < len(ids):
        raise SystemExit(f"no tile cache covering tier {args.tier} under {args.feature_cache}")
    goods_by_rep = [session_goods(parts) for parts in reps]
    need = sorted({p for g in goods_by_rep for votes in g.values() for click, p in votes if click <= max(states)})
    feats = load_or_extract(need, {p: pages[p].path for p in need}, args.feature_cache, args.tier, args.workers)
    print(f"tier {args.tier}: tiles for {len(tiles)} pages, features for {len(feats)} Goods in {time.time() - t0:.0f}s")

    rows: list[dict[str, Any]] = []
    for r, parts in enumerate(reps):
        swap = r == 1
        goods_all = goods_by_rep[r]
        for cid in sorted(session_classes(parts)):
            z = np.load(args.matrix / f"{cid.replace('/', '__', 1)}.npz")
            pool = [str(p) for p in z["pool_ids"]]
            positive = z["positives"].astype(bool)
            test = np.array([in_test_half(p) != swap for p in pool])
            targets = [p for p, y, t in zip(pool, positive, test) if y and t]
            snap: dict[str, dict] = {p: {"embedder": "sift_vlad_doc", "tile_vectors": tiles[p]} for p in pool}
            crop = _extract(classes[cid]["query_crop"])
            for v in states:
                t1 = time.time()
                if v == 0:
                    queries = s1.example_queries([crop])
                else:
                    gs = [p for click, p in goods_all.get(cid, []) if click <= v]  # the Goods of clicks 1..v
                    for p in gs:
                        snap[p] = dict(snap[p], local_features=feats[p])
                    boxes = {p: b for p in gs if (b := tm.largest_box(pages[p], cid)) is not None}
                    queries = s1.vote_queries({p: None for p in gs}, snap, boxes) if gs else None
                if queries is None:
                    continue
                order = [e["id"] for e in s1.tiled_stage1(snap, queries)]
                rank = {p: i for i, p in enumerate(order)}
                for p in targets:
                    rows.append(
                        {"rep": r + 1, "class_id": cid, "v": v, "page_id": p, "rank": rank[p], "pool": len(pool)}
                    )
                print(f"  rep{r + 1} {cid} v{v}: {len(targets)} test positives, {time.time() - t1:.1f}s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "ranks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    text = "\n".join([f"tier {args.tier}: test positives within Stage 1's top K", "", *summarise(rows, states)]) + "\n"
    (args.out / "reach.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
