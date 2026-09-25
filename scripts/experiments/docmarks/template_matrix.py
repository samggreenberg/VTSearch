"""Every template of a class verified against every pool page, once (#4162).

The vote-curve arms (``vote_curve.py``) differ only in which templates they
combine and how, so the expensive part -- SIFT matching plus RANSAC -- is done
here once and replayed there.  A class's templates are its query crop and, for
every positive page, that page's own features restricted to its DocMarks box:
:func:`~vtscore.training.structural_similarity.filter_features_to_box`, the
production RegionYes template.  A page carrying the class more than once gives
its largest box.

For each class, ``<out>/<class>.npz`` holds:

``template_ids``   the query page id first, then the positive pages
``pool_ids``       the ``own_verified`` pool (query page excluded), sorted
``stats``          ``float16 [T, N, 9]``: :func:`match_stats_to_features` of
                   template *t* verified against pool page *n*
``inliers``        ``int16 [T, N]``: inlier count when the model is sane, else 0
                   (what ``eval_sift_rank.py`` ranks by)
``tentative``      ``int32 [T, N]``: ratio-test matches, its tie-break
``positives``      bool ``[N]``

and ``<out>/vectors-<shard>.npz`` the page VLADs and the classes' query
VLAD / SigLIP vectors the SVM arms need.

    python template_matrix.py --tier s --out <dir> [--shard 0/4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import eval_retrieval as ev  # noqa: E402
import eval_sift_rank as esr  # noqa: E402
from eval_splg_rank import _gray  # noqa: E402

#: Pages verified per worker task; each task matches every template against them.
CHUNK = 32

_TEMPLATES: list[Any] = []


def slug(cid: str) -> str:
    return cid.replace("/", "__")


def largest_box(page: Any, cid: str) -> Optional[tuple[float, float, float, float]]:
    """The class's largest mark on *page*, normalised ``(x0, y0, x1, y1)``."""
    marks = [m for m in page.marks if m.class_id == cid and m.box]
    if not marks:
        return None
    x, y, w, h = max(marks, key=lambda m: m.area()).box
    return (x / page.width, y / page.height, (x + w) / page.width, (y + h) / page.height)


def _verify_chunk(page_ids: list[str]) -> tuple[list[str], np.ndarray]:
    from vtscore.media.structural import match_stats_to_features  # noqa: PLC0415

    cands = [esr._FEATURES[p] for p in page_ids]
    out = np.zeros((len(_TEMPLATES), len(page_ids), 9), dtype=np.float32)
    for t, tpl in enumerate(_TEMPLATES):
        for n, stats in enumerate(esr._matcher().verify_many(tpl, cands)):
            out[t, n] = match_stats_to_features(stats)
    return page_ids, out


def _init_worker() -> None:
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)


def inliers_from_stats(stats: np.ndarray) -> np.ndarray:
    """``int16`` inlier counts, 0 where the model is not sane -- ``eval_sift_rank``'s ranking key."""
    counts = np.rint(np.expm1(stats[..., 0].astype(np.float32)))
    return np.where(stats[..., 8] > 0.5, counts, 0).astype(np.int16)


def verify_all(templates: list[Any], pool_ids: list[str], workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(stats float16 [T, N, 9], inliers int16 [T, N], tentative int32 [T, N])``."""
    global _TEMPLATES
    _TEMPLATES = templates
    chunks = [pool_ids[i : i + CHUNK] for i in range(0, len(pool_ids), CHUNK)]
    stats = np.zeros((len(templates), len(pool_ids), 9), dtype=np.float16)
    inliers = np.zeros((len(templates), len(pool_ids)), dtype=np.int16)
    tentative = np.zeros((len(templates), len(pool_ids)), dtype=np.int32)
    if not templates:
        return stats, inliers, tentative
    col = {p: i for i, p in enumerate(pool_ids)}
    # Fork after the templates are set, so workers inherit them and the page features.
    with get_context("fork").Pool(workers, initializer=_init_worker) as pool:
        for chunk_ids, block in pool.imap_unordered(_verify_chunk, chunks):
            cols = [col[p] for p in chunk_ids]
            stats[:, cols, :] = block
            # From the float32 block: log1p counts do not survive float16.
            inliers[:, cols] = inliers_from_stats(block)
            tentative[:, cols] = np.rint(np.expm1(block[..., 2]))
    return stats, inliers, tentative


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vtscore.media import get_embedder  # noqa: PLC0415
    from vtscore.training.structural_similarity import filter_features_to_box  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--shard", default="0/1", help="i/n: this job does classes i, i+n, ... in sorted order")
    ap.add_argument("--classes", default="", help="comma-separated subset (default: every roster class)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument(
        "--goods-in-top",
        type=int,
        default=0,
        help="only positives in the exemplar's top N become templates (the shared sequence to N votes)",
    )
    ap.add_argument("--no-vectors", action="store_true", help="skip the VLAD/SigLIP vectors the SVM arms need")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        ap.error("--out is inside the corpus; an eval never writes there")
    args.out.mkdir(parents=True, exist_ok=True)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    classes = {
        cid: meta
        for cid, meta in json.loads((args.corpus / "classes.json").read_text(encoding="utf-8")).items()
        if meta.get("on_roster") and meta.get("query_crop")
    }
    wanted = sorted(c for c in args.classes.split(",") if c) or sorted(classes)
    mine = [cid for i, cid in enumerate(wanted) if i % shard_n == shard_i]
    mine = [cid for cid in mine if not (args.out / f"{slug(cid)}.npz").exists()]
    print(f"=== tier {args.tier}: shard {args.shard}, {len(mine)} class(es) to do", flush=True)
    if not mine and args.no_vectors:
        # A continuation job after a timeout: nothing left, so skip the extraction.
        return 0

    pages = embed_corpus.pages_for_tier(args.corpus, args.tier)
    page_by_id = {p.page_id: p for p in pages}
    pages_by_source: dict[str, list[str]] = {}
    industry_of: dict[str, Optional[str]] = {}
    for page in pages:
        pages_by_source.setdefault(page.source, []).append(page.page_id)
        industry_of[page.page_id] = page.meta.get("industry")

    t0 = time.time()
    vlads: dict[str, np.ndarray] = {}
    with get_context("fork").Pool(args.workers, initializer=esr._init_budget, initargs=(args.budget,)) as pool:
        for i, (page_id, feats, _n, vlad) in enumerate(
            pool.imap_unordered(esr._extract, [(p.page_id, p.path) for p in pages], chunksize=8)
        ):
            esr._FEATURES[page_id] = feats
            vlads[page_id] = vlad
            if (i + 1) % 1000 == 0:
                print(f"  extracted {i + 1}/{len(pages)} in {time.time() - t0:.0f}s", flush=True)

    if args.no_vectors:
        mine_vectors: list[str] = []
    else:
        mine_vectors = mine
    siglip_emb = get_embedder("siglip") if mine_vectors else None
    esr._init_budget(args.budget)
    query_vlad, query_siglip = {}, {}
    if siglip_emb is not None:
        siglip_emb.load_models()
    for cid in mine_vectors:
        crop = esr._matcher().detect_and_describe(_gray(classes[cid]["query_crop"]), max_features=args.budget)
        query_vlad[slug(cid)] = esr._vlad(crop)
        vec, _ = ev.embed_query(siglip_emb, Path(classes[cid]["query_crop"]))
        query_siglip[slug(cid)] = np.asarray(vec, dtype=np.float32)
    ids = sorted(vlads)
    np.savez(  # an empty vectors file when --no-vectors, so a rerun still finds one per shard
        args.out / f"vectors-{shard_i}.npz",
        page_ids=np.array(ids if mine_vectors else []),
        page_vlad=np.stack([vlads[p] for p in ids]).astype(np.float32) if mine_vectors else np.zeros((0, 0)),
        **{f"qvlad__{k}": v for k, v in query_vlad.items()},
        **{f"qsiglip__{k}": v for k, v in query_siglip.items()},
    )

    for cid in mine:
        t1 = time.time()
        meta = classes[cid]
        pools = ev.class_pools(meta, pages_by_source, industry_of)
        pool_ids = sorted(pools[ev.HEADLINE_POOL])
        positives = pools["positives"]
        crop = esr._matcher().detect_and_describe(_gray(meta["query_crop"]), max_features=args.budget)
        template_ids = [meta["query_page_id"]]
        templates = [crop.compact()]
        candidates = sorted(positives)
        if args.goods_in_top:
            # Only the Goods the shared sequence can reveal: positives in the exemplar's top N.
            c_stats, c_inl, c_tent = verify_all(templates, pool_ids, args.workers)
            top = np.lexsort((np.arange(len(pool_ids)), -c_tent[0].astype(np.int64), -c_inl[0].astype(np.int64)))
            reach = {pool_ids[i] for i in top[: args.goods_in_top]}
            candidates = [p for p in candidates if p in reach]
        unboxed = []
        for pid in candidates:
            box = largest_box(page_by_id[pid], cid)
            if box is None:
                unboxed.append(pid)
                continue
            template_ids.append(pid)
            templates.append(filter_features_to_box(esr._FEATURES[pid], box))
        if args.goods_in_top:
            g_stats, g_inl, g_tent = verify_all(templates[1:], pool_ids, args.workers)
            stats = np.concatenate([c_stats, g_stats])
            inliers = np.concatenate([c_inl, g_inl])
            tentative = np.concatenate([c_tent, g_tent])
        else:
            stats, inliers, tentative = verify_all(templates, pool_ids, args.workers)
        np.savez_compressed(
            args.out / f"{slug(cid)}.npz",
            template_ids=np.array(template_ids),
            template_sizes=np.array([t.count for t in templates]),
            pool_ids=np.array(pool_ids),
            stats=stats,
            inliers=inliers,
            tentative=tentative,
            positives=np.array([p in positives for p in pool_ids]),
            unboxed=np.array(unboxed),
        )
        print(
            f"  {cid}: {len(templates)} templates x {len(pool_ids)} pages in {time.time() - t1:.0f}s"
            + (f"; {len(unboxed)} positive(s) without a box" if unboxed else ""),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
