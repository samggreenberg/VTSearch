"""Arm 6 of #4162: learn Good from Bad in SIFT space with a descriptor stop-list.

The one arm the template matrix cannot replay, because it changes the templates.
After *v* votes from the shared sequence (the top *v* of the exemplar ranking,
as ``vote_curve.py``), every template -- the query crop and each Good's box --
drops the descriptors that survive the ratio test against any Bad page's
features.  Those descriptors are exactly what let a Bad match: a letterhead
rule, a signature flourish, a font's glyphs.  The pruned templates are then
re-verified against the top 1,000 of ``a1_max``'s ranking of the unlabelled
remainder and that head re-ordered by the new max-over-templates inliers; the
tail keeps ``a1_max``'s order.  A stop-list can only demote, so pages outside the
head cannot gain.

Rows go to ``<out>/rows-a6-<shard>.csv`` in ``vote_curve.py``'s schema, readout
``shared``, arm ``a6_stoplist``; ``vote_curve.py --summarise`` folds them in.

    python vote_stoplist.py --matrix <dir>/matrix-s --tier s --out <dir>/curves-s [--shard 0/4]
"""

from __future__ import annotations

import argparse
import csv
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
import eval_sift_rank as esr  # noqa: E402
import vote_curve as vc  # noqa: E402
from eval_splg_rank import _gray  # noqa: E402
from template_matrix import CHUNK, _init_worker, largest_box, slug  # noqa: E402

ARM = "a6_stoplist"
HEAD = 1000
#: A stop-list needs a Bad; v=0 is the exemplar itself.
CHECKPOINTS = tuple(v for v in vc.CHECKPOINTS if v > 0)

_TEMPLATES: list[Any] = []


def prune(template: Any, bad_feats: Sequence[Any]) -> tuple[Any, int]:
    """*template* without the descriptors that ratio-match any Bad; ``(pruned, n_dropped)``."""
    from vtscore.media.structural import StructuralFeatures, ratio_test_matches  # noqa: PLC0415

    desc = template.descriptors_f32()
    if not bad_feats or desc.shape[0] < 2:
        return template, 0
    drop = np.zeros(desc.shape[0], dtype=bool)
    for t_idx, _c in ratio_test_matches(desc, [b.descriptors_f32() for b in bad_feats], ratio=0.75):
        drop[t_idx] = True
    keep = ~drop
    return StructuralFeatures(keypoints=template.keypoints_f32()[keep], descriptors=desc[keep]), int(drop.sum())


def _verify_chunk(page_ids: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    cands = [esr._FEATURES[p] for p in page_ids]
    inl = np.zeros((len(_TEMPLATES), len(page_ids)), dtype=np.int32)
    tent = np.zeros_like(inl)
    for t, tpl in enumerate(_TEMPLATES):
        for n, stats in enumerate(esr._matcher().verify_many(tpl, cands)):
            inl[t, n] = stats.inlier_count if stats.model_ok else 0
            tent[t, n] = stats.tentative_count
    return page_ids, inl, tent


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _TEMPLATES
    from vtscore.training.structural_similarity import filter_features_to_box  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--classes", default="")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        ap.error("--out is inside the corpus; an eval never writes there")
    args.out.mkdir(parents=True, exist_ok=True)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    files = sorted(f for f in args.matrix.glob("*.npz") if not f.name.startswith("vectors-"))
    wanted = {slug(c) for c in args.classes.split(",") if c}
    files = [f for i, f in enumerate(f for f in files if not wanted or f.stem in wanted) if i % shard_n == shard_i]
    print(f"=== tier {args.tier}: shard {args.shard}, {len(files)} class(es)", flush=True)

    pages = embed_corpus.pages_for_tier(args.corpus, args.tier)
    page_by_id = {p.page_id: p for p in pages}
    t0 = time.time()
    with get_context("fork").Pool(args.workers, initializer=esr._init_budget, initargs=(args.budget,)) as pool:
        for i, (page_id, feats, _n, _vlad) in enumerate(
            pool.imap_unordered(esr._extract, [(p.page_id, p.path) for p in pages], chunksize=8)
        ):
            esr._FEATURES[page_id] = feats
            if (i + 1) % 1000 == 0:
                print(f"  extracted {i + 1}/{len(pages)} in {time.time() - t0:.0f}s", flush=True)
    esr._init_budget(args.budget)

    rows: list[dict[str, Any]] = []
    out_csv = args.out / f"rows-a6-{shard_i}.csv"
    for f in files:
        t1 = time.time()
        cid = f.stem.replace("__", "/", 1)
        meta = classes[cid]
        dummy = np.zeros((1, 1), dtype=np.float32)
        z = np.load(f)
        n = len(z["pool_ids"])
        cd = vc.ClassData(f, np.zeros((n, 1), np.float32), np.zeros((n, 1), np.float32), dummy[0], dummy[0])
        if not cd.positive.any():
            continue
        crop = esr._matcher().detect_and_describe(_gray(meta["query_crop"]), max_features=args.budget).compact()
        a0 = vc.rank("a0_exemplar", cd, [], [])
        notes = []
        for v in CHECKPOINTS:
            seq = [int(i) for i in a0[:v]]
            goods = [i for i in seq if cd.positive[i]]
            bads = [i for i in seq if not cd.positive[i]]
            labelled = set(seq)
            rest = vc.remainder(vc.rank("a1_max", cd, goods, bads), labelled)
            left = int(cd.positive[rest].sum())
            if left == 0:
                continue
            templates = [crop]
            for g in goods:
                if g in cd.template_of:
                    pid = cd.pool_ids[g]
                    templates.append(filter_features_to_box(esr._FEATURES[pid], largest_box(page_by_id[pid], cid)))
            bad_feats = [esr._FEATURES[cd.pool_ids[b]] for b in bads]
            pruned, dropped = zip(*(prune(t, bad_feats) for t in templates))
            _TEMPLATES = [t for t in pruned if t.count >= 2]
            head = rest[:HEAD]
            head_ids = [cd.pool_ids[i] for i in head]
            col = {p: j for j, p in enumerate(head_ids)}
            inl = np.zeros((len(_TEMPLATES), len(head)), dtype=np.int32)
            tent = np.zeros_like(inl)
            if _TEMPLATES:
                chunks = [head_ids[i : i + CHUNK] for i in range(0, len(head_ids), CHUNK)]
                with get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
                    for ids, bi, bt in pool.imap_unordered(_verify_chunk, chunks):
                        cols = [col[p] for p in ids]
                        inl[:, cols], tent[:, cols] = bi, bt
            if len(_TEMPLATES):
                arg = inl.argmax(axis=0)
                best_inl, best_tent = inl[arg, np.arange(len(head))], tent[arg, np.arange(len(head))]
            else:
                best_inl = best_tent = np.zeros(len(head), dtype=np.int32)
            order = np.concatenate([head[vc.order_by(best_inl, best_tent)], rest[HEAD:]])
            p10 = float(cd.positive[order[:10]].mean())
            rows.append(
                {
                    "class_id": cid,
                    "readout": "shared",
                    "arm": ARM,
                    "v": v,
                    "found": len(goods),
                    "ap": vc.average_precision(order, cd.positive),
                    "p10": p10,
                    "left": left,
                    "n_positive": int(cd.positive.sum()),
                }
            )
            kept = sum(t.count for t in pruned)
            notes.append(f"v{v}: {len(bads)} bad, dropped {sum(dropped)}/{sum(dropped) + kept}")
        print(f"  {cid}: {time.time() - t1:.0f}s; " + "; ".join(notes), flush=True)
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["class_id", "readout", "arm", "v", "found", "ap", "p10", "left", "n_positive"]
            )
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
