"""Stage 1 with votes on the tiled-VLAD cell (#3928, M3).

With votes, what should the cached Stage 1 rank by?

``t0_crop``   cosine to the query crop only, max over tiles; never learns
``t1_maxq``   max over queries x tiles, the queries being the crop and each Good page's
              VLAD inside its FullMarks box (projected like the tiles); Bads unused
``t2_svm``    the production linear SVM head over tile rows: the crop and the Good-box
              vectors are positives, every tile of every Bad page a negative; a page
              scores its best tile.  Cosine to the crop before a Bad exists

Replays #4162's shared vote sequence: the top *v* of the query crop's exhaustive SIFT
ranking (``matrix-<tier>``'s template 0), labelled from ground truth.  Reports, on the
unlabelled remainder:

* Stage-1 AP and recall@K;
* the pipeline AP: Stage 1's top K re-ordered by max-over-templates inliers (the crop and
  each Good's box template, #4162's ``a1_max``), the tail left in Stage-1 order -- what the
  app would show.

    python stage1_votes.py --tier s --cell <cell>/d512 --matrix <votes-4162>/matrix-s --out <dir>
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

import fullmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import stage1_cell as s1  # noqa: E402
import template_matrix as tm  # noqa: E402
import vote_curve as vc  # noqa: E402

ARMS = ("t0_crop", "t1_maxq", "t2_svm")
CHECKPOINTS = (0, 5, 10, 20)
KS = (500, 1000)
BUDGET = 8192


def box_vlad(page: Any, cid: str) -> Optional[np.ndarray]:
    """The raw VLAD of the page's features inside its largest box of *cid*."""
    from eval_splg_rank import _gray  # noqa: PLC0415
    from vtscore.media.structural import SiftMatcher, aggregate_vlad, load_vlad_codebook  # noqa: PLC0415

    box = tm.largest_box(page, cid)
    if box is None:
        return None
    feats = SiftMatcher().detect_and_describe(_gray(page.path), max_features=BUDGET).compact()
    kp, desc = feats.keypoints_f32(), feats.descriptors_f32()
    inside = (kp[:, 0] >= box[0]) & (kp[:, 0] <= box[2]) & (kp[:, 1] >= box[1]) & (kp[:, 1] <= box[3])
    if not inside.any():
        return None
    return aggregate_vlad(desc[inside], load_vlad_codebook()).astype(np.float32)


def svm_page_scores(cell: s1.Cell, rows: np.ndarray, pos: np.ndarray, bad_rows: np.ndarray) -> np.ndarray:
    """The linear SVM head over tile rows; a page scores its best tile."""
    import torch  # noqa: PLC0415

    from vtscore.training.svm import fit_linear_svm_head  # noqa: PLC0415

    X = np.vstack([pos, bad_rows]).astype(np.float32)
    y = np.array([1.0] * len(pos) + [0.0] * len(bad_rows), dtype=np.float32)
    model = fit_linear_svm_head(X, y, X.shape[1])
    out = []
    with torch.no_grad():
        dev = next(model.parameters()).device
        for i in range(0, len(rows), 200_000):
            chunk = torch.from_numpy(np.ascontiguousarray(rows[i : i + 200_000], dtype=np.float32)).to(dev)
            out.append(model(chunk).squeeze(1).cpu().numpy())
    return np.maximum.reduceat(np.concatenate(out), cell.starts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--max-v", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = [v for v in CHECKPOINTS if v <= args.max_v]

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    page_by_id = {p.page_id: p for p in embed_corpus.pages_for_tier(args.corpus, args.tier)}
    cell = s1.Cell(args.cell)
    col = {p: i for i, p in enumerate(cell.page_ids)}
    tiles = cell.tiles.astype(np.float32)
    tile_page = np.repeat(np.arange(len(cell.page_ids)), cell.counts)
    rows_out: list[dict[str, Any]] = []

    for f in sorted(args.matrix.glob("*.npz")):
        if f.name.startswith("vectors-"):
            continue
        t0 = time.time()
        cid = f.stem.replace("__", "/", 1)
        n = len(np.load(f)["pool_ids"])
        none = np.zeros((n, 1), dtype=np.float32)
        cd = vc.ClassData(f, none, none, np.zeros(1, np.float32), np.zeros(1, np.float32))
        if not cd.positive.any() or cid not in classes:
            continue
        missing = [p for p in cd.pool_ids if p not in col]
        if missing:
            raise SystemExit(f"{cid}: {len(missing)} pool pages are not in the cell, e.g. {missing[:3]}")
        pool_cells = np.array([col[p] for p in cd.pool_ids])
        crop_q = cell.prepare_query(s1.query_vlad(classes[cid]["query_crop"], BUDGET))
        a0 = vc.rank("a0_exemplar", cd, [], [])
        box_q: dict[int, np.ndarray] = {}
        for v in checkpoints:
            seq = [int(i) for i in a0[:v]]
            goods = [i for i in seq if cd.positive[i]]
            bads = [i for i in seq if not cd.positive[i]]
            for g in goods:
                if g not in box_q:
                    raw = box_vlad(page_by_id[cd.pool_ids[g]], cid)
                    box_q[g] = cell.prepare_query(raw) if raw is not None else np.zeros_like(crop_q)
            queries = np.vstack([crop_q, *[box_q[g] for g in goods]])
            inl, tent, _ = cd.best([0, *cd.templates_for(goods)])
            labelled = set(seq)
            for arm in ARMS:
                if arm == "t0_crop":
                    page = s1.page_scores(tiles, cell.starts, crop_q)
                elif arm == "t1_maxq":
                    page = np.max(np.stack([s1.page_scores(tiles, cell.starts, q) for q in queries]), axis=0)
                elif not bads:
                    page = s1.page_scores(tiles, cell.starts, crop_q)
                else:
                    bad_rows = tiles[np.isin(tile_page, pool_cells[bads])]
                    page = svm_page_scores(cell, tiles, queries, bad_rows)
                stage1 = vc.order_by(page[pool_cells])
                rest1 = vc.remainder(stage1, labelled)
                row: dict[str, Any] = {
                    "class_id": cid,
                    "arm": arm,
                    "v": v,
                    "found": len(goods),
                    "left": int(cd.positive[rest1].sum()),
                    "n_positive": int(cd.positive.sum()),
                    "stage1_ap": vc.average_precision(rest1, cd.positive),
                }
                for k in KS:
                    head = rest1[:k]
                    row[f"recall_{k}"] = float(cd.positive[head].sum() / max(1, cd.positive[rest1].sum()))
                    piped = vc.shortlist(rest1, k, inl, tent)
                    row[f"pipe_ap_{k}"] = vc.average_precision(piped, cd.positive)
                rows_out.append(row)
        print(f"  {cid}: {cd.n} pages, {cd.positive.sum()} positives, {time.time() - t0:.0f}s", flush=True)
        with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
