"""What the float16 patch grid does to the POOLED region score (#3159, part 2).

    python grid_drift_3159.py --datasets visual_genome_m,coco_val,caltech101_m --out <dir>

Reads the published pile cell (float16 grid) and this study's float32 rebuild of
it (``build_grid_3159.py``; its ``verify`` must have passed, which makes the pile
cell exactly the float32 cell through the storage cast) and measures drift where
a vote reads it.  The grid is an intermediate; the number that decides anything
is ``max over rows of row . query`` - the MaxPatch score - so that is what is
differenced, not the raw grid.

**Both arms score exactly as the app does.**  Rows come from
:func:`vtscore.embedding.matrix.media_score_rows` (CLS row + every patch) at the
arm's dtype, upcast to float32 for the dot product as
:func:`~vtscore.embedding.matrix.chunked_row_scores` does, and each arm's query
is built from *its own* grid - a Good box vote in a float16 dataset trains on a
float16 patch, so a float32 query against float16 rows would measure a mix no
user has.

Two query kinds, because a trained head is neither of them exactly:

* ``exemplar`` - one Good box vote: the production nearest-patch rule
  (:func:`vtscore.detectors.training.pool_box_from_media`) on one positive.  On
  a boxless dataset the image-level vector, which is what a boxless Good vote
  trains on.  The query row itself sits in the gallery, so this is also the
  case max-pooling amplifies most: a self-match at cosine ~1.
* ``centroid`` - the unit mean of a category's exemplars: closer to what a
  linear head learns after a few votes, and never itself a row.

Per query: absolute pooled-score drift over the whole gallery, how often the
winning row changes (and across what gap), Spearman over the gallery, top-k
overlap, and AP against the category's labels.  Every query also carries the
box area of the exemplar that made it, so the **sub-patch** regime (box under
one DINOv3 patch, 1/196 of the image) - where the issue expects a quantisation
floor to bite - is broken out rather than averaged away.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))

import common  # noqa: E402

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_grid_3159 import EMBEDDER, SHARED_EMBEDDINGS, STUDY, arm_datadir  # noqa: E402

PATCH_AREA = 1 / 196
LEAF_AREA = 1 / 12
#: The calibration harness's scale bands (experiment_config.SCALE_BANDS), so a
#: drift band and a bench band mean the same thing.
BANDS = [
    ("sub_patch", 0.0, PATCH_AREA),
    ("patch_to_leaf", PATCH_AREA, LEAF_AREA),
    ("leaf_to_4x", LEAF_AREA, 4 * LEAF_AREA),
    ("above_4x", 4 * LEAF_AREA, 1.01),
]
EXEMPLARS_PER_CATEGORY = 8
MIN_POSITIVES = 10
TOPK = (10, 100)
QUERY_BATCH = 64


def log(msg: str) -> None:
    print(f"[drift-3159] {msg}", flush=True)


def band_of(area: float | None) -> str:
    if area is None:
        return "image_level"
    for name, lo, hi in BANDS:
        if lo <= area < hi:
            return name
    return "above_4x"


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Non-interpolated AP, ties broken by index (stable) - identical in both arms."""
    order = np.argsort(-scores, kind="stable")
    hits = labels[order].astype(np.float64)
    n_pos = hits.sum()
    if n_pos == 0:
        return float("nan")
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / n_pos)


def ranks_of(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(len(scores))
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = ranks_of(a).astype(np.float64), ranks_of(b).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def load_pair(ds: str) -> tuple[list[int], dict, dict]:
    from _cells_io import load_medias  # noqa: PLC0415

    f16 = load_medias(SHARED_EMBEDDINGS / f"{ds}__{EMBEDDER}.pkl")
    f32 = load_medias(arm_datadir("fp32") / "embeddings" / f"{ds}__{EMBEDDER}.pkl")
    if sorted(f16) != sorted(f32):
        raise SystemExit(f"{ds}: media sets differ; run build_grid_3159.py verify first")
    return sorted(f16), f16, f32


def stack_rows(medias: dict, ids: list[int], dtype) -> tuple[np.ndarray, np.ndarray]:
    """``(rows float32, seg_starts)``: every media's score rows at *dtype*, upcast."""
    from vtscore.embedding.matrix import media_score_rows  # noqa: PLC0415

    blocks = [np.asarray(media_score_rows(medias[i], EMBEDDER, dtype=dtype), dtype=np.float32) for i in ids]
    starts = np.zeros(len(blocks), dtype=np.int64)
    np.cumsum([b.shape[0] for b in blocks[:-1]], out=starts[1:])
    return np.concatenate(blocks, axis=0), starts


def row_drift(r16: np.ndarray, r32: np.ndarray, starts: np.ndarray) -> dict:
    """Row-level drift, CLS rows separated from patch rows (CLS is never cast)."""
    is_cls = np.zeros(len(r16), dtype=bool)
    is_cls[starts] = True
    out = {}
    for name, mask in (("patch", ~is_cls), ("cls", is_cls)):
        a, b = r16[mask], r32[mask]
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        cos = (a * b).sum(axis=1) / np.maximum(na * nb, 1e-12)
        one_minus_cos = np.clip(1.0 - cos, 0, None)
        norm_err = np.abs(na / np.maximum(nb, 1e-12) - 1.0)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        nz = np.abs(b) > 1e-6
        out[name] = {
            "n_rows": int(mask.sum()),
            "one_minus_cos": _quantiles(one_minus_cos),
            "norm_rel_err": _quantiles(norm_err),
            "element_rel_err": _quantiles(rel[nz]),
            "rows_bit_identical": float(np.mean(np.all(a == b, axis=1))),
        }
    return out


def _quantiles(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    return {
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
        "mean": float(x.mean()),
    }


def build_queries(ds: str, ids: list[int], f16: dict, f32: dict) -> list[dict]:
    """Exemplar and centroid queries, each built from its own arm's grid."""
    from vtscore.detectors.training import pool_box_from_media  # noqa: PLC0415
    from vtscore.eval.labels import media_is_positive, region_box_for_category, voted_box_area  # noqa: PLC0415

    cats: dict[str, int] = {}
    for i in ids:
        m = f32[i]
        for c in m.get("categories") or [m.get("category")]:
            if c:
                cats[c] = cats.get(c, 0) + 1
    queries: list[dict] = []
    for cat in sorted(c for c, n in cats.items() if n >= MIN_POSITIVES):
        pos = [i for i in ids if media_is_positive(f32[i], cat)]
        boxed = [i for i in pos if region_box_for_category(f32[i], cat) is not None]
        pool = boxed or pos
        rng = np.random.RandomState(_stable_seed(ds, cat))
        chosen = [
            int(c) for c in rng.choice(np.array(pool), size=min(EXEMPLARS_PER_CATEGORY, len(pool)), replace=False)
        ]
        ex16, ex32 = [], []
        for cid in chosen:
            box = region_box_for_category(f32[cid], cat)
            area = voted_box_area(f32[cid], cat)
            q16 = pool_box_from_media(f16[cid], box)
            q32 = pool_box_from_media(f32[cid], box)
            if q16 is None:  # boxless: the image-level vector (row 0), identical in both arms
                q16 = f16[cid]["embeddings"][EMBEDDER]
                q32 = f32[cid]["embeddings"][EMBEDDER]
            q16, q32 = _unit(q16), _unit(q32)
            ex16.append(q16)
            ex32.append(q32)
            queries.append(
                {
                    "dataset": ds,
                    "category": cat,
                    "kind": "exemplar",
                    "exemplar_id": cid,
                    "box_area": area,
                    "band": band_of(area),
                    "q16": q16,
                    "q32": q32,
                }
            )
        areas = [voted_box_area(f32[c], cat) for c in pos]
        areas = [a for a in areas if a is not None]
        med = float(np.median(areas)) if areas else None
        queries.append(
            {
                "dataset": ds,
                "category": cat,
                "kind": "centroid",
                "exemplar_id": -1,
                "box_area": med,
                "band": band_of(med),
                "q16": _unit(np.mean(ex16, axis=0)),
                "q32": _unit(np.mean(ex32, axis=0)),
            }
        )
    return queries


def _stable_seed(ds: str, cat: str) -> int:
    import zlib  # noqa: PLC0415

    return zlib.crc32(f"{ds}::{cat}".encode()) & 0x7FFFFFFF


def pooled(rows: np.ndarray, starts: np.ndarray, qmat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per media: max score and the winning row index, for each query column."""
    flat = rows @ qmat  # (R, B) float32, as chunked_row_scores computes it
    mx = np.maximum.reduceat(flat, starts, axis=0)
    # winner = first row achieving the max (segmented_max_pool's tie rule)
    ends = np.append(starts[1:], len(rows))
    win = np.empty_like(mx, dtype=np.int32)
    for j, (s, e) in enumerate(zip(starts, ends, strict=True)):
        win[j] = np.argmax(flat[s:e], axis=0)
    return mx.astype(np.float64), win


def analyse_dataset(ds: str, outdir: Path) -> tuple[pd.DataFrame, dict, list[dict]]:
    from vtscore.eval.labels import media_is_positive, voted_box_area  # noqa: PLC0415

    t0 = time.time()
    ids, f16, f32 = load_pair(ds)
    log(f"{ds}: {len(ids)} medias loaded in {time.time() - t0:.0f}s")
    r16, starts = stack_rows(f16, ids, np.float16)
    r32, starts32 = stack_rows(f32, ids, np.float32)
    assert np.array_equal(starts, starts32)
    drift = row_drift(r16, r32, starts)
    log(f"{ds}: row drift {json.dumps(drift)}")

    queries = build_queries(ds, ids, f16, f32)
    log(f"{ds}: {len(queries)} queries")
    records: list[dict] = []
    examples: list[dict] = []
    # Per category, once: the labels, and which positives hold it below one
    # patch -- the regime the issue is about, measured on the medias rather
    # than on the query.
    labels_of: dict[str, np.ndarray] = {}
    subpatch_of: dict[str, np.ndarray] = {}
    for cat in sorted({q["category"] for q in queries}):
        lab = np.array([media_is_positive(f32[i], cat) for i in ids])
        areas = [voted_box_area(f32[i], cat) if lab[k] else None for k, i in enumerate(ids)]
        labels_of[cat] = lab
        subpatch_of[cat] = np.array([a is not None and a < PATCH_AREA for a in areas])
    for b0 in range(0, len(queries), QUERY_BATCH):
        batch = queries[b0 : b0 + QUERY_BATCH]
        s16, w16 = pooled(r16, starts, np.stack([q["q16"] for q in batch], axis=1))
        s32, w32 = pooled(r32, starts, np.stack([q["q32"] for q in batch], axis=1))
        for j, q in enumerate(batch):
            a, b = s16[:, j], s32[:, j]
            labels = labels_of[q["category"]]
            d = np.abs(a - b)
            flips = w16[:, j] != w32[:, j]
            # How close the fp32 winner was to its runner-up on the medias
            # whose winner flipped: a flip across a gap smaller than the cast's
            # own error is a tie broken differently, not a changed answer.
            flip_gaps = []
            for mi in np.flatnonzero(flips)[:200]:
                seg = r32[starts[mi] : (starts[mi + 1] if mi + 1 < len(starts) else len(r32))] @ q["q32"]
                top2 = np.sort(seg)[-2:]
                flip_gaps.append(float(top2[1] - top2[0]))
            ra, rb = ranks_of(a), ranks_of(b)
            move = np.abs(ra - rb)
            rec = {
                "dataset": ds,
                "category": q["category"],
                "kind": q["kind"],
                "exemplar_id": q["exemplar_id"],
                "box_area": q["box_area"],
                "band": q["band"],
                "n_pos": int(labels.sum()),
                "abs_drift_median": float(np.median(d)),
                "abs_drift_p99": float(np.quantile(d, 0.99)),
                "abs_drift_max": float(d.max()),
                "signed_drift_mean": float((a - b).mean()),
                "winner_flip_rate": float(flips.mean()),
                "flip_gap_median": float(np.median(flip_gaps)) if flip_gaps else float("nan"),
                "flip_gap_max": float(max(flip_gaps)) if flip_gaps else float("nan"),
                "spearman": spearman(a, b),
                "top1_same": bool(ra.argmin() == rb.argmin()),
                "max_rank_move": int(move.max()),
                "max_rank_move_pos": int(move[labels].max()) if labels.any() else 0,
                "ap16": average_precision(a, labels),
                "ap32": average_precision(b, labels),
            }
            for k in TOPK:
                top_a = set(np.argsort(-a, kind="stable")[:k])
                top_b = set(np.argsort(-b, kind="stable")[:k])
                rec[f"top{k}_overlap"] = len(top_a & top_b) / k
            sub = subpatch_of[q["category"]]
            rec["n_subpatch_pos"] = int(sub.sum())
            rec["max_rank_move_subpatch_pos"] = int(move[sub].max()) if sub.any() else 0
            records.append(rec)

            # Literal examples: the biggest rank move per query, with the score
            # gap it crossed, so a reader can check it is a tie and not a miss.
            mi = int(move.argmax())
            if move[mi] > 0:
                order32 = np.argsort(-b, kind="stable")
                neighbour = order32[min(rb[mi] + 1, len(order32) - 1)]
                examples.append(
                    {
                        "dataset": ds,
                        "category": q["category"],
                        "kind": q["kind"],
                        "exemplar_id": q["exemplar_id"],
                        "band": q["band"],
                        "media_id": ids[mi],
                        "filename": f32[ids[mi]].get("filename") or f32[ids[mi]].get("origin_name"),
                        "is_positive": bool(labels[mi]),
                        "rank_fp32": int(rb[mi]),
                        "rank_fp16": int(ra[mi]),
                        "rank_move": int(move[mi]),
                        "score_fp32": float(b[mi]),
                        "score_fp16": float(a[mi]),
                        "gap_to_next_fp32": float(b[mi] - b[neighbour]),
                    }
                )
        log(f"  {ds}: {min(b0 + QUERY_BATCH, len(queries))}/{len(queries)} queries ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(records)
    df["d_ap"] = df["ap16"] - df["ap32"]
    return df, drift, examples


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, kind, band): the decision-relevant numbers, paired SE on dAP."""
    rows = []
    for key, g in df.groupby(["dataset", "kind", "band"]):
        n = len(g)
        se = float(g["d_ap"].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append(
            {
                "dataset": key[0],
                "kind": key[1],
                "band": key[2],
                "n_queries": n,
                "abs_drift_median": float(g["abs_drift_median"].median()),
                "abs_drift_max": float(g["abs_drift_max"].max()),
                "winner_flip_rate": float(g["winner_flip_rate"].mean()),
                "spearman_min": float(g["spearman"].min()),
                "top10_overlap_mean": float(g["top10_overlap"].mean()),
                "top100_overlap_mean": float(g["top100_overlap"].mean()),
                "top1_same_rate": float(g["top1_same"].mean()),
                "ap32_mean": float(g["ap32"].mean()),
                "d_ap_mean": float(g["d_ap"].mean()),
                "d_ap_se": se,
                "d_ap_max_abs": float(g["d_ap"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="visual_genome_m,coco_val,caltech101_m")
    ap.add_argument("--out", default=str(STUDY / "drift"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verify = STUDY / "verify.json"
    if not verify.exists():
        raise SystemExit(f"{verify} missing: run build_grid_3159.py verify first")
    vrec = json.loads(verify.read_text())
    frames, drifts, examples = [], {}, []
    for ds in [d for d in args.datasets.split(",") if d]:
        if not str(vrec.get(ds, {}).get("verdict", "")).startswith("PASS"):
            raise SystemExit(f"{ds}: verify did not PASS; the comparison would not isolate the cast")
        df, drift, ex = analyse_dataset(ds, out)
        frames.append(df)
        drifts[ds] = drift
        examples.extend(ex)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "queries.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "row_drift.json").write_text(json.dumps(drifts, indent=2) + "\n")
    examples.sort(key=lambda e: -e["rank_move"])
    (out / "examples.json").write_text(json.dumps(examples[:60], indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        log("\n" + summ.to_string(index=False))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
