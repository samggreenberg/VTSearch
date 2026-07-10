#!/usr/bin/env python3
"""Visual interpretability for the SOD sweep — offline, from cache (no GPU).

Two modes:

* ``--kind split`` — montage galleries of each train/test bucket
  (annotation-pool positives, test positives, training negatives, eval negatives),
  positives drawn with their GT boxes. Answers "what do the splits look like?".
* ``--kind predict`` — for one config (embedder × proposal × head=mlp × K × seed),
  rebuild the MLP head from cached vectors, score every test image, and montage the
  predictions: the winning region box + score, the GT boxes, tagged TP/FP/FN/TN at
  the cross-calibrated threshold. Answers "what did the detector actually fire on?".

Everything is read from the sweep's npz cache + the image zips, so this runs on the
login node without loading any model (MLP-head predictions only; the cosine head
would need the text encoder). Split buckets are recomputed deterministically via
``features.partition_split`` (same seed → same split as the sweep).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from datasets import Box, SodDataset
from features import partition_split, slugify

_EMBEDDER_ALIASES = {"dinov2": "dinov2_patch", "dinov3": "dinov3_patch"}
_GT_COLOR = (60, 220, 60)
_PRED_COLOR = (240, 60, 60)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw(img: Image.Image, gt: list[Box], pred: Box | None, score: float | None) -> Image.Image:
    """Draw GT boxes (green) and, optionally, the predicted box + score (red)."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.width, im.height
    for x0, y0, x1, y1 in gt:
        d.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), outline=_GT_COLOR, width=3)
    if pred is not None:
        x0, y0, x1, y1 = pred
        d.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), outline=_PRED_COLOR, width=2)
        if score is not None:
            d.text((max(0, x0 * w) + 2, max(0, y0 * h) + 2), f"{score:.2f}", fill=_PRED_COLOR)
    return im


def _montage(items: list[tuple[Image.Image, str]], out_path: Path, suptitle: str, cols: int = 6) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not items:
        return
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (img, title) in zip(axes.flat, items, strict=False):
        ax.imshow(np.asarray(img))
        ax.set_title(title, fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _sample(ids: list[int], n: int, seed: int) -> list[int]:
    if len(ids) <= n:
        return list(ids)
    return sorted(int(x) for x in np.random.default_rng(seed).choice(ids, size=n, replace=False))


# ---------------------------------------------------------------------------
# split galleries
# ---------------------------------------------------------------------------


def render_split_gallery(
    ds: SodDataset, split, *, out_dir: Path, dataset: str, cls: str, gallery_n: int, sample_seed: int
) -> None:
    """Montage each split bucket; positives drawn with their GT boxes."""
    out = out_dir / f"{dataset}_{slugify(cls)}"
    buckets = {
        "train_pos": (split.train_pos, True),
        "test_pos": (split.test_pos, True),
        "train_neg": (split.train_neg, False),
        "test_neg": (split.test_neg, False),
    }
    for name, (ids, positive) in buckets.items():
        items: list[tuple[Image.Image, str]] = []
        for iid in _sample(ids, gallery_n, sample_seed):
            try:
                img = ds.load_image(iid)
            except Exception as exc:
                print(f"  [{iid}] load failed ({exc})", flush=True)
                continue
            gt = split.gt_boxes.get(iid, []) if positive else []
            items.append((_draw(img, gt, None, None), str(iid)))
        _montage(items, out / f"{name}.png", f"{dataset} / {cls} — {name} (n={len(ids)})")
        print(f"wrote {out / f'{name}.png'}  ({len(items)} of {len(ids)})", flush=True)


def cmd_split(args, ds: SodDataset, split) -> None:
    render_split_gallery(
        ds,
        split,
        out_dir=args.out_dir,
        dataset=args.dataset,
        cls=args.cls,
        gallery_n=args.gallery_n,
        sample_seed=args.split_seed,
    )


# ---------------------------------------------------------------------------
# prediction overlays (MLP head, from cache)
# ---------------------------------------------------------------------------


def _resolve_slug(regions_root: Path, proposal: str, alpha: float, explicit: str | None) -> str | None:
    """Cache slug for a proposal, or ``None`` if nothing matches (caller warns)."""
    if explicit:
        return explicit
    if not regions_root.exists():
        return None
    cands = sorted(
        d.name
        for d in regions_root.iterdir()
        if d.is_dir() and (d.name == proposal or d.name.startswith(proposal + "_"))
    )
    if proposal == "hac":
        cands = [c for c in cands if f"a{alpha}" in c] or cands
    return cands[0] if cands else None


def _load_stack(npz_dir: Path, ids: list[int], key: str) -> tuple[np.ndarray, list[int]]:
    """Vstack ``key`` arrays across ids that have a cache file; return (stack, used_ids)."""
    arrs, used = [], []
    for iid in ids:
        p = npz_dir / f"{iid}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            a = z[key]
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[0] > 0:
            arrs.append(a)
            used.append(iid)
    return (np.vstack(arrs).astype(np.float32) if arrs else np.zeros((0, 0), np.float32)), used


def _load_neg_bags(regions_dir: Path, ids: list[int]) -> list[np.ndarray]:
    """One CLS+leaf bag per negative image (from cached vecs + leaf_mask), for the
    region-voting overlay. Skips images with no cache or no childless nodes."""
    bags: list[np.ndarray] = []
    for iid in ids:
        p = regions_dir / f"{iid}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            vecs = z["vecs"]
            mask = z["leaf_mask"] if "leaf_mask" in z else np.ones(vecs.shape[0], dtype=bool)
        bag = vecs[mask] if mask.any() else vecs
        if bag.shape[0] > 0:
            bags.append(np.asarray(bag, dtype=np.float32))
    return bags


def render_predictions(
    ds: SodDataset,
    split,
    *,
    cache_dir: Path,
    out_dir: Path,
    dataset: str,
    cls: str,
    embedder: str,
    proposal: str,
    alpha: float,
    slug: str | None,
    k: int,
    seed: int,
    neg_ratio: int,
    inclusion: int,
    safe_thresholds: bool,
    gallery_n: int,
    neg_regions: bool = False,
    region_voting: bool = False,
) -> None:
    """MLP prediction overlays for one config, from cache. Skips (warns) if data is missing.

    With ``region_voting`` the training reproduces the faithful DINO-patch path
    (snapped positives, leaf-flooded per-image bag negatives, bag-aware weighting)
    via the shared :func:`vtscore.eval.region_curve.train_rv_head`, so overlays
    match a ``--region-voting`` sweep's metrics.
    """
    reg = _EMBEDDER_ALIASES.get(embedder, embedder)
    regions_root = cache_dir / "regions" / dataset / reg
    slug = _resolve_slug(regions_root, proposal, alpha, slug)
    if slug is None:
        print(f"  [predict] skip {embedder}/{proposal}: no cache under {regions_root}", flush=True)
        return
    regions_dir = regions_root / slug
    exem_dir = cache_dir / "exemplars" / dataset / slugify(cls) / reg / slug

    pos_ex, _ = _load_stack(exem_dir, split.train_pos, "exemplars")
    if pos_ex.shape[0] == 0:
        print(f"  [predict] skip {embedder}/{proposal} (slug={slug}): empty exemplar cache", flush=True)
        return
    dim = pos_ex.shape[1]
    k = min(k, pos_ex.shape[0])
    if k < 1:
        return

    if region_voting:
        from vtscore.eval.region_curve import sample_rv_budget, train_rv_head

        neg_bags = _load_neg_bags(regions_dir, split.train_neg)
        budget = sample_rv_budget(pos_ex, neg_bags, k, neg_ratio, seed) if neg_bags else None
        trained = (
            train_rv_head(
                budget[0],
                budget[1],
                dim,
                seed,
                inclusion=inclusion,
                safe_thresholds=safe_thresholds,
                calibrate_count=2,
                cal_fraction=0.5,
            )
            if budget is not None
            else None
        )
        if trained is None:
            print(f"  [predict] skip {embedder}/{proposal} (slug={slug}): region-voting budget unmet", flush=True)
            return
        predict, raw_thr, n_votes = trained
    else:
        from vtscore.eval.scoring_heads import MLPHead
        from vtscore.eval.xcal import cross_calibrated_threshold

        neg_train, _ = _load_stack(regions_dir, split.train_neg, "vecs" if neg_regions else "whole_vec")
        if neg_train.shape[0] == 0:
            print(f"  [predict] skip {embedder}/{proposal} (slug={slug}): empty negative cache", flush=True)
            return
        rng = np.random.default_rng(seed)
        P, N = pos_ex.shape[0], neg_train.shape[0]
        n_neg = min(max(1, neg_ratio * k), N)
        x = np.vstack([pos_ex[rng.permutation(P)[:k]], neg_train[rng.permutation(N)[:n_neg]]]).astype(np.float32)
        y = np.array([1.0] * k + [0.0] * n_neg, dtype=np.float32)
        head = MLPHead(dim)
        raw_thr = cross_calibrated_threshold(x, y, head.trainer_fn(), seed, inclusion_value=inclusion)
        head.fit(x, y, seed)
        predict = head.score_rows
        n_votes = k + n_neg

    # Pass 1: score every test image (winning region), collecting the distribution
    # so the GMM safe-threshold blends over the same scores production sees.
    scored: list[tuple[int, bool, float, tuple]] = []
    all_scores: list[float] = []
    for iid, is_pos in [(i, True) for i in split.test_pos] + [(i, False) for i in split.test_neg]:
        p = regions_dir / f"{iid}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            vecs, boxes = z["vecs"], z["boxes"]
        if vecs.shape[0] == 0:
            continue
        s = np.asarray(predict(vecs))
        best = int(s.argmax())
        score = float(s[best])
        scored.append((iid, is_pos, score, tuple(float(b) for b in boxes[best])))
        all_scores.append(score)

    thr = raw_thr
    if safe_thresholds:
        from vtscore.training.thresholds import calculate_safe_threshold

        thr = calculate_safe_threshold(raw_thr, all_scores, n_votes)

    # Pass 2: bucket by confusion outcome at thr and draw overlays.
    groups: dict[str, list[tuple[float, Image.Image, str]]] = {"TP": [], "FP": [], "FN": [], "TN": []}
    for iid, is_pos, score, best_box in scored:
        pred_pos = score >= thr
        tag = ("TP" if pred_pos else "FN") if is_pos else ("FP" if pred_pos else "TN")
        try:
            img = ds.load_image(iid)
        except Exception:
            continue
        gt = split.gt_boxes.get(iid, []) if is_pos else []
        overlay = _draw(img, gt, best_box, score)
        groups[tag].append((score, overlay, f"{iid}  {score:.2f}"))

    alpha_tag = f"_a{alpha}" if proposal == "hac" else ""
    tag_out = out_dir / f"{dataset}_{slugify(cls)}_{embedder}_{proposal}{alpha_tag}_mlp_k{k}"
    print(
        f"  [predict] {embedder}/{proposal} K={k} thr={thr:.3f}: "
        f"TP={len(groups['TP'])} FP={len(groups['FP'])} FN={len(groups['FN'])} TN={len(groups['TN'])}",
        flush=True,
    )
    for tag, items in groups.items():
        # Sort TP/FP by descending score (most confident first); FN/TN ascending.
        items.sort(key=lambda t: t[0], reverse=tag in ("TP", "FP"))
        capped = [(im, ti) for _s, im, ti in items[:gallery_n]]
        _montage(
            capped,
            tag_out / f"{tag}.png",
            f"{embedder}/{proposal} MLP K={k} thr={thr:.3f} — {tag} ({len(items)})",
        )


def cmd_predict(args, ds: SodDataset, split) -> None:
    render_predictions(
        ds,
        split,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        dataset=args.dataset,
        cls=args.cls,
        embedder=args.embedder,
        proposal=args.proposal,
        alpha=args.alpha,
        slug=args.slug,
        k=args.k,
        seed=args.seed,
        neg_ratio=args.neg_ratio,
        inclusion=args.inclusion,
        safe_thresholds=args.safe_thresholds,
        gallery_n=args.gallery_n,
        neg_regions=args.neg_regions,
        region_voting=args.region_voting,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("split", "predict"), required=True)
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--cls", "--class", dest="cls", default="stop sign")
    ap.add_argument("--out-dir", type=Path, required=True)
    # split params (must match the sweep run being inspected)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--neg-count", type=int, default=690)
    ap.add_argument("--test-fraction", type=float, default=0.5)
    ap.add_argument("--gallery-n", type=int, default=24, help="max images per montage")
    # predict-only
    ap.add_argument("--cache-dir", type=Path, default=None, help="sweep cache dir (predict mode)")
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--proposal", default="sliding")
    ap.add_argument("--slug", default=None, help="exact cache slug (else auto-match by proposal)")
    ap.add_argument("--alpha", type=float, default=0.5, help="hac alpha (to disambiguate the slug)")
    ap.add_argument("--head", choices=("mlp",), default="mlp", help="only MLP is supported offline")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg-ratio", type=int, default=1)
    ap.add_argument("--inclusion", type=int, default=0)
    ap.add_argument("--safe-thresholds", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--neg-regions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train MLP negatives on proposed-region crops of negative images (match the sweep run)",
    )
    ap.add_argument(
        "--region-voting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="faithful DINO-patch training for hac overlays (snap positives + leaf-flood bag negatives); "
        "point --cache-dir at a --region-voting sweep's cache and use --proposal hac",
    )
    args = ap.parse_args()

    with SodDataset(args.dataset) as ds:
        cs = ds.class_split(args.cls, neg_count=args.neg_count, seed=args.split_seed)
        if not cs.positive_ids:
            raise SystemExit(f"no positives for {args.cls!r} in {args.dataset}")
        split = partition_split(cs, args.test_fraction, args.split_seed)
        if args.kind == "split":
            cmd_split(args, ds, split)
        else:
            if args.cache_dir is None:
                raise SystemExit("--cache-dir is required for --kind predict")
            cmd_predict(args, ds, split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
