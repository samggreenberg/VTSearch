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

from datasets import GUI_MIN_BOX_FRAC, Box, SodDataset
from features import partition_split, slugify

_EMBEDDER_ALIASES = {"dinov2": "dinov2_patch", "dinov3": "dinov3_patch"}
_GT_COLOR = (60, 220, 60)
_PRED_COLOR = (240, 60, 60)
_MATCH_COLOR = (240, 60, 60)  # surfacing argmax — the region that made the model pick this image
_SNAP_COLOR = (60, 120, 240)  # good-vote snapped best-IoU-to-GT node — the training target


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw(
    img: Image.Image, gt: list[Box], pred: Box | None, score: float | None, snap: Box | None = None
) -> Image.Image:
    """Draw GT boxes (green), optionally the predicted box + score (red), and optionally the
    good-vote **snapped** region (blue) — the patch that becomes the training positive for the
    next MLP. Blue is drawn last (on top) so it stays visible even when it coincides with red
    (i.e. the detector already fired on the region it will train on)."""
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
    if snap is not None:
        x0, y0, x1, y1 = snap
        d.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), outline=_SNAP_COLOR, width=2)
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


def _save_captioned(img: Image.Image, out_path: Path, caption: str) -> None:
    """Save one image with a wrapped caption above it, sized so the full caption fits.

    Used by the labeling trace (the metadata caption is long and overflowed the tiny
    single-tile ``_montage`` figure)."""
    import textwrap  # noqa: PLC0415

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    arr = np.asarray(img)
    h, w = arr.shape[:2]
    fig_w = 5.0  # wide enough for the wrapped caption at fontsize 8
    wrapped = "\n".join(textwrap.wrap(caption, width=64)) or caption
    n_lines = wrapped.count("\n") + 1
    fig_h = fig_w * (h / max(w, 1)) + 0.28 * n_lines + 0.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(arr)
    ax.axis("off")
    ax.set_title(wrapped, fontsize=8, loc="left", family="monospace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confidence-sorted prediction gallery (--confidence-gallery)
# ---------------------------------------------------------------------------


def _hstack2(left: Image.Image, right: Image.Image) -> Image.Image:
    """Two RGB images side by side; the right is scaled to the left's height."""
    left = left.convert("RGB")
    right = right.convert("RGB")
    if right.height != left.height:
        rw = max(1, round(right.width * left.height / right.height))
        right = right.resize((rw, left.height))
    out = Image.new("RGB", (left.width + right.width, left.height), (255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out


def _save_caption_pil(img: Image.Image, out_path: Path, caption: str) -> None:
    """Fast (matplotlib-free) captioned save: a dark bar with *caption* above *img*.

    The confidence gallery renders every test image (thousands per class), so it can't
    afford a matplotlib figure per image the way :func:`_save_captioned` does."""
    im = img.convert("RGB")
    bar_h = 20
    canvas = Image.new("RGB", (im.width, im.height + bar_h), (18, 18, 18))
    canvas.paste(im, (0, bar_h))
    ImageDraw.Draw(canvas).text((4, 5), caption, fill=(240, 240, 240))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=88)


def _confidence_panel(
    img: Image.Image, *, good: bool, best_box: Box, best_mask: "np.ndarray | None", is_hac: bool, gt=()
) -> Image.Image:
    """One test image's composite for the confidence gallery.

    HAC cells: the right panel is always the top-scoring (highest-MLP) node's **own patches**
    (a colour cell-crop, like the training nodes). On the left: a predicted-good (score >= thr)
    draws a blue box on that node plus the green ground-truth box(es) when the image is a *true*
    positive; a predicted-bad draws a red box on that node. Non-HAC cells: just the plain image."""
    if not is_hac:
        return img.convert("RGB")
    full = np.asarray(img.convert("RGB"), dtype=np.uint8)
    right = _cell_thumb(full, best_mask, best_box, (img.height, img.height))  # highest-MLP node's patches (colour)
    if good:
        left = _draw(img, list(gt), None, None, snap=best_box)  # green GT (true positives) + blue box on the node
    else:
        left = _draw(img, [], best_box, None)  # red box on the node
    return _hstack2(left, right)


def render_confidence_gallery(
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
    predict,
    thr: float,
    t: int,
    seed: int,
) -> None:
    """Every test image, sorted by descending detector confidence, one captioned composite
    each (see :func:`_confidence_panel`). Additional to the TP/FP/FN/TN split gallery; one
    call per *seed*'s final head (output namespaced under ``seed{seed}/``). Files are named
    ``{rank:04d}_conf{score}_id{iid}_{good|bad}.png`` so they sort by confidence."""
    reg = _EMBEDDER_ALIASES.get(embedder, embedder)
    regions_root = cache_dir / "regions" / dataset / reg
    slug = _resolve_slug(regions_root, proposal, alpha, slug)
    if slug is None:
        print(f"  [conf-gallery] skip {embedder}/{proposal}: no cache under {regions_root}", flush=True)
        return
    regions_dir = regions_root / slug
    is_hac = proposal == "hac"

    scored: list[tuple] = []
    for iid, is_pos in [(i, True) for i in split.test_pos] + [(i, False) for i in split.test_neg]:
        p = regions_dir / f"{iid}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            vecs, boxes = z["vecs"], z["boxes"]
            cell_masks = z["cell_masks"] if ("cell_masks" in z and is_hac) else None
        if vecs.shape[0] == 0:
            continue
        s = np.asarray(predict(vecs))
        best = int(s.argmax())
        best_mask = cell_masks[best] if cell_masks is not None else None
        gt = split.gt_boxes.get(iid, []) if is_pos else []  # green GT box only for true positives
        scored.append((iid, float(s[best]), tuple(float(b) for b in boxes[best]), best_mask, gt))

    scored.sort(key=lambda r: r[1], reverse=True)
    alpha_tag = f"_a{alpha}" if proposal == "hac" else ""
    gdir = out_dir / "confidence_gallery" / f"{dataset}_{slugify(cls)}_{embedder}_{proposal}{alpha_tag}" / f"seed{seed}"
    n_good = sum(1 for r in scored if r[1] >= thr)
    print(
        f"  [conf-gallery] {embedder}/{proposal} seed{seed} (t={t}) thr={thr:.3f}: {len(scored)} test imgs "
        f"({n_good} good / {len(scored) - n_good} bad) -> {gdir.parent.name}/{gdir.name}",
        flush=True,
    )
    for rank, (iid, score, best_box, best_mask, gt) in enumerate(scored):
        try:
            img = ds.load_image(iid)
        except Exception:
            continue
        good = score >= thr
        panel = _confidence_panel(img, good=good, best_box=best_box, best_mask=best_mask, is_hac=is_hac, gt=gt)
        _save_caption_pil(
            panel,
            gdir / f"{rank:04d}_conf{score:.3f}_id{iid}_{'good' if good else 'bad'}.png",
            f"id={iid} conf={score:.2f} thr={thr:.2f}",
        )


def render_training_nodes(
    ds: SodDataset,
    split,
    trace: list[dict],
    *,
    cache_dir: Path,
    out_dir: Path,
    dataset: str,
    cls: str,
    embedder: str,
    proposal: str,
    alpha: float,
    slug: str | None,
    seed: int,
    bagged: bool = True,
    snapped: bool = True,
) -> None:
    """Dump the rows that went into TRAINING at the final vote set (after the loop consumed all
    its annotations), in labeling order, as captioned crops under
    ``training_nodes/<config>/seed{seed}/``.

    *bagged* / *snapped* must describe the run being rendered, because the two label
    constructions put genuinely different rows in front of the trainer:

    * ``bagged=True`` (``--region-voting`` / ``--neg-regions``, via ``train_rv_head``) -- each Bad
      vote floods the childless nodes (CLS + HAC leaves) as one bag, and rows carry the per-bag
      loss weights of :func:`vtscore.training.thresholds._per_bag_fit_weights` (good =
      ``n_bad_bags/n_good``, bad = ``1/bag_size``), so both are captioned.
    * ``bagged=False`` (box-pool, via ``make_head``) -- each Bad vote contributes exactly ONE
      whole-image CLS row, and that path applies no sample weights at all, so a single full-frame
      crop is drawn per Bad vote and no weight is captioned.
    * ``snapped=True`` -- a Good vote trains on the covering box snapped to its best-IoU node.
      ``snapped=False`` -- it trains on one grid-pooled row per GT box, so every GT box is drawn.

    Rendering the bagged construction for a box-pool run (the pre-2026-08-07 behaviour) drew 33
    NEG crops and a weight caption for a run that used one unweighted CLS row, so the flags are
    required rather than assumed. HAC only."""
    if proposal != "hac":
        print(f"  [train-nodes] skip {embedder}/{proposal}: HAC only", flush=True)
        return
    reg = _EMBEDDER_ALIASES.get(embedder, embedder)
    regions_root = cache_dir / "regions" / dataset / reg
    slug = _resolve_slug(regions_root, proposal, alpha, slug)
    if slug is None:
        print(f"  [train-nodes] skip {embedder}/{proposal}: no cache under {regions_root}", flush=True)
        return
    regions_dir = regions_root / slug
    good_ids = [int(e["image_id"]) for e in trace if e["gt_label"] == "good"]
    bad_ids = [int(e["image_id"]) for e in trace if e["gt_label"] == "bad"]
    n_good, n_bad_bags = len(good_ids), len(bad_ids)
    if n_good == 0 or n_bad_bags == 0:
        print(
            f"  [train-nodes] skip {embedder}/{proposal} seed{seed}: single-class ({n_good} pos / {n_bad_bags} neg)",
            flush=True,
        )
        return
    # Weights exist only on the bagged path; make_head trains on raw unweighted rows.
    good_w = (n_bad_bags / n_good) if bagged else None
    wtag = (lambda w: f"_w{w:.3f}") if bagged else (lambda w: "")
    wcap = (lambda w: f"  weight={w:.3f}") if bagged else (lambda w: "")
    gdir = out_dir / "training_nodes" / f"{dataset}_{slugify(cls)}_{embedder}_{proposal}_a{alpha}" / f"seed{seed}"
    rank = 0
    n_neg_nodes = 0
    for e in trace:  # labeling order
        iid = int(e["image_id"])
        viz = _load_region_viz(regions_dir, iid)
        if viz is None:
            continue
        boxes, children, cell_masks, _sal = viz
        try:
            full = np.asarray(ds.load_image(iid).convert("RGB"), dtype=np.uint8)
        except Exception:
            continue
        gt = [tuple(float(v) for v in b) for b in split.gt_boxes.get(iid, [])]
        blist = [tuple(float(v) for v in b) for b in boxes]
        if e["gt_label"] == "good":
            # Snapped: one row, the best-IoU node for the covering box. Grid-pooled: one row per
            # GT box, pooled over the box's cells rather than snapped to any node.
            if snapped:
                j = _snapped_index(gt, blist)
                if j is None:
                    continue
                rows = [(cell_masks[j] if cell_masks is not None else None, boxes[j], f"node={j}")]
            else:
                rows = [(None, np.asarray(b, dtype=np.float32), f"gtbox{bi}") for bi, b in enumerate(gt)]
            for mask, box, kind in rows:
                thumb = _cell_thumb(full, mask, box, (384, 384))
                _save_caption_pil(
                    thumb,
                    gdir / f"{rank:04d}_t{int(e['t']):03d}_id{iid}_POS{wtag(good_w)}.png",
                    f"id={iid} POS  bag=g:{iid}  {kind}{wcap(good_w)}  (t{int(e['t'])})",
                )
                rank += 1
        elif bagged:
            childless = [i for i in range(len(children)) if int(children[i][0]) < 0]
            bag_size = len(childless)
            bad_w = 1.0 / bag_size if bag_size else 0.0  # _per_bag_fit_weights: every bad row weighs 1/bag_size
            for node_i, j in enumerate(childless):
                mask = cell_masks[j] if cell_masks is not None else None
                thumb = _cell_thumb(full, mask, boxes[j], (384, 384))
                node_kind = "CLS" if j == 0 else f"leaf{j}"
                _save_caption_pil(
                    thumb,
                    gdir / f"{rank:04d}_t{int(e['t']):03d}_id{iid}_NEG_node{node_i:02d}_w{bad_w:.3f}.png",
                    f"id={iid} NEG  bag=b:{iid} ({bag_size} nodes)  {node_kind}  weight={bad_w:.3f}  (t{int(e['t'])})",
                )
                rank += 1
                n_neg_nodes += 1
        else:
            # Box-pool: the Bad vote is ONE whole-image CLS row (pool_whole_vecs), unweighted.
            thumb = _cell_thumb(full, None, np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32), (384, 384))
            _save_caption_pil(
                thumb,
                gdir / f"{rank:04d}_t{int(e['t']):03d}_id{iid}_NEG_wholeCLS.png",
                f"id={iid} NEG  whole-image CLS (1 row, unweighted)  (t{int(e['t'])})",
            )
            rank += 1
            n_neg_nodes += 1
    shape = "bagged childless nodes" if bagged else "one whole-image CLS row"
    wnote = f" (w={good_w:.3f})" if bagged else " (unweighted)"
    print(
        f"  [train-nodes] {embedder}/{proposal} seed{seed}: {n_good} pos{wnote} + "
        f"{n_bad_bags} neg [{shape}] = {n_neg_nodes} neg rows -> {rank} crops in "
        f"{gdir.parent.name}/{gdir.name}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# HAC region-tree interpretability (labeling trace)
# ---------------------------------------------------------------------------


def _iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _covering_box(boxes: list[Box]) -> Box:
    """Union rectangle of all GT instance boxes (matches region_sources._covering_box)."""
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _snapped_index(gt_boxes: list[Box], region_boxes: list[Box]) -> int | None:
    """The region a Good vote snaps to: best-IoU node vs the GT covering box (region_sources.py:357)."""
    if not gt_boxes or not region_boxes:
        return None
    cover = _covering_box(gt_boxes)
    ious = [_iou(cover, tuple(rb)) for rb in region_boxes]
    return int(np.argmax(ious)) if ious else None


# Reference-style HAC-tree rendering, adapted from scripts/run_hac_tree_sweep.py
# (that script is throwaway; this is the reusable SOD-tier port). Node thumbnails are
# masked to their patch-cell union — the true footprint, not the loose bbox — laid out
# by HAC merge depth, with a twin attention panel. We add MLP-score labels + matched
# (surfacing) / snapped (good-vote) rings on top.
_LEAF_RING = (255, 215, 0)  # yellow — HAC leaf
_INTERNAL_RING = (40, 170, 200)  # cyan — HAC merge node


def _cell_thumb(full_rgb: np.ndarray, mask: np.ndarray | None, box, size: tuple[int, int]) -> Image.Image:
    """Thumbnail of a node. With a patch ``mask`` (H,W bool): block-upsample it to image
    res, crop to its tight bbox, dim outside-mask pixels, fit into ``size`` (the reference
    masked-cell look). Without a mask (old cache): fall back to a plain ``box`` crop."""
    h_img, w_img = full_rgb.shape[:2]
    full = np.ascontiguousarray(full_rgb, dtype=np.uint8)
    pad = (250, 250, 250)
    if mask is None:
        x0, y0, x1, y1 = (float(v) for v in box)
        px0, py0 = int(x0 * w_img), int(y0 * h_img)
        px1, py1 = max(px0 + 1, int(x1 * w_img)), max(py0 + 1, int(y1 * h_img))
        crop_img = Image.fromarray(full[py0:py1, px0:px1, :], mode="RGB")
    else:
        h_grid, w_grid = mask.shape
        row_edges = np.linspace(0, h_img, h_grid + 1).astype(int)
        col_edges = np.linspace(0, w_img, w_grid + 1).astype(int)
        full_mask = np.zeros((h_img, w_img), dtype=bool)
        for r in range(h_grid):
            for c in range(w_grid):
                if mask[r, c]:
                    full_mask[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]] = True
        if not full_mask.any():
            return Image.new("RGB", size, pad)
        ys, xs = np.where(full_mask)
        py0, py1 = int(ys.min()), int(ys.max()) + 1
        px0, px1 = int(xs.min()), int(xs.max()) + 1
        crop = full[py0:py1, px0:px1, :]
        inside = full_mask[py0:py1, px0:px1, None]
        masked = np.where(inside, crop, np.array((210, 210, 210), dtype=np.uint8)[None, None, :])
        crop_img = Image.fromarray(masked, mode="RGB")
    target_w, target_h = size
    cw, ch = crop_img.size
    scale = min(target_w / cw, target_h / ch)
    new = crop_img.resize((max(1, int(cw * scale)), max(1, int(ch * scale))), Image.LANCZOS)
    cell = Image.new("RGB", size, pad)
    cell.paste(new, ((target_w - new.width) // 2, (target_h - new.height) // 2))
    return cell


def _saliency_overlay(image: Image.Image, saliency: np.ndarray) -> np.ndarray:
    """Image-resolution (H,W,3) uint8 inferno attention overlay: block-upsample the
    (side,side) per-patch saliency, colormap it, blend over a dimmed grayscale of the
    image (port of run_hac_tree_sweep._saliency_full_rgb)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")

    side_h, side_w = saliency.shape
    w_img, h_img = image.size
    sal = saliency.astype(np.float32)
    sal = sal / max(float(sal.max()), 1e-8)
    row_edges = np.linspace(0, h_img, side_h + 1).astype(int)
    col_edges = np.linspace(0, w_img, side_w + 1).astype(int)
    sal_full = np.zeros((h_img, w_img), dtype=np.float32)
    for r in range(side_h):
        for c in range(side_w):
            sal_full[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]] = sal[r, c]
    heat = (matplotlib.colormaps["inferno"](sal_full)[:, :, :3] * 255.0).astype(np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    base = (gray * 0.55 + 40.0)[:, :, None].repeat(3, axis=2)
    blended = base * 0.4 + heat.astype(np.float32) * 0.6
    return np.clip(blended, 0, 255).astype(np.uint8)


def _layout_tree(boxes: np.ndarray, children: np.ndarray, k: int):
    """(col, depth, ch, max_depth) over the HAC nodes (global indices 1..N-1; CLS at 0 is
    excluded — it's shown as the corner image). In-order DFS from the root (last node),
    visiting the child whose subtree holds the leftmost leaf first so left→right roughly
    tracks the image (port of run_hac_tree_sweep._layout_tree)."""
    n = len(boxes)
    ch = {g: (int(children[g][0]), int(children[g][1])) for g in range(1, n) if int(children[g][0]) >= 0}
    leaf_cx = {g: (float(boxes[g][0]) + float(boxes[g][2])) / 2.0 for g in range(1, n) if g not in ch}
    min_cx = dict(leaf_cx)
    depth = {g: 0 for g in leaf_cx}
    for g in range(1, n):  # increasing order: an internal's children have smaller indices
        if g in ch:
            a, b = ch[g]
            min_cx[g] = min(min_cx[a], min_cx[b])
            depth[g] = max(depth[a], depth[b]) + 1
    col: dict[int, float] = {}
    slot = [0]

    def visit(g: int) -> None:
        if g not in ch:
            col[g] = float(slot[0])
            slot[0] += 1
            return
        a, b = ch[g]
        first, second = (a, b) if min_cx[a] <= min_cx[b] else (b, a)
        visit(first)
        visit(second)
        col[g] = (col[first] + col[second]) / 2.0

    # Visit every root (a node no other node lists as a child). A proper HAC tree has one
    # (the last-built internal); a hierarchy-less cache (old npz whose ``children`` back-filled
    # to all -1) has every node as its own root, so they lay out as a flat row instead of
    # collapsing to a single node.
    referenced = {c for pair in ch.values() for c in pair}
    for r in sorted(g for g in range(1, n) if g not in referenced):
        visit(r)
    return col, depth, ch, (max(depth.values()) if depth else 0)


def _tree_panel(
    base_rgb: np.ndarray,
    corner_rgb: np.ndarray,
    boxes: np.ndarray,
    cell_masks: np.ndarray | None,
    children: np.ndarray,
    k: int,
    scores: list | None,
    matched: int | None,
    snapped: int | None,
    *,
    title: str,
    thumb: int = 84,
    gap_x: int = 8,
    gap_y: int = 30,
    margin: int = 14,
) -> Image.Image:
    """One HAC panel: masked-cell node thumbnails cut from ``base_rgb``, laid out by merge
    depth with parent→child edges, the full ``corner_rgb`` tucked in the top-left. Leaves
    ringed yellow / internals cyan, the surfacing match red and the snapped match blue, each
    node labeled with its MLP score."""
    col, depth, ch, max_depth = _layout_tree(boxes, children, k)
    cell_w = thumb + gap_x
    row_pitch = thumb + gap_y
    title_h = 18
    tree_top = title_h + margin
    # Width by the actual number of leaf columns: k for a proper tree, up to 2K-1 for a
    # hierarchy-less (old-cache) flat row — so nodes never overflow the canvas.
    n_cols = int(max(col.values())) + 1 if col else 1
    canvas_w = int(n_cols * cell_w + 2 * margin)
    canvas_h = int(tree_top + thumb + max_depth * row_pitch + margin + 12)  # +12 for score labels
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    nodes = list(col)  # global indices 1..N-1

    def center(g: int) -> tuple[int, int]:
        cx = int(margin + thumb / 2 + col[g] * cell_w)
        cy = int(tree_top + thumb / 2 + (max_depth - depth[g]) * row_pitch)
        return cx, cy

    # Corner image: largest scale that clears every node thumbnail (nodes leave the
    # upper-left empty), clamped to the canvas.
    corner = Image.fromarray(np.ascontiguousarray(corner_rgb, dtype=np.uint8), mode="RGB")
    iw, ih = corner.size
    caps = [max((cx - thumb / 2 - margin) / iw, (cy - thumb / 2 - tree_top) / ih) for cx, cy in map(center, nodes)]
    caps += [(canvas_w - 2 * margin) / iw, (canvas_h - margin - tree_top) / ih]
    scale = max(0.0, min(caps)) if caps else 0.0
    if scale > 0:
        cimg = corner.resize((max(1, int(iw * scale)), max(1, int(ih * scale))), Image.LANCZOS)
        canvas.paste(cimg, (margin, tree_top))
        # The CLS full-image node is global index 0, shown as this corner image (not a tree
        # node); ring it when the surfacing/snapped match is the whole-image node, and label
        # its score, so a CLS match is still visible.
        if matched == 0:
            outline, ow = _MATCH_COLOR, 3
        elif snapped == 0:
            outline, ow = _SNAP_COLOR, 3
        else:
            outline, ow = (60, 60, 60), 1
        draw.rectangle(
            (margin - 1, tree_top - 1, margin + cimg.width, tree_top + cimg.height), outline=outline, width=ow
        )
        if scores is not None and len(scores):
            draw.text((margin + 1, tree_top + 1), f"CLS {scores[0]:.2f}", fill=(255, 255, 255))

    draw.rectangle((0, 0, canvas_w, title_h), fill=(0, 0, 0))
    draw.text((4, 3), title, fill=(255, 255, 255))

    for g in nodes:  # edges first, thumbnails on top
        if g in ch:
            pcx, pcy = center(g)
            for cgi in ch[g]:
                ccx, ccy = center(cgi)
                draw.line([(pcx, pcy + thumb // 2), (ccx, ccy - thumb // 2)], fill=(120, 120, 120), width=1)

    for g in nodes:
        cx, cy = center(g)
        tx, ty = cx - thumb // 2, cy - thumb // 2
        mask = cell_masks[g] if cell_masks is not None else None
        canvas.paste(_cell_thumb(base_rgb, mask, boxes[g], (thumb, thumb)), (tx, ty))
        if g == matched:
            ring, wd = _MATCH_COLOR, 3
        elif g == snapped:
            ring, wd = _SNAP_COLOR, 3
        else:
            ring, wd = (_LEAF_RING if g not in ch else _INTERNAL_RING), 2
        draw.rectangle((tx, ty, tx + thumb, ty + thumb), outline=ring, width=wd)
        if scores is not None and g < len(scores):
            draw.text((tx + 1, ty + thumb + 1), f"{scores[g]:.2f}", fill=(20, 20, 20))
    return canvas


def _render_hac_composite(
    image: Image.Image,
    boxes: np.ndarray,
    cell_masks: np.ndarray | None,
    saliency: np.ndarray | None,
    children: np.ndarray,
    scores: list | None,
    matched: int | None,
    snapped: int | None,
    out_path: Path,
    caption: str,
) -> None:
    """The reference render_config_tree look, per labeling step: the HAC tree drawn twice
    side by side — left over the image pixels, right over the inferno attention overlay —
    with MLP scores + matched/snapped rings, captioned. Saved to ``out_path``."""
    n = len(boxes)
    if n < 2:  # need at least CLS + one leaf
        return
    k = n // 2  # CLS + K leaves + (K-1) internals = 2K nodes
    img_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    panels = [_tree_panel(img_rgb, img_rgb, boxes, cell_masks, children, k, scores, matched, snapped, title="image")]
    if saliency is not None:
        heat = _saliency_overlay(image, saliency)
        panels.append(
            _tree_panel(heat, heat, boxes, cell_masks, children, k, scores, matched, snapped, title="attention")
        )

    gap, bar = 12, 20
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    body_h = max(p.height for p in panels)
    out = Image.new("RGB", (total_w, body_h + bar), (250, 250, 250))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, total_w, bar), fill=(0, 0, 0))
    d.text((4, 5), caption, fill=(255, 255, 255))
    x = 0
    for i, p in enumerate(panels):
        out.paste(p, (x, bar))
        x += p.width
        if i < len(panels) - 1:
            d.line([(x + gap // 2, bar), (x + gap // 2, bar + body_h)], fill=(180, 180, 180), width=1)
            x += gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


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


def _render_overlays(
    ds: SodDataset,
    split,
    regions_dir: Path,
    predict,
    *,
    out_dir: Path,
    dataset: str,
    cls: str,
    embedder: str,
    proposal: str,
    alpha: float,
    gallery_n: int,
    label: str,
    thr: float | None = None,
    blend: tuple[float, int] | None = None,
) -> None:
    """Score the test set with ``predict``, bucket TP/FP/FN/TN at the threshold, and
    montage. Shared by the controlled and realistic prediction paths.

    Threshold: pass ``blend=(raw_thr, n_votes)`` to GMM-blend over the test score
    distribution (the controlled path's ``--safe-thresholds`` behavior), or a final
    ``thr`` directly (the realistic path passes the loop's already-blended threshold,
    which is what the sweep row used). ``label`` is the filename/title suffix
    (e.g. ``"k8"`` or ``"realistic_t50"``).
    """
    # Pass 1: score every test image (winning region), collecting the distribution
    # so a GMM blend (if requested) fits over the same scores.
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

    if blend is not None:
        from vtscore.training.thresholds import calculate_safe_threshold

        raw_thr, n_votes = blend
        thr = calculate_safe_threshold(raw_thr, all_scores, n_votes)
    if thr is None:
        thr = 0.5

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
    tag_out = out_dir / f"{dataset}_{slugify(cls)}_{embedder}_{proposal}{alpha_tag}_mlp_{label}"
    print(
        f"  [predict] {embedder}/{proposal} {label} thr={thr:.3f}: "
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
            f"{embedder}/{proposal} MLP {label} thr={thr:.3f} — {tag} ({len(items)})",
        )


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

    _render_overlays(
        ds,
        split,
        regions_dir,
        predict,
        out_dir=out_dir,
        dataset=dataset,
        cls=cls,
        embedder=embedder,
        proposal=proposal,
        alpha=alpha,
        gallery_n=gallery_n,
        label=f"k{k}",
        blend=(raw_thr, n_votes) if safe_thresholds else None,
        thr=raw_thr,
    )


def render_predictions_realistic(
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
    predict,
    thr: float,
    t: int,
    gallery_n: int,
) -> None:
    """Prediction overlays for a realistic run's FINAL detector (at the max ``t``).

    ``predict``/``thr`` come from ``evaluate_realistic_curve(return_finals=True)`` — the
    in-memory head and its loop-final blended threshold (used as-is, not re-blended over
    the test set). Test region vectors are read from the same npz cache the sweep wrote.
    """
    reg = _EMBEDDER_ALIASES.get(embedder, embedder)
    regions_root = cache_dir / "regions" / dataset / reg
    slug = _resolve_slug(regions_root, proposal, alpha, slug)
    if slug is None:
        print(f"  [predict] skip {embedder}/{proposal}: no cache under {regions_root}", flush=True)
        return
    _render_overlays(
        ds,
        split,
        regions_root / slug,
        predict,
        out_dir=out_dir,
        dataset=dataset,
        cls=cls,
        embedder=embedder,
        proposal=proposal,
        alpha=alpha,
        gallery_n=gallery_n,
        label=f"realistic_t{t}",
        thr=thr,
    )


def _load_region_viz(regions_dir: Path, iid: int):
    """Per-image HAC geometry for the trace viz, straight from the region npz:
    ``(boxes (N,4), children (N,2), cell_masks (N,H,W)|None, saliency (H,W)|None)``.
    Returns ``None`` if the npz is missing; ``cell_masks``/``saliency`` are ``None`` for
    caches written before this feature (renderer degrades: box crops / no attention)."""
    path = regions_dir / f"{iid}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        boxes = z["boxes"]
        children = z["children"] if "children" in z else np.full((boxes.shape[0], 2), -1, dtype=int)
        cell_masks = z["cell_masks"] if "cell_masks" in z else None
        saliency = z["saliency"] if "saliency" in z else None
    return boxes, children, cell_masks, saliency


def render_labeling_trace(
    ds: SodDataset,
    split,
    trace: list[dict],
    *,
    cache_dir: Path,
    slug: str | None,
    out_dir: Path,
    dataset: str,
    cls: str,
    embedder: str,
    proposal: str,
    alpha: float,
    seed: int,
    images: bool = True,
) -> None:
    """Write one seed's labeling trace: numbered images in the order the realistic loop
    labeled them, plus ``trace.csv`` / ``trace.json``.

    With ``images=False`` only ``trace.json`` / ``trace.csv`` are written (the per-step
    PNGs — the bulk of the output, ~2 files/step — are skipped). Use this when you only
    need the per-step vote/threshold/cost record, e.g. the threshold-stability spike
    analysis or Stage-A replay, and can't afford tens of thousands of images on a
    space-tight volume. ``snapped_region`` is left blank in that mode (it is computed
    during image rendering). Per step (``images=True``), two images (named
    ``{t:03d}_{iid}_{good|bad}_*`` so they sort in labeling order):

    * ``…_pred.png`` — GT green + the detector's surfacing box/score red (+ good-vote snapped
      region blue). For a **good** vote it also appends, on the right, the snapped node's own
      patches (a colour cell-crop) — the exact sub-image that becomes the next MLP's training positive.
    * ``…_hac.png``  — the reference-style HAC composite (see :func:`_render_hac_composite`):
      the region tree drawn twice — masked-cell pixel thumbnails on the left, the same tree
      over the inferno attention overlay on the right — with each node's MLP score and the
      surfacing match (red) / good-vote snapped match (blue) ringed.

    HAC geometry (region boxes, HAC children, per-node cell masks, patch saliency) is read
    from the cell's region npz under ``regions_dir``; only the head-dependent per-region MLP
    scores + matched index come from the trace. The snapped good-vote match (best-IoU node
    vs the GT covering box) is computed here from the npz boxes + class GT (good votes only)."""
    import csv  # noqa: PLC0415
    import json  # noqa: PLC0415

    alpha_tag = f"_a{alpha}" if proposal == "hac" else ""
    d = out_dir / f"{dataset}_{slugify(cls)}_{embedder}_{proposal}{alpha_tag}" / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)

    # Resolve the same region-npz dir the sweep wrote (holds boxes/children/cell_masks/saliency).
    regions_root = cache_dir / "regions" / dataset / _EMBEDDER_ALIASES.get(embedder, embedder)
    resolved = _resolve_slug(regions_root, proposal, alpha, slug)
    regions_dir = regions_root / resolved if resolved is not None else None

    (d / "trace.json").write_text(json.dumps(trace, indent=2))
    snapped_by_t: dict[int, int | None] = {}

    for e in trace if images else []:
        iid = e["image_id"]
        try:
            img = ds.load_image(iid)
        except Exception:
            continue
        stem = f"{e['t']:03d}_{iid}_{e['gt_label']}"
        gt = split.gt_boxes.get(iid, []) if e["gt_label"] == "good" else []
        pred = tuple(e["pred_box"]) if e.get("pred_box") else None
        # ``surf`` = the surfacing head's top-region score (the head that CHOSE this image,
        # i.e. the previous step). ``surf_thr`` = that same head's decision threshold
        # (score − margin) — compare the node scores against THIS, not ``thr``. ``thr`` is
        # the post-label head's threshold (retrained after this image was labeled), shown
        # for reference; the two heads differ by one vote.
        surf = ""
        if e.get("surface_score") is not None:
            surf = f" surf={e['surface_score']:.2f}"
            if e.get("surface_margin") is not None:
                surf += f" surf_thr={e['surface_score'] - e['surface_margin']:.2f}"
        caption = (
            f"t{e['t']} id{iid} {e['gt_label']} | {e['select_mode']}->{e['phase']} | "
            f"{e['head']}/{e['calib_mode']} thr={e['threshold']:.2f}{surf}"
        )
        # Load the region npz first (holds the boxes) so the snapped good-vote region — the
        # patch that becomes the next MLP's training positive — can be drawn on BOTH outputs.
        viz = _load_region_viz(regions_dir, iid) if regions_dir is not None else None
        snapped = None
        snap_box = None
        if viz is not None:
            boxes, children, cell_masks, saliency = viz
            snapped = _snapped_index(gt, [tuple(b) for b in boxes]) if gt else None
            if snapped is not None:
                snap_box = tuple(float(v) for v in boxes[snapped])
        snapped_by_t[e["t"]] = snapped
        # 1. Single-pred overlay: GT green + surfacing region red + snapped (next-MLP) region blue.
        #    For a GOOD vote, append the snapped node's own patches (colour cell-crop) on the right,
        #    so you see exactly the sub-image that becomes the training positive.
        pred_img = _draw(img, gt, pred, e.get("surface_score"), snap=snap_box)
        if snapped is not None and viz is not None:
            mask = cell_masks[snapped] if cell_masks is not None else None
            node = _cell_thumb(
                np.asarray(img.convert("RGB"), dtype=np.uint8), mask, boxes[snapped], (img.height, img.height)
            )
            pred_img = _hstack2(pred_img, node)
        _save_captioned(pred_img, d / f"{stem}_pred.png", caption)
        # 2. The reference-style HAC composite (needs the region npz for this image).
        if viz is None:
            continue
        matched = e.get("matched_region")
        _render_hac_composite(
            img, boxes, cell_masks, saliency, children, e.get("region_scores"),
            matched, snapped, d / f"{stem}_hac.png",
            caption + f" | match={matched} snap={snapped}",
        )  # fmt: skip

    if trace:
        keys = [
            "t", "image_id", "gt_label", "select_mode", "phase", "head", "calib_mode", "threshold",
            "surface_score", "surface_margin", "pred_box", "matched_region", "snapped_region",
            "n_good", "n_bad", "n_votes",
            "smart", "stable", "span", "cost", "fpr", "fnr", "f1", "stop_recommended",
        ]  # fmt: skip
        with (d / "trace.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for e in trace:
                row = dict(e)
                pb = e.get("pred_box")
                row["pred_box"] = " ".join(f"{v:.4f}" for v in pb) if pb else ""
                row["snapped_region"] = snapped_by_t.get(e["t"])
                w.writerow(row)
    print(f"  [trace] {embedder}/{proposal} seed{seed}: {len(trace)} steps -> {d}", flush=True)


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
    ap.add_argument(
        "--neg-multiple", type=int, default=10, help="neg pool = neg_multiple × positives (match the sweep)"
    )
    ap.add_argument(
        "--min-box-frac",
        type=float,
        default=GUI_MIN_BOX_FRAC,
        help="drop GT boxes below this fraction of the image on either axis (match the sweep; 0 disables)",
    )
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
        cs = ds.class_split(
            args.cls, neg_multiple=args.neg_multiple, seed=args.split_seed, min_box_frac=args.min_box_frac
        )
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
