#!/usr/bin/env python3
"""Crop-based CLIP vs whole-image for small-object detection on Visual Genome.

Tests whether scoring overlapping crops with a single-vector embedder (SigLIP)
beats scoring the whole image for "does this image match the prompt?".  The
intuition: a tight crop zooms a sub-resolution object up to where SigLIP can
resolve it, whereas a whole-image vector averages it into the background.

Three methods, all text-query (cosine between SigLIP's text embedding of the
``--prompt`` and each image/crop vector):

* ``whole``   - one SigLIP vector for the full image (single-vector baseline)
* ``sliding`` - multiscale overlapping square crops; score = max cos over crops
* ``dino``    - crops are the DINO/HAC region boxes (learned proposals)

Input is a **directory of positive images whose filenames are VG image ids**
(e.g. from ``annotate_vg_noun.py`` / a train-test split).  The directory is used
only for the id list; the **clean** pixels are re-fetched from the VG zips by id
(so drawn-on annotation boxes never contaminate the embedding).

* Positives only (default): reports each method's score distribution over the
  positives (mean/median/min/max) - useful for comparing methods relatively.
* With ``--neg-dir`` (another id directory): also reports presence metrics
  AP / AUROC / best-F1 over positives-vs-negatives.
* With ``--viz-dir``: writes per-image overlays drawing every crop box colored
  by its confidence (red=low -> green=high), top-K labeled numerically.
  ``sliding`` is split one file per scale (``<viz-dir>/sliding/<id>_s<scale>.png``)
  sharing one global color scale; ``dino`` is ``<viz-dir>/dino/<id>.png``.

Reads the dataset staged at ``/exp/scale26/datasets/external/VisualGenome``.

GPU run::

    srun --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 --mem=46G --time=1:00:00 \\
        /exp/mlucio/projects/VTSearch/.venv/bin/python \\
        scripts/vg/eval_crop_clip.py data/vg/vg_hat_100_max --prompt "a photo of a hat" \\
        --out /exp/$USER/crop_clip_hat.json

CPU run (small image/crop counts only)::

    srun --partition=cpu --cpus-per-task=4 --mem=16G --time=0:30:00 \\
        /exp/mlucio/projects/VTSearch/.venv/bin/python \\
        scripts/vg/eval_crop_clip.py data/vg/vg_hat_100_max --device cpu \\
        --prompt "a photo of a hat" --scales 0.5,0.25

Matthew Usage::

    cd /exp/mlucio/projects/VTSearch
    source .venv/bin/activate
    srun --partition=cpu --cpus-per-task=4 --mem=16G --time=0:30:00 bash -lc 'export HF_HOME=/exp/$USER/.cache/huggingface OMP_NUM_THREADS=4; python ./scripts/vg/eval_crop_clip.py ./data/vg/vg_hat_100_max/all --device cpu --prompt "a photo of a hat" --scales 1,0.7,0.4 --viz-dir ./data/vg/vg_hat_100_max'
"""

from __future__ import annotations

import argparse
import io
import json
import statistics as st
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score

from vtscore.config import DINOV2_MODEL_ID, SIGLIP_MODEL_ID
from vtscore.eval.metrics import compute_average_precision, compute_binary_classification_metrics
from vtscore.media.patch_embed import build_region_tree, hf_vit_to_patch_output

DEFAULT_VG_DIR = Path("/exp/scale26/datasets/external/VisualGenome")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
Box = tuple[float, float, float, float]


def _pooled(out: object) -> torch.Tensor:
    """SigLIP ``get_*_features`` returns a tensor in some transformers versions
    and a ``BaseModelOutputWithPooling`` in others; normalize to the tensor."""
    return out if torch.is_tensor(out) else out.pooler_output


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------
def load_image_dims(image_data_path: Path) -> dict[int, tuple[int, int]]:
    """image_id -> (width, height) from image_data.json."""
    with image_data_path.open("rb") as fh:
        data = json.load(fh)
    dims: dict[int, tuple[int, int]] = {}
    for entry in data:
        iid = entry.get("image_id")
        w = entry.get("width")
        h = entry.get("height")
        if iid is not None and w and h:
            dims[int(iid)] = (int(w), int(h))
    return dims


def build_member_index(zip_paths: list[Path]) -> dict[int, tuple[Path, str]]:
    """image_id -> (zip_path, member) over all archives (keyed on int stem)."""
    index: dict[int, tuple[Path, str]] = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for member in zf.namelist():
                if not member.endswith(".jpg"):
                    continue
                try:
                    index[int(Path(member).stem)] = (zp, member)
                except ValueError:
                    continue
    return index


def ids_from_dir(d: Path) -> set[int]:
    """Integer VG image ids parsed from a directory's image filenames."""
    ids: set[int] = set()
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            try:
                ids.add(int(p.stem))
            except ValueError:
                continue
    return ids


def read_pil(members: dict[int, tuple[Path, str]], iid: int) -> Image.Image:
    """Read a clean JPEG in-memory from its VG zip member."""
    zp, member = members[iid]
    with zipfile.ZipFile(zp) as zf:
        raw = zf.read(member)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# --------------------------------------------------------------------------
# Crop generation
# --------------------------------------------------------------------------
def sliding_boxes_by_scale(
    w: int, h: int, scales: list[float], overlap: float, min_window: int
) -> dict[float, list[Box]]:
    """Multiscale square windows (pixel coords), grouped by scale."""
    short = min(w, h)
    out: dict[float, list[Box]] = {}
    for f in scales:
        side = min(int(round(f * short)), short)
        if side < min_window:
            continue
        stride = max(1, int(round(side * (1.0 - overlap))))
        xs = list(range(0, max(1, w - side + 1), stride))
        ys = list(range(0, max(1, h - side + 1), stride))
        if w - side > 0 and xs[-1] != w - side:
            xs.append(w - side)
        if h - side > 0 and ys[-1] != h - side:
            ys.append(h - side)
        boxes = [(float(x), float(y), float(x + side), float(y + side)) for x in xs for y in ys]
        if boxes:
            out[f] = list(dict.fromkeys(boxes))
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class Siglip:
    """SigLIP image/text features via transformers, L2-normalized."""

    def __init__(self, model_id: str, device: str, batch_size: int) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_id).eval().to(device)
        self.proc = AutoProcessor.from_pretrained(model_id)

    def text_vec(self, prompt: str) -> np.ndarray:
        inputs = self.proc(text=[prompt], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = _pooled(self.model.get_text_features(**inputs))
        v = feat[0].float().cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-12)

    def image_feats(self, images: list[Image.Image]) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            inputs = self.proc(images=chunk, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feat = _pooled(self.model.get_image_features(**inputs))
            out.append(feat.float().cpu().numpy())
        feats = np.concatenate(out, axis=0) if out else np.zeros((0, 1), np.float32)
        return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)


class Dino:
    """DINOv2 patch forward -> HAC region boxes (normalized)."""

    def __init__(self, model_id: str, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.model = AutoModel.from_pretrained(model_id, attn_implementation="eager").eval().to(device)
        self.proc = AutoImageProcessor.from_pretrained(model_id)

    def region_boxes(self, img: Image.Image) -> list[Box]:
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        pe = hf_vit_to_patch_output(outputs, num_register_tokens=0)
        if pe is None:
            return []
        return [r.box for r in build_region_tree(pe, k=12, alpha=0.5)]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def crops_from_boxes(img: Image.Image, boxes_px: list[Box]) -> list[Image.Image]:
    return [img.crop((int(b[0]), int(b[1]), int(b[2]), int(b[3]))) for b in boxes_px]


def crop_sims(img: Image.Image, boxes_px: list[Box], siglip: Siglip, q: np.ndarray) -> np.ndarray:
    """Cosine of each crop against the query (empty array when no crops)."""
    if not boxes_px:
        return np.zeros(0, dtype=np.float64)
    return siglip.image_feats(crops_from_boxes(img, boxes_px)) @ q


# --------------------------------------------------------------------------
# Visualization (--viz-dir)
# --------------------------------------------------------------------------
def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=12)
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None  # type: ignore[return-value]


def _heat_color(t: float) -> tuple[int, int, int]:
    """Red (low) -> yellow -> green (high) for t in [0, 1]."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    if t < 0.5:
        return (255, int(round(510 * t)), 0)
    return (int(round(510 * (1 - t))), 255, 0)


def render_crops_viz(
    img: Image.Image,
    boxes_px: list[Box],
    sims: np.ndarray,
    label_topk: int,
    font: ImageFont.ImageFont,
    lo: float | None = None,
    hi: float | None = None,
) -> Image.Image:
    """Draw every crop box colored by confidence; label the top-K (all if 0).

    ``lo``/``hi`` set the color-normalization range; pass the global min/max so
    per-scale images share one color scale (green = globally high)."""
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    lo = float(sims.min()) if lo is None else lo
    hi = float(sims.max()) if hi is None else hi
    span = hi - lo if hi > lo else 1.0
    order = np.argsort(-sims)
    label_idx = set(range(len(sims))) if label_topk <= 0 else set(order[:label_topk].tolist())
    best = int(order[0])
    # Draw lower-confidence boxes first so hot/labeled ones sit on top.
    for i in reversed(order.tolist()):
        x0, y0, x1, y1 = boxes_px[i]
        color = _heat_color((sims[i] - lo) / span)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3 if i == best else 1)
        if i in label_idx:
            txt = f"{sims[i]:.3f}"
            tx, ty = int(x0) + 2, int(y0) + 2
            try:
                tb = draw.textbbox((tx, ty), txt, font=font)
                draw.rectangle(tb, fill=(0, 0, 0))
            except Exception:
                pass
            draw.text((tx, ty), txt, fill=color, font=font)
    return canvas


def summarize(scores: list[float]) -> dict[str, float]:
    finite = [s for s in scores if np.isfinite(s)]
    if not finite:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(st.mean(finite)),
        "median": float(st.median(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    """Sweep thresholds; return (f1, precision, recall, threshold) at the best F1."""
    labels_list = labels.tolist()
    best = (0.0, 0.0, 0.0, 0.0)
    for t in np.unique(scores):
        preds = (scores >= t).astype(int).tolist()
        _acc, p, r, f1 = compute_binary_classification_metrics(preds, labels_list)
        if f1 > best[0]:
            best = (f1, p, r, float(t))
    return best


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:  # noqa: C901 - a single linear experiment driver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pos_dir", type=Path, help="directory of positive images (filenames are VG image ids)")
    ap.add_argument(
        "--neg-dir", type=Path, default=None, help="optional directory of negative images; enables AP/AUROC/F1"
    )
    ap.add_argument("--prompt", required=True, help='text query, e.g. "a photo of a hat"')
    ap.add_argument("--vg-dir", type=Path, default=DEFAULT_VG_DIR)
    ap.add_argument("--methods", default="whole,sliding,dino", help="comma list of: whole,sliding,dino")
    ap.add_argument("--scales", default="1.0,0.5,0.25", help="sliding-window sizes as fraction of min side")
    ap.add_argument("--min-window", type=int, default=64, help="drop sliding windows smaller than this (px)")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--siglip-model", default=SIGLIP_MODEL_ID)
    ap.add_argument("--dino-model", default=DINOV2_MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument(
        "--out", type=Path, default=None, help="results JSON (defaults to <viz-dir>/results.json when --viz-dir is set)"
    )
    ap.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="write per-image overlays of every crop box colored by confidence to <viz-dir>/<method>/",
    )
    ap.add_argument(
        "--viz-label-topk",
        type=int,
        default=8,
        help="number of highest-confidence crops to label numerically in the viz (0 = label all)",
    )
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    scales = [float(s) for s in args.scales.split(",")]
    if args.out is None and args.viz_dir is not None:
        args.out = args.viz_dir / "results.json"
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    image_data = args.vg_dir / "annotations" / "image_data.json"
    zips = [args.vg_dir / "images" / "images.zip", args.vg_dir / "images" / "images2.zip"]
    for p in [image_data, *zips]:
        if not p.exists():
            raise SystemExit(f"missing {p}\nrun fetch_visual_genome.sbatch first.")
    if not args.pos_dir.is_dir():
        raise SystemExit(f"not a directory: {args.pos_dir}")

    print(f"indexing dataset (device={device})…", flush=True)
    dims = load_image_dims(image_data)
    members = build_member_index(zips)
    have = set(members) & set(dims)

    pos = sorted(ids_from_dir(args.pos_dir) & have)
    if not pos:
        raise SystemExit(f"no VG-id images found in {args.pos_dir}")
    neg = sorted(ids_from_dir(args.neg_dir) & have) if args.neg_dir else []
    eval_ids = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))
    print(f"  {len(pos)} positives" + (f" + {len(neg)} negatives" if neg else " (positives only)"), flush=True)

    siglip = Siglip(args.siglip_model, device, args.batch_size)
    q = siglip.text_vec(args.prompt)
    dino = Dino(args.dino_model, device) if "dino" in methods else None

    score: dict[str, list[float]] = {m: [] for m in methods}
    crop_count: dict[str, int] = dict.fromkeys(methods, 0)
    viz_font = _font() if args.viz_dir else None
    if args.viz_dir:
        for m in ("sliding", "dino"):
            if m in methods:
                (args.viz_dir / m).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for n, iid in enumerate(eval_ids, 1):
        img = read_pil(members, iid)
        w, h = img.size
        if "whole" in methods:
            score["whole"].append(float((siglip.image_feats([img]) @ q)[0]))
            crop_count["whole"] += 1
        if "sliding" in methods:
            by_scale = sliding_boxes_by_scale(w, h, scales, args.overlap, args.min_window)
            flat = [b for boxes in by_scale.values() for b in boxes]
            sims = crop_sims(img, flat, siglip, q)
            score["sliding"].append(float(np.max(sims)) if len(sims) else float("-inf"))
            crop_count["sliding"] += len(flat)
            if args.viz_dir and len(sims):
                # One overlay per scale, sharing one global color scale.
                lo, hi = float(sims.min()), float(sims.max())
                off = 0
                for sc, boxes in by_scale.items():
                    seg = sims[off : off + len(boxes)]
                    off += len(boxes)
                    viz = render_crops_viz(img, boxes, seg, args.viz_label_topk, viz_font, lo, hi)
                    viz.save(args.viz_dir / "sliding" / f"{iid}_s{sc:g}.png")
        if dino is not None:
            boxes = [(b[0] * w, b[1] * h, b[2] * w, b[3] * h) for b in dino.region_boxes(img)]
            sims = crop_sims(img, boxes, siglip, q)
            score["dino"].append(float(np.max(sims)) if len(sims) else float("-inf"))
            crop_count["dino"] += len(boxes)
            if args.viz_dir and len(sims):
                render_crops_viz(img, boxes, sims, args.viz_label_topk, viz_font).save(
                    args.viz_dir / "dino" / f"{iid}.png"
                )
        if n % 200 == 0:
            print(f"  scored {n}/{len(eval_ids)}…", flush=True)
    elapsed = time.perf_counter() - t0

    # ---- metrics ----
    results: dict[str, dict] = {}
    pos_set = set(pos)
    for m in methods:
        scores = np.array(score[m], dtype=np.float64)
        rec: dict[str, object] = {"crops": crop_count[m], **summarize(score[m])}
        if neg:
            order = np.argsort(-scores, kind="stable")
            ranked_ids = [eval_ids[i] for i in order]
            rec["ap"] = compute_average_precision(ranked_ids, pos_set)
            rec["auroc"] = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else float("nan")
            f1, prec, recall, _thr = best_f1(scores, labels)
            rec.update({"best_f1": f1, "precision": prec, "recall": recall})
        results[m] = rec

    # ---- table ----
    mode = f"{len(pos)} pos + {len(neg)} neg" if neg else f"{len(pos)} positives"
    print(f"\n=== prompt={args.prompt!r}  {mode}  device={device}  {elapsed:.1f}s ===")
    if neg:
        print(f"{'method':<9}{'AP':>8}{'AUROC':>8}{'bestF1':>8}{'prec':>7}{'rec':>7}{'mean':>8}{'crops':>9}")
        for m in methods:
            r = results[m]
            print(
                f"{m:<9}{r['ap']:>8.3f}{r['auroc']:>8.3f}{r['best_f1']:>8.3f}"
                f"{r['precision']:>7.2f}{r['recall']:>7.2f}{r['mean']:>8.3f}{r['crops']:>9,}"
            )
    else:
        print(f"{'method':<9}{'mean':>8}{'median':>8}{'min':>8}{'max':>8}{'crops':>9}")
        for m in methods:
            r = results[m]
            print(f"{m:<9}{r['mean']:>8.3f}{r['median']:>8.3f}{r['min']:>8.3f}{r['max']:>8.3f}{r['crops']:>9,}")

    if args.out:
        payload = {
            "prompt": args.prompt,
            "device": device,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "scales": scales,
            "elapsed_s": elapsed,
            "results": results,
            "per_image": {str(iid): {m: score[m][i] for m in methods} for i, iid in enumerate(eval_ids)},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
