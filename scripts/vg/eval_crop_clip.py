#!/usr/bin/env python3
"""Crop-based CLIP vs whole-image for small-object detection on Visual Genome.

Tests whether scoring overlapping crops with a single-vector embedder (SigLIP)
beats scoring the whole image for "does this image match the prompt?".  The
intuition: a tight crop zooms a sub-resolution object up to where SigLIP can
resolve it, whereas a whole-image vector averages it into the background.

Methods (text-query: cosine between the model's text embedding of ``--prompt``
and each image/crop vector; image score = max over crops):

* ``whole``          - one SigLIP vector for the full image (single-vector baseline)
* ``sliding_siglip`` - multiscale overlapping square crops, SigLIP-embedded
* ``sliding_clip``   - the same crops, CLIP-embedded (CLIP vs SigLIP for crops)
* ``dino_v2``        - DINOv2 proposes HAC region boxes; crops SigLIP-scored
* ``dino_v3``        - DINOv3 proposes the region boxes (gated; needs HF_TOKEN), SigLIP-scored

DINOv2/v3 are vision-only (no text encoder), so they are used only to *propose*
region crops; scoring is always SigLIP text-cosine.

Input is a **directory of positive images whose filenames are VG image ids**
(e.g. from ``annotate_vg_noun.py`` / a train-test split).  The directory is used
only for the id list; the **clean** pixels are re-fetched from the VG zips by id
(so drawn-on annotation boxes never contaminate the embedding).

* Positives only (default): reports each method's score distribution over the
  positives (mean/median/min/max) - useful for comparing methods relatively.
* With ``--neg-dir`` (another id directory): also reports presence metrics
  AP / AUROC / best-F1 over positives-vs-negatives.
* With ``--viz-dir``: writes per-image overlays under ``<viz-dir>/<method>/``,
  every crop box colored by confidence (red=low -> green=high), top-K labeled.
  sliding methods split one file per scale (``<id>_s<scale>.png``, shared color
  scale); dino methods write one ``<id>.png``.

Scores are SigLIP-calibrated confidences ``sigmoid(t*cos + b)`` in [0, 1] by
default (raw SigLIP cosines sit in a tiny band, match boundary ~0.11); pass
``--score cosine`` for the bare cosine. The transform is monotonic, so
ranking/AP/AUROC are identical either way - it only changes the readability of
the reported numbers and viz labels.

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

    CPU
    
    cd /exp/mlucio/projects/VTSearch
    source .venv/bin/activate
    echo $HF_TOKEN # make sure set
    echo $HF_HOME # make sure set
    export OMP_NUM_THREADS=4
    srun --partition=cpu --cpus-per-task=4 --mem=16G --time=0:30:00 python ./scripts/vg/eval_crop_clip.py ./data/vg/vg_hat_100_max/test --device cpu --prompt "hat" --scales 1,0.75,0.5 --score cosine --viz-dir ./data/vg/vg_hat_100_max/test/eval_hat_all
    
    GPU
    
    cd /exp/mlucio/projects/VTSearch
    source .venv/bin/activate
    echo $HF_TOKEN # make sure set
    echo $HF_HOME # make sure set
    srun --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=4 --mem=16G --time=1:00:00 python ./scripts/vg/eval_crop_clip.py ./data/vg/vg_hat_100_max/test --prompt "hat" --scales 1,0.75,0.5,0.3 --score cosine --viz-dir ./data/vg/vg_hat_100_max/test/eval_hat_all
    
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

from vtscore.config import CLIP_MODEL_ID, DINOV2_MODEL_ID, DINOV3_MODEL_ID, SIGLIP_MODEL_ID
from vtscore.eval.metrics import compute_average_precision, compute_binary_classification_metrics

# sliding_boxes_by_scale / crops_from_boxes now live in the shared region-source
# core so the sweep and this script use one implementation.
from vtscore.eval.region_sources import crops_from_boxes, sliding_boxes_by_scale
from vtscore.eval._hac_compat import build_region_tree
from vtscore.media.patch_embed import hf_vit_to_patch_output

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
# Crop generation: sliding_boxes_by_scale is imported from
# vtscore.eval.region_sources (shared with the sweep).
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class TextImageEmbedder:
    """SigLIP- or CLIP-style image/text features via transformers, L2-normalized."""

    def __init__(self, model_id: str, device: str, batch_size: int, text_padding: object = "max_length") -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.batch_size = batch_size
        # SigLIP tokenizer needs padding="max_length"; CLIP uses padding=True.
        self.text_padding = text_padding
        self.model = AutoModel.from_pretrained(model_id).eval().to(device)
        self.proc = AutoProcessor.from_pretrained(model_id)
        # SigLIP decides matches via sigmoid(t*cos + b), so raw cosines sit in a
        # tiny band (boundary at cos = -b/t ≈ 0.11); expose t, b to map cosine ->
        # [0,1]. CLIP has logit_scale but no logit_bias (softmax, not per-pair
        # sigmoid), so calib is None there and confidence() returns raw cosine.
        ls = getattr(self.model, "logit_scale", None)
        lb = getattr(self.model, "logit_bias", None)
        self.calib: tuple[float, float] | None = (
            (float(ls.detach().exp()), float(lb.detach())) if ls is not None and lb is not None else None
        )

    def confidence(self, cos: np.ndarray) -> np.ndarray:
        """Map raw cosine -> calibrated [0,1] via sigmoid(t*cos + b).

        Falls back to the raw cosine if the model exposes no logit_scale/bias
        (e.g. CLIP). Monotonic in cosine, so ranking/AP is unchanged."""
        if self.calib is None:
            return cos
        t, b = self.calib
        return 1.0 / (1.0 + np.exp(-(t * cos + b)))

    def text_vec(self, prompt: str) -> np.ndarray:
        inputs = self.proc(text=[prompt], padding=self.text_padding, return_tensors="pt").to(self.device)
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


class DinoProposer:
    """DINOv2/v3 patch forward -> HAC region boxes (normalized).

    Proposal only; the proposed crops are scored by SigLIP. DINOv3 carries
    register tokens (skip them) and is gated on HF (needs a token)."""

    def __init__(self, model_id: str, device: str, num_register_tokens: int = 0) -> None:
        from transformers import AutoImageProcessor, AutoModel

        from vtscore.media.embedder import hf_token

        self.device = device
        self.num_register_tokens = num_register_tokens
        self.model = (
            AutoModel.from_pretrained(model_id, attn_implementation="eager", token=hf_token() or None).eval().to(device)
        )
        self.proc = AutoImageProcessor.from_pretrained(model_id, token=hf_token() or None)

    def region_boxes(self, img: Image.Image) -> list[Box]:
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        pe = hf_vit_to_patch_output(outputs, num_register_tokens=self.num_register_tokens)
        if pe is None:
            return []
        return [r.box for r in build_region_tree(pe, k=12, alpha=0.5)]


# --------------------------------------------------------------------------
# Scoring (crops_from_boxes imported from vtscore.eval.region_sources)
# --------------------------------------------------------------------------
def crop_sims(img: Image.Image, boxes_px: list[Box], emb: "TextImageEmbedder", q: np.ndarray) -> np.ndarray:
    """Cosine of each crop against the query (empty array when no crops)."""
    if not boxes_px:
        return np.zeros(0, dtype=np.float64)
    return emb.image_feats(crops_from_boxes(img, boxes_px)) @ q


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
    ap.add_argument(
        "--methods",
        default="whole,sliding_siglip,sliding_clip,dino_v2,dino_v3",
        help=(
            "comma list of: whole, sliding_siglip, sliding_clip, dino_v2, dino_v3. "
            "dino_v2/dino_v3 propose regions (DINOv2/DINOv3) scored by SigLIP; "
            "dino_v3 is gated (needs HF_TOKEN)."
        ),
    )
    ap.add_argument("--scales", default="1.0,0.5,0.25", help="sliding-window sizes as fraction of min side")
    ap.add_argument("--min-window", type=int, default=64, help="drop sliding windows smaller than this (px)")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--siglip-model", default=SIGLIP_MODEL_ID)
    ap.add_argument("--clip-model", default=CLIP_MODEL_ID, help="CLIP model for the sliding_clip method")
    ap.add_argument("--dinov2-model", default=DINOV2_MODEL_ID, help="DINOv2 model for the dino_v2 proposer")
    ap.add_argument("--dinov3-model", default=DINOV3_MODEL_ID, help="DINOv3 model for the dino_v3 proposer (gated)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument(
        "--score",
        choices=("sigmoid", "cosine"),
        default="sigmoid",
        help="report SigLIP-calibrated sigmoid(t*cos+b) confidence in [0,1] (default) or raw cosine",
    )
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

    # SigLIP scores every method's crops; CLIP scores sliding_clip; DINOv2/v3
    # only PROPOSE region boxes for dino_v2/dino_v3 (still SigLIP-scored).
    siglip = TextImageEmbedder(args.siglip_model, device, args.batch_size, "max_length")
    q_sig = siglip.text_vec(args.prompt)
    clip = TextImageEmbedder(args.clip_model, device, args.batch_size, True) if "sliding_clip" in methods else None
    q_clip = clip.text_vec(args.prompt) if clip is not None else None
    dino_v2 = DinoProposer(args.dinov2_model, device, 0) if "dino_v2" in methods else None
    dino_v3 = None
    if "dino_v3" in methods:
        try:
            dino_v3 = DinoProposer(args.dinov3_model, device, 4)
        except Exception as exc:  # gated model: missing/invalid HF_TOKEN, etc.
            print(f"  (dino_v3 unavailable: {exc}; set HF_TOKEN for the gated DINOv3 — skipping)", flush=True)

    if args.score == "sigmoid" and siglip.calib is None:
        print("  (siglip exposes no logit_scale/bias; falling back to raw cosine)", flush=True)

    def score_of(emb: TextImageEmbedder, cos: np.ndarray) -> np.ndarray:
        # sigmoid(t*cos+b) for SigLIP; CLIP has no bias so confidence() returns cosine.
        return emb.confidence(cos) if args.score == "sigmoid" else cos

    # method -> (embedder, query_vec, crop_source, region_proposer|None)
    known = {
        "whole": (siglip, q_sig, "whole", None),
        "sliding_siglip": (siglip, q_sig, "sliding", None),
        "sliding_clip": (clip, q_clip, "sliding", None),
        "dino_v2": (siglip, q_sig, "dino", dino_v2),
        "dino_v3": (siglip, q_sig, "dino", dino_v3),
    }
    unknown = [m for m in methods if m not in known]
    if unknown:
        print(f"  (ignoring unknown methods: {', '.join(unknown)})", flush=True)

    def _usable(spec: tuple) -> bool:
        emb, _q, src, prop = spec
        return emb is not None and (prop is not None if src == "dino" else True)

    specs = {m: known[m] for m in methods if m in known and _usable(known[m])}
    methods = list(specs)

    score: dict[str, list[float]] = {m: [] for m in specs}
    crop_count: dict[str, int] = dict.fromkeys(specs, 0)
    viz_font = _font() if args.viz_dir else None
    if args.viz_dir:
        for m, (_e, _q, src, _p) in specs.items():
            if src in ("sliding", "dino"):
                (args.viz_dir / m).mkdir(parents=True, exist_ok=True)
    need_sliding = any(src == "sliding" for _e, _q, src, _p in specs.values())

    t0 = time.perf_counter()
    for n, iid in enumerate(eval_ids, 1):
        img = read_pil(members, iid)
        w, h = img.size
        by_scale = sliding_boxes_by_scale(w, h, scales, args.overlap, args.min_window) if need_sliding else {}
        flat_sliding = [b for boxes in by_scale.values() for b in boxes]
        for m, (emb, qv, src, prop) in specs.items():
            if src == "whole":
                score[m].append(float(score_of(emb, emb.image_feats([img]) @ qv)[0]))
                crop_count[m] += 1
                continue
            if src == "sliding":
                boxes = flat_sliding
            else:  # dino proposer (v2 or v3) -> region boxes, scaled to pixels
                boxes = [(b[0] * w, b[1] * h, b[2] * w, b[3] * h) for b in prop.region_boxes(img)]
            sims = score_of(emb, crop_sims(img, boxes, emb, qv))
            score[m].append(float(np.max(sims)) if len(sims) else float("-inf"))
            crop_count[m] += len(boxes)
            if not (args.viz_dir and len(sims)):
                continue
            if src == "sliding":
                lo, hi = float(sims.min()), float(sims.max())  # shared color scale across scales
                off = 0
                for sc, sbx in by_scale.items():
                    seg = sims[off : off + len(sbx)]
                    off += len(sbx)
                    render_crops_viz(img, sbx, seg, args.viz_label_topk, viz_font, lo, hi).save(
                        args.viz_dir / m / f"{iid}_s{sc:g}.png"
                    )
            else:
                render_crops_viz(img, boxes, sims, args.viz_label_topk, viz_font).save(args.viz_dir / m / f"{iid}.png")
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
    score_desc = "sigmoid(t*cos+b)" if (args.score == "sigmoid" and siglip.calib) else "cosine"
    print(f"\n=== prompt={args.prompt!r}  {mode}  score={score_desc}  device={device}  {elapsed:.1f}s ===")
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
            "score": score_desc,
            "siglip_logit_scale_bias": siglip.calib,
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
