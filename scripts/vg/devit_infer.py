#!/usr/bin/env python3
"""Run the real DE-ViT few-shot detector on Visual Genome and emit per-image
presence scores in the same JSON schema as ``eval_crop_clip.py``.

For each VG **test** image (clean pixels re-fetched by id), DE-ViT (frozen
DINOv2 ViT-L/14 + offline R50-FPN RPN + trained region/box/classification heads)
is run with a one-shot class **prototype** (see ``devit_build_prototype.py``).
Each image is reduced to a **presence score = max detection score for the class**
and the **best box**, written as ``per_image[<id>][<method>]`` so it sits
directly next to ``eval_crop_clip``'s ``whole`` / ``sliding`` / ``dino_v2``
columns. With ``--neg-dir`` it also reports AP / AUROC / best-F1.

DE-ViT scores are its own classification confidence — NOT comparable in
*magnitude* to SigLIP cosine. Compare via ranking (AP/AUROC) or the boxes, not
raw values vs the crop-CLIP methods.

Runs in the **devit** conda env on a **V100/A100** (cu117, sm_86 max — NOT
L40S/H100/H200). Needs the 2 checkpoints under ``<devit-root>/weights`` (see the
plan); construction repoints ``DE.CLASS_PROTOTYPES`` at the prototype, so the
LVIS prototype pkls are not required.

GPU run::

    srun --partition=gpu --gres=gpu:v100:1 --cpus-per-task=8 --mem=32G --time=1:00:00 \\
        /exp/mlucio/.conda/envs/devit/bin/python \\
        /exp/mlucio/projects/VTSearch/scripts/vg/devit_infer.py \\
        /exp/mlucio/projects/VTSearch/data/vg/vg_hat_100_max/test \\
        --neg-dir /exp/mlucio/projects/VTSearch/data/vg/vg_neg_100/test \\
        --prototype weights/hat_prototype.vitl14.pth --label hat \\
        --out /exp/mlucio/projects/VTSearch/data/vg/vg_hat_100_max/test/devit/results.json \\
        --viz-dir /exp/mlucio/projects/VTSearch/data/vg/vg_hat_100_max/test/devit
        
Matthew Usage::
srun --partition=gpu --gres=gpu:v100:1 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty bash -l
cd /exp/mlucio/projects/devit
PY=/exp/mlucio/.conda/envs/devit/bin/python

# 1) build on 12 distractors
$PY /exp/mlucio/projects/VTSearch/scripts/vg/devit_build_prototype.py hat \
    --num-refs 8 \
    --distractors person,dog,car,tree,building,sign,bottle,chair,bird,bus,clock,window \
    --out weights/hat_prototype.vitl14.pth

# 2) re-run with top-5 debug dump (prints + draws the 5 best hat dets per image)
$PY /exp/mlucio/projects/VTSearch/scripts/vg/devit_infer.py \
    /exp/mlucio/projects/VTSearch/data/vg/vg_hat_100_max/test \
    --prototype weights/hat_prototype.vitl14.pth --label hat \
    --debug-topk 5 \
    --viz-dir /exp/mlucio/projects/VTSearch/data/vg/vg_hat_100_max/test/devit
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics as st
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

DEFAULT_VG_DIR = Path("/exp/scale26/datasets/external/VisualGenome")
DEFAULT_DEVIT_ROOT = Path("/exp/mlucio/projects/devit")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------
# VG dataset helpers (duplicated from eval_crop_clip so this script is
# self-contained in the devit env, which has no vtscore on its path).
# --------------------------------------------------------------------------
def build_member_index(zip_paths: list[Path]) -> dict[int, tuple[Path, str]]:
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
    ids: set[int] = set()
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            try:
                ids.add(int(p.stem))
            except ValueError:
                continue
    return ids


def read_pil(members: dict[int, tuple[Path, str]], iid: int):
    from PIL import Image

    zp, member = members[iid]
    with zipfile.ZipFile(zp) as zf:
        raw = zf.read(member)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# --------------------------------------------------------------------------
# Metrics (sklearn is in the devit env; mirror eval_crop_clip's definitions).
# --------------------------------------------------------------------------
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
    best = (0.0, 0.0, 0.0, 0.0)
    for t in np.unique(scores):
        preds = scores >= t
        tp = int(np.sum(preds & (labels == 1)))
        fp = int(np.sum(preds & (labels == 0)))
        fn = int(np.sum(~preds & (labels == 1)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best[0]:
            best = (f1, p, r, float(t))
    return best


# --------------------------------------------------------------------------
# DE-ViT model
# --------------------------------------------------------------------------
def build_devit(devit_root: Path, config_file: str, rpn_config: str, model_path: str, prototype: str,
                topk: int, score_thresh: float, mult_rpn_score: bool, device: str):
    """Build DE-ViT exactly like demo/demo.py, but repoint CLASS_PROTOTYPES at our
    one-shot prototype and override the category space at inference."""
    import torch
    from detectron2.config import get_cfg
    import detectron2.data.transforms as T
    import detectron2.data.detection_utils as utils
    from tools.train_net import Trainer

    proto_abs = str(Path(prototype).resolve())
    cs = torch.load(proto_abs, map_location=device)
    n_classes = int(cs["prototypes"].shape[0])

    config = get_cfg()
    config.merge_from_file(config_file)
    config.DE.OFFLINE_RPN_CONFIG = rpn_config
    # DE.TOPK = #top classes scored per region; torch.topk requires it <= #prototypes.
    config.DE.TOPK = max(1, min(topk, n_classes))
    config.MODEL.MASK_ON = True  # the LVIS checkpoint carries mask weights; keep arch matching
    # Repoint construction-time prototypes at ours so the big LVIS pkls aren't needed.
    config.DE.CLASS_PROTOTYPES = proto_abs
    # Low threshold so weak detections still yield a presence score (max over dets).
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    # MULTIPLY_RPN_SCORE folds RPN objectness into the score (score=(prob*rpn)**0.5),
    # which lets large high-objectness regions dominate small objects like hats.
    # Off by default here so the score is the pure class prob (best prototype match wins).
    config.DE.MULTIPLY_RPN_SCORE = mult_rpn_score
    config.freeze()

    model = Trainer.build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    model.eval()
    model = model.to(device)

    # Override the category space with our one-shot prototype (as demo.py does).
    model.label_names = cs["label_names"]
    model.test_class_weight = cs["prototypes"].to(device)

    augmentations = T.AugmentationList(utils.build_augmentation(config, False))
    return model, augmentations, T


def detect(model, augmentations, T, pil_img, device):
    """Run DE-ViT on one PIL image; return (scores[N], boxes[N,4]) as numpy."""
    import torch

    image = np.asarray(pil_img, dtype=np.uint8)  # HWC RGB, like utils.read_image(..., "RGB")
    h, w = image.shape[:2]
    aug_input = T.AugInput(image)
    augmentations(aug_input)
    dd = {
        "height": h,
        "width": w,
        "image": torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))).to(device),
    }
    with torch.no_grad():
        inst = model([dd])[0]["instances"].to("cpu")
    scores = inst.scores.numpy() if len(inst) else np.zeros(0, np.float32)
    boxes = inst.pred_boxes.tensor.numpy() if len(inst) else np.zeros((0, 4), np.float32)
    classes = inst.pred_classes.numpy() if len(inst) else np.zeros(0, np.int64)
    return scores, boxes, classes


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
def draw_best(pil_img, box: Box | None, score: float):
    from PIL import ImageDraw, ImageFont

    canvas = pil_img.copy()
    if box is None:
        return canvas
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=14)
    except Exception:
        font = None
    draw.rectangle(tuple(box), outline=(0, 255, 0), width=3)
    txt = f"{score:.3f}"
    try:
        tb = draw.textbbox((box[0] + 2, box[1] + 2), txt, font=font)
        draw.rectangle(tb, fill=(0, 0, 0))
    except Exception:
        pass
    draw.text((box[0] + 2, box[1] + 2), txt, fill=(0, 255, 0), font=font)
    return canvas


def draw_topk(pil_img, dets):
    """Draw ranked detections (best first): #1 thick green, rest thin orange."""
    from PIL import ImageDraw, ImageFont

    canvas = pil_img.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=13)
    except Exception:
        font = None
    for rank, (box, score) in enumerate(dets):
        color = (0, 255, 0) if rank == 0 else (255, 170, 0)
        draw.rectangle(tuple(box), outline=color, width=3 if rank == 0 else 1)
        draw.text((box[0] + 2, box[1] + 2), f"{score:.2f}", fill=color, font=font)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pos_dir", type=Path, help="directory of positive images (filenames are VG image ids)")
    ap.add_argument("--neg-dir", type=Path, default=None, help="optional negatives dir; enables AP/AUROC/F1")
    ap.add_argument("--prototype", required=True, help="one-shot prototype .pth from devit_build_prototype.py")
    ap.add_argument("--label", default=None, help="class label (default: from the prototype)")
    ap.add_argument("--method-name", default="devit", help="key under results/per_image (default: devit)")
    ap.add_argument("--vg-dir", type=Path, default=DEFAULT_VG_DIR)
    ap.add_argument("--devit-root", type=Path, default=DEFAULT_DEVIT_ROOT)
    ap.add_argument("--config", default="configs/open-vocabulary/lvis/vitl.yaml")
    ap.add_argument("--rpn-config", default="configs/RPN/mask_rcnn_R_50_FPN_1x.yaml")
    ap.add_argument("--model-path", default="weights/trained/open-vocabulary/lvis/vitl_0069999.pth")
    ap.add_argument("--topk", type=int, default=10, help="DE.TOPK proposals per class (default: 10)")
    ap.add_argument(
        "--score-thresh", type=float, default=0.0, help="ROI_HEADS.SCORE_THRESH_TEST; low keeps weak dets (default: 0.0)"
    )
    ap.add_argument(
        "--mult-rpn-score",
        action="store_true",
        help="fold RPN objectness into the score ((prob*rpn)**0.5); off by default so small objects "
        "aren't drowned out by large high-objectness regions (score = pure class prob)",
    )
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--out", type=Path, default=None, help="results JSON (default: <viz-dir>/results.json)")
    ap.add_argument("--viz-dir", type=Path, default=None, help="write <id>.png with the best box + score")
    ap.add_argument(
        "--debug-topk",
        type=int,
        default=0,
        help="if >0: print the top-K target-class dets (score+box) for the first few images and draw all K (ranked) in the viz",
    )
    args = ap.parse_args()

    devit_root = args.devit_root.resolve()
    sys.path.insert(0, str(devit_root))
    os.chdir(devit_root)  # demo.py uses paths relative to the repo root
    os.environ.setdefault("TORCH_HOME", f"/exp/{os.environ.get('USER', 'mlucio')}/.cache/torch")
    if args.out is None and args.viz_dir is not None:
        args.out = args.viz_dir / "results.json"

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    method = args.method_name

    zips = [args.vg_dir / "images" / "images.zip", args.vg_dir / "images" / "images2.zip"]
    for p in [*zips, devit_root / args.model_path, Path(args.prototype)]:
        if not Path(p).exists():
            raise SystemExit(f"missing {p}")
    if not args.pos_dir.is_dir():
        raise SystemExit(f"not a directory: {args.pos_dir}")

    print(f"indexing dataset (device={device})…", flush=True)
    members = build_member_index(zips)
    pos = sorted(ids_from_dir(args.pos_dir) & set(members))
    if not pos:
        raise SystemExit(f"no VG-id images found in {args.pos_dir}")
    neg = sorted(ids_from_dir(args.neg_dir) & set(members)) if args.neg_dir else []
    eval_ids = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))
    print(f"  {len(pos)} positives" + (f" + {len(neg)} negatives" if neg else " (positives only)"), flush=True)

    print("building DE-ViT…", flush=True)
    model, augs, T = build_devit(
        devit_root, args.config, args.rpn_config, args.model_path, args.prototype,
        args.topk, args.score_thresh, args.mult_rpn_score, device,
    )
    label = (args.label or (model.label_names[0] if model.label_names else "class"))
    if label not in model.label_names:
        raise SystemExit(f"label {label!r} not in prototype classes {model.label_names}")
    target_idx = model.label_names.index(label)
    print(
        f"  classes={model.label_names}  target={label!r} (idx {target_idx})  "
        f"weights={tuple(model.test_class_weight.shape)}",
        flush=True,
    )

    if args.viz_dir:
        args.viz_dir.mkdir(parents=True, exist_ok=True)

    scores_l: list[float] = []
    boxes_l: list[Box | None] = []
    ndet = 0
    t0 = time.perf_counter()
    for n, iid in enumerate(eval_ids, 1):
        img = read_pil(members, iid)
        s, b, c = detect(model, augs, T, img, device)
        keep = np.where(c == target_idx)[0] if len(s) else np.empty(0, dtype=int)
        order = keep[np.argsort(-s[keep])] if len(keep) else np.empty(0, dtype=int)  # hat dets, best first
        ndet += len(keep)
        if len(order):
            j = int(order[0])
            scores_l.append(float(s[j]))
            boxes_l.append(tuple(float(x) for x in b[j]))
        else:
            scores_l.append(0.0)
            boxes_l.append(None)
        if args.debug_topk and n <= 5:
            top = order[: args.debug_topk]
            print(f"  [{iid}] {len(keep)} {label}-dets; top {len(top)}:", flush=True)
            for r in top:
                bx = b[r]
                print(f"      score={s[r]:.3f}  box=({bx[0]:.0f},{bx[1]:.0f},{bx[2]:.0f},{bx[3]:.0f})", flush=True)
        if args.viz_dir:
            if args.debug_topk:
                dets = [(tuple(float(x) for x in b[r]), float(s[r])) for r in order[: args.debug_topk]]
                draw_topk(img, dets).save(args.viz_dir / f"{iid}.png")
            else:
                draw_best(img, boxes_l[-1], scores_l[-1]).save(args.viz_dir / f"{iid}.png")
        if n % 50 == 0:
            print(f"  scored {n}/{len(eval_ids)}…", flush=True)
    elapsed = time.perf_counter() - t0

    scores = np.array(scores_l, dtype=np.float64)
    rec: dict[str, object] = {"detections": ndet, **summarize(scores_l)}
    if neg:
        from sklearn.metrics import average_precision_score, roc_auc_score

        rec["ap"] = float(average_precision_score(labels, scores))
        rec["auroc"] = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else float("nan")
        f1, prec, recall, _thr = best_f1(scores, labels)
        rec.update({"best_f1": f1, "precision": prec, "recall": recall})
    results = {method: rec}

    mode = f"{len(pos)} pos + {len(neg)} neg" if neg else f"{len(pos)} positives"
    print(f"\n=== DE-ViT label={label!r}  {mode}  device={device}  {elapsed:.1f}s ===")
    if neg:
        print(f"{'method':<9}{'AP':>8}{'AUROC':>8}{'bestF1':>8}{'prec':>7}{'rec':>7}{'mean':>8}{'dets':>9}")
        print(
            f"{method:<9}{rec['ap']:>8.3f}{rec['auroc']:>8.3f}{rec['best_f1']:>8.3f}"
            f"{rec['precision']:>7.2f}{rec['recall']:>7.2f}{rec['mean']:>8.3f}{rec['detections']:>9,}"
        )
    else:
        print(f"{'method':<9}{'mean':>8}{'median':>8}{'min':>8}{'max':>8}{'dets':>9}")
        print(
            f"{method:<9}{rec['mean']:>8.3f}{rec['median']:>8.3f}{rec['min']:>8.3f}{rec['max']:>8.3f}{rec['detections']:>9,}"
        )

    if args.out:
        payload = {
            "label": label,
            "method": method,
            "model_path": str(devit_root / args.model_path),
            "prototype": str(Path(args.prototype).resolve()),
            "device": device,
            "score": "devit_confidence",
            "n_pos": len(pos),
            "n_neg": len(neg),
            "elapsed_s": elapsed,
            "results": results,
            "per_image": {str(iid): {method: scores_l[i]} for i, iid in enumerate(eval_ids)},
            "per_image_box": {str(iid): boxes_l[i] for i, iid in enumerate(eval_ids)},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
