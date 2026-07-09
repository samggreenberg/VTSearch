#!/usr/bin/env python3
"""Build a one-shot DE-ViT class prototype from Visual Genome reference images.

DE-ViT classifies region proposals by cosine similarity to per-class
*prototypes* — mean foreground DINOv2 patch tokens of a few reference crops.
This script builds such a prototype for one noun (e.g. "hat") from clean VG
images: it pulls the noun's bounding boxes from the staged extract, masks the
object region, and averages the DINOv2 ViT-L/14 patch tokens inside the mask
(mirroring ``demo/build_prototypes.ipynb``).

Output is a ``{'prototypes': [1, D], 'label_names': [noun]}`` ``.pth`` that both
the DE-ViT meta-arch constructor (``DE.CLASS_PROTOTYPES``) and the inference-time
``category_space`` override accept — see ``devit_infer.py``.

Runs in the **devit** conda env (NOT the VTSearch venv). The DINOv2 ViT-L/14 it
encodes references with is DE-ViT's own vendored ``lib.dinov2`` arch loaded from
the **trained checkpoint's frozen backbone** (``backbone.*`` keys of
``vitl_0069999.pth``). That is the exact feature space the detector classifies
in, needs no extra download or ``torch.hub`` (whose current ``main`` requires
Python 3.10), and matches what the authors' ``build_prototypes.ipynb`` did
(DINOv2 is frozen during DE-ViT training, so its weights are unchanged).

GPU run (V100/A100 — the devit env is cu117, sm_86 max)::

    cd /exp/mlucio/projects/devit
    srun --partition=gpu --gres=gpu:v100:1 --cpus-per-task=4 --mem=32G --time=0:30:00 \\
        /exp/mlucio/.conda/envs/devit/bin/python \\
        /exp/mlucio/projects/VTSearch/scripts/vg/devit_build_prototype.py hat \\
        --num-refs 8 --out weights/hat_prototype.vitl14.pth
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import random
import zipfile
from pathlib import Path

import numpy as np

DEFAULT_VG_DIR = Path("/exp/scale26/datasets/external/VisualGenome")
DEFAULT_DEVIT_ROOT = Path("/exp/mlucio/projects/devit")
DEFAULT_MODEL_PATH = "weights/trained/open-vocabulary/lvis/vitl_0069999.pth"

# DINOv2 ImageNet normalization on the 0-255 pixel scale, matching the authors'
# build_prototypes.ipynb. The shipped demo/ycb_prototypes.pth is built this way and
# produces correct detections with this detector (validated via demo.py), so this is
# the proven recipe — NOT the config's CLIP-style PIXEL_MEAN.
PIXEL_MEAN = (123.675, 116.280, 103.530)
PIXEL_STD = (58.395, 57.120, 57.375)


# --------------------------------------------------------------------------
# VG dataset helpers (duplicated from eval_crop_clip / annotate_vg_noun so this
# script is self-contained in the devit env, which has no vtscore on its path).
# --------------------------------------------------------------------------
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


def read_pil(members: dict[int, tuple[Path, str]], iid: int):
    """Read a clean JPEG in-memory from its VG zip member (PIL RGB)."""
    from PIL import Image

    zp, member = members[iid]
    with zipfile.ZipFile(zp) as zf:
        raw = zf.read(member)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _row_matches(row: dict, mode: str, q_syn: str, q_name: str) -> bool:
    name = str(row.get("name", "")).strip().lower()
    if mode == "name":
        return name == q_name
    if mode == "substring":
        return q_name in name
    syn = str(row.get("synset", "")).strip().lower()
    if syn:
        return syn.split(".")[0] == q_syn
    return name == q_name


def find_matches(extract_path: Path, mode: str, q_syn: str, q_name: str) -> dict[int, list[dict]]:
    """Stream the extract, returning image_id -> matching object rows."""
    tok = q_name.split()[0] if q_name else ""
    by_image: dict[int, list[dict]] = {}
    with gzip.open(extract_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if tok and tok not in line:
                continue
            row = json.loads(line)
            if _row_matches(row, mode, q_syn, q_name):
                by_image.setdefault(int(row["image_id"]), []).append(row)
    return by_image


# --------------------------------------------------------------------------
# DINOv2 foreground-token extraction
# --------------------------------------------------------------------------
def _resize_shortest_edge(img, short: int, maxs: int, *, nearest: bool):
    """Resize a CHW float tensor so its short side is ``short`` (cap long at ``maxs``)."""
    import torch.nn.functional as F

    _c, h, w = img.shape
    scale = short / min(h, w)
    if max(h, w) * scale > maxs:
        scale = maxs / max(h, w)
    nh, nw = max(int(round(h * scale)), 1), max(int(round(w * scale)), 1)
    mode = "nearest" if nearest else "bilinear"
    kw = {} if nearest else {"align_corners": False, "antialias": True}
    return F.interpolate(img[None].float(), size=(nh, nw), mode=mode, **kw)[0]


def _to_closest_14x(img):
    """Resize a CHW tensor so H and W are the nearest positive multiples of 14."""
    import torch.nn.functional as F

    _c, h, w = img.shape
    nh, nw = max(round(h / 14), 1) * 14, max(round(w / 14), 1) * 14
    return F.interpolate(img[None].float(), size=(nh, nw), mode="bicubic", align_corners=False)[0]


def foreground_token(model, device, pil_img, boxes_norm: list[tuple[float, float, float, float]]):
    """Average DINOv2 patch tokens inside the union of ``boxes_norm`` (normalized xyxy).

    Returns a 1-D tensor (the mean foreground token) or ``None`` if the mask is
    empty after downsampling to the patch grid.
    """
    import torch

    arr = np.asarray(pil_img, dtype=np.uint8)  # HWC RGB
    h, w = arr.shape[:2]
    img = torch.from_numpy(arr).permute(2, 0, 1).float()  # CHW 0-255

    mask = torch.zeros((1, h, w), dtype=torch.float32)
    for x0, y0, x1, y1 in boxes_norm:
        ix0, iy0 = int(round(x0 * w)), int(round(y0 * h))
        ix1, iy1 = int(round(x1 * w)), int(round(y1 * h))
        ix0, ix1 = sorted((max(0, ix0), min(w, ix1)))
        iy0, iy1 = sorted((max(0, iy0), min(h, iy1)))
        mask[:, iy0:iy1, ix0:ix1] = 1.0
    if mask.sum() <= 0:
        return None

    img = _resize_shortest_edge(img, 800, 1333, nearest=False)
    mask = _resize_shortest_edge(mask, 800, 1333, nearest=True)
    img14 = _to_closest_14x(img)
    gh, gw = img14.shape[1] // 14, img14.shape[2] // 14

    mean = torch.tensor(PIXEL_MEAN).view(3, 1, 1)
    std = torch.tensor(PIXEL_STD).view(3, 1, 1)
    nimg = ((img14 - mean) / std)[None].to(device)  # 0-255 ImageNet norm (matches build_prototypes.ipynb)

    with torch.no_grad():
        layers = model.get_intermediate_layers(nimg, return_class_token=True, reshape=True)
    patch = layers[0][0][0].cpu()  # [C, gh, gw]

    import torch.nn.functional as F

    mask14 = F.interpolate(mask[None], size=(gh, gw), mode="bilinear", align_corners=False)[0]  # [1, gh, gw]
    denom = float(mask14.sum())
    if denom <= 0.5:
        return None
    return (mask14 * patch).flatten(1).sum(1) / denom  # [C]


def load_dinov2_backbone(devit_root: Path, model_path: Path, device: str):
    """Build DE-ViT's vendored DINOv2 ViT-L/14 and load the frozen backbone from
    the trained checkpoint's ``backbone.*`` weights."""
    import torch

    sys.path.insert(0, str(devit_root))
    # Import detectron2's backbone package FIRST so its circular dependency with
    # lib.dinov2.vit resolves (detectron2.modeling.backbone.vit imports from
    # lib.dinov2.vit and vice-versa); importing lib.dinov2.vit standalone first
    # deadlocks the cycle.
    import detectron2.modeling.backbone  # noqa: F401
    from lib.dinov2.vit import vit_large  # vendored arch, py3.9-compatible

    model = vit_large(img_size=518, patch_size=14, init_values=1, out_indices=[2, 12, 23]).eval()
    sd = torch.load(model_path, map_location="cpu")
    sd = sd.get("model", sd)
    bb = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
    if not bb:
        prefixes = sorted({k.split(".")[0] for k in sd})
        raise SystemExit(f"no 'backbone.*' keys in {model_path}; top-level prefixes were: {prefixes}")
    missing, unexpected = model.load_state_dict(bb, strict=False)
    # detectron2 Backbone adds non-DINOv2 buffers (e.g. _out_feature_*); those
    # legitimately stay missing. Real trouble is unexpected checkpoint keys.
    if unexpected:
        print(f"  (warn: {len(unexpected)} unexpected backbone keys, e.g. {unexpected[:3]})", flush=True)
    print(f"  loaded backbone: {len(bb)} tensors ({len(missing)} arch buffers left at init)", flush=True)
    return model.to(device)


def _mean_norm(tokens):
    import torch

    return torch.nn.functional.normalize(torch.stack(tokens).mean(dim=0), dim=0)


def collect_noun_token(model, device, members, by_image, ref_ids, noun):
    """Mean L2-normalized foreground token for one noun over its reference images."""
    tokens = []
    for iid in ref_ids:
        boxes = [(r["x0"], r["y0"], r["x1"], r["y1"]) for r in by_image[iid]]
        tok = foreground_token(model, device, read_pil(members, iid), boxes)
        if tok is None:
            print(f"    [{iid}] empty mask — skipping", flush=True)
            continue
        tokens.append(tok)
    if not tokens:
        raise SystemExit(f"no usable reference tokens for '{noun}'")
    print(f"    '{noun}': {len(tokens)} refs -> class token", flush=True)
    return _mean_norm(tokens)


def collect_other_token(model, device, members, exclude_ids, k, seed):
    """Generic 'other' class: mean whole-image token over random non-target images.

    DE-ViT's classifier needs >=2 foreground classes (it contrasts the target
    against 'other' classes), so a lone target prototype crashes the forward.
    This builds a single coarse contrast class from random non-target scenes."""
    pool = [i for i in sorted(members) if i not in exclude_ids]
    sel = sorted(random.Random(seed + 1).sample(pool, min(k, len(pool))))
    tokens = []
    for iid in sel:
        tok = foreground_token(model, device, read_pil(members, iid), [(0.0, 0.0, 1.0, 1.0)])
        if tok is not None:
            tokens.append(tok)
    if not tokens:
        raise SystemExit("no usable 'other' tokens")
    print(f"    'other': {len(tokens)} random non-target images -> class token", flush=True)
    return _mean_norm(tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("noun", help="object/noun the prototype represents (e.g. 'hat')")
    ap.add_argument("--vg-dir", type=Path, default=DEFAULT_VG_DIR, help="staged Visual Genome dir")
    ap.add_argument(
        "--ref-ids",
        default=None,
        help="comma list of VG image ids to use as references (default: sample --num-refs containing the noun)",
    )
    ap.add_argument("--num-refs", type=int, default=8, help="how many reference images to average (default: 8)")
    ap.add_argument(
        "--match",
        choices=("synset", "name", "substring"),
        default="synset",
        help="how the noun matches VG object names (default: synset-canonical)",
    )
    ap.add_argument("--devit-root", type=Path, default=DEFAULT_DEVIT_ROOT, help="DE-ViT repo root")
    ap.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="trained checkpoint to lift the frozen DINOv2 backbone from (relative to --devit-root or absolute)",
    )
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed when sampling references")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="prototype .pth output (default: weights/<noun>_prototype.vitl14.pth under cwd)",
    )
    ap.add_argument("--label", default=None, help="label name stored in the prototype (default: the noun, lowercased)")
    ap.add_argument(
        "--distractors",
        default=None,
        help="comma list of extra class nouns for contrast (default: one generic 'other' from random non-target images)",
    )
    ap.add_argument(
        "--other-refs",
        type=int,
        default=16,
        help="random non-target images to average for the generic 'other' class (default: 16)",
    )
    args = ap.parse_args()

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    devit_root = args.devit_root.resolve()
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = devit_root / model_path

    q_name = args.noun.strip().lower()
    q_syn = q_name.replace(" ", "_")
    label = (args.label or q_name).strip()
    out = args.out or Path(f"weights/{q_syn}_prototype.vitl14.pth")

    extract = args.vg_dir / "derived" / "objects_flat.jsonl.gz"
    zips = [args.vg_dir / "images" / "images.zip", args.vg_dir / "images" / "images2.zip"]
    for p in [extract, *zips]:
        if not p.exists():
            raise SystemExit(f"missing {p}\nrun fetch_visual_genome.sbatch first.")
    if not model_path.exists():
        raise SystemExit(f"missing checkpoint {model_path}\ndownload vitl_0069999.pth first (see the plan).")

    print(f"finding VG images with '{q_name}' (match={args.match})…", flush=True)
    by_image = find_matches(extract, args.match, q_syn, q_name)
    members = build_member_index(zips)
    have = sorted(set(by_image) & set(members))
    if not have:
        raise SystemExit(f"no VG images with object '{q_name}'")

    if args.ref_ids:
        want = [int(x) for x in args.ref_ids.split(",") if x.strip()]
        ref_ids = [i for i in want if i in by_image and i in members]
        missing = [i for i in want if i not in ref_ids]
        if missing:
            print(f"  (skipping ids without '{q_name}' boxes or image: {missing})", flush=True)
    else:
        ref_ids = sorted(random.Random(args.seed).sample(have, min(args.num_refs, len(have))))
    if not ref_ids:
        raise SystemExit("no usable reference ids")
    print(f"  {len(have):,} candidates; using {len(ref_ids)} references: {ref_ids}", flush=True)

    print(f"loading vendored DINOv2 ViT-L/14 backbone from {model_path}…", flush=True)
    model = load_dinov2_backbone(devit_root, model_path, device)

    print("building class prototypes…", flush=True)
    vecs = [collect_noun_token(model, device, members, by_image, ref_ids, label)]
    names = [label]
    exclude = set(by_image)  # never sample target images into 'other'
    if args.distractors:
        for d in [x.strip().lower() for x in args.distractors.split(",") if x.strip()]:
            by_d = find_matches(extract, args.match, d.replace(" ", "_"), d)
            d_have = sorted(set(by_d) & set(members))
            if not d_have:
                print(f"  (no images for distractor '{d}' — skipping)", flush=True)
                continue
            d_refs = sorted(random.Random(args.seed).sample(d_have, min(args.num_refs, len(d_have))))
            vecs.append(collect_noun_token(model, device, members, by_d, d_refs, d))
            names.append(d)
            exclude |= set(by_d)
    if len(vecs) == 1:  # no (usable) distractors -> one generic 'other' contrast class
        vecs.append(collect_other_token(model, device, members, exclude, args.other_refs, args.seed))
        names.append("other")

    prototypes = torch.stack(vecs)  # [C, D]
    payload = {"prototypes": prototypes, "label_names": names}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"\nwrote {out}  (classes={names}, shape={tuple(prototypes.shape)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
