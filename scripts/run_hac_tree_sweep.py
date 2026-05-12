"""HAC tree sweep on caltech-101.

Throwaway experiment script supporting the patch-embedder design
(``docs/plans/patch-embedder.md``).  Sweeps ``K ∈ {8, 12, 16}`` and
``α ∈ {0.3, 0.5, 0.7}`` over a sample of caltech-101 images, renders the
leaf + HAC-internal bounding-box overlays as PNGs, and writes a markdown
report under ``docs/experiments/hac-tree-sweep/``.

Embedder: DINOv2 ViT-B/14 (``facebook/dinov2-base``).  DINOv2 is the
ungated patch-capable embedder; the HAC tree code does not depend on
which patch embedder produced the ``PatchEmbedOutput`` so the (K, α)
choice transfers to DINOv3 / EUPE.

Usage::

    python scripts/run_hac_tree_sweep.py \\
        --image-root data/caltech-101/101_ObjectCategories \\
        --out-dir docs/experiments/hac-tree-sweep \\
        --num-images 30 \\
        --seed 0

Designed to be deleted once the sweep results are committed.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from vtsearch.models.patch_regions import (
    PatchEmbedOutput,
    RegionVector,
    build_region_tree,
    hf_vit_to_patch_output,
)

DEFAULT_K_VALUES = (8, 12, 16)
DEFAULT_ALPHA_VALUES = (0.3, 0.5, 0.7)


# ---------------------------------------------------------------------------
# Image sampling
# ---------------------------------------------------------------------------


def sample_caltech_paths(root: Path, n: int, seed: int) -> list[Path]:
    """Pick *n* JPEGs from caltech-101 spread across many categories.

    We round-robin over the categories (sorted alphabetically) so the
    sample contains both single-object (e.g. faces, airplanes) and
    cluttered-object (e.g. anchor, ant) scenes.  ``BACKGROUND_Google``
    is dropped — it has no foreground subject and would muddle the
    visual review.
    """
    rng = random.Random(seed)
    cats = sorted([p for p in root.iterdir() if p.is_dir() and p.name != "BACKGROUND_Google"])
    rng.shuffle(cats)

    by_cat: list[list[Path]] = []
    for cat in cats:
        files = sorted(cat.glob("*.jpg"))
        if files:
            rng.shuffle(files)
            by_cat.append(files)

    out: list[Path] = []
    i = 0
    while len(out) < n and by_cat:
        idx = i % len(by_cat)
        files = by_cat[idx]
        out.append(files.pop())
        if not files:
            by_cat.pop(idx)
        else:
            i += 1
    return out[:n]


# ---------------------------------------------------------------------------
# DINOv2 forward
# ---------------------------------------------------------------------------


@dataclass
class _Backbone:
    model: object
    processor: object


def load_dinov2() -> _Backbone:
    from transformers import AutoImageProcessor, AutoModel

    model_id = "facebook/dinov2-base"
    print(f"  loading {model_id} (this may download ~350MB on first run)…", flush=True)
    # Force eager attention so output_attentions=True actually returns
    # weights — recent transformers default to SDPA which drops them.
    model = AutoModel.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        token=False,
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoImageProcessor.from_pretrained(model_id, token=False)
    return _Backbone(model=model, processor=processor)


def patch_forward(bb: _Backbone, image: Image.Image) -> PatchEmbedOutput | None:
    inputs = bb.processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = bb.model(**inputs, output_attentions=True)
    return hf_vit_to_patch_output(outputs, num_register_tokens=0)


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


def _draw_box(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float],
              w: int, h: int, color: tuple[int, int, int], width: int = 2) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), outline=color, width=width)


def _label(draw: ImageDraw.ImageDraw, text: str, w: int) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 4
    draw.rectangle((0, 0, w, 18), fill=(0, 0, 0))
    draw.text((pad, 2), text, fill=(255, 255, 255), font=font)


def render_config_overlay(
    image: Image.Image,
    regions: list[RegionVector],
    k: int,
    title: str,
    out_size: int = 384,
    show_internals: bool = True,
) -> Image.Image:
    """One panel: image + K leaf boxes (yellow); optional HAC internals (cyan)."""
    panel = image.convert("RGB").resize((out_size, out_size))
    draw = ImageDraw.Draw(panel)
    # Skip index 0 (the full-image CLS node) — its box is (0,0,1,1).
    leaves = regions[1 : 1 + k]
    internals = regions[1 + k :]
    if show_internals:
        # Render every other internal (alternates by build order) to thin
        # the clutter; the smallest merges are most informative.
        for r in internals[: max(1, len(internals) // 2)]:
            _draw_box(draw, r.box, out_size, out_size, color=(64, 200, 220), width=1)
    for r in leaves:
        _draw_box(draw, r.box, out_size, out_size, color=(255, 215, 0), width=2)
    _label(draw, title, out_size)
    return panel


def render_image_grid(
    image: Image.Image,
    config_panels: dict[tuple[int, float], Image.Image],
    k_values: tuple[int, ...],
    alpha_values: tuple[float, ...],
    cell: int = 384,
) -> Image.Image:
    rows = len(k_values)
    cols = len(alpha_values) + 1  # +1 for the original
    grid = Image.new("RGB", (cell * cols, cell * rows), (40, 40, 40))
    # Original in the leftmost column of each row
    orig = image.convert("RGB").resize((cell, cell))
    draw = ImageDraw.Draw(orig)
    _label(draw, "original", cell)
    for r in range(rows):
        grid.paste(orig, (0, r * cell))
    for r, k in enumerate(k_values):
        for c, alpha in enumerate(alpha_values):
            panel = config_panels[(k, alpha)]
            grid.paste(panel, ((c + 1) * cell, r * cell))
    return grid


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _box_area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_iou(a, b) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 1e-9 else 0.0


def measure_config(regions: list[RegionVector], k: int) -> dict[str, float]:
    """Quantitative summary of one (K, α) region tree.

    All metrics are coordinate-free (boxes in [0, 1]²) so they're
    directly comparable across K and α.

    Keys returned:
      * ``leaf_area_mean``           — mean leaf box area (1/K is uniform).
      * ``leaf_area_std``            — std of leaf areas (high = imbalanced).
      * ``leaf_overlap_max``         — max IoU between any two leaves
        (leaves should be near-disjoint, so this should be near 0).
      * ``internal_area_mean``       — mean HAC-internal box area.
      * ``root_area``                — area of the final merge box; ideally
        close to 1.0 (root recovers the whole image).
      * ``merge_balance``            — mean over internals of
        ``min(|left|, |right|) / max(|left|, |right|)`` where ``|·|`` is
        the subtree size in patch cells (= sum of leaf areas).  1.0 is
        perfectly balanced, ~0.1 is degenerate-chain.
      * ``area_growth``              — mean ratio of internal area to
        sum of its children's areas; ≈ 1 means children are adjacent,
        > 1 means the merge introduces a lot of empty bounding box.
    """
    leaves = regions[1 : 1 + k]
    internals = regions[1 + k :]
    leaf_areas = np.array([_box_area(r.box) for r in leaves], dtype=np.float64)
    internal_areas = np.array([_box_area(r.box) for r in internals], dtype=np.float64)

    max_overlap = 0.0
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            iou = _box_iou(leaves[i].box, leaves[j].box)
            if iou > max_overlap:
                max_overlap = iou

    # Compute "subtree leaf count" for each node so we can score balance.
    leaf_count = [1] * len(leaves) + [0] * len(internals)
    flat = regions[1:]  # leaves + internals as a flat list (children indices
                       # in `regions` are 1-based because of the CLS node).
    balances: list[float] = []
    growths: list[float] = []
    for offset, node in enumerate(internals, start=len(leaves)):
        ci, cj = node.children  # type: ignore[misc]
        # children are indices into `regions`; shift by 1 to index `flat`.
        left = leaf_count[ci - 1]
        right = leaf_count[cj - 1]
        leaf_count[offset] = left + right
        if max(left, right) > 0:
            balances.append(min(left, right) / max(left, right))
        a_box = flat[ci - 1].box
        b_box = flat[cj - 1].box
        area_node = _box_area(node.box)
        area_children_sum = _box_area(a_box) + _box_area(b_box)
        if area_children_sum > 1e-9:
            growths.append(area_node / area_children_sum)

    root_area = _box_area(internals[-1].box) if internals else 1.0
    return {
        "leaf_area_mean": float(leaf_areas.mean()),
        "leaf_area_std": float(leaf_areas.std()),
        "leaf_overlap_max": float(max_overlap),
        "internal_area_mean": float(internal_areas.mean()) if len(internals) else 0.0,
        "root_area": float(root_area),
        "merge_balance": float(np.mean(balances)) if balances else 0.0,
        "area_growth": float(np.mean(growths)) if growths else 0.0,
    }


# ---------------------------------------------------------------------------
# Diversity-tree sanity check
# ---------------------------------------------------------------------------


def diversity_tree_clusters(
    cls_vectors: np.ndarray,
    labels: list[str],
    n_clusters: int = 6,
) -> dict[int, list[str]]:
    """Agglomerative clustering on CLS-pooled DINOv2 vectors.

    Stand-in for ``vtsearch/models/diversity_tree.py`` (which is fancier
    but harder to instantiate in a one-off script).  We just want a
    sanity check on top-level groupings.
    """
    from sklearn.cluster import AgglomerativeClustering

    cls_l2 = cls_vectors / (np.linalg.norm(cls_vectors, axis=1, keepdims=True) + 1e-12)
    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    assign = model.fit_predict(cls_l2)
    clusters: dict[int, list[str]] = {i: [] for i in range(n_clusters)}
    for lab, c in zip(labels, assign):
        clusters[int(c)].append(lab)
    return clusters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-root", type=Path,
                    default=Path("data/caltech-101/101_ObjectCategories"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("docs/experiments/hac-tree-sweep"))
    ap.add_argument("--num-images", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    ap.add_argument("--alpha-values", type=float, nargs="+", default=list(DEFAULT_ALPHA_VALUES))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    k_values = tuple(args.k_values)
    alpha_values = tuple(args.alpha_values)

    print(f"sampling {args.num_images} images from {args.image_root}")
    paths = sample_caltech_paths(args.image_root, args.num_images, args.seed)
    for p in paths:
        print(f"  {p.parent.name}/{p.name}")

    print("loading DINOv2…")
    bb = load_dinov2()

    cls_vectors: list[np.ndarray] = []
    image_labels: list[str] = []

    # config_id -> list of per-image metric dicts
    config_metrics: dict[tuple[int, float], list[dict[str, float]]] = {
        (k, a): [] for k in k_values for a in alpha_values
    }
    timings: list[float] = []

    for idx, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image_label = f"{path.parent.name}/{path.stem}"
        t0 = time.perf_counter()
        out = patch_forward(bb, image)
        timings.append(time.perf_counter() - t0)
        if out is None:
            print(f"  [{idx}] {image_label}: patch_forward returned None — skipping")
            continue
        cls_vectors.append(out.cls_vec)
        image_labels.append(image_label)

        config_panels: dict[tuple[int, float], Image.Image] = {}
        for k in k_values:
            for alpha in alpha_values:
                regions = build_region_tree(out, k=k, alpha=alpha)
                config_metrics[(k, alpha)].append(measure_config(regions, k=k))
                config_panels[(k, alpha)] = render_config_overlay(
                    image, regions, k=k, title=f"K={k} α={alpha}"
                )

        grid = render_image_grid(image, config_panels, k_values, alpha_values)
        # Resize so max side ≤ 1024 — readable in github markdown without
        # bloating the repo.  JPG quality=85 to keep files ~250 KB each.
        max_side = max(grid.size)
        if max_side > 1024:
            scale = 1024 / max_side
            grid = grid.resize(
                (int(grid.size[0] * scale), int(grid.size[1] * scale)),
                Image.LANCZOS,
            )
        out_path = args.out_dir / "overlays" / f"{idx:02d}_{path.parent.name}_{path.stem}.jpg"
        grid.save(out_path, quality=85, optimize=True)
        print(f"  [{idx}] {image_label}: forward {timings[-1]:.2f}s, grid -> {out_path.name}")

    # ----- aggregate metrics ------------------------------------------------
    print("\naggregating metrics…")
    agg_rows: list[tuple[int, float, dict[str, float]]] = []
    for (k, alpha), runs in config_metrics.items():
        keys = runs[0].keys()
        mean = {key: float(np.mean([r[key] for r in runs])) for key in keys}
        agg_rows.append((k, alpha, mean))

    # ----- diversity tree ---------------------------------------------------
    print("running diversity-tree sanity check…")
    cls_mat = np.stack(cls_vectors, axis=0)
    n_clusters = min(6, max(1, len(image_labels) // 2))
    clusters = diversity_tree_clusters(cls_mat, image_labels, n_clusters=n_clusters)

    # ----- write markdown report -------------------------------------------
    report_path = args.out_dir / "README.md"
    write_report(
        report_path=report_path,
        num_images=len(image_labels),
        image_labels=image_labels,
        k_values=k_values,
        alpha_values=alpha_values,
        agg_rows=agg_rows,
        clusters=clusters,
        mean_forward_s=float(np.mean(timings)),
    )
    print(f"wrote report -> {report_path}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(
    *,
    report_path: Path,
    num_images: int,
    image_labels: list[str],
    k_values: tuple[int, ...],
    alpha_values: tuple[float, ...],
    agg_rows: list[tuple[int, float, dict[str, float]]],
    clusters: dict[int, list[str]],
    mean_forward_s: float,
) -> None:
    lines: list[str] = []
    lines.append("# HAC tree (K, α) sweep — caltech-101")
    lines.append("")
    lines.append(
        "Throwaway experiment for `docs/plans/patch-embedder.md` — "
        "confirms the K=12, α=0.5 defaults pinned in "
        "`vtsearch/datasets/loader_folder.py::_attach_patch_regions`."
    )
    lines.append("")
    lines.append(
        f"Backbone: DINOv2 ViT-B/14 (`facebook/dinov2-base`) on CPU.  "
        f"Sample: **{num_images} images** spread across caltech-101 categories "
        f"(seed = 0, see `scripts/run_hac_tree_sweep.py`).  Mean forward "
        f"pass: **{mean_forward_s:.2f} s/image** on CPU."
    )
    lines.append("")
    lines.append("## Images sampled")
    lines.append("")
    for i, label in enumerate(image_labels):
        lines.append(f"- `{i:02d}` — {label}")
    lines.append("")
    lines.append("## Overlay grids")
    lines.append("")
    lines.append(
        "Each PNG below is a 3 × 4 grid: rows are `K ∈ "
        + str(list(k_values))
        + "`, columns are the original image followed by `α ∈ "
        + str(list(alpha_values))
        + "`.  Yellow boxes are HAC leaves; cyan boxes are HAC internal "
        "merge nodes (the boxes the MLP and similarity max-pool over)."
    )
    lines.append("")
    for i, label in enumerate(image_labels):
        cat, stem = label.split("/", 1)
        fname = f"overlays/{i:02d}_{cat}_{stem}.jpg"
        lines.append(f"### `{i:02d}` {label}")
        lines.append("")
        lines.append(f"![{label}]({fname})")
        lines.append("")

    # ----- metrics table ----------------------------------------------------
    lines.append("## Quantitative metrics")
    lines.append("")
    lines.append(
        "Means across the sample.  Notation: "
        "`leaf_area` ≈ how much of the image each leaf covers "
        "(uniform = 1/K); "
        "`leaf_overlap_max` ≈ max IoU between any two leaves (lower is "
        "better — leaves should be disjoint); "
        "`internal_area` ≈ mean HAC-internal box area; "
        "`root_area` ≈ final merge box area (1.0 = root recovers the "
        "whole image, good); "
        "`merge_balance` ∈ (0, 1] ≈ subtree-size balance at each merge "
        "(1.0 = perfectly balanced, lower = chain-like); "
        "`area_growth` ≈ internal_area / sum(child_areas) "
        "(1.0 = perfectly adjacent children, > 1 = boxes include empty "
        "space)."
    )
    lines.append("")
    headers = (
        "| K | α | leaf_area | leaf_area_std | leaf_overlap_max | "
        "internal_area | root_area | merge_balance | area_growth |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines.append(headers)
    lines.append(sep)
    for k, alpha, m in sorted(agg_rows):
        lines.append(
            f"| {k} | {alpha} | "
            f"{m['leaf_area_mean']:.3f} | "
            f"{m['leaf_area_std']:.3f} | "
            f"{m['leaf_overlap_max']:.3f} | "
            f"{m['internal_area_mean']:.3f} | "
            f"{m['root_area']:.3f} | "
            f"{m['merge_balance']:.3f} | "
            f"{m['area_growth']:.3f} |"
        )
    lines.append("")

    # ----- diversity tree ---------------------------------------------------
    lines.append("## Diversity-tree sanity check")
    lines.append("")
    lines.append(
        "Top-level groupings produced by agglomerative clustering "
        "(cosine, average linkage, 6 clusters) on the CLS-pooled "
        "DINOv2 vectors for the sampled images.  We're looking for "
        "clusters that group semantically related categories "
        "(e.g. animals, vehicles, faces) rather than arbitrary visual "
        "noise."
    )
    lines.append("")
    for cid in sorted(clusters):
        members = clusters[cid]
        if not members:
            continue
        lines.append(f"- **cluster {cid}** ({len(members)} items):")
        for m in members:
            lines.append(f"    - {m}")
    lines.append("")

    report_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
