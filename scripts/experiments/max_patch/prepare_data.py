"""Stage 0: load + embed each (dataset, embedder) pair once; cache pickles + exemplar crops.

For every embedder in the grid, each dataset is loaded and embedded through
``load_demo_dataset`` (which also attaches the patch side-channels -
``patch_grid`` + ``patch_regions`` - for patch-capable embedders), then the
demo cache pickle is copied to a per-(dataset, embedder) name so the three
embedders don't evict each other from the single demo cache slot.  Array tasks
load these warm pickles directly.

Also pre-computes the **startup-sort exemplar crops**: for each selected
category, up to ``EXEMPLAR_CANDIDATES`` positive images are chosen
deterministically; each one's ground-truth region is cropped out of the image
and embedded as a stand-alone image (the "cropped exemplar"), giving the vector
the runner seeds every style's startup sort with.  Boxless datasets
(caltech101) fall back to the whole-image vector - their photos are already
roughly object crops.

DINOv3 weights are gated on Hugging Face: ``HF_TOKEN`` must be set to a token
that has accepted the licence, or the dinov3 rows are skipped with a loud
warning.

Usage::

    python prepare_data.py            # full grid from experiment_config
    python prepare_data.py --datasets visual_genome_m --embedders dinov2_patch
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402

import experiment_config as cfg  # noqa: E402


def _crop_vector(media: dict, box, embedder) -> "object":
    """Embed the pixel crop of *box* (normalised coords) as a stand-alone image.

    Returns the whole-image embedding of the cropped exemplar - the vector the
    user's "here's an example" startup sort would run on.  ``box=None`` (or a
    degenerate crop) falls back to the media's stored whole-image vector.
    """
    from PIL import Image  # noqa: PLC0415

    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415
    from vtscore.media.embedder import media_from_path  # noqa: PLC0415

    if box is None:
        return media_embedding(media)
    img = Image.open(io.BytesIO(media["media_bytes"])).convert("RGB")
    w, h = img.size
    x0, y0, x1, y1 = box
    px0, py0 = int(min(x0, x1) * w), int(min(y0, y1) * h)
    px1, py1 = int(max(x0, x1) * w), int(max(y0, y1) * h)
    # Enforce a minimum 16-px crop side by symmetric expansion (clamped), so a
    # tiny annotation still produces something the embedder can resize sanely.
    if px1 - px0 < 16:
        cx = (px0 + px1) // 2
        px0, px1 = max(0, cx - 8), min(w, cx + 8)
    if py1 - py0 < 16:
        cy = (py0 + py1) // 2
        py0, py1 = max(0, cy - 8), min(h, cy + 8)
    if px1 <= px0 or py1 <= py0:
        return media_embedding(media)
    crop = img.crop((px0, py0, px1, py1))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        crop.save(tmp, format="PNG")
        tmp_path = Path(tmp.name)
    try:
        vec = embedder.embed_media(media_from_path(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
    return vec if vec is not None else media_embedding(media)


def _exemplar_crops(medias: dict, categories: list[str], embedder) -> tuple[dict, dict]:
    """Return ``(vectors, candidates)`` for the startup-sort exemplars.

    ``vectors`` maps ``"{category}::{media_id}"`` to the cropped exemplar's
    embedding; ``candidates`` maps category to the ordered candidate id list
    (seed *s* uses entry ``s % len``).  Candidates prefer positives that carry a
    ground-truth box for the category; boxless datasets use whole images.
    """
    from vtscore.eval.labels import media_is_positive, region_box_for_category  # noqa: PLC0415

    vectors: dict[str, object] = {}
    candidates: dict[str, list[int]] = {}
    for cat in categories:
        pos = [cid for cid in sorted(medias) if media_is_positive(medias[cid], cat)]
        boxed = [cid for cid in pos if region_box_for_category(medias[cid], cat) is not None]
        pool = boxed or pos
        if not pool:
            continue
        rng = np.random.RandomState(cfg.category_rng_seed(cat))
        n = min(cfg.EXEMPLAR_CANDIDATES, len(pool))
        chosen = [int(c) for c in rng.choice(np.array(pool, dtype=np.int64), size=n, replace=False)]
        for cid in chosen:
            box = region_box_for_category(medias[cid], cat)
            vectors[f"{cat}::{cid}"] = np.asarray(_crop_vector(medias[cid], box, embedder), dtype=np.float32)
        candidates[cat] = chosen
        common.log(f"  exemplars[{cat}]: {n} candidates ({'boxed' if boxed else 'whole-image'})")
    return vectors, candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load + embed each (dataset, embedder) pair for the Max-Patch study.")
    parser.add_argument("--datasets", nargs="+", default=cfg.DATASETS)
    parser.add_argument("--embedders", nargs="+", default=cfg.EMBEDDERS)
    args = parser.parse_args(argv)

    from vtscore.datasets import loader as _loader
    from vtscore.datasets.loader_demo import load_demo_dataset
    from vtscore.embedding import initialize_models
    from vtscore.embedding.media_vectors import media_embedding
    from vtscore.media import get_embedder

    initialize_models()

    crops_dir = common.RESULTS / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    info_path = common.RESULTS / "prepare_info.json"
    info: dict = {"datasets": {}, "failed": []}
    if info_path.exists():
        # Re-runs (e.g. after adding an embedder) keep earlier entries.
        info = json.loads(info_path.read_text())

    for emb_name in args.embedders:
        if emb_name.startswith("dinov3") and not os.environ.get("HF_TOKEN"):
            common.log(f"SKIPPING {emb_name}: HF_TOKEN is not set (DINOv3 weights are licence-gated on HF).")
            info["failed"].append(emb_name)
            continue
        embedder = get_embedder(emb_name)
        for ds in args.datasets:
            common.log(f"\n=== {ds} x {emb_name} ===")
            timings: dict[str, float] = {}
            medias: dict[int, dict] = {}
            try:
                with common.timed(f"load:{ds}:{emb_name}", timings):
                    load_demo_dataset(ds, medias, embedder_name=emb_name)
                    # load_demo_dataset attaches the CLS vector but not the
                    # patch-region side-channels; run the same patch back-fill
                    # the app runs at dataset creation so patch_grid /
                    # patch_regions exist to persist (embed_missing is a no-op
                    # for the already-present CLS vectors and only does the
                    # patch pass for this patch-capable embedder).
                    from vtscore.datasets.stages.embedding import embed_missing  # noqa: PLC0415

                    embed_missing(medias, emb_name)
            except Exception as e:  # noqa: BLE001 - one bad pair must not lose the others
                import traceback

                common.log(f"FAILED to load {ds} with {emb_name}: {e}")
                traceback.print_exc()
                info["failed"].append(f"{ds}:{emb_name}")
                info_path.write_text(json.dumps(info, indent=2))
                continue

            # Serialize the in-memory medias (which carry patch_grid /
            # patch_regions / regions / categories) directly.  The demo *cache*
            # pickle drops those fields (see _cells_io), so a copy would leave
            # every style scoring on the whole-image vector alone.
            from _cells_io import dump_medias  # noqa: PLC0415

            dst = _loader.EMBEDDINGS_DIR / cfg.pickle_name(ds, emb_name)
            nbytes = dump_medias(medias, dst)
            common.log(f"  wrote cell pickle {dst.name}: {nbytes / 1e6:.0f} MB")

            cats: dict[str, int] = {}
            for m in medias.values():
                for c in m.get("categories") or [m.get("category")]:
                    if c:
                        cats[c] = cats.get(c, 0) + 1
            selected, sel_report = cfg.select_categories(medias, cats)
            dim = int(len(media_embedding(next(iter(medias.values()))))) if medias else None
            n_patch = sum(1 for m in medias.values() if m.get("patch_grid") is not None)
            common.log(
                f"{ds} x {emb_name}: {len(medias)} medias (patch grids on {n_patch}), dim={dim}, "
                f"{len(cats)} categories -> selected {len(selected)} by {sel_report.get('mode')}"
            )
            # Report the scale sample explicitly: an under-populated band is the
            # difference between testing the crossover and merely bracketing it.
            for band, info_b in (sel_report.get("bands") or {}).items():
                lo, hi = info_b["range"]
                common.log(
                    f"  band {band:14s} [{lo * 100:5.2f}%, {hi * 100:6.2f}%): "
                    f"{len(info_b['selected'])}/{info_b['target']} from {info_b['n_candidates']} candidates"
                    f"{'  ** UNDER-POPULATED **' if info_b['under_populated'] else ''} -> {info_b['selected']}"
                )
            for cat, area in sel_report.get("dropped_above_max_voted_area") or []:
                common.log(
                    f"  dropped {cat!r}: median voted box covers {area * 100:.0f}% of the image "
                    f"(> {cfg.MAX_VOTED_AREA * 100:.0f}% cap) - its region vote is an image-level vote"
                )

            with common.timed(f"crops:{ds}:{emb_name}", timings):
                vectors, candidates = _exemplar_crops(medias, selected, embedder)
            np.savez_compressed(crops_dir / f"{cfg.crops_basename(ds, emb_name)}.npz", **vectors)
            (crops_dir / f"{cfg.crops_basename(ds, emb_name)}.json").write_text(json.dumps(candidates, indent=2))

            info["datasets"].setdefault(ds, {})[emb_name] = {
                "n_medias": len(medias),
                "n_patch_grids": n_patch,
                "dim": dim,
                "category_counts": cats,
                "selected_categories": selected,
                "category_selection": sel_report,
                "load_seconds": timings.get(f"load:{ds}:{emb_name}"),
                "embed_seconds_per_image": round(timings.get(f"load:{ds}:{emb_name}", 0.0) / max(len(medias), 1), 4),
                "crops_seconds": timings.get(f"crops:{ds}:{emb_name}"),
                "pickle": cfg.pickle_name(ds, emb_name),
            }
            info_path.write_text(json.dumps(info, indent=2))
            del medias

    common.log(f"\nWrote {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
