"""Stage 0: ensure a per-(dataset, embedder) pickle + exemplar crops for every arm.

Reuse-aware: where the Max-Patch runner already embedded a ``(dataset, embedder)``
pair, ``setup_reuse.sh`` symlinks its pickle (and, for boxed datasets, its
exemplar crops) into this experiment's datadir, and this stage just **loads the
cached pickle** to (re)derive the selected categories — no model, no re-embed.
For a pair with no cached pickle (``siglip_l`` here), it loads + embeds the demo
dataset through ``load_demo_dataset`` + ``embed_missing`` and writes the pickle.

Exemplar crops: boxed datasets (Visual Genome) crop each category's
ground-truth region and embed it (needs the embedder + image bytes, so only done
on the fresh path); boxless datasets (Caltech) use the whole-image vector already
in the pickle (no model, no bytes).

Usage::

    python prepare_data.py                         # every (dataset, embedder) in the grid
    python prepare_data.py --embedders siglip_l    # just the pair(s) that need embedding
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
    """Embed the pixel crop of *box* as a stand-alone image, or fall back.

    Returns the whole-image embedding when there is no box, no embedder, or no
    stored raster (the reuse path has neither model nor bytes) — exactly an
    image-level exemplar, which is what a boxless dataset wants anyway.
    """
    from PIL import Image  # noqa: PLC0415

    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415
    from vtscore.media.embedder import media_from_path  # noqa: PLC0415

    if box is None or embedder is None or media.get("media_bytes") is None:
        return media_embedding(media)
    img = Image.open(io.BytesIO(media["media_bytes"])).convert("RGB")
    w, h = img.size
    x0, y0, x1, y1 = box
    px0, py0 = int(min(x0, x1) * w), int(min(y0, y1) * h)
    px1, py1 = int(max(x0, x1) * w), int(max(y0, y1) * h)
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
    """``(vectors, candidates)`` for the startup-sort exemplars.

    Candidates prefer positives carrying a ground-truth box for the category;
    boxless datasets use whole images.  With ``embedder=None`` (reuse path) every
    crop degrades to the whole-image vector.
    """
    from vtscore.eval.labels import (  # noqa: PLC0415
        evaluable_pool,
        media_is_positive,
        region_box_for_category,
    )

    vectors: dict[str, object] = {}
    candidates: dict[str, list[int]] = {}
    for cat in categories:
        # Wrong-band images are excluded from the cell, so they must not seed it
        # either: an exemplar crop taken from one is a crop of the right object
        # at the wrong scale.
        medias_cat = evaluable_pool(medias, cat)
        pos = [cid for cid in sorted(medias_cat) if media_is_positive(medias_cat[cid], cat)]
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


def _category_counts(medias: dict) -> dict[str, int]:
    cats: dict[str, int] = {}
    for m in medias.values():
        for c in m.get("categories") or [m.get("category")]:
            if c:
                cats[c] = cats.get(c, 0) + 1
    return cats


def _prepare_pair(ds: str, emb_name: str, info: dict) -> None:
    from vtscore.datasets import loader as _loader
    from vtscore.embedding.media_vectors import media_embedding

    from _cells_io import dump_medias, load_medias  # noqa: PLC0415

    pkl = _loader.EMBEDDINGS_DIR / cfg.pickle_name(ds, emb_name)
    crops_np = common.RESULTS / "crops" / f"{cfg.crops_basename(ds, emb_name)}.npz"
    crops_js = crops_np.with_suffix(".json")

    timings: dict[str, float] = {}
    medias: dict[int, dict] = {}
    embedder = None

    if pkl.exists():
        common.log(f"\n=== {ds} x {emb_name} === (reusing cached pickle {pkl.name})")
        with common.timed(f"load_cache:{ds}:{emb_name}", timings):
            medias = load_medias(pkl)
    else:
        common.log(f"\n=== {ds} x {emb_name} === (embedding fresh)")
        if emb_name.startswith("dinov3") and not os.environ.get("HF_TOKEN"):
            common.log(f"SKIPPING {emb_name}: HF_TOKEN not set (DINOv3 weights are licence-gated).")
            info.setdefault("failed", []).append(f"{ds}:{emb_name}")
            return
        from vtscore.datasets.loader_demo import load_demo_dataset  # noqa: PLC0415
        from vtscore.datasets.stages.embedding import embed_missing  # noqa: PLC0415
        from vtscore.media import get_embedder  # noqa: PLC0415

        embedder = get_embedder(emb_name)
        with common.timed(f"load:{ds}:{emb_name}", timings):
            load_demo_dataset(ds, medias, embedder_name=emb_name)
            embed_missing(medias, emb_name)
        nbytes = dump_medias(medias, pkl)
        common.log(f"  wrote cell pickle {pkl.name}: {nbytes / 1e6:.0f} MB")

    cats = _category_counts(medias)
    selected, sel_report = cfg.select_categories(medias, cats)
    if cfg.MIN_SIM_POSITIVES > 0:
        # Keep only categories deep enough to sustain a long horizon.  The
        # simulation set is SIM_FRACTION of the medias, so a category's usable
        # positives are its count scaled by that fraction.
        kept, dropped = [], []
        for c in selected:
            n_sim_pos = int(cats.get(c, 0) * cfg.SIM_FRACTION)
            (kept if n_sim_pos >= cfg.MIN_SIM_POSITIVES else dropped).append((c, n_sim_pos))
        sel_report["min_sim_positives"] = cfg.MIN_SIM_POSITIVES
        sel_report["dropped_too_shallow"] = sorted(dropped, key=lambda kv: kv[1])
        sel_report["kept_sim_positives"] = dict(sorted(kept, key=lambda kv: -kv[1]))
        selected = sorted(c for c, _ in kept)
        common.log(
            f"  deep-category filter (>= {cfg.MIN_SIM_POSITIVES} sim positives): "
            f"kept {len(kept)}, dropped {len(dropped)}"
        )
    dim = int(len(media_embedding(next(iter(medias.values()))))) if medias else None
    n_patch = sum(1 for m in medias.values() if m.get("patch_grid") is not None)
    common.log(
        f"{ds} x {emb_name}: {len(medias)} medias (patch grids on {n_patch}), dim={dim}, "
        f"{len(cats)} categories -> selected {len(selected)} by {sel_report.get('mode')}"
    )
    for band, info_b in (sel_report.get("bands") or {}).items():
        lo, hi = info_b["range"]
        common.log(
            f"  band {band:14s} [{lo * 100:5.2f}%, {hi * 100:6.2f}%): "
            f"{len(info_b['selected'])}/{info_b['target']} from {info_b['n_candidates']} candidates"
            f"{'  ** UNDER-POPULATED **' if info_b['under_populated'] else ''} -> {info_b['selected']}"
        )

    if crops_np.exists() and crops_js.exists():
        common.log(f"  reusing cached crops {crops_np.name}")
    else:
        crops_np.parent.mkdir(parents=True, exist_ok=True)
        with common.timed(f"crops:{ds}:{emb_name}", timings):
            vectors, candidates = _exemplar_crops(medias, selected, embedder)
        np.savez_compressed(crops_np, **vectors)
        crops_js.write_text(json.dumps(candidates, indent=2))
        common.log(f"  wrote crops {crops_np.name} ({len(vectors)} exemplars)")

    info["datasets"].setdefault(ds, {})[emb_name] = {
        "n_medias": len(medias),
        "n_patch_grids": n_patch,
        "dim": dim,
        "category_counts": cats,
        "selected_categories": selected,
        "category_selection": sel_report,
        "reused_pickle": pkl.exists() and embedder is None,
        "load_seconds": timings.get(f"load:{ds}:{emb_name}") or timings.get(f"load_cache:{ds}:{emb_name}"),
        "crops_seconds": timings.get(f"crops:{ds}:{emb_name}"),
        "pickle": cfg.pickle_name(ds, emb_name),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare (or reuse) each (dataset, embedder) pair for #2781.")
    parser.add_argument("--datasets", nargs="+", default=cfg.DATASETS)
    parser.add_argument(
        "--embedders", nargs="+", default=None, help="Restrict to these embedders (default: all in grid)."
    )
    args = parser.parse_args(argv)

    from vtscore.embedding import initialize_models

    initialize_models()

    (common.RESULTS / "crops").mkdir(parents=True, exist_ok=True)
    info_path = common.RESULTS / "prepare_info.json"
    info: dict = {"datasets": {}, "failed": []}
    if info_path.exists():
        info = json.loads(info_path.read_text())
        info.setdefault("datasets", {})
        info.setdefault("failed", [])

    for ds in args.datasets:
        embs = cfg.embedders_for_dataset(ds)
        if args.embedders is not None:
            embs = [e for e in embs if e in args.embedders]
        for emb_name in embs:
            try:
                _prepare_pair(ds, emb_name, info)
            except Exception as e:  # noqa: BLE001 - one bad pair must not lose the others
                import traceback

                common.log(f"FAILED {ds} x {emb_name}: {e}")
                traceback.print_exc()
                info["failed"].append(f"{ds}:{emb_name}")
            info_path.write_text(json.dumps(info, indent=2))

    common.log(f"\nWrote {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
