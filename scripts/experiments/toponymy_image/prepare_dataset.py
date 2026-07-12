"""Download + embed an image demo dataset via the vtscore library tier.

Produces, under ``RESULTS/<dataset>/``:

- ``embeddings_<embedder>.npy`` — (n, d) float32 matrix, ingest-normalized,
  exactly what VTSearch's in-memory embedding matrix would hold.
- ``meta.json`` — aligned per-image records: filename, category (ground
  truth), absolute image path (on node scratch) for the image→text stages,
  and for ``mixed`` a ``domain`` field.

Datasets cover the domain-diversity axis of the study:

- ``caltech101`` — generic object photos (101 categories).
- ``stanford_dogs`` — fine-grained photos (120 breeds); stands in for the
  "results of a dog-photo detector" Find→Browse scenario.
- ``enrico`` — mobile-UI screenshots (20 design topics).
- ``rvl_cdip`` — scanned/faxed document pages (16 types); the "faxed
  spreadsheet detector" scenario.
- ``mixed`` — an even sample of the four above, category =
  ``<domain>:<class>``; tests the heterogeneous-corpus case where coarse
  layers should find the domains.

Usage::

    python prepare_dataset.py caltech101 --per-cat 20
    python prepare_dataset.py stanford_dogs --per-cat 15
    python prepare_dataset.py enrico
    python prepare_dataset.py rvl_cdip --per-cat 100
    python prepare_dataset.py mixed --per-source 500   # after the four above
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402

SOURCES = {
    "caltech101": "caltech101",
    "stanford_dogs": "stanford_dogs",
    "enrico": "enrico",
    "rvl_cdip": "rvl_cdip",
}

# Domain tag used for the mixed corpus's coarse ground truth.
DOMAIN = {
    "caltech101": "photo",
    "stanford_dogs": "dog",
    "enrico": "screenshot",
    "rvl_cdip": "document",
}

MIXED_PARTS = ["caltech101", "stanford_dogs", "enrico", "rvl_cdip"]


def discover_categories(source: str) -> list[str]:
    """Download the source (idempotent) and list its ground-truth categories."""
    from vtscore.datasets import downloader

    if source == "caltech101":
        img_dir = downloader.download_caltech101()
        # BACKGROUND_Google is Caltech's junk/clutter class - not a category.
        return sorted(p.name for p in img_dir.iterdir() if p.is_dir() and p.name != "BACKGROUND_Google")
    if source == "stanford_dogs":
        img_dir = downloader.download_stanford_dogs()
        return sorted(p.name.split("-", 1)[1] for p in img_dir.iterdir() if p.is_dir() and "-" in p.name)
    if source == "enrico":
        from vtscore.media.image._demo_categories import ENRICO_CATEGORIES

        downloader.download_enrico()
        return list(ENRICO_CATEGORIES)
    if source == "rvl_cdip":
        img_dir = downloader.download_rvl_cdip()
        return sorted(p.name for p in img_dir.iterdir() if p.is_dir())
    raise ValueError(source)


def prepare_single(args) -> None:
    from vtscore import media as media_registry

    timings: dict = {}
    source = SOURCES[args.dataset]
    with common.timed("discover_categories", timings):
        categories = discover_categories(source)
    print(f"{args.dataset}: {len(categories)} categories")

    mt = media_registry.get("image")
    embedder = media_registry.get_embedder(args.embedder)

    clips: dict = {}
    with common.timed("load_demo_source(download+embed)", timings):
        mt.load_demo_source(
            source=source,
            categories=categories,
            slice_start=0,
            slice_end=args.per_cat,
            clips=clips,
            embedder=embedder,
        )
    print(f"loaded {len(clips)} images")

    ordered = [clips[k] for k in sorted(clips)]
    emb = np.stack([c["embeddings"][embedder.name] for c in ordered]).astype(np.float32)

    # Write each image's bytes to a flat per-dataset cache on scratch so the
    # image→text stages never have to guess the source layout.
    img_cache = common.WORK / "imgs" / args.dataset
    img_cache.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, c in enumerate(ordered):
        ext = Path(c["filename"]).suffix.lower() or ".jpg"
        img_path = img_cache / f"{i:05d}{ext}"
        if not img_path.exists():
            img_path.write_bytes(c["media_bytes"])
        meta.append(
            {
                "filename": c["filename"],
                "category": c["category"],
                "img_path": str(img_path),
            }
        )

    out = common.ds_dir(args.dataset)
    np.save(out / f"embeddings_{embedder.name}.npy", emb)
    common.save_json(out / "meta.json", meta)
    common.save_json(
        out / "prepare_info.json",
        {
            "dataset": args.dataset,
            "source": source,
            "embedder": embedder.name,
            "n_images": len(meta),
            "dim": int(emb.shape[1]),
            "n_categories": len(categories),
            "timings_s": timings,
        },
    )


def prepare_mixed(args) -> None:
    """Even per-source sample of already-prepared datasets; category = domain:class."""
    rng = np.random.default_rng(args.seed)
    all_meta, all_emb = [], []
    part_indices: dict[str, list[int]] = {}
    for part in MIXED_PARTS:
        src = common.ds_dir(part)
        if not (src / "meta.json").exists():
            raise SystemExit(f"prepare {part} first")
        meta = common.load_json(src / "meta.json")
        emb = np.load(src / f"embeddings_{args.embedder}.npy")
        idx = rng.permutation(len(meta))[: args.per_source]
        idx.sort()
        part_indices[part] = [int(i) for i in idx]
        for i in idx:
            m = dict(meta[i])
            m["domain"] = DOMAIN[part]
            m["source_dataset"] = part
            m["category"] = f"{DOMAIN[part]}:{m['category']}"
            all_meta.append(m)
        all_emb.append(emb[idx])
        print(f"mixed <- {part}: {len(idx)} images")

    emb = np.concatenate(all_emb).astype(np.float32)
    out = common.ds_dir("mixed")
    np.save(out / f"embeddings_{args.embedder}.npy", emb)
    common.save_json(out / "meta.json", all_meta)
    common.save_json(
        out / "prepare_info.json",
        {
            "dataset": "mixed",
            "parts": MIXED_PARTS,
            "part_indices": part_indices,
            "per_source": args.per_source,
            "embedder": args.embedder,
            "n_images": len(all_meta),
            "dim": int(emb.shape[1]),
            "seed": args.seed,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=[*sorted(SOURCES), "mixed"])
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--per-cat", type=int, default=None, help="images per category")
    ap.add_argument("--per-source", type=int, default=500, help="mixed: images per source dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset == "mixed":
        prepare_mixed(args)
    else:
        prepare_single(args)


if __name__ == "__main__":
    main()
