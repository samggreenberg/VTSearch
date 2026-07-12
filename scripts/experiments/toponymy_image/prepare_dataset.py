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
        return sorted(_stanford_dogs_by_breed())
    if source == "enrico":
        from vtscore.config import DATA_DIR
        from vtscore.media.image._demo_categories import ENRICO_CATEGORIES

        # NOTE: found during this study — the Aalto server 404s on
        # design_topics.csv (URL rot); the Enrico GitHub repo still serves
        # it. Pre-place the file so download_enrico skips its dead URL.
        topics_csv = DATA_DIR / "enrico" / "design_topics.csv"
        if not topics_csv.exists():
            import requests

            topics_csv.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(
                "https://raw.githubusercontent.com/luileito/enrico/master/design_topics.csv", timeout=120
            )
            resp.raise_for_status()
            topics_csv.write_text(resp.text)
        downloader.download_enrico()
        return list(ENRICO_CATEGORIES)
    if source == "rvl_cdip":
        img_dir = downloader.download_rvl_cdip()
        return sorted(p.name for p in img_dir.iterdir() if p.is_dir())
    raise ValueError(source)


def _stanford_dogs_by_breed() -> dict[str, list]:
    """Group the flat Stanford Dogs mirror by breed.

    NOTE: found during this study — the HF mirror tarball
    (``Alanox/stanford-dogs`` images.tar.gz) extracts to a FLAT ``images/``
    dir of ``<wnid>_<num>.jpg`` files, not the ``Images/<wnid>-<Breed>/``
    tree ``download_stanford_dogs`` documents and ``check_path`` expects, so
    the vtscore demo loader finds no breed folders (and re-downloads 750 MB
    on every load). Here we recover breed names from the WordNet synset
    offset in each filename instead.
    """
    from vtscore.config import DATA_DIR

    flat = DATA_DIR / "stanford_dogs" / "images"
    if not flat.is_dir() or sum(1 for _ in flat.glob("*.jpg")) < 10000:
        from vtscore.datasets import downloader

        downloader.download_stanford_dogs()

    import nltk

    nltk.data.path.insert(0, str(common.WORK / "nltk_data"))
    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", download_dir=str(common.WORK / "nltk_data"), quiet=True)
        from nltk.corpus import wordnet as wn

    by_breed: dict[str, list] = {}
    breed_of: dict[str, str] = {}
    for p in sorted(flat.glob("*.jpg")):
        wnid = p.name.split("_", 1)[0]
        if wnid not in breed_of:
            offset = int(wnid[1:])
            breed_of[wnid] = wn.synset_from_pos_and_offset("n", offset).lemma_names()[0].replace("_", " ")
        by_breed.setdefault(breed_of[wnid], []).append(p)
    return by_breed


def prepare_stanford_dogs(args) -> None:
    """Flat-mirror path: embed per-breed slices directly (see note above)."""
    from vtscore import media as media_registry
    from vtscore.media.embedder import media_from_path

    timings: dict = {}
    with common.timed("discover+group", timings):
        by_breed = _stanford_dogs_by_breed()
    print(f"stanford_dogs: {len(by_breed)} breeds")

    embedder = media_registry.get_embedder(args.embedder)
    embedder.load_models()

    img_cache = common.WORK / "imgs" / args.dataset
    img_cache.mkdir(parents=True, exist_ok=True)
    meta, vecs = [], []
    with common.timed("embed", timings):
        i = 0
        for breed in sorted(by_breed):
            for p in by_breed[breed][: args.per_cat]:
                v = embedder.embed_media(media_from_path(p))
                if v is None:
                    continue
                img_path = img_cache / f"{i:05d}{p.suffix.lower()}"
                if not img_path.exists():
                    img_path.write_bytes(p.read_bytes())
                meta.append({"filename": f"{breed}/{p.name}", "category": breed, "img_path": str(img_path)})
                vecs.append(v)
                i += 1
    emb = np.stack(vecs).astype(np.float32)
    print(f"loaded {len(meta)} images")

    out = common.ds_dir(args.dataset)
    np.save(out / f"embeddings_{args.embedder}.npy", emb)
    common.save_json(out / "meta.json", meta)
    common.save_json(
        out / "prepare_info.json",
        {
            "dataset": args.dataset,
            "source": "stanford_dogs(flat-mirror workaround)",
            "embedder": args.embedder,
            "n_images": len(meta),
            "dim": int(emb.shape[1]),
            "n_categories": len(by_breed),
            "timings_s": timings,
        },
    )


RVL_PARQUET_URLS = [
    "https://huggingface.co/api/datasets/jordyvl/rvl_cdip_100_examples_per_class/parquet/default/train/0.parquet",
    "https://huggingface.co/api/datasets/jordyvl/rvl_cdip_100_examples_per_class/parquet/default/test/0.parquet",
    "https://huggingface.co/api/datasets/jordyvl/rvl_cdip_100_examples_per_class/parquet/default/validation/0.parquet",
]


def prepare_rvl_cdip(args) -> None:
    """Direct parquet path for RVL-CDIP.

    NOTE: found during this study — the vtscore demo mirror
    (``umair894/rvl_cdip_300_examples_per_class``) actually contains ONLY the
    300 ``invoice`` examples in its single train shard, so the demo importer
    yields a one-class dataset. This uses the balanced
    ``jordyvl/rvl_cdip_100_examples_per_class`` mirror (16 classes) instead.
    """
    import io

    import pyarrow.parquet as pq
    import requests
    from PIL import Image

    from vtscore import media as media_registry
    from vtscore.media.embedder import media_from_path
    from vtscore.media.image._demo_categories import RVL_CDIP_CATEGORIES

    timings: dict = {}
    pq_dir = common.WORK / "rvl_parquet"
    pq_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with common.timed("download+decode_parquet", timings):
        for k, url in enumerate(RVL_PARQUET_URLS):
            path = pq_dir / f"{k}.parquet"
            if not path.exists():
                resp = requests.get(url, timeout=600)
                resp.raise_for_status()
                path.write_bytes(resp.content)
            t = pq.read_table(path, columns=["image", "label"])
            for img_struct, label in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
                if isinstance(label, int) and 0 <= label < len(RVL_CDIP_CATEGORIES):
                    rows.append((RVL_CDIP_CATEGORIES[label], img_struct["bytes"]))

    by_cat: dict[str, list[bytes]] = {}
    for cat, blob in rows:
        by_cat.setdefault(cat, []).append(blob)
    print(f"rvl_cdip: {len(by_cat)} classes, {len(rows)} pages")

    embedder = media_registry.get_embedder(args.embedder)
    embedder.load_models()
    img_cache = common.WORK / "imgs" / args.dataset
    img_cache.mkdir(parents=True, exist_ok=True)
    meta, vecs = [], []
    with common.timed("embed", timings):
        i = 0
        for cat in sorted(by_cat):
            for blob in by_cat[cat][: args.per_cat]:
                img_path = img_cache / f"{i:05d}.png"
                if not img_path.exists():
                    Image.open(io.BytesIO(blob)).convert("RGB").save(img_path)
                v = embedder.embed_media(media_from_path(img_path))
                if v is None:
                    continue
                meta.append({"filename": f"{cat}/{i:05d}.png", "category": cat, "img_path": str(img_path)})
                vecs.append(v)
                i += 1
    emb = np.stack(vecs).astype(np.float32)
    print(f"loaded {len(meta)} images")

    out = common.ds_dir(args.dataset)
    np.save(out / f"embeddings_{args.embedder}.npy", emb)
    common.save_json(out / "meta.json", meta)
    common.save_json(
        out / "prepare_info.json",
        {
            "dataset": args.dataset,
            "source": "rvl_cdip(jordyvl balanced mirror workaround)",
            "embedder": args.embedder,
            "n_images": len(meta),
            "dim": int(emb.shape[1]),
            "n_categories": len(by_cat),
            "timings_s": timings,
        },
    )


def prepare_enrico(args) -> None:
    """Direct path for Enrico.

    NOTE: found during this study — the Aalto screenshots.zip now unpacks to
    flat ``<screen_id>.jpg`` names (the ``<id>-screenshot.jpg`` convention
    vtscore's collector and completion probe expect is gone), and the labels
    CSV 404s at its documented URL (GitHub still serves it). Both make the
    vtscore enrico demo load 0 images; this collects the new layout directly.
    """
    import csv

    from vtscore import media as media_registry
    from vtscore.config import DATA_DIR
    from vtscore.media.embedder import media_from_path
    from vtscore.media.image._demo_categories import ENRICO_CATEGORIES

    timings: dict = {}
    with common.timed("download", timings):
        discover_categories("enrico")  # pre-places CSV + downloads/extracts zip
    enrico_dir = DATA_DIR / "enrico"
    display_by_norm = {c.lower(): c for c in ENRICO_CATEGORIES}
    id_to_cat: dict[str, str] = {}
    with open(enrico_dir / "design_topics.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = (row.get("screen_id") or "").strip()
            cat = display_by_norm.get((row.get("topic") or "").strip().lower())
            if sid and cat:
                id_to_cat[sid] = cat

    files = []
    for p in sorted(enrico_dir.rglob("*.jpg")):
        sid = p.stem.replace("-screenshot", "")
        if sid in id_to_cat:
            files.append((p, id_to_cat[sid]))
    print(f"enrico: {len(files)} labeled screenshots")

    embedder = media_registry.get_embedder(args.embedder)
    embedder.load_models()
    img_cache = common.WORK / "imgs" / args.dataset
    img_cache.mkdir(parents=True, exist_ok=True)
    meta, vecs = [], []
    with common.timed("embed", timings):
        for i, (p, cat) in enumerate(files):
            v = embedder.embed_media(media_from_path(p))
            if v is None:
                continue
            img_path = img_cache / f"{i:05d}.jpg"
            if not img_path.exists():
                img_path.write_bytes(p.read_bytes())
            meta.append({"filename": f"{cat}/{p.name}", "category": cat, "img_path": str(img_path)})
            vecs.append(v)
    emb = np.stack(vecs).astype(np.float32)
    print(f"loaded {len(meta)} images")

    out = common.ds_dir(args.dataset)
    np.save(out / f"embeddings_{args.embedder}.npy", emb)
    common.save_json(out / "meta.json", meta)
    common.save_json(
        out / "prepare_info.json",
        {
            "dataset": args.dataset,
            "source": "enrico(flat-zip workaround)",
            "embedder": args.embedder,
            "n_images": len(meta),
            "dim": int(emb.shape[1]),
            "n_categories": len({c for _, c in files}),
            "timings_s": timings,
        },
    )


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
    elif args.dataset == "stanford_dogs":
        prepare_stanford_dogs(args)
    elif args.dataset == "rvl_cdip":
        prepare_rvl_cdip(args)
    elif args.dataset == "enrico":
        prepare_enrico(args)
    else:
        prepare_single(args)


if __name__ == "__main__":
    main()
