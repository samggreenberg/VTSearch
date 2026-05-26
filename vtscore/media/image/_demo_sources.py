"""Demo dataset construction and loading for image media.

Builds the :class:`~vtscore.media.base.DemoDataset` list returned by
``ImageMediaType.demo_datasets`` and implements the per-source download
+ embed dispatcher behind ``ImageMediaType.load_demo_source``.

Both live here rather than on the class because they are pure functions
of the demo-category constants - they don't touch instance state and
splitting them out keeps ``media_type.py`` focused on the
``MediaType`` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from vtscore.config import DATA_DIR
from vtscore.media.base import DemoDataset, demo_slice
from vtscore.media.image._demo_categories import (
    DEMO_CATEGORIES_CALTECH101,
    DEMO_CATEGORIES_CALTECH256,
    EUROSAT_CATEGORIES,
    FOOD101_CATEGORIES,
    OXFORD_FLOWERS_CATEGORIES,
    PLACES365_CATEGORIES,
    STANFORD_DOGS_CATEGORIES,
    UCSF_DOCUMENTS_CATEGORIES,
)


_MEDIA_TYPE_ID = "image"


def build_demo_datasets() -> list[DemoDataset]:
    """Build the demo-dataset catalog exposed by :class:`ImageMediaType`."""
    from vtscore.datasets.downloader import (  # noqa: PLC0415
        CALTECH101_DOWNLOAD_SIZE_MB,
        CALTECH256_DOWNLOAD_SIZE_MB,
        EUROSAT_DOWNLOAD_SIZE_MB,
        FOOD101_DOWNLOAD_SIZE_MB,
        OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
        PLACES365_DOWNLOAD_SIZE_MB,
        STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        UCSF_IDL_DOWNLOAD_SIZE_MB,
    )

    cats101 = DEMO_CATEGORIES_CALTECH101
    cats256 = DEMO_CATEGORIES_CALTECH256
    ct101_desc = "Centered object photos"
    ct101_folder = DATA_DIR / "caltech-101" / "101_ObjectCategories"
    food_desc = "Crowd-sourced food photos"
    food_folder = DATA_DIR / "food-101" / "images"
    euro_desc = "Satellite imagery by land use"
    euro_folder = DATA_DIR / "EuroSAT_RGB"
    dogs_desc = "Dog breeds"
    dogs_folder = DATA_DIR / "stanford_dogs" / "Images"
    places_desc = "Indoor & outdoor scenes"
    places_folder = DATA_DIR / "places365" / "val_256"
    return [
        DemoDataset(
            id="caltech101_s",
            label="Caltech-101 (S)",
            description=ct101_desc,
            categories=cats101,
            source="caltech101",
            required_folder=ct101_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=86,
            download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="caltech101_m",
            label="Caltech-101 (M)",
            description=ct101_desc,
            categories=cats101,
            source="caltech101",
            required_folder=ct101_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=86,
            download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="caltech101_l",
            label="Caltech-101 (L)",
            description=ct101_desc,
            categories=cats101,
            source="caltech101",
            required_folder=ct101_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=86,
            download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="caltech101_a",
            label="Caltech-101 (A)",
            description=ct101_desc,
            categories=cats101,
            source="caltech101",
            required_folder=ct101_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=86,
            download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="caltech256_a",
            label="Caltech-256 (A)",
            description="Cluttered object photos",
            categories=cats256,
            source="caltech256",
            required_folder=DATA_DIR / "caltech-256" / "256_ObjectCategories",
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=119,
            download_size_mb=CALTECH256_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="oxford_flowers_102_a",
            label="Oxford Flowers 102 (A)",
            description="Close-up flower photos",
            categories=OXFORD_FLOWERS_CATEGORIES,
            source="oxford_flowers_102",
            required_folder=DATA_DIR / "oxford_flowers",
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=80,
            download_size_mb=OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="food101_s",
            label="Food-101 (S)",
            description=food_desc,
            categories=FOOD101_CATEGORIES,
            source="food101",
            required_folder=food_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=1000,
            download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="food101_m",
            label="Food-101 (M)",
            description=food_desc,
            categories=FOOD101_CATEGORIES,
            source="food101",
            required_folder=food_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=1000,
            download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="food101_l",
            label="Food-101 (L)",
            description=food_desc,
            categories=FOOD101_CATEGORIES,
            source="food101",
            required_folder=food_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=1000,
            download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="food101_a",
            label="Food-101 (A)",
            description=food_desc,
            categories=FOOD101_CATEGORIES,
            source="food101",
            required_folder=food_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=1000,
            download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="eurosat_s",
            label="EuroSAT (S)",
            description=euro_desc,
            categories=EUROSAT_CATEGORIES,
            source="eurosat",
            required_folder=euro_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=2700,
            download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="eurosat_m",
            label="EuroSAT (M)",
            description=euro_desc,
            categories=EUROSAT_CATEGORIES,
            source="eurosat",
            required_folder=euro_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=2700,
            download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="eurosat_l",
            label="EuroSAT (L)",
            description=euro_desc,
            categories=EUROSAT_CATEGORIES,
            source="eurosat",
            required_folder=euro_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=2700,
            download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="eurosat_a",
            label="EuroSAT (A)",
            description=euro_desc,
            categories=EUROSAT_CATEGORIES,
            source="eurosat",
            required_folder=euro_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=2700,
            download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="stanford_dogs_s",
            label="Stanford Dogs (S)",
            description=dogs_desc,
            categories=STANFORD_DOGS_CATEGORIES,
            source="stanford_dogs",
            required_folder=dogs_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=171,
            download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="stanford_dogs_m",
            label="Stanford Dogs (M)",
            description=dogs_desc,
            categories=STANFORD_DOGS_CATEGORIES,
            source="stanford_dogs",
            required_folder=dogs_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=171,
            download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="stanford_dogs_l",
            label="Stanford Dogs (L)",
            description=dogs_desc,
            categories=STANFORD_DOGS_CATEGORIES,
            source="stanford_dogs",
            required_folder=dogs_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=171,
            download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="stanford_dogs_a",
            label="Stanford Dogs (A)",
            description=dogs_desc,
            categories=STANFORD_DOGS_CATEGORIES,
            source="stanford_dogs",
            required_folder=dogs_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=171,
            download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="places365_s",
            label="Places365 (S)",
            description=places_desc,
            categories=PLACES365_CATEGORIES,
            source="places365",
            required_folder=places_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=100,
            download_size_mb=PLACES365_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="places365_m",
            label="Places365 (M)",
            description=places_desc,
            categories=PLACES365_CATEGORIES,
            source="places365",
            required_folder=places_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=100,
            download_size_mb=PLACES365_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="places365_l",
            label="Places365 (L)",
            description=places_desc,
            categories=PLACES365_CATEGORIES,
            source="places365",
            required_folder=places_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=100,
            download_size_mb=PLACES365_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="places365_a",
            label="Places365 (A)",
            description=places_desc,
            categories=PLACES365_CATEGORIES,
            source="places365",
            required_folder=places_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=100,
            download_size_mb=PLACES365_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="ucsf_documents_a",
            label="UCSF Documents (A)",
            description="Scanned document pages",
            categories=UCSF_DOCUMENTS_CATEGORIES,
            source="ucsf_documents",
            required_folder=DATA_DIR / "ucsf_documents",
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=25,
            download_size_mb=UCSF_IDL_DOWNLOAD_SIZE_MB,
        ),
    ]


_FILE_SOURCE_DOWNLOADERS: dict[str, str] = {
    "caltech101": "download_caltech101",
    "caltech256": "download_caltech256",
    "food101": "download_food101",
    "eurosat": "download_eurosat",
}


def _slice_by_category(by_cat: dict, categories, slice_start, slice_end, slice_frac_start, slice_frac_end) -> list:
    out: list = []
    for cat in categories:
        out.extend(demo_slice(by_cat.get(cat, []), slice_start, slice_end, slice_frac_start, slice_frac_end))
    return out


def _collect_simple_folder_files(source, categories, slice_args, on_progress) -> list:
    """Sources that just download → load_image_metadata_from_folders → group by category."""
    from vtscore.datasets import downloader  # noqa: PLC0415
    from vtscore.datasets.loader import load_image_metadata_from_folders  # noqa: PLC0415

    download_fn = getattr(downloader, _FILE_SOURCE_DOWNLOADERS[source])
    img_dir = download_fn(on_progress=on_progress)
    metadata = load_image_metadata_from_folders(img_dir, categories)
    by_cat: dict = {}
    for _fname, meta in sorted(metadata.items()):
        by_cat.setdefault(meta["category"], []).append((meta["path"], meta["category"]))
    return _slice_by_category(by_cat, categories, *slice_args)


def _collect_oxford_flowers_files(categories, slice_args, on_progress) -> list:
    from vtscore.datasets.downloader import download_oxford_flowers  # noqa: PLC0415
    from vtscore.datasets.loader import load_oxford_flowers_metadata  # noqa: PLC0415

    flowers_dir = download_oxford_flowers(on_progress=on_progress)
    metadata = load_oxford_flowers_metadata(flowers_dir, OXFORD_FLOWERS_CATEGORIES)

    by_cat: dict = {}
    for _fname, meta in sorted(metadata.items()):
        if meta["category"] in categories:
            by_cat.setdefault(meta["category"], []).append((meta["path"], meta["category"]))
    return _slice_by_category(by_cat, categories, *slice_args)


def _collect_stanford_dogs_files(categories, slice_args, on_progress) -> list:
    from vtscore.datasets.downloader import download_stanford_dogs  # noqa: PLC0415

    images_dir = download_stanford_dogs(on_progress=on_progress)
    breed_to_folder: dict[str, Path] = {}
    if images_dir.exists():
        for folder in images_dir.iterdir():
            if folder.is_dir() and "-" in folder.name:
                breed_to_folder[folder.name.split("-", 1)[1]] = folder

    by_cat: dict = {}
    for cat in categories:
        folder = breed_to_folder.get(cat)
        if folder is None:
            continue
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in sorted(folder.glob(ext)):
                by_cat.setdefault(cat, []).append((img_path, cat))
    return _slice_by_category(by_cat, categories, *slice_args)


def _collect_places365_files(categories, slice_args, on_progress) -> list:
    from vtscore.datasets.downloader import download_places365  # noqa: PLC0415
    from vtscore.datasets.loader import load_places365_metadata  # noqa: PLC0415

    places_dir = download_places365(on_progress=on_progress)
    metadata = load_places365_metadata(places_dir, PLACES365_CATEGORIES)

    by_cat: dict = {}
    for _fname, meta in sorted(metadata.items()):
        if meta["category"] in categories:
            by_cat.setdefault(meta["category"], []).append((meta["path"], meta["category"]))
    return _slice_by_category(by_cat, categories, *slice_args)


def _collect_ucsf_doc_pages(categories, slice_args, on_progress) -> list:
    """Returns a list of (page_name, PIL.Image, category) tuples."""
    from vtscore.datasets.downloader import download_ucsf_documents  # noqa: PLC0415
    from vtscore.datasets.pdf import render_pdf_pages  # noqa: PLC0415

    docs_dir = download_ucsf_documents(categories, on_progress=on_progress)

    by_cat_pages: dict = {}
    for cat in categories:
        cat_dir = docs_dir / cat
        if not cat_dir.is_dir():
            continue
        for pdf_path in sorted(cat_dir.glob("*.pdf")):
            try:
                pages = render_pdf_pages(pdf_path, dpi=150)
                if pages:
                    by_cat_pages.setdefault(cat, []).append(pages[0])
            except Exception:
                continue

    selected_pages: list = []
    slice_start, slice_end, slice_frac_start, slice_frac_end = slice_args
    for cat in categories:
        for page_name, pil_image in demo_slice(
            by_cat_pages.get(cat, []), slice_start, slice_end, slice_frac_start, slice_frac_end
        ):
            selected_pages.append((page_name, pil_image, cat))
    return selected_pages


def _collect_cifar10_images(categories, slice_args, on_progress) -> list:
    """Returns a list of (image_array, category) tuples."""
    from vtscore.datasets.downloader import download_cifar10  # noqa: PLC0415
    from vtscore.datasets.loader import load_cifar10_batch  # noqa: PLC0415

    cifar_dir = download_cifar10(on_progress=on_progress)
    images, labels, label_names = load_cifar10_batch(cifar_dir / "data_batch_1")
    category_indices = {label_names[i]: i for i in range(len(label_names))}

    slice_start, slice_end, slice_frac_start, slice_frac_end = slice_args
    selected: list = []
    for cat in categories:
        if cat not in category_indices:
            continue
        cat_idx = category_indices[cat]
        cat_mask = [i for i, lbl in enumerate(labels) if lbl == cat_idx]
        for idx in demo_slice(cat_mask, slice_start, slice_end or len(cat_mask), slice_frac_start, slice_frac_end):
            selected.append((images[idx], cat))
    return selected


def _ensure_image_embedder_loaded(embedder, on_progress) -> None:
    if getattr(embedder, "_model", None) is None:
        on_progress("loading", "Loading image embedding model…", 0, 0)
        original_cb = embedder._on_progress
        embedder._on_progress = on_progress
        try:
            embedder.load_models()
        finally:
            embedder._on_progress = original_cb


def _embed_file_images(selected, clips, embedder, on_progress, demo_origin) -> None:
    """Embed a list of (img_path, category) tuples into ``clips``."""
    import hashlib  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    _ensure_image_embedder_loaded(embedder, on_progress)

    clip_id = max(clips.keys(), default=0) + 1
    total = len(selected)
    on_progress("embedding", f"Starting embedding for {total} images...", 0, total)

    from vtscore.media.embedder import media_from_path  # noqa: PLC0415

    for i, (img_path, category) in enumerate(selected):
        on_progress("embedding", f"Embedding {category}/{img_path.name}", i + 1, total)
        embedding = embedder.embed_media(media_from_path(img_path))
        if embedding is None:
            continue
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        try:
            with Image.open(img_path) as img:
                width, height = img.width, img.height
        except Exception:
            width, height = None, None
        clips[clip_id] = {
            "id": clip_id,
            "media_type": _MEDIA_TYPE_ID,
            "embedder": embedder.name,
            "duration": 0,
            "file_size": len(image_bytes),
            "md5": hashlib.md5(image_bytes).hexdigest(),
            "embedding": embedding,
            "media_bytes": image_bytes,
            "media_string": None,
            "filename": f"{category}/{img_path.name}",
            "category": category,
            "width": width,
            "height": height,
            "origin": demo_origin,
            "origin_name": f"{category}/{img_path.name}",
        }
        clip_id += 1


def _embed_pil_pages(selected_pages, clips, embedder, on_progress, demo_origin) -> None:
    """Embed a list of (page_name, PIL.Image, category) tuples into ``clips``."""
    import hashlib  # noqa: PLC0415
    import io as _io  # noqa: PLC0415

    _ensure_image_embedder_loaded(embedder, on_progress)

    clip_id = max(clips.keys(), default=0) + 1
    total = len(selected_pages)
    on_progress("embedding", f"Starting embedding for {total} document pages...", 0, total)

    for i, (page_name, pil_image, category) in enumerate(selected_pages):
        on_progress("embedding", f"Embedding {page_name}", i + 1, total)
        embedding = cast(Any, embedder).embed_pil_image(pil_image)
        if embedding is None:
            continue
        img_buffer = _io.BytesIO()
        pil_image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()
        rel_name = f"{category}/{page_name}"
        clips[clip_id] = {
            "id": clip_id,
            "media_type": _MEDIA_TYPE_ID,
            "embedder": embedder.name,
            "duration": 0,
            "file_size": len(image_bytes),
            "md5": hashlib.md5(image_bytes).hexdigest(),
            "embedding": embedding,
            "media_bytes": image_bytes,
            "media_string": None,
            "filename": f"{rel_name}.png",
            "category": category,
            "width": pil_image.width,
            "height": pil_image.height,
            "origin": demo_origin,
            "origin_name": rel_name,
        }
        clip_id += 1


def _embed_cifar_arrays(selected, clips, embedder, on_progress, demo_origin) -> None:
    """Embed a list of (image_array, category) tuples into ``clips``."""
    import hashlib  # noqa: PLC0415
    import io as _io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    _ensure_image_embedder_loaded(embedder, on_progress)

    clip_id = max(clips.keys(), default=0) + 1
    total = len(selected)
    on_progress("embedding", f"Starting embedding for {total} images...", 0, total)

    for i, (image_array, category) in enumerate(selected):
        on_progress("embedding", f"Embedding {category}", i + 1, total)
        img = Image.fromarray(image_array.astype("uint8"), "RGB")
        img_buffer = _io.BytesIO()
        img.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()
        embedding = cast(Any, embedder).embed_pil_image(img)
        if embedding is None:
            continue
        fname = f"{category}/{category}_{clip_id}.png"
        clips[clip_id] = {
            "id": clip_id,
            "media_type": _MEDIA_TYPE_ID,
            "embedder": embedder.name,
            "duration": 0,
            "file_size": len(image_bytes),
            "md5": hashlib.md5(image_bytes).hexdigest(),
            "embedding": embedding,
            "media_bytes": image_bytes,
            "media_string": None,
            "filename": fname,
            "category": category,
            "width": img.width,
            "height": img.height,
            "origin": demo_origin,
            "origin_name": fname,
        }
        clip_id += 1


def load_demo_source(
    source,
    categories,
    slice_start,
    slice_end,
    clips,
    on_progress=None,
    embedder=None,
    slice_frac_start=None,
    slice_frac_end=None,
    **kwargs,
):
    if on_progress is None:
        from vtscore.concurrency.progress import update_progress

        on_progress = update_progress

    if embedder is None:
        from vtscore.media import embedders_for_type

        avail = embedders_for_type(_MEDIA_TYPE_ID)
        if not avail:
            raise ValueError(f"No embedders registered for media type {_MEDIA_TYPE_ID!r}")
        embedder = avail[0]

    demo_origin: dict = {"importer": "demo", "params": {}}
    slice_args = (slice_start, slice_end, slice_frac_start, slice_frac_end)

    if source in _FILE_SOURCE_DOWNLOADERS:
        _embed_file_images(
            _collect_simple_folder_files(source, categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
        )
        return None

    if source == "oxford_flowers_102":
        _embed_file_images(
            _collect_oxford_flowers_files(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
        )
        return None

    if source == "stanford_dogs":
        _embed_file_images(
            _collect_stanford_dogs_files(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
        )
        return None

    if source == "places365":
        _embed_file_images(
            _collect_places365_files(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
        )
        return None

    if source == "ucsf_documents":
        _embed_pil_pages(
            _collect_ucsf_doc_pages(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
        )
        return None

    if source == "cifar10_sample" or not source:
        _embed_cifar_arrays(
            _collect_cifar10_images(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
        )
        return None

    raise ValueError(f"Unsupported image source: {source!r}")
