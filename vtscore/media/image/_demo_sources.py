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
    ROXFORD_CATEGORIES,
    STANFORD_DOGS_CATEGORIES,
    UCSF_DOCUMENTS_CATEGORIES,
    VISUAL_GENOME_CATEGORIES,
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
        ROXFORD_IMAGES_DOWNLOAD_SIZE_MB,
        STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        UCSF_IDL_DOWNLOAD_SIZE_MB,
        VISUAL_GENOME_IMAGES2_DOWNLOAD_SIZE_MB,
        VISUAL_GENOME_IMAGES_DOWNLOAD_SIZE_MB,
        VISUAL_GENOME_OBJECTS_DOWNLOAD_SIZE_MB,
    )

    vg_size = (
        VISUAL_GENOME_IMAGES_DOWNLOAD_SIZE_MB
        + VISUAL_GENOME_IMAGES2_DOWNLOAD_SIZE_MB
        + VISUAL_GENOME_OBJECTS_DOWNLOAD_SIZE_MB
    )
    vg_desc = "Dense multi-label scenes with object boxes"
    vg_folder = DATA_DIR / "visual_genome"

    cats101 = DEMO_CATEGORIES_CALTECH101
    cats256 = DEMO_CATEGORIES_CALTECH256
    ct101_desc = "Centered object photos"
    ct101_folder = DATA_DIR / "caltech-101" / "101_ObjectCategories"
    ct256_desc = "Cluttered object photos"
    ct256_folder = DATA_DIR / "caltech-256" / "256_ObjectCategories"
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
            id="caltech256_s",
            label="Caltech-256 (S)",
            description=ct256_desc,
            categories=cats256,
            source="caltech256",
            required_folder=ct256_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=119,
            download_size_mb=CALTECH256_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="caltech256_m",
            label="Caltech-256 (M)",
            description=ct256_desc,
            categories=cats256,
            source="caltech256",
            required_folder=ct256_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=119,
            download_size_mb=CALTECH256_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="caltech256_l",
            label="Caltech-256 (L)",
            description=ct256_desc,
            categories=cats256,
            source="caltech256",
            required_folder=ct256_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=119,
            download_size_mb=CALTECH256_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="caltech256_a",
            label="Caltech-256 (A)",
            description=ct256_desc,
            categories=cats256,
            source="caltech256",
            required_folder=ct256_folder,
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
            id="roxford5k_s",
            label="ROxford5k (S)",
            description="Oxford landmarks — instance matching",
            categories=ROXFORD_CATEGORIES,
            source="roxford5k",
            required_folder=DATA_DIR / "roxford5k" / "jpg",
            slice_frac_start=0.0,
            slice_frac_end=1 / 10,
            items_per_category=1500,
            download_size_mb=ROXFORD_IMAGES_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="roxford5k_a",
            label="ROxford5k (A)",
            description="Oxford landmarks — instance matching",
            categories=ROXFORD_CATEGORIES,
            source="roxford5k",
            required_folder=DATA_DIR / "roxford5k" / "jpg",
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=1500,
            download_size_mb=ROXFORD_IMAGES_DOWNLOAD_SIZE_MB,
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
        # Visual Genome is multi-label and sliced flat over the image list (not
        # per-category), so the advertised count is only approximate — see
        # docs/plans/visual-genome-dataset.md (real-download verification is a
        # tracked follow-up).
        DemoDataset(
            id="visual_genome_s",
            label="Visual Genome (S)",
            description=vg_desc,
            categories=VISUAL_GENOME_CATEGORIES,
            source="visual_genome",
            required_folder=vg_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 50,
            items_per_category=1000,
            download_size_mb=vg_size,
        ),
        DemoDataset(
            id="visual_genome_m",
            label="Visual Genome (M)",
            description=vg_desc,
            categories=VISUAL_GENOME_CATEGORIES,
            source="visual_genome",
            required_folder=vg_folder,
            slice_frac_start=1 / 50,
            slice_frac_end=3 / 50,
            items_per_category=1000,
            download_size_mb=vg_size,
        ),
        DemoDataset(
            id="visual_genome_l",
            label="Visual Genome (L)",
            description=vg_desc,
            categories=VISUAL_GENOME_CATEGORIES,
            source="visual_genome",
            required_folder=vg_folder,
            slice_frac_start=3 / 50,
            slice_frac_end=7 / 50,
            items_per_category=1000,
            download_size_mb=vg_size,
        ),
        DemoDataset(
            id="visual_genome_a",
            label="Visual Genome (A)",
            description=vg_desc,
            categories=VISUAL_GENOME_CATEGORIES,
            source="visual_genome",
            required_folder=vg_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=1000,
            download_size_mb=vg_size,
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


def _roxford_category_for(stem: str, landmark_set: frozenset[str]) -> str:
    """Map an Oxford Buildings filename stem to its landmark category.

    Filenames look like ``radcliffe_camera_000158``; the landmark is the stem
    with the trailing ``_<digits>`` index removed.  Anything not in the known
    query-landmark set falls into ``"other"`` (the distractor haystack).
    """
    import re  # noqa: PLC0415

    prefix = re.sub(r"_\d+$", "", stem)
    return prefix if prefix in landmark_set else "other"


def _collect_roxford_files(categories, slice_args, on_progress) -> list:
    from vtscore.datasets.downloader import download_roxford5k  # noqa: PLC0415

    roxford_dir = download_roxford5k(on_progress=on_progress)
    jpg_dir = roxford_dir / "jpg"
    # "other" is a catch-all, not a filename prefix, so it never matches a stem.
    landmark_set = frozenset(c for c in categories if c != "other")

    by_cat: dict = {}
    if jpg_dir.exists():
        for img_path in sorted(jpg_dir.glob("*.jpg")):
            cat = _roxford_category_for(img_path.stem, landmark_set)
            if cat in categories:
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


# Visual Genome object names are messy (case, plurals, synonyms).  This maps
# the irregular plurals that show up often in VG onto the singular forms used in
# our category vocab; regular plurals are handled by stripping a trailing "s".
_VG_IRREGULAR_PLURALS = {
    "men": "man",
    "women": "woman",
    "feet": "foot",
    "leaves": "leaf",
    "buses": "bus",
    "glasses": "glass",
    "bushes": "bush",
    "boxes": "box",
}


def _vg_category_for(name: str, vocab: frozenset[str]) -> str | None:
    """Map a raw Visual Genome object name onto a vocab category, or ``None``.

    Normalizes case/whitespace, then tries: exact match, irregular-plural fold,
    and a naive trailing-``s`` singularization.  Returns the matched category
    name (always a member of *vocab*) or ``None`` when nothing matches.
    """
    n = name.strip().lower()
    if n in vocab:
        return n
    if n in _VG_IRREGULAR_PLURALS:
        mapped = _VG_IRREGULAR_PLURALS[n]
        return mapped if mapped in vocab else None
    if n.endswith("s") and n[:-1] in vocab:
        return n[:-1]
    return None


def _resolve_vg_image_path(vg_dir: Path, image_id: int) -> Path | None:
    """Return the on-disk path for a VG image id, or ``None`` if missing.

    VG images are split across the ``VG_100K`` and ``VG_100K_2`` folders; the id
    determines neither, so both are probed.
    """
    for sub in ("VG_100K", "VG_100K_2"):
        candidate = vg_dir / sub / f"{image_id}.jpg"
        if candidate.exists():
            return candidate
    return None


def _vg_objects_to_labels(objects, vocab: frozenset[str]) -> tuple[list[str], list]:
    """Map one image's VG objects onto ``(positive_categories, pixel_regions)``.

    ``positive_categories`` is the de-duplicated list of in-vocab categories the
    image belongs to; ``pixel_regions`` is one ``(x, y, w, h, label)`` per
    in-vocab object (source pixel coordinates, normalized at embed time).
    """
    positive: list[str] = []
    regions: list = []
    for obj in objects:
        label = None
        for name in obj.get("names", []):
            label = _vg_category_for(name, vocab)
            if label is not None:
                break
        if label is None:
            continue
        if label not in positive:
            positive.append(label)
        try:
            regions.append((int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"]), label))
        except (KeyError, TypeError, ValueError):
            continue
    return positive, regions


def _collect_visual_genome_files(categories, slice_args, on_progress) -> list:
    """Parse VG ``objects.json`` into per-image multi-label + region records.

    Returns a flat, image-id-sorted list of
    ``(img_path, positive_categories, pixel_regions)`` for images with at least
    one in-vocab object, sliced *flat* (Visual Genome is not folder-per-class,
    so the per-category slicing the other sources use doesn't apply).
    """
    import json  # noqa: PLC0415

    from vtscore.datasets.downloader import download_visual_genome  # noqa: PLC0415

    vg_dir = download_visual_genome(on_progress=on_progress)

    on_progress("loading", "Reading Visual Genome annotations…", 0, 0)
    with open(vg_dir / "objects.json", encoding="utf-8") as f:
        annotations = json.load(f)

    vocab = frozenset(categories)
    records: list = []
    for entry in annotations:
        image_id = entry.get("image_id")
        if image_id is None:
            continue
        positive, regions = _vg_objects_to_labels(entry.get("objects", []), vocab)
        if not positive:
            continue
        img_path = _resolve_vg_image_path(vg_dir, image_id)
        if img_path is None:
            continue
        records.append((image_id, img_path, positive, regions))

    records.sort(key=lambda r: r[0])
    return demo_slice([(p, pos, reg) for _id, p, pos, reg in records], *slice_args)


def _ensure_image_embedder_loaded(embedder, on_progress) -> None:
    if getattr(embedder, "_model", None) is None:
        on_progress("loading", "Loading image embedding model…", 0, 0)
        original_cb = embedder._on_progress
        embedder._on_progress = on_progress
        try:
            embedder.load_models()
        finally:
            embedder._on_progress = original_cb


def _embed_file_images(selected, clips, embedder, on_progress, demo_origin, skip_embedding=False) -> None:
    """Embed a list of (img_path, category) tuples into ``clips``."""
    import hashlib  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    if not skip_embedding:
        _ensure_image_embedder_loaded(embedder, on_progress)

    clip_id = max(clips.keys(), default=0) + 1
    total = len(selected)
    status = "loading" if skip_embedding else "embedding"
    verb = "Loading" if skip_embedding else "Embedding"
    on_progress(status, f"{verb} {total} images...", 0, total)

    from vtscore.media.embedder import media_from_path  # noqa: PLC0415

    for i, (img_path, category) in enumerate(selected):
        if skip_embedding:
            on_progress("loading", f"Loading {category}/{img_path.name}", i + 1, total)
            embedding = None
        else:
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
            "embeddings": {} if skip_embedding else {embedder.name: embedding},
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


def _embed_pil_pages(selected_pages, clips, embedder, on_progress, demo_origin, skip_embedding=False) -> None:
    """Embed a list of (page_name, PIL.Image, category) tuples into ``clips``."""
    import hashlib  # noqa: PLC0415
    import io as _io  # noqa: PLC0415

    if not skip_embedding:
        _ensure_image_embedder_loaded(embedder, on_progress)

    clip_id = max(clips.keys(), default=0) + 1
    total = len(selected_pages)
    status = "loading" if skip_embedding else "embedding"
    verb = "Loading" if skip_embedding else "Embedding"
    on_progress(status, f"{verb} {total} document pages...", 0, total)

    for i, (page_name, pil_image, category) in enumerate(selected_pages):
        if skip_embedding:
            on_progress("loading", f"Loading {page_name}", i + 1, total)
            embedding = None
        else:
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
            "embeddings": {} if skip_embedding else {embedder.name: embedding},
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


def _embed_cifar_arrays(selected, clips, embedder, on_progress, demo_origin, skip_embedding=False) -> None:
    """Embed a list of (image_array, category) tuples into ``clips``."""
    import hashlib  # noqa: PLC0415
    import io as _io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    if not skip_embedding:
        _ensure_image_embedder_loaded(embedder, on_progress)

    clip_id = max(clips.keys(), default=0) + 1
    total = len(selected)
    status = "loading" if skip_embedding else "embedding"
    verb = "Loading" if skip_embedding else "Embedding"
    on_progress(status, f"{verb} {total} images...", 0, total)

    for i, (image_array, category) in enumerate(selected):
        on_progress(status, f"{verb} {category}", i + 1, total)
        img = Image.fromarray(image_array.astype("uint8"), "RGB")
        img_buffer = _io.BytesIO()
        img.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()
        if skip_embedding:
            embedding = None
        else:
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
            "embeddings": {} if skip_embedding else {embedder.name: embedding},
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


def _normalize_regions(pixel_regions, width, height) -> list:
    """Convert ``(x, y, w, h, label)`` pixel boxes to normalized region dicts.

    Boxes are divided by the image dimensions and clamped to ``[0, 1]``.  Boxes
    are dropped when the image dimensions are unknown or non-positive.
    """
    if not width or not height:
        return []
    out: list = []
    for x, y, w, h, label in pixel_regions:
        x0 = min(max(x / width, 0.0), 1.0)
        y0 = min(max(y / height, 0.0), 1.0)
        x1 = min(max((x + w) / width, 0.0), 1.0)
        y1 = min(max((y + h) / height, 0.0), 1.0)
        if x1 <= x0 or y1 <= y0:
            continue
        out.append({"box": [round(x0, 5), round(y0, 5), round(x1, 5), round(y1, 5)], "label": label})
    return out


def _embed_vg_images(selected, clips, embedder, on_progress, demo_origin, skip_embedding=False) -> None:
    """Embed Visual Genome images, stamping multi-label categories + regions.

    ``selected`` is a list of ``(img_path, positive_categories, pixel_regions)``.
    Each clip gets a ``categories`` list (the multi-label positives), a
    ``category`` primary (first positive, for legacy single-label readers), and
    a store-only ``regions`` list of normalized ground-truth boxes.
    """
    import hashlib  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    from vtscore.media.embedder import media_from_path  # noqa: PLC0415

    if not skip_embedding:
        _ensure_image_embedder_loaded(embedder, on_progress)

    clip_id = max(clips.keys(), default=0) + 1
    total = len(selected)
    status = "loading" if skip_embedding else "embedding"
    verb = "Loading" if skip_embedding else "Embedding"
    on_progress(status, f"{verb} {total} images...", 0, total)

    for i, (img_path, positive_categories, pixel_regions) in enumerate(selected):
        primary = positive_categories[0]
        if skip_embedding:
            on_progress("loading", f"Loading {primary}/{img_path.name}", i + 1, total)
            embedding = None
        else:
            on_progress("embedding", f"Embedding {primary}/{img_path.name}", i + 1, total)
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
            "embeddings": {} if skip_embedding else {embedder.name: embedding},
            "media_bytes": image_bytes,
            "media_string": None,
            "filename": img_path.name,
            "category": primary,
            "categories": list(positive_categories),
            "regions": _normalize_regions(pixel_regions, width, height),
            "width": width,
            "height": height,
            "origin": demo_origin,
            "origin_name": img_path.name,
        }
        clip_id += 1


def load_demo_source(  # noqa: C901 - flat per-source dispatch; one branch per demo source
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

    # When a clipper will re-embed every produced crop, skip embedding the full
    # parent images here (the result would be discarded) - see skip_embedding in
    # load_demo_dataset.
    skip_embedding = bool(kwargs.get("skip_embedding", False))

    demo_origin: dict = {"importer": "demo", "params": {}}
    slice_args = (slice_start, slice_end, slice_frac_start, slice_frac_end)

    if source in _FILE_SOURCE_DOWNLOADERS:
        _embed_file_images(
            _collect_simple_folder_files(source, categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
            skip_embedding=skip_embedding,
        )
        return None

    if source == "oxford_flowers_102":
        _embed_file_images(
            _collect_oxford_flowers_files(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
            skip_embedding=skip_embedding,
        )
        return None

    if source == "stanford_dogs":
        _embed_file_images(
            _collect_stanford_dogs_files(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
            skip_embedding=skip_embedding,
        )
        return None

    if source == "roxford5k":
        _embed_file_images(
            _collect_roxford_files(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
            skip_embedding=skip_embedding,
        )
        return None

    if source == "places365":
        _embed_file_images(
            _collect_places365_files(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
            skip_embedding=skip_embedding,
        )
        return None

    if source == "ucsf_documents":
        _embed_pil_pages(
            _collect_ucsf_doc_pages(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
            skip_embedding=skip_embedding,
        )
        return None

    if source == "visual_genome":
        _embed_vg_images(
            _collect_visual_genome_files(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
            skip_embedding=skip_embedding,
        )
        return None

    if source == "cifar10_sample" or not source:
        _embed_cifar_arrays(
            _collect_cifar10_images(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
            skip_embedding=skip_embedding,
        )
        return None

    raise ValueError(f"Unsupported image source: {source!r}")
