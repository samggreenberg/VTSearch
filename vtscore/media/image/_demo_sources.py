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

from pathlib import Path, PurePosixPath
from typing import Any, cast

from vtscore.config import DATA_DIR
from vtscore.media._toponymy_demo import SOURCE_ID as _TOPONYMY_SOURCE_ID
from vtscore.media._toponymy_demo import TAXONOMY as _TOPONYMY_TAXONOMY
from vtscore.media.base import DemoDataset, demo_slice
from vtscore.media.image._demo_categories import (
    DEMO_CATEGORIES_CALTECH101,
    DEMO_CATEGORIES_CALTECH256,
    ENRICO_CATEGORIES,
    EUROSAT_CATEGORIES,
    FOOD101_CATEGORIES,
    OPENLOGO_CATEGORIES,
    OXFORD_FLOWERS_CATEGORIES,
    PLACES365_CATEGORIES,
    RICO_ICON_CATEGORIES,
    RICO_ICON_IDENTITIES,
    RICO_SCREEN2WORDS_CATEGORIES,
    ROXFORD_CATEGORIES,
    RVL_CDIP_CATEGORIES,
    VGGFACE2_CATEGORIES,
    VGGFACE2_IDENTITIES,
    VISUAL_GENOME_CATEGORIES,
)
from vtscore.utils.hashing import content_md5


_MEDIA_TYPE_ID = "image"


def build_demo_datasets() -> list[DemoDataset]:
    """Build the demo-dataset catalog exposed by :class:`ImageMediaType`."""
    from vtscore.datasets.downloader import (  # noqa: PLC0415
        CALTECH101_DOWNLOAD_SIZE_MB,
        CALTECH256_DOWNLOAD_SIZE_MB,
        ENRICO_DOWNLOAD_SIZE_MB,
        EUROSAT_DOWNLOAD_SIZE_MB,
        FOOD101_DOWNLOAD_SIZE_MB,
        OPENLOGO_DOWNLOAD_SIZE_MB,
        OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
        PLACES365_DOWNLOAD_SIZE_MB,
        RICO_ICONS_MANIFEST_MB,
        RICO_ICONS_SHARD_COUNT,
        RICO_ICONS_SHARD_MB,
        RICO_SCREEN2WORDS_DOWNLOAD_SIZE_MB,
        ROXFORD_IMAGES_DOWNLOAD_SIZE_MB,
        RVL_CDIP_DOWNLOAD_SIZE_MB,
        VGGFACE2_TEST_DOWNLOAD_SIZE_MB,
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
    places_desc = "Indoor & outdoor scenes"
    places_folder = DATA_DIR / "places365" / "val_256"
    rico_folder = DATA_DIR / "rico_screen2words" / "screenshots"
    rvl_folder = DATA_DIR / "rvl_cdip" / "images"
    rico_icons_folder = DATA_DIR / "rico_icons" / "data"
    rico_icons_desc = "Mobile UI screenshots with boxed, labelled icons"

    def rico_icons_mb(frac_start: float, frac_end: float | None) -> int:
        """Advertised download for one Rico-icons variant.

        Unlike every other demo, the four variants do *not* share one figure:
        the loader fetches the 535 MB manifest plus only the ~116 MB image
        shard folders its slice lands in, so (S) really is ~0.9 GB rather than
        the whole corpus' ~8.3 GB.  A slice spanning a fraction of the corpus
        touches that fraction of the 67 shards, plus one for straddling a
        boundary.
        """
        if frac_end is None and frac_start == 0.0:
            shards = RICO_ICONS_SHARD_COUNT
        else:
            span = (1.0 if frac_end is None else frac_end) - frac_start
            shards = min(RICO_ICONS_SHARD_COUNT, round(span * RICO_ICONS_SHARD_COUNT) + 1)
        return RICO_ICONS_MANIFEST_MB + shards * RICO_ICONS_SHARD_MB

    faces_desc = "In-the-wild celebrity photos, one label per person"
    faces_folder = DATA_DIR / "vggface2" / "test"
    return [
        DemoDataset(
            id="synthetic_world_image",
            label="Synthetic World Map (signposts demo)",
            description=(
                "Pre-baked 4-level toponymy (Continent → Country → State → City) "
                "with cheating ground-truth signposts — no download, loads instantly."
            ),
            categories=list(_TOPONYMY_TAXONOMY.keys()),
            source=_TOPONYMY_SOURCE_ID,
            items_per_category=0,
            download_size_mb=0,
        ),
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
        # Enrico: born-digital mobile-UI *screenshots* (not natural photos),
        # labeled by screen function.  ~1,460 images over 20 topics (~73/topic,
        # unevenly), so the S/M/L slices are small; A is the whole set.
        DemoDataset(
            id="enrico_s",
            label="Enrico UI (S)",
            description="Mobile app UI screenshots by screen type",
            categories=ENRICO_CATEGORIES,
            source="enrico",
            required_folder=DATA_DIR / "enrico",
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=73,
            download_size_mb=ENRICO_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="enrico_m",
            label="Enrico UI (M)",
            description="Mobile app UI screenshots by screen type",
            categories=ENRICO_CATEGORIES,
            source="enrico",
            required_folder=DATA_DIR / "enrico",
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=73,
            download_size_mb=ENRICO_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="enrico_l",
            label="Enrico UI (L)",
            description="Mobile app UI screenshots by screen type",
            categories=ENRICO_CATEGORIES,
            source="enrico",
            required_folder=DATA_DIR / "enrico",
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=73,
            download_size_mb=ENRICO_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="enrico_a",
            label="Enrico UI (A)",
            description="All Enrico mobile UI screenshots across 20 screen-function topics",
            categories=ENRICO_CATEGORIES,
            source="enrico",
            required_folder=DATA_DIR / "enrico",
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=73,
            download_size_mb=ENRICO_DOWNLOAD_SIZE_MB,
        ),
        # RICO-Screen2Words: mobile-UI screenshots labeled by *app genre* (Google
        # Play category) rather than screen function.  All variants pull the same
        # ~1.7 GB train split and extract the 16 curated categories; the slices
        # differ only in how much of each category they take.
        DemoDataset(
            id="rico_screen2words_s",
            label="RICO App UIs (S)",
            description="Mobile app UI screenshots by app category",
            categories=RICO_SCREEN2WORDS_CATEGORIES,
            source="rico_screen2words",
            required_folder=rico_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=400,
            download_size_mb=RICO_SCREEN2WORDS_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="rico_screen2words_m",
            label="RICO App UIs (M)",
            description="Mobile app UI screenshots by app category",
            categories=RICO_SCREEN2WORDS_CATEGORIES,
            source="rico_screen2words",
            required_folder=rico_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=400,
            download_size_mb=RICO_SCREEN2WORDS_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="rico_screen2words_l",
            label="RICO App UIs (L)",
            description="Mobile app UI screenshots by app category",
            categories=RICO_SCREEN2WORDS_CATEGORIES,
            source="rico_screen2words",
            required_folder=rico_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=400,
            download_size_mb=RICO_SCREEN2WORDS_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="rico_screen2words_a",
            label="RICO App UIs (A)",
            description="Mobile app UI screenshots across 16 Google Play app categories",
            categories=RICO_SCREEN2WORDS_CATEGORIES,
            source="rico_screen2words",
            required_folder=rico_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=400,
            download_size_mb=RICO_SCREEN2WORDS_DOWNLOAD_SIZE_MB,
        ),
        # Rico UI semantics: the same born-digital screenshots as the demo above,
        # but labelled at the *element* level instead of the screen level.  Each
        # media is one screen; its categories are the icon semantics visible on
        # it (multi-label, ~2.2 distinct icon classes per screen) and its
        # ``regions`` carry one ground-truth box per icon.  This is the only demo
        # in the tree that can answer "box this search icon, then find every
        # other search icon" — the workflow VTSearch exists for, on the media
        # type where a semantic embedder alone struggles most.
        #
        # Multi-label and sliced *flat* over the image list (not per-category),
        # like OpenLogo and Visual Genome, so ``items_per_category`` is a tuned
        # estimate rather than a real per-category count: 66,261 screens of which
        # a measured ~68.5% carry at least one in-vocab icon (~45,400), over 32
        # categories, gives 45,400 / 32 ≈ 1,419.
        DemoDataset(
            id="rico_icons_s",
            label="Rico Icons (S)",
            description=rico_icons_desc,
            categories=RICO_ICON_CATEGORIES,
            source="rico_icons",
            required_folder=rico_icons_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 32,
            items_per_category=1419,
            download_size_mb=rico_icons_mb(0.0, 1 / 32),
        ),
        DemoDataset(
            id="rico_icons_m",
            label="Rico Icons (M)",
            description=rico_icons_desc,
            categories=RICO_ICON_CATEGORIES,
            source="rico_icons",
            required_folder=rico_icons_folder,
            slice_frac_start=1 / 32,
            slice_frac_end=3 / 32,
            items_per_category=1419,
            download_size_mb=rico_icons_mb(1 / 32, 3 / 32),
        ),
        DemoDataset(
            id="rico_icons_l",
            label="Rico Icons (L)",
            description=rico_icons_desc,
            categories=RICO_ICON_CATEGORIES,
            source="rico_icons",
            required_folder=rico_icons_folder,
            slice_frac_start=3 / 32,
            slice_frac_end=7 / 32,
            items_per_category=1419,
            download_size_mb=rico_icons_mb(3 / 32, 7 / 32),
        ),
        DemoDataset(
            id="rico_icons_a",
            label="Rico Icons (A)",
            description="All Rico screenshots carrying one of 32 boxed icon classes",
            categories=RICO_ICON_CATEGORIES,
            source="rico_icons",
            required_folder=rico_icons_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=1419,
            download_size_mb=rico_icons_mb(0.0, None),
        ),
        # RVL-CDIP: scanned grayscale *document images* across 16 balanced types
        # (letter, form, email, invoice, resume, memo, …).  A demo-sized,
        # class-balanced 100-per-class mirror (~1,600 images); the "document
        # screenshot" corner of digitally-native imagery with a clean 16-way
        # label set.
        DemoDataset(
            id="rvl_cdip_s",
            label="RVL-CDIP Docs (S)",
            description="Scanned document images by type",
            categories=RVL_CDIP_CATEGORIES,
            source="rvl_cdip",
            required_folder=rvl_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=100,
            download_size_mb=RVL_CDIP_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="rvl_cdip_m",
            label="RVL-CDIP Docs (M)",
            description="Scanned document images by type",
            categories=RVL_CDIP_CATEGORIES,
            source="rvl_cdip",
            required_folder=rvl_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=100,
            download_size_mb=RVL_CDIP_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="rvl_cdip_l",
            label="RVL-CDIP Docs (L)",
            description="Scanned document images by type",
            categories=RVL_CDIP_CATEGORIES,
            source="rvl_cdip",
            required_folder=rvl_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=100,
            download_size_mb=RVL_CDIP_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="rvl_cdip_a",
            label="RVL-CDIP Docs (A)",
            description="Scanned document images across 16 RVL-CDIP document types",
            categories=RVL_CDIP_CATEGORIES,
            source="rvl_cdip",
            required_folder=rvl_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=100,
            download_size_mb=RVL_CDIP_DOWNLOAD_SIZE_MB,
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
        # OpenLogo is multi-label (an image may show several brands) and sliced
        # flat over the image list, so the per-category ``items_per_category``
        # estimate does not apply; the advertised count comes from the measured
        # DEMO_MEDIA_COUNTS entries instead (see vtscore/datasets/demo_counts.py).
        DemoDataset(
            id="openlogo_s",
            label="OpenLogo (S)",
            description="Brand logos in the wild — instance matching",
            categories=OPENLOGO_CATEGORIES,
            source="openlogo",
            required_folder=DATA_DIR / "openlogo" / "data",
            slice_frac_start=0.0,
            slice_frac_end=1 / 10,
            items_per_category=156,
            download_size_mb=OPENLOGO_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="openlogo_a",
            label="OpenLogo (A)",
            description="Brand logos in the wild — instance matching",
            categories=OPENLOGO_CATEGORIES,
            source="openlogo",
            required_folder=DATA_DIR / "openlogo" / "data",
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=156,
            download_size_mb=OPENLOGO_DOWNLOAD_SIZE_MB,
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
        # Every curated identity has 469+ photos, so an absolute per-person cap
        # (slice_start..slice_end) yields a uniform, exact count - no fractional
        # slicing needed.  S and M take the first 15 / 40 photos of each person.
        DemoDataset(
            id="vggface2_faces_s",
            label="Faces - VGGFace2 (S)",
            description=faces_desc,
            categories=VGGFACE2_CATEGORIES,
            source="vggface2",
            required_folder=faces_folder,
            slice_start=0,
            slice_end=15,
            items_per_category=15,
            download_size_mb=VGGFACE2_TEST_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="vggface2_faces_m",
            label="Faces - VGGFace2 (M)",
            description=faces_desc,
            categories=VGGFACE2_CATEGORIES,
            source="vggface2",
            required_folder=faces_folder,
            slice_start=0,
            slice_end=40,
            items_per_category=40,
            download_size_mb=VGGFACE2_TEST_DOWNLOAD_SIZE_MB,
        ),
    ]


_FILE_SOURCE_DOWNLOADERS: dict[str, str] = {
    "caltech101": "download_caltech101",
    "caltech256": "download_caltech256",
    "food101": "download_food101",
    "eurosat": "download_eurosat",
    # RICO-Screen2Words decodes its HF parquet into a <category>/<id>.jpg tree,
    # so it plugs straight into the folder-per-class collect path.
    "rico_screen2words": "download_rico_screen2words",
    # RVL-CDIP likewise decodes its parquet mirror into <class>/<idx>.png.
    "rvl_cdip": "download_rvl_cdip",
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


def _collect_vggface2_files(categories, slice_args, on_progress) -> list:
    """Collect VGGFace2 face photos grouped by person (the Faces demo).

    The download unpacks to ``test/n######/*.jpg`` - one folder per identity.
    We map the curated ``(class_id, display_name)`` subset in
    ``VGGFACE2_IDENTITIES`` to human-readable ``category`` labels, emitting
    ``(path, display_name)`` for every photo of each requested person, then
    slice within each person's photo list.
    """
    from vtscore.datasets.downloader import download_vggface2  # noqa: PLC0415

    test_dir = download_vggface2(on_progress=on_progress)
    wanted = set(categories)
    by_cat: dict = {}
    for class_id, name in VGGFACE2_IDENTITIES:
        if name not in wanted:
            continue
        person_dir = test_dir / class_id
        if not person_dir.is_dir():
            continue
        for img_path in sorted(person_dir.glob("*.jpg")):
            by_cat.setdefault(name, []).append((img_path, name))
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


def _collect_enrico_files(categories, slice_args, on_progress) -> list:
    """Collect Enrico screenshots grouped by their 20-way design-topic label.

    Screenshots are JPEGs keyed on the Rico screen id; upstream drift means the
    id shows up either as a flat ``<screen_id>.jpg`` or the older
    ``<screen_id>-screenshot.jpg``, so we accept both and recover the id from
    the stem.  The label lives in ``design_topics.csv`` (``screen_id,topic``),
    whose ``topic`` values are lowercase single tokens (e.g. ``mediaplayer``).
    We fold each to its display category (``MediaPlayer``) with a
    case-insensitive lookup so the stored ``category`` matches
    ``ENRICO_CATEGORIES`` (and the eval queries).
    """
    import csv  # noqa: PLC0415

    from vtscore.datasets.downloader import download_enrico  # noqa: PLC0415

    extract_dir = download_enrico(on_progress=on_progress)
    topics_csv = extract_dir / "design_topics.csv"

    display_by_norm = {c.lower(): c for c in ENRICO_CATEGORIES}
    id_to_cat: dict[str, str] = {}
    if topics_csv.exists():
        with open(topics_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = (row.get("screen_id") or "").strip()
                cat = display_by_norm.get((row.get("topic") or "").strip().lower())
                if sid and cat:
                    id_to_cat[sid] = cat

    by_cat: dict = {}
    for img_path in sorted(extract_dir.rglob("*.jpg")):
        stem = img_path.stem
        sid = stem[: -len("-screenshot")] if stem.endswith("-screenshot") else stem
        cat = id_to_cat.get(sid)
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


def _region_from_norm_xywh(box, label: str) -> dict | None:
    """Convert a normalized ``[x, y, w, h]`` box to a region dict, or ``None``.

    Both FiftyOne-exported sources (OpenLogo, Rico UI semantics) store detections
    as a normalized ``[x, y, w, h]`` in the sample manifest, while the rest of the
    system speaks ``{"box": [x0, y0, x1, y1], "label": ...}`` with corners clamped
    to ``[0, 1]``.  Malformed, unparseable and degenerate (zero-area) boxes return
    ``None`` so the caller can drop them without special-casing.
    """
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    try:
        x, y, w, h = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    except (TypeError, ValueError):
        return None
    x0, y0 = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    x1, y1 = min(max(x + w, 0.0), 1.0), min(max(y + h, 0.0), 1.0)
    if x1 <= x0 or y1 <= y0:
        return None
    return {"box": [round(x0, 5), round(y0, 5), round(x1, 5), round(y1, 5)], "label": label}


def _openlogo_norm(name: str) -> str:
    """Normalize a brand label to a punctuation/case-insensitive match key.

    OpenLogo stores class labels in a normalized lowercase-alphanumeric form
    ("coca-cola" -> "cocacola", "Stella Artois" -> "stellaartois"), so both the
    dataset labels and our display categories are reduced to ``[a-z0-9]`` before
    comparison — the display name's spacing/casing/punctuation is irrelevant.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _openlogo_detections_to_labels(detections, norm_to_display) -> tuple[list[str], list]:
    """Map one image's OpenLogo detections onto ``(positive_categories, regions)``.

    Detections whose brand is not in *norm_to_display* (keyed by
    :func:`_openlogo_norm`) are skipped.  ``bounding_box`` is a normalized
    ``[x, y, w, h]``; each in-vocab box becomes a ``{"box": [x0, y0, x1, y1],
    "label": brand}`` dict (clamped to ``[0, 1]``, degenerate boxes dropped).
    """
    positive: list[str] = []
    regions: list = []
    for det in detections:
        display = norm_to_display.get(_openlogo_norm(det.get("label") or ""))
        if display is None:
            continue
        if display not in positive:
            positive.append(display)
        region = _region_from_norm_xywh(det.get("bounding_box"), display)
        if region is not None:
            regions.append(region)
    return positive, regions


def _collect_openlogo_files(categories, slice_args, on_progress) -> list:
    """Parse OpenLogo's ``samples.json`` into per-image multi-label + region records.

    Each FiftyOne sample carries a ``ground_truth`` Detections field whose
    detections give a brand ``label`` and a normalized ``bounding_box``.  Images
    with no in-vocab brand are skipped.  Returns a flat, filename-sorted, sliced
    list of ``(img_path, positive_categories, regions)``.
    """
    import json  # noqa: PLC0415

    from vtscore.datasets.downloader import download_openlogo  # noqa: PLC0415

    ds_dir = download_openlogo(on_progress=on_progress)
    data_dir = ds_dir / "data"

    on_progress("loading", "Reading OpenLogo annotations…", 0, 0)
    with open(ds_dir / "samples.json", encoding="utf-8") as f:
        doc = json.load(f)
    samples = doc.get("samples", []) if isinstance(doc, dict) else doc

    norm_to_display = {_openlogo_norm(c): c for c in categories}
    records: list = []
    for sample in samples:
        filepath = sample.get("filepath") or ""
        if not filepath:
            continue
        fname = Path(filepath).name
        img_path = data_dir / fname
        if not img_path.is_file():
            continue
        detections = (sample.get("ground_truth") or {}).get("detections") or []
        positive, regions = _openlogo_detections_to_labels(detections, norm_to_display)
        if not positive:
            continue
        records.append((fname, img_path, positive, regions))

    records.sort(key=lambda r: r[0])
    return demo_slice([(p, pos, reg) for _fname, p, pos, reg in records], *slice_args)


def _iter_fiftyone_samples(path: Path):
    """Yield the sample dicts from a FiftyOne ``samples.json``, one at a time.

    Rico's manifest is a single 535 MB JSON document.  ``json.load`` would
    materialise every one of its 66k samples — each carrying a 64-float
    ``ui_vector`` and dozens of detections — as live Python objects at once,
    costing multiple gigabytes for data we reduce to a handful of fields per
    screen.  Decoding one sample at a time with ``raw_decode`` keeps the peak at
    the file's text plus a single sample, and still uses the stdlib parser, so
    string escaping and number formats are handled exactly as ``json.load``
    would.

    Falls back to a whole-document parse if the expected ``{"samples": [...]}``
    envelope isn't found, so a differently-shaped export still loads.
    """
    import json  # noqa: PLC0415

    text = path.read_text(encoding="utf-8")
    key = text.find('"samples"')
    start = text.find("[", key) if key != -1 else -1
    if start == -1:
        doc = json.loads(text)
        yield from (doc.get("samples", []) if isinstance(doc, dict) else doc)
        return

    decoder = json.JSONDecoder()
    i, n = start + 1, len(text)
    while True:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            return
        obj, i = decoder.raw_decode(text, i)
        yield obj


def _rico_icon_detections_to_labels(detections, token_to_display) -> tuple[list[str], list]:
    """Map one Rico screen's detections onto ``(positive_categories, regions)``.

    Only ``Icon`` detections are considered: every element type carries a
    ``content_or_function``, but on a ``Text`` element that field holds the
    element's text ("Cosmos 1455 Rocket"), not an icon semantic, so filtering on
    the component label is what keeps the vocabulary meaningful.  Icons whose
    semantic is outside *token_to_display* (keyed by the raw snake_case token)
    are skipped, as are icons with no annotated semantics at all.
    """
    positive: list[str] = []
    regions: list = []
    for det in detections:
        if det.get("label") != "Icon":
            continue
        token = (det.get("content_or_function") or "").strip().lower()
        display = token_to_display.get(token)
        if display is None:
            continue
        if display not in positive:
            positive.append(display)
        region = _region_from_norm_xywh(det.get("bounding_box"), display)
        if region is not None:
            regions.append(region)
    return positive, regions


def _rico_icons_relpath(filepath: str) -> str:
    """Normalize a manifest ``filepath`` to a repo-relative ``data/data_K/x.jpg``.

    Rico's export stores paths already relative to the repo root, but a FiftyOne
    manifest can also carry the absolute path of the machine that exported it
    (OpenLogo's does).  Rebuilding from the last two components covers both, and
    keeps the shard folder — which is what decides *which* images get downloaded
    — recoverable either way.
    """
    parts = PurePosixPath(filepath.replace("\\", "/")).parts
    if len(parts) >= 2 and parts[-2].startswith("data_"):
        return f"data/{parts[-2]}/{parts[-1]}"
    return filepath.lstrip("/")


def _collect_rico_icons_files(categories, slice_args, on_progress) -> list:
    """Collect Rico screenshots carrying at least one in-vocab boxed icon.

    Two-phase by design: the manifest is downloaded and sliced *before* any image
    is fetched, so only the shard folders the slice actually lands in are pulled.
    An (S) load therefore costs the 535 MB manifest plus a couple of ~116 MB
    folders rather than the corpus's full ~7.7 GB of screenshots.

    Returns a flat, path-sorted, sliced list of
    ``(img_path, positive_categories, regions)`` — the same record shape the
    OpenLogo and Visual Genome demos produce, so it shares their embed path.
    """
    from vtscore.datasets.downloader import (  # noqa: PLC0415
        download_rico_icons_manifest,
        download_rico_icons_shards,
    )

    ds_dir = download_rico_icons_manifest(on_progress=on_progress)

    on_progress("loading", "Reading Rico UI annotations…", 0, 0)
    wanted = set(categories)
    token_to_display = {tok: disp for tok, disp in RICO_ICON_IDENTITIES if disp in wanted}

    records: list = []
    for sample in _iter_fiftyone_samples(ds_dir / "samples.json"):
        filepath = sample.get("filepath") or ""
        if not filepath:
            continue
        detections = (sample.get("detections") or {}).get("detections") or []
        positive, regions = _rico_icon_detections_to_labels(detections, token_to_display)
        if not positive:
            continue
        records.append((_rico_icons_relpath(filepath), positive, regions))

    records.sort(key=lambda r: r[0])
    selected = demo_slice(records, *slice_args)

    shard_dirs = sorted({PurePosixPath(rel).parent.as_posix() for rel, _pos, _reg in selected})
    if shard_dirs:
        download_rico_icons_shards(shard_dirs, on_progress=on_progress)

    return [(ds_dir / rel, pos, reg) for rel, pos, reg in selected if (ds_dir / rel).is_file()]


def _ensure_image_embedder_loaded(embedder, on_progress) -> None:
    if getattr(embedder, "_model", None) is None:
        on_progress("loading", "Loading image embedding model…", 0, 0)
        with embedder.progress_scope(on_progress):
            embedder.load_models()


def _synthetic_tile_png(item) -> bytes:
    """Render a synthetic place as a small solid-colour PNG tile.

    Hue is a continent family (four quadrants of the wheel), lightness/saturation
    drift with the leaf-city index, so sibling cities read as shades of one
    regional colour while continents stay clearly distinct on the map.
    """
    import colorsys  # noqa: PLC0415
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    from vtscore.media._toponymy_demo import total_cities  # noqa: PLC0415

    n_cities = total_cities()
    hue = (item.continent_index / 4.0 + (item.city_index / max(1, n_cities)) * 0.08) % 1.0
    sat = 0.45 + 0.4 * ((item.city_index % 9) / 8.0)
    val = 0.55 + 0.35 * ((item.city_index % 5) / 4.0)
    r, g, b = (int(round(c * 255)) for c in colorsys.hsv_to_rgb(hue, sat, val))
    img = Image.new("RGB", (96, 96), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _load_synthetic_toponymy(clips, embedder, on_progress, demo_origin) -> None:
    """Populate *clips* with the synthetic world-map demo (no model, no download).

    Image sibling of the audio synthetic demo: each leaf city is a solid colour
    tile tagged with its ``Continent/Country/State/City`` path and a pre-baked
    hierarchical embedding, so browsing lights up the ground-truth signpost
    layer straight from those paths.  See :mod:`vtscore.media._toponymy_demo`.
    """

    from vtscore.media._toponymy_demo import generate_items  # noqa: PLC0415
    from vtscore.media.image.thumbnail import make_image_thumbnail  # noqa: PLC0415

    # SigLIP's image/text space is 768-D; match it so the baked vectors slot into
    # the primary embedder's slot and text queries don't dimension-clash.
    items = generate_items(dim=768)
    total = len(items)
    on_progress("loading", f"Generating {total} synthetic tiles…", 0, total)

    emb_name = embedder.name if embedder is not None else "siglip"
    clip_id = 1
    for i, item in enumerate(items):
        png_bytes = _synthetic_tile_png(item)
        thumb = make_image_thumbnail(png_bytes)
        filename = f"{item.category}/tile{i:04d}.png"
        clips[clip_id] = {
            "id": clip_id,
            "media_type": _MEDIA_TYPE_ID,
            "embedder": emb_name,
            "duration": 0,
            "file_size": len(png_bytes),
            "md5": content_md5(png_bytes),
            "embeddings": {emb_name: item.embedding},
            "media_bytes": png_bytes,
            "media_string": None,
            "thumbnail_bytes": thumb[0] if thumb is not None else None,
            "filename": filename,
            "category": item.category,
            "width": 96,
            "height": 96,
            "origin": demo_origin,
            "origin_name": filename,
        }
        clip_id += 1
        if (i + 1) % 100 == 0:
            on_progress("loading", f"Generating synthetic tiles… ({i + 1}/{total})", i + 1, total)


def _embed_file_images(selected, clips, embedder, on_progress, demo_origin, skip_embedding=False) -> None:
    """Embed a list of (img_path, category) tuples into ``clips``."""

    from vtscore.media.image.decode import upright_size  # noqa: PLC0415

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
            width, height = upright_size(img_path)
        except Exception:
            width, height = None, None
        clips[clip_id] = {
            "id": clip_id,
            "media_type": _MEDIA_TYPE_ID,
            "embedder": embedder.name,
            "duration": 0,
            "file_size": len(image_bytes),
            "md5": content_md5(image_bytes),
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
            "md5": content_md5(image_bytes),
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
            "md5": content_md5(image_bytes),
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

    from vtscore.media.embedder import media_from_path  # noqa: PLC0415
    from vtscore.media.image.decode import upright_size  # noqa: PLC0415

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
            width, height = upright_size(img_path)
        except Exception:
            width, height = None, None
        clips[clip_id] = {
            "id": clip_id,
            "media_type": _MEDIA_TYPE_ID,
            "embedder": embedder.name,
            "duration": 0,
            "file_size": len(image_bytes),
            "md5": content_md5(image_bytes),
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


def _embed_boxed_multilabel_images(selected, clips, embedder, on_progress, demo_origin, skip_embedding=False) -> None:
    """Embed ``(img_path, positive_categories, regions)`` records into ``clips``.

    Shared by every demo whose records arrive already boxed and multi-label —
    OpenLogo (brands) and Rico UI semantics (icons) today.  Each clip gets a
    ``categories`` list (all in-vocab labels present), a ``category`` primary
    (the first, for single-label readers), and store-only normalized
    ground-truth ``regions``: the boxed region is the natural template seed for
    a structural detector, and the boxes let the Calibration & Evaluation flow
    score against ground truth.

    Visual Genome has its own variant only because its boxes arrive in source
    pixel coordinates and must be normalized against each image's decoded size.
    """

    from vtscore.media.embedder import media_from_path  # noqa: PLC0415
    from vtscore.media.image.decode import upright_size  # noqa: PLC0415

    if not skip_embedding:
        _ensure_image_embedder_loaded(embedder, on_progress)

    clip_id = max(clips.keys(), default=0) + 1
    total = len(selected)
    status = "loading" if skip_embedding else "embedding"
    verb = "Loading" if skip_embedding else "Embedding"
    on_progress(status, f"{verb} {total} images...", 0, total)

    for i, (img_path, positive_categories, regions) in enumerate(selected):
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
            width, height = upright_size(img_path)
        except Exception:
            width, height = None, None
        clips[clip_id] = {
            "id": clip_id,
            "media_type": _MEDIA_TYPE_ID,
            "embedder": embedder.name,
            "duration": 0,
            "file_size": len(image_bytes),
            "md5": content_md5(image_bytes),
            "embeddings": {} if skip_embedding else {embedder.name: embedding},
            "media_bytes": image_bytes,
            "media_string": None,
            "filename": f"{primary}/{img_path.name}",
            "category": primary,
            "categories": list(positive_categories),
            "regions": list(regions),
            "width": width,
            "height": height,
            "origin": demo_origin,
            "origin_name": f"{primary}/{img_path.name}",
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

    # Synthetic signposts demo: generated in-memory, no download, no model.
    if source == _TOPONYMY_SOURCE_ID:
        _load_synthetic_toponymy(clips, embedder, on_progress, demo_origin)
        return None

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

    if source == "vggface2":
        _embed_file_images(
            _collect_vggface2_files(categories, slice_args, on_progress),
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

    if source == "openlogo":
        _embed_boxed_multilabel_images(
            _collect_openlogo_files(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
            skip_embedding=skip_embedding,
        )
        return None

    if source == "rico_icons":
        _embed_boxed_multilabel_images(
            _collect_rico_icons_files(categories, slice_args, on_progress),
            clips,
            embedder,
            on_progress,
            demo_origin,
            skip_embedding=skip_embedding,
        )
        return None

    if source == "enrico":
        _embed_file_images(
            _collect_enrico_files(categories, slice_args, on_progress),
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
