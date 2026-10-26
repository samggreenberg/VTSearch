"""Demo dataset importer: loads pre-configured demo datasets.

Each registered :class:`~vtscore.media.base.MediaType` defines a list of
:class:`~vtscore.media.base.DemoDataset` entries.  This importer exposes
them through the standard importer interface so that demo datasets are
discovered, loaded, and managed exactly like any other dataset source.

The ``name`` field selects which demo dataset to load.  Optional
``embedder`` and ``converter`` fields allow overriding the default
embedding model and applying a media-type converter respectively.

A demo load can also make several linked datasets from one demo (the Add
Dataset dialog's Multi-Dataset box on the Demo tab): ``/api/dataset/load-demo``
hands the multi-dataset pipeline one output per dataset, each fed by the
demo's own type, directly or through one converter (the photos, and the faces
cut out of them).  Each output reaches :meth:`DemoDatasetImporter.run` with its
converter in ``source_specs`` rather than in ``converter``; see
:func:`_demo_converter`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtscore.datasets.importers.base import ImporterBase, PluginField
from vtscore.datasets.loader import load_demo_dataset


def _demo_converter(field_values: dict[str, Any]) -> str:
    """The converter a demo load applies, or ``""`` for the demo's own type.

    The ``converter`` field when set (a single-dataset load).  Otherwise the
    converter of the ``source_specs`` row, which is how one output of a
    multi-dataset demo load names it: as a list of rows in the narrowed field
    values, or as their JSON in the origin a reload reads back.
    """
    name = str(field_values.get("converter") or "")
    if name:
        return name
    specs: Any = field_values.get("source_specs") or []
    if isinstance(specs, str):
        try:
            specs = json.loads(specs)
        except ValueError:
            return ""
    for spec in specs if isinstance(specs, list) else []:
        converter = spec.get("converter") if isinstance(spec, dict) else getattr(spec, "converter", None)
        if converter:
            return str(converter)
    return ""


class DemoDatasetImporter(ImporterBase):
    """Load a pre-configured demo dataset by name.

    Demo datasets are defined by media types and collected via
    :func:`~vtscore.media.all_demo_datasets`.  This importer wraps the
    existing ``load_demo_dataset()`` function so that the demo loading
    pipeline participates in the standard importer registry.
    """

    name = "demo"
    display_name = "Downloaded Media"
    description = "Choose from a selection of pre-configured demo datasets"
    icon = "\U0001f5c4"  # 🗄 (frontend renders as a database icon)
    ui_mode = "custom"
    picker_view = "demo"
    category = "demo"
    # ``load_demo_dataset`` applies the clipper itself (and caches the
    # clipped + embedded result), so the shared pipeline must not clip again.
    handles_own_clipping = True
    #: Closed to the generic import route's ``outputs`` (#4703): a demo has no
    #: ``media_type`` field to multiply.  ``/api/dataset/load-demo`` takes its
    #: own ``outputs`` instead, one per type the demo can be loaded as.
    multi_output = False

    fields = [
        PluginField(
            key="name",
            label="Dataset",
            field_type="select",
            description="Which demo dataset to load.",
        ),
        PluginField(
            key="embedder",
            label="Embedder",
            field_type="text",
            description="Override the default embedder (optional).",
            required=False,
            include_in_origin=False,
        ),
        PluginField(
            key="converter",
            label="Converter",
            field_type="text",
            description="Apply a media-type converter after loading (optional).",
            required=False,
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._refresh_options()

    def _refresh_options(self) -> None:
        """Populate the ``name`` field's options from the demo-dataset registry."""
        from vtscore.datasets.config import DEMO_DATASETS

        for f in self.fields:
            if f.key == "name":
                f.options = list(DEMO_DATASETS.keys())
                break

    def default_display_name(self, field_values: dict[str, Any]) -> str:
        from vtscore.datasets.config import DEMO_DATASETS

        dataset_name = field_values.get("name", "")
        entry = DEMO_DATASETS.get(dataset_name)
        if entry:
            return entry.get("label", self.display_name)
        return self.display_name

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        dataset_name = field_values.get("name", "")
        if not dataset_name:
            raise ValueError("Missing required field: name")

        embedder_name = field_values.get("embedder", "") or ""
        converter_name = _demo_converter(field_values)
        clipper_name = field_values.get("clipper", "") or ""
        clipper_params = field_values.get("clipper_params") or None

        load_demo_dataset(
            dataset_name,
            medias,
            embedder_name=embedder_name,
            converter_name=converter_name,
            clipper_name=clipper_name,
            clipper_params=clipper_params,
        )

    def origin_display(self, origin: dict[str, Any]) -> str:
        params = origin.get("params", {})
        return f"demo:{params.get('name', '')}"

    def can_reload_from_origin(self, origin: dict[str, Any]) -> bool:
        from vtscore.datasets.config import DEMO_DATASETS

        params = origin.get("params", {})
        return params.get("name", "") in DEMO_DATASETS

    def reload_from_origin(self, origin: dict[str, Any]) -> dict[str, Any] | None:
        params = origin.get("params", {})
        demo_name = params.get("name", "")
        if not demo_name:
            return None
        field_values: dict[str, Any] = {"name": demo_name}
        converter = _demo_converter(params)
        if converter:
            field_values["converter"] = converter
        return field_values

    def resolve_file(self, origin: dict[str, Any], origin_name: str = "", filename: str = "") -> Path | None:  # noqa: C901
        """Resolve a media file from a demo dataset origin.

        Maps the demo dataset name to its source, then resolves the file
        within the expected download directory on disk.
        """
        from vtscore.datasets.config import DEMO_DATASETS

        params = origin.get("params", {})
        demo_name = params.get("name", "")
        if not demo_name:
            return None

        entry = DEMO_DATASETS.get(demo_name)
        if not entry:
            return None

        source = entry.get("source", "")
        root = _source_directory(source)
        # Fall back to required_folder (e.g. ESC-50 has no explicit source)
        if root is None:
            rf = entry.get("required_folder")
            if rf is not None:
                root = Path(rf)
            else:
                return None

        # Try origin_name first (e.g. "kangaroo/image_0017.jpg"), then filename
        for name in (origin_name, filename):
            if name:
                candidate = root / name
                if candidate.is_file():
                    return candidate

        # Basename fallback: some demo datasets store origin_name with a
        # category prefix (e.g. "dog/1-100032-A-0.wav") but the on-disk
        # layout is flat (ESC-50) or uses a different hierarchy (UrbanSound8K
        # fold dirs, Oxford Flowers flat jpg/).  Try the bare filename.
        for name in (origin_name, filename):
            if name:
                basename = Path(name).name
                # Direct child first (flat directories like ESC-50)
                candidate = root / basename
                if candidate.is_file():
                    return candidate
                # Recursive search (fold-based layouts like UrbanSound8K)
                matches = list(root.rglob(basename))
                if len(matches) == 1:
                    return matches[0]

        return None


# Source name → expected download directory (without triggering downloads).
_SOURCE_DIRS: dict[str, Path] | None = None


def _source_directory(source: str) -> Path | None:
    """Return the on-disk root directory for a demo dataset *source*."""
    global _SOURCE_DIRS  # noqa: PLW0603
    if _SOURCE_DIRS is None:
        from vtscore.config import DATA_DIR

        video_dir = DATA_DIR / "video"
        _SOURCE_DIRS = {
            # Image sources
            "caltech101": DATA_DIR / "caltech-101" / "101_ObjectCategories",
            "caltech256": DATA_DIR / "caltech-256" / "256_ObjectCategories",
            "oxford_flowers_102": DATA_DIR / "oxford_flowers",
            "food101": DATA_DIR / "food-101" / "images",
            "eurosat": DATA_DIR / "EuroSAT_RGB",
            "enrico": DATA_DIR / "enrico",
            "rico_screen2words": DATA_DIR / "rico_screen2words" / "screenshots",
            "rvl_cdip": DATA_DIR / "rvl_cdip" / "images",
            "openlogo": DATA_DIR / "openlogo" / "data",
            "ucsf_documents": DATA_DIR / "ucsf_documents",
            "cifar10_sample": DATA_DIR / "cifar-10-batches-py",
            "vggface2": DATA_DIR / "vggface2" / "test",
            "celebrity_together": DATA_DIR / "celebrity_together" / "images",
            # Audio sources
            "esc50": DATA_DIR / "ESC-50-master" / "audio",
            "gtzan": DATA_DIR / "gtzan" / "genres",
            "speech_commands_v2": DATA_DIR / "speech_commands_v2",
            "urbansound8k": DATA_DIR / "UrbanSound8K" / "audio",
            "tut_sound_events_2017": DATA_DIR / "tut_sound_events_2017",
            "clotho": DATA_DIR / "clotho",
            # Video sources
            "ucf101": video_dir / "ucf101",
            "hmdb51": video_dir / "hmdb51",
            "ucf101_full": video_dir / "ucf101_full",
            "kth": video_dir / "kth",
        }
    return _SOURCE_DIRS.get(source)


IMPORTER = DemoDatasetImporter()
