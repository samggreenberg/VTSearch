"""Demo dataset importer — loads pre-configured demo datasets.

Each registered :class:`~vtsearch.media.base.MediaType` defines a list of
:class:`~vtsearch.media.base.DemoDataset` entries.  This importer exposes
them through the standard importer interface so that demo datasets are
discovered, loaded, and managed exactly like any other dataset source.

The ``name`` field selects which demo dataset to load.  Optional
``embedder`` and ``converter`` fields allow overriding the default
embedding model and applying a media-type converter respectively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import load_demo_dataset


class DemoDatasetImporter(DatasetImporter):
    """Load a pre-configured demo dataset by name.

    Demo datasets are defined by media types and collected via
    :func:`~vtsearch.media.all_demo_datasets`.  This importer wraps the
    existing ``load_demo_dataset()`` function so that the demo loading
    pipeline participates in the standard importer registry.
    """

    name = "demo"
    display_name = "Demo Dataset"
    description = "Load a built-in demo dataset for exploring VTSearch"
    icon = "\U0001f3ac"  # 🎬
    ui_mode = "custom"
    hidden_from_picker = True

    fields = [
        ImporterField(
            key="name",
            label="Dataset",
            field_type="select",
            description="Which demo dataset to load.",
        ),
        ImporterField(
            key="embedder",
            label="Embedder",
            field_type="text",
            description="Override the default embedder (optional).",
            required=False,
        ),
        ImporterField(
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
        from vtsearch.datasets.config import DEMO_DATASETS

        for f in self.fields:
            if f.key == "name":
                f.options = list(DEMO_DATASETS.keys())
                break

    def resolve_display_name(self, field_values: dict[str, Any]) -> str:
        from vtsearch.datasets.config import DEMO_DATASETS

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
        converter_name = field_values.get("converter", "") or ""
        clipper_name = field_values.get("clipper", "") or ""

        load_demo_dataset(
            dataset_name,
            medias,
            embedder_name=embedder_name,
            converter_name=converter_name,
            clipper_name=clipper_name,
        )

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, str] = {"name": field_values.get("name", "")}
        converter = field_values.get("converter", "")
        if converter:
            params["converter"] = converter
        return {"importer": self.name, "params": params}

    def origin_display(self, origin: dict[str, Any]) -> str:
        params = origin.get("params", {})
        return f"demo:{params.get('name', '')}"

    def can_reload_from_origin(self, origin: dict[str, Any]) -> bool:
        from vtsearch.datasets.config import DEMO_DATASETS

        params = origin.get("params", {})
        return params.get("name", "") in DEMO_DATASETS

    def reload_from_origin(self, origin: dict[str, Any]) -> dict[str, Any] | None:
        params = origin.get("params", {})
        demo_name = params.get("name", "")
        if not demo_name:
            return None
        field_values: dict[str, Any] = {"name": demo_name}
        converter = params.get("converter", "")
        if converter:
            field_values["converter"] = converter
        return field_values

    def resolve_file(self, origin: dict[str, Any], origin_name: str = "", filename: str = "") -> Path | None:
        """Resolve a media file from a demo dataset origin.

        Maps the demo dataset name to its source, then resolves the file
        within the expected download directory on disk.
        """
        from vtsearch.datasets.config import DEMO_DATASETS

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

        # Basename fallback — some demo datasets store origin_name with a
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
        from vtsearch.config import DATA_DIR

        video_dir = DATA_DIR / "video"
        _SOURCE_DIRS = {
            # Image sources
            "caltech101": DATA_DIR / "caltech-101" / "101_ObjectCategories",
            "caltech256": DATA_DIR / "caltech-256" / "256_ObjectCategories",
            "oxford_flowers_102": DATA_DIR / "oxford_flowers",
            "food101": DATA_DIR / "food-101" / "images",
            "eurosat": DATA_DIR / "EuroSAT_RGB",
            "stanford_dogs": DATA_DIR / "stanford_dogs" / "Images",
            "ucsf_documents": DATA_DIR / "ucsf_documents",
            "cifar10_sample": DATA_DIR / "cifar-10-batches-py",
            # Audio sources
            "esc50": DATA_DIR / "ESC-50-master" / "audio",
            "gtzan": DATA_DIR / "gtzan" / "genres",
            "speech_commands_v2": DATA_DIR / "speech_commands_v2",
            "urbansound8k": DATA_DIR / "UrbanSound8K" / "audio",
            # Video sources
            "ucf101": video_dir / "ucf101",
            "hmdb51": video_dir / "hmdb51",
            "ucf101_full": video_dir / "ucf101_full",
            "kth": video_dir / "kth",
        }
    return _SOURCE_DIRS.get(source)


IMPORTER = DemoDatasetImporter()
