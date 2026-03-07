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

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        dataset_name = field_values.get("name", "")
        if not dataset_name:
            raise ValueError("Missing required field: name")

        embedder_name = field_values.get("embedder", "") or ""
        converter_name = field_values.get("converter", "") or ""

        load_demo_dataset(
            dataset_name,
            medias,
            embedder_name=embedder_name,
            converter_name=converter_name,
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


IMPORTER = DemoDatasetImporter()
