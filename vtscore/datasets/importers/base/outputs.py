"""Output specs: one importer run, several datasets (#4703).

A single-dataset import is described by two form values the importer reads
itself: ``media_type`` (the type of the dataset it builds) and
``source_specs`` (which source types feed that dataset, through which
converters).  A **multi-dataset** import is a list of :class:`OutputSpec`
rows, each carrying exactly those two values for one dataset the run should
produce, plus the per-dataset load options that the *pipeline* applies after
the importer returns: the embedders, the clipper and its chain, the cleanup
gates, and the dataset's name.

The importer never sees the pipeline half.  :meth:`OutputSpec.narrow` folds
one output's ``media_type`` / ``source_specs`` (and name) back into a copy of
the shared ``field_values``, so the default multi-output hooks on
:class:`~vtscore.datasets.importers.base.core.ImporterBase` drive the existing
single-dataset :meth:`~ImporterBase.run` / :meth:`~ImporterBase.run_chunked`
once per output, and an importer that overrides those hooks to acquire its
source once (download an archive, extract it) still reads each output through
the same two keys its single-dataset path already understands.

Two outputs may share a dataset media type.  The Add Dataset form's
**Document** box, say, builds an *image* dataset of rendered pages
(``source_specs=[document → document2image]``) beside the **Image** box's image
dataset of the photos themselves; what tells them apart is
:attr:`OutputSpec.category`, the ingestion category the output stands for,
which also names the dataset (``"<base name> – Document"``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from typing import Any

from .specs import SourceSpec, _parse_multi_media_specs

__all__ = ["OutputSpec", "parse_output_specs", "output_dataset_name"]

#: Keys an :class:`OutputSpec` row may carry on the wire, besides the two the
#: importer reads (``media_type``, ``source_specs``) and ``category``.
_PIPELINE_KEYS = (
    "dataset_name",
    "embedder",
    "embedders",
    "clipper",
    "clipper_params",
    "clipper_chain",
    "cleaners",
)


def _resolve_type_id(raw: Any) -> str:
    """Resolve a folder name (``"images"``) or type id (``"image"``) to the type id.

    Raises :class:`ValueError` for a string naming no registered media type, so
    a typo in a request fails at validation rather than deep inside a loader.
    """
    from vtscore.media import all_type_ids, get_by_folder_name  # noqa: PLC0415

    value = str(raw or "").strip()
    if not value:
        raise ValueError("every output needs a 'media_type'")
    try:
        return get_by_folder_name(value).type_id
    except KeyError:
        pass
    if value in all_type_ids():
        return value
    raise ValueError(f"Unknown media type: {value!r}")


def _folder_name(type_id: str) -> str:
    """The ``folder_import_name`` importers' ``media_type`` fields speak."""
    from vtscore.media import get  # noqa: PLC0415

    try:
        return get(type_id).folder_import_name
    except KeyError:
        return type_id


def _type_label(type_id: str) -> str:
    """The media type's display name, or the id when it is not registered."""
    from vtscore.media import get  # noqa: PLC0415

    try:
        return get(type_id).name.strip()
    except KeyError:
        return type_id


def output_dataset_name(base_name: str, output: OutputSpec) -> str:
    """The dataset name one output of a multi-dataset import lands under.

    An explicit :attr:`OutputSpec.dataset_name` wins.  Otherwise the shared
    base name (what the user typed, or the importer's own
    :meth:`~ImporterBase.default_display_name`) gains the output's category
    label as a suffix, so the three datasets one archive produces read as
    siblings on the dashboard: ``"holiday – Image"``, ``"holiday – Face"``,
    ``"holiday – Document"``.
    """
    if output.dataset_name:
        return output.dataset_name
    label = output.category_label()
    base = (base_name or "").strip()
    return f"{base} – {label}" if base else label


@dataclass
class OutputSpec:
    """One dataset a multi-dataset import should produce.

    :param media_type: Type id of the dataset to build (``"image"``, ``"face"``…).
        What the importer's ``media_type`` field would hold in a single-dataset
        import of this output alone.
    :param source_specs: The source rows feeding this dataset; the same
        :class:`SourceSpec` list a single-dataset import submits.  Every
        converter row must bridge its source type to *media_type*.
    :param category: The ingestion category this output stands for, as a type
        id.  Defaults to *media_type*; differs when a category is only ever
        reached through a converter (a **Document** output whose dataset is
        the rendered pages, ``media_type="image"``).  Names the dataset.
    :param dataset_name: Explicit dataset name; empty derives one from the
        import's base name and the category (see :func:`output_dataset_name`).
    :param embedder: Primary create-time embedder for this dataset; empty for
        the media type's default.
    :param embedders: The v3 trio (text / patch / structural picks) when the
        form bound more than the primary; ``None`` for the single-embedder path.
    :param clipper: Clipper name; empty for none.
    :param clipper_params: The clipper's parameter values.
    :param clipper_chain: A ``clipper_chain`` step list, when the form sent one.
    :param cleaners: The ``cleaners`` selection the form sent (a list of
        ``{"name", "params"}`` rows, or its JSON string).
    """

    media_type: str
    source_specs: list[SourceSpec] = dc_field(default_factory=list)
    category: str = ""
    dataset_name: str = ""
    embedder: str = ""
    embedders: list[str] | None = None
    clipper: str = ""
    clipper_params: dict[str, Any] | None = None
    clipper_chain: list[dict[str, Any]] | None = None
    cleaners: Any = None

    def __post_init__(self) -> None:
        if not self.category:
            self.category = self.media_type

    def category_label(self) -> str:
        """The display name of this output's category (``"Image"``, ``"Document"``…)."""
        return _type_label(self.category or self.media_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_type": self.media_type,
            "source_specs": [s.to_dict() for s in self.source_specs],
            "category": self.category,
            "dataset_name": self.dataset_name,
            "embedder": self.embedder,
            "embedders": list(self.embedders) if self.embedders else None,
            "clipper": self.clipper,
            "clipper_params": dict(self.clipper_params) if self.clipper_params else None,
            "clipper_chain": list(self.clipper_chain) if self.clipper_chain else None,
            "cleaners": self.cleaners,
        }

    def narrow(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Fold this output into a copy of the shared *field_values*.

        The copy is what a single-dataset import of this output alone would
        have submitted: ``media_type`` holds the dataset type as the
        ``folder_import_name`` importers' select fields speak, ``source_specs``
        this output's rows (as plain dicts, the form's own encoding), and the
        per-dataset ``dataset_name`` / ``embedder`` / ``clipper`` the few
        importers that read them.  Hand the result to :meth:`ImporterBase.run`,
        :meth:`ImporterBase.run_chunked`, :meth:`ImporterBase.build_origin` or
        :meth:`ImporterBase.resolve_display_name` exactly as the single-dataset
        path does.
        """
        narrowed = dict(field_values)
        narrowed.pop("outputs", None)
        narrowed["media_type"] = _folder_name(self.media_type)
        narrowed["source_specs"] = [s.to_dict() for s in self.source_specs]
        if self.dataset_name:
            narrowed["dataset_name"] = self.dataset_name
        narrowed["embedder"] = self.embedder
        narrowed["clipper"] = self.clipper
        return narrowed


def _coerce_list(raw: Any) -> list[Any]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid outputs JSON: {exc}") from exc
    else:
        decoded = raw
    if decoded is None:
        return []
    if not isinstance(decoded, list):
        raise ValueError(f"outputs must be a list, got {type(decoded).__name__}")
    return decoded


def _coerce_embedders(raw: Any) -> list[str] | None:
    """Parse an output's ``embedders`` (a list, a JSON array string, or a
    comma-separated string) into an ordered, deduplicated name list; ``None``
    when nothing was sent, so the pipeline falls back to the single ``embedder``."""
    if raw is None:
        return None
    items: list[Any]
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        text = str(raw).strip()
        if not text:
            return None
        try:
            decoded = json.loads(text)
            items = decoded if isinstance(decoded, list) else [decoded]
        except (TypeError, ValueError):
            items = text.split(",")
    names: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in names:
            names.append(name)
    return names or None


def parse_output_specs(raw: Any) -> list[OutputSpec]:
    """Parse and validate a request's ``outputs`` value.

    *raw* is the list the Add Dataset form submits (one object per ticked
    category), or its JSON encoding on a multipart body.  ``None``, ``""`` and
    an empty list all mean "a single-dataset import": the caller then takes
    the ordinary ``media_type`` / ``source_specs`` path, so a client that does
    not know about outputs is unaffected.

    Each entry's ``media_type`` and ``category`` may be a folder name or a type
    id; both resolve to the type id.  ``source_specs`` goes through the same
    validation a single-dataset import's does
    (:func:`~vtscore.datasets.importers.base.specs._parse_multi_media_specs`),
    so an unknown converter, or one that does not produce the output's type,
    is rejected here with that validator's message.

    Raises:
        ValueError: On malformed JSON, a non-object entry, an unknown media
            type, or an invalid source-spec row.
    """
    specs: list[OutputSpec] = []
    for index, item in enumerate(_coerce_list(raw)):
        if not isinstance(item, dict):
            raise ValueError(f"outputs[{index}] must be an object, got {type(item).__name__}")
        media_type = _resolve_type_id(item.get("media_type"))
        category = _resolve_type_id(item.get("category")) if item.get("category") else media_type
        source_specs = _parse_multi_media_specs(item.get("source_specs"), media_type)
        clipper_params = item.get("clipper_params")
        if isinstance(clipper_params, str) and clipper_params.strip():
            try:
                clipper_params = json.loads(clipper_params)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"outputs[{index}].clipper_params is not valid JSON: {exc}") from exc
        if clipper_params is not None and not isinstance(clipper_params, dict):
            raise ValueError(f"outputs[{index}].clipper_params must be an object")
        specs.append(
            OutputSpec(
                media_type=media_type,
                source_specs=source_specs,
                category=category,
                dataset_name=str(item.get("dataset_name") or "").strip(),
                embedder=str(item.get("embedder") or "").strip(),
                embedders=_coerce_embedders(item.get("embedders")),
                clipper=str(item.get("clipper") or "").strip(),
                clipper_params=clipper_params or None,
                clipper_chain=item.get("clipper_chain") or None,
                cleaners=item.get("cleaners") or None,
            )
        )
    return specs
