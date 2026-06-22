"""Source-spec value objects and converter-ingestion helpers.

A multi-media import is described by a list of :class:`SourceSpec` rows; each
asks the importer to fetch media of one ``source_type`` and, when a converter
is set, run those media through it to reach the dataset's output media type.

The helpers here parse and validate the user-submitted spec list and stamp the
framework-required bookkeeping fields onto converter outputs.  They live apart
from the importer classes so the value object + validation logic can be
imported (e.g. by tests, or by spec-aware importers that drive their own
converter loop) without pulling in the full importer base.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dc_field
from typing import Any

__all__ = ["SourceSpec", "PickerView"]


@dataclass
class SourceSpec:
    """Declarative description of one media stream an importer should pull in.

    A multi-media import is a list of these.  Each spec asks the importer
    to fetch media of ``source_type`` and, when ``converter`` is set,
    pass them through that converter (with the supplied ``params``) to
    produce media of the dataset's output media type.

    When ``converter`` is ``None`` the source media is included directly
    and ``source_type`` must equal the importer's chosen output media
    type.

    See :meth:`DatasetImporter.effective_source_specs` for how importers
    obtain this list.
    """

    source_type: str
    converter: str | None = None
    params: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "converter": self.converter,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSpec:
        return cls(
            source_type=str(data.get("source_type") or ""),
            converter=(str(data["converter"]) if data.get("converter") else None),
            params=dict(data.get("params") or {}),
        )


def _parse_multi_media_specs(raw: Any, output_type: str) -> list[SourceSpec]:
    """Parse and validate the explicit ``source_specs`` form value.

    Falls back to a single pass-through spec when no value is submitted
    (or when the submitted value parses to an empty list) so an importer
    whose form omits ``source_specs`` still loads cleanly, and an empty
    spec-grid does not silently produce a zero-media dataset.
    """
    from vtscore.converters import get_converter  # noqa: PLC0415
    from vtscore.media import get_by_folder_name  # noqa: PLC0415

    specs_raw: list[dict[str, Any]]
    if raw is None or raw == "":
        specs_raw = []
    elif isinstance(raw, str):
        try:
            specs_raw = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid source_specs JSON: {exc}") from exc
    else:
        specs_raw = list(raw)

    if not output_type:
        raise ValueError("import requires a 'media_type' (output) field")

    if not specs_raw:
        specs_raw = [{"source_type": output_type, "converter": None, "params": {}}]

    specs: list[SourceSpec] = []
    for item in specs_raw:
        if not isinstance(item, dict):
            raise ValueError(f"source_specs entries must be objects, got {type(item).__name__}")
        spec = SourceSpec.from_dict(item)
        try:
            spec.source_type = get_by_folder_name(spec.source_type).type_id
        except (KeyError, AttributeError) as exc:
            raise ValueError(f"Unknown source_type: {spec.source_type!r}") from exc
        _validate_spec_converter(spec, output_type, get_converter)
        specs.append(spec)
    return specs


def _validate_spec_converter(spec: SourceSpec, output_type: str, get_converter) -> None:
    """Validate that *spec*'s converter (if any) bridges source→output.

    Also validates ``spec.params`` against the converter's own
    :class:`PluginField` schema, so declared ``min`` / ``max`` ranges
    are enforced before the params reach :meth:`MediaConverter.convert`.
    """
    from marshmallow import ValidationError  # noqa: PLC0415

    if spec.converter is None:
        if spec.source_type != output_type:
            raise ValueError(
                f"Direct (no-converter) source_type {spec.source_type!r} "
                f"does not match output media_type {output_type!r}",
            )
        return
    converter = get_converter(spec.converter)
    if converter is None:
        raise ValueError(f"Unknown converter: {spec.converter!r}")
    if converter.source_type != spec.source_type:
        raise ValueError(
            f"Converter {spec.converter!r} expects source_type {converter.source_type!r}, not {spec.source_type!r}",
        )
    if converter.target_type != output_type:
        raise ValueError(
            f"Converter {spec.converter!r} produces "
            f"{converter.target_type!r}, but output media_type is {output_type!r}",
        )
    try:
        converter.validate_params(spec.params)
    except ValidationError as exc:
        raise ValueError(f"Invalid params for converter {spec.converter!r}: {exc.messages}") from exc


PickerView = str  # one of: "form", "demo", "server_folder", "local"


def _fill_converter_output_fields(media: dict[str, Any]) -> None:
    """Fill framework-required fields that converters don't produce.

    Converter implementations return only content fields (``media_bytes``,
    ``media_string``, ``filename``, ``duration``, ``width``, ``height``).
    The ingestion layer is responsible for stamping the bookkeeping fields
    that the rest of the framework (``export_dataset_to_file``, the embed
    stage, etc.) expects on every media dict.
    """
    if "file_size" not in media:
        mb = media.get("media_bytes") or b""
        ms = (media.get("media_string") or "").encode("utf-8")
        media["file_size"] = len(mb) if mb else len(ms)
    if "md5" not in media:
        mb = media.get("media_bytes")
        if mb:
            media["md5"] = hashlib.md5(mb).hexdigest()
        elif media.get("media_string"):
            media["md5"] = hashlib.md5((media["media_string"] or "").encode("utf-8")).hexdigest()
        else:
            media["md5"] = hashlib.md5(b"").hexdigest()
    media.setdefault("embedding", None)
    media.setdefault("embedder", "")
    media.setdefault("category", "custom")
    media.setdefault("duration", 0)
