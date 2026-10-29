"""Import labels into a detector from the CLI (``--import-labels-into``, a pipeline file's ``import_labels:``).

:func:`import_labels_into_detector` runs a label importer and merges its
``good``/``bad`` entries into the detector's labelset, creating the detector
when asked to - the way the Dashboard's New Detector would.
:func:`_label_import_media_type` decides, before any media loads, what type a
detector the import creates gets.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vtscore.cli.sources import _SourceSpec
    from vtscore.datasets.labelset import LabelSet


class DetectorNotFoundError(ValueError):
    """The detector a label import names has no file in the detectors dir.

    A ``ValueError`` so callers that already report import failures keep
    doing so; the subclass lets a CLI surface add its own hint (the flag or
    YAML key that would create the detector) without matching on the text.
    """

    def __init__(self, det_name: str) -> None:
        super().__init__(f"Detector '{det_name}' not found.")
        self.det_name = det_name


def import_labels_into_detector_from_file(
    det_name: str,
    importer_name: str,
    filepath: str,
) -> tuple[int, int]:
    """Run a label importer against a single file and merge into a detector.

    Shorthand for :func:`import_labels_into_detector` with
    ``{"filepath": filepath}`` as the importer's field values.
    """
    return import_labels_into_detector(det_name, importer_name, {"filepath": filepath})


def import_labels_into_detector(
    det_name: str,
    importer_name: str,
    field_values: dict[str, Any],
    *,
    create_media_type: str = "",
) -> tuple[int, int]:
    """Run a label importer with *field_values* and merge its labels into a detector.

    *field_values* maps the importer's :attr:`PluginField.key` s to values,
    exactly as the web form would submit them, so importers that take no
    file (a database query, a remote service) work from the CLI too.
    Required fields are checked, and the values normalized, the same way
    the dataset importer and exporter CLI paths do.

    A missing detector raises :class:`DetectorNotFoundError` unless
    *create_media_type* names a media type, in which case the detector is
    created with that type from the imported labels - written and
    registered the way the Dashboard's New Detector does, so it shows up in
    the creating user's Drafts.  Nothing is created when the import yields
    no ``good``/``bad`` label.  An existing detector ignores
    *create_media_type* and is merged into as usual.
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.store import _detector_path, _read_detector, _write_detector

    path = _detector_path(det_name)
    data = _read_detector(path)
    if data is None:
        if not create_media_type:
            raise DetectorNotFoundError(det_name)
        _check_detector_media_type(create_media_type, "create_media_type")

    label_entries, import_record = _run_label_importer(importer_name, field_values)
    existing = LabelSet.from_dict((data or {}).get("labelset") or {})
    applied, skipped = _merge_label_entries(existing, label_entries)

    created = data is None
    if data is None:
        if not applied:
            raise ValueError(
                f"Label importer {importer_name!r} produced no good/bad labels to create detector '{det_name}' from."
            )
        data = _new_detector_data(det_name, create_media_type)
    data["labelset"] = existing.to_dict()
    _write_detector(path, data)
    if created:
        _register_created_detector(det_name, create_media_type, applied, imported=import_record)
    return applied, skipped


def _run_label_importer(
    importer_name: str, field_values: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate *field_values* for the named label importer, run it, and return its entries.

    Also returns the run's import record (the importer, its normalized field
    values, the time), which a detector the import creates keeps (#5334).
    """
    from vtscore.detectors.import_record import build_import_record
    from vtscore.labels.importers import get_label_importer

    importer = get_label_importer(importer_name)
    if importer is None:
        raise ValueError(f"Unknown label importer: {importer_name!r}.")

    field_values = dict(field_values)
    importer.validate_cli_field_values(field_values)
    label_entries = importer.run_cli(field_values)
    if not isinstance(label_entries, list):
        raise ValueError(f"Label importer {importer_name!r} returned {type(label_entries).__name__}, expected list.")
    return label_entries, build_import_record(importer, field_values)


def _merge_label_entries(existing: "LabelSet", label_entries: list[dict[str, Any]]) -> tuple[int, int]:
    """Append each new ``good``/``bad`` entry to *existing* in place; return ``(applied, skipped)``.

    Entries with any other label, and ``(md5, label)`` pairs *existing*
    already holds, are skipped.
    """
    from vtscore.datasets.labelset import LabeledElement

    existing_keys: set[tuple[str, str]] = {(el.md5, el.label) for el in existing.elements if el.md5}
    applied = 0
    skipped = 0
    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        md5 = entry.get("md5", "")
        if md5 and (md5, label) in existing_keys:
            skipped += 1
            continue
        existing.elements.append(LabeledElement.from_dict(entry))
        if md5:
            existing_keys.add((md5, label))
        applied += 1
    return applied, skipped


def _check_detector_media_type(media_type: str, option: str) -> None:
    """Reject a media type no registered media type answers to, naming *option*."""
    from vtscore.media import all_type_ids

    valid = all_type_ids()
    if media_type not in valid:
        raise ValueError(f"Unknown media type for {option}: {media_type!r}. Valid values: {', '.join(sorted(valid))}.")


def _new_detector_data(det_name: str, media_type: str) -> dict[str, Any]:
    """The detector JSON the Dashboard's New Detector writes, with no seed examples.

    Mirrors ``POST /api/detectors/registry``.  ``embedder_type`` stays empty,
    as it does for a detector created with no dataset loaded: the first train
    resolves it.  Origins and labels only - never vectors or weights.
    """
    import time

    return {
        "name": det_name,
        "text_query": "",
        "media_example": "",
        "media_type": media_type,
        "examples": [],
        "created_at": time.time(),
        "embedder_type": "",
        "labelset": {},
    }


def _register_created_detector(
    det_name: str, media_type: str, num_labels: int, *, imported: dict[str, Any] | None = None
) -> None:
    """Add a CLI-created detector to the registry so it appears in its creator's Drafts.

    *imported* is the import record of the label import that created it.

    Skipped when an entry already owns the name - one whose labelset file had
    been deleted - since that entry now finds its file again, and a second
    entry would share it (the Dashboard refuses such a duplicate with a 409).
    """
    from vtscore.detectors.registry import find_by_name, register_detector
    from vtscore.state.current_user import get_current_user

    if find_by_name(det_name) is not None:
        return
    register_detector(
        name=det_name,
        media_type=media_type,
        num_training=num_labels,
        created_by=get_current_user(),
        imported=imported,
    )


def _source_media_type(spec: _SourceSpec) -> str:
    """The media type *spec*'s source declares, read without loading any media.

    A dataset pickle records it in its container's ``meta.json``; an importer
    that asks for one (``server_folder``, ``http_archive``, ...) has it in its
    ``media_type`` field.  ``""`` when the source declares none - a legacy
    pickle, or an importer with no such field.
    """
    if spec.kind == "pickle":
        path = Path(spec.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {spec.dataset_path}")
        from vtscore.datasets.loader_pickle import _read_pkl_meta_safe

        return str(_read_pkl_meta_safe(path).get("media_type") or "")
    if spec.outputs:
        # Several datasets declare one type only when they all share it; a
        # mixed run leaves the caller to say which type it means.
        types = {output.media_type for output in spec.outputs}
        return next(iter(types)) if len(types) == 1 else ""
    from vtscore.datasets.importers import get_importer

    importer = get_importer(spec.importer_name)
    field_def = next((f for f in importer.fields if f.key == "media_type"), None) if importer else None
    if field_def is None:
        return ""
    return str(spec.field_values.get("media_type") or field_def.default or "")


def _label_import_media_type(
    det_name: str,
    spec: _SourceSpec,
    *,
    create: bool,
    media_type: str = "",
    media_type_option: str,
) -> str:
    """What a label import into *det_name* should create it with, if anything.

    Returns ``""`` when the detector exists (the import merges into it) or
    *create* is off (a missing detector then fails in
    :func:`import_labels_into_detector`).  Otherwise returns the media type to
    create it with: *media_type* when given, else the one the run's source
    declares (:func:`_source_media_type`).  Raises :class:`ValueError` naming
    *media_type_option* when neither supplies a known media type, so a run
    that would have to guess fails before any media is loaded.
    """
    from vtscore.detectors.store import _detector_path, _read_detector

    if media_type:
        # Checked even when it goes unused, so a typo never waits for the one
        # run that has to create the detector.
        _check_detector_media_type(media_type, media_type_option)
    if not create or _read_detector(_detector_path(det_name)) is not None:
        return ""
    resolved = media_type or _source_media_type(spec)
    if not resolved:
        raise ValueError(
            f"Cannot tell which media type to create detector '{det_name}' with: "
            f"the source does not declare one. Set {media_type_option}."
        )
    _check_detector_media_type(resolved, media_type_option)
    return resolved
