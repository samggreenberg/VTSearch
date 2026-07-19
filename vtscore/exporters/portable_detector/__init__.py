"""Portable-detector exporter – write standalone ONNX scoring bundles headlessly.

The GUI has a dedicated modal that exports a saved detector as a portable,
scoring-only zip (an ONNX MLP + ``manifest.json`` + ``README.md``; see
:mod:`vtscore.detectors.portable_bundle`).  This exporter is the CLI / CI
counterpart: run ``--autodetect`` (or a ``--pipeline`` YAML) with
``--exporter portable_detector`` and, instead of exporting the scored hits, it
writes one bundle per detector the run trained.

It is the sanctioned exception to the "No Persisted Vectors or MLPs" rule
(``CLAUDE.md``): the bundle persists the trained MLP - never embeddings or raw
media - so a third party can score their own media without VTSearch.

Unlike every other exporter, this one consumes the *trained classifiers*, not
the scored results, so it sets :attr:`needs_trained_detectors` and the pipeline
hands it the detectors via :meth:`export_cli_detectors`.  Structural (SIFT/VLAD)
detectors are skipped with a note rather than aborting the whole export - their
stage-2 RANSAC verification isn't representable as a scoring-only ONNX graph
(see :func:`vtscore.detectors.portable_bundle.check_exportable`).  Patch
(DINOv2/v3, EUPE) detectors export normally, in a degraded whole-item-only
scoring mode (see :func:`vtscore.detectors.portable_bundle.caveats_for_embedder_type`);
see ``docs/plans/detector-standalone-export.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vtscore.exporters.base import LabelsetExporter, PluginField
from vtscore.io import atomic_write_bytes

logger = logging.getLogger(__name__)

_DEFAULT_BUNDLE_PATH = "data/{detector_name}-detector.zip"


class PortableDetectorLabelsetExporter(LabelsetExporter):
    """Write each trained detector as a standalone, portable scoring bundle.

    CLI-only: it needs the trained MLP, which only the ``--autodetect`` /
    ``--pipeline`` path produces, so it is hidden from the GUI picker (the GUI
    has its own portable-export modal).  One zip is written per detector; use
    ``{detector_name}`` in the path to disambiguate a multi-detector run.
    """

    name = "portable_detector"
    display_name = "Portable Detector Bundle"
    description = "Write each trained detector as a standalone ONNX scoring bundle (zip)."
    icon = "\U0001f4e6"  # package
    # Needs the trained MLP, only available on the CLI/pipeline path; the GUI
    # has its own dedicated portable-export modal, so keep it out of the picker.
    hidden_from_picker = True
    fields = [
        PluginField(
            key="filepath",
            label="Save bundle to (server path)",
            field_type="server_path",
            description="Where on the server to write the portable-detector zip.",
            hint=(
                "One zip is written per detector. Use {detector_name} to give each detector its own "
                "file; without it, a multi-detector run inserts the detector name before the "
                "extension so bundles don't overwrite each other.\n"
                "Date template variables are also supported: {YYYYMMDD-HHMMSS}, {YYYYMMDD}, {YYYY}, "
                "{MM}, {DD}, {username}."
            ),
            placeholder=_DEFAULT_BUNDLE_PATH,
            default=_DEFAULT_BUNDLE_PATH,
            # NOTE: ``detector_name`` is intentionally *not* declared here - the
            # framework would resolve it once against the active detector
            # context, but this exporter substitutes it per-detector itself
            # (below) so a multi-detector run gets one file each.
            template_vars=("YYYYMMDD-HHMMSS", "YYYYMMDD", "YYYY", "MM", "DD", "username"),
        ),
    ]

    @property
    def needs_trained_detectors(self) -> bool:
        return True

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Not supported: this exporter needs the trained classifier, not hits.

        The results-only path (the GUI Auto-Find auto-export) can't build a
        bundle because it never sees the trained MLP.  Use the CLI/pipeline
        path (``--exporter portable_detector`` with ``--autodetect``) instead;
        for interactive one-off exports, the GUI's portable-export modal.
        """
        raise NotImplementedError(
            "The portable_detector exporter needs the trained classifier, which is only produced by "
            "the CLI autodetect/pipeline path. Run `--autodetect --exporter portable_detector`, or use "
            "the GUI's portable-export modal for a single saved detector."
        )

    def export_cli_detectors(
        self,
        detectors: list[dict[str, Any]],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Build and write one portable bundle per trained detector."""
        from vtscore.detectors import portable_bundle as pb  # noqa: PLC0415

        template = field_values["filepath"]
        multi = len(detectors) > 1

        written: list[str] = []
        skipped: list[tuple[str, str]] = []
        for descriptor in detectors:
            name = descriptor.get("detector_name", "") or "detector"
            weights = descriptor.get("weights") or {}
            embedder_type = descriptor.get("embedder_type", "") or ""
            try:
                pb.check_exportable(embedder_type)
                manifest = self._build_manifest(pb, descriptor)
                bundle = pb.build_bundle(weights=weights, manifest=manifest)
            except ValueError as exc:
                # Either the embedder type is blocked outright (structural), or
                # the weight shape can't be modelled by the fixed ONNX graph.
                # Skip this detector, keep exporting the rest.
                logger.warning("portable_detector: skipping %r (%s)", name, exc)
                skipped.append((name, str(exc)))
                continue

            out_path = self._resolve_path(template, name, disambiguate=multi)
            atomic_write_bytes(out_path, bundle)
            written.append(str(out_path.resolve()))

        return self._status(written, skipped)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _build_manifest(pb: Any, descriptor: dict[str, Any]) -> dict[str, Any]:
        """Assemble the bundle manifest for one trained detector."""
        embedder_name = descriptor.get("embedder", "") or ""
        embedder_type = descriptor.get("embedder_type", "") or ""
        embedder_display = embedder_name
        embedder_model_id: str | None = None
        if embedder_name:
            try:
                from vtscore.media import get_embedder  # noqa: PLC0415

                emb_obj = get_embedder(embedder_name)
                embedder_display = emb_obj.display_name
                embedder_model_id = emb_obj.model_id
            except Exception:  # noqa: BLE001 - cosmetic only; fall back to the raw name.
                embedder_display = embedder_name

        return pb.build_manifest(
            detector_name=descriptor.get("detector_name", ""),
            media_type=descriptor.get("media_type", "") or "",
            embedder=embedder_name,
            embedder_display_name=embedder_display,
            embedder_model_id=embedder_model_id,
            embedder_type=embedder_type,
            embedding_dim=pb.embedding_dim_from_weights(descriptor.get("weights") or {}),
            threshold=float(descriptor.get("threshold", 0.5)),
            good_count=int(descriptor.get("good_count", 0)),
            bad_count=int(descriptor.get("bad_count", 0)),
            exported_by=_exported_by(),
            exported_at=_utc_now_iso(),
            caveats=pb.caveats_for_embedder_type(embedder_type),
        )

    @staticmethod
    def _resolve_path(template: str, detector_name: str, *, disambiguate: bool) -> Path:
        """Turn the path template into a concrete per-detector zip path.

        Substitutes ``{detector_name}`` with a filesystem-safe slug of the
        detector's name.  When the template carries no ``{detector_name}`` and
        the run trained more than one detector, the slug is inserted before the
        file extension so the bundles don't overwrite one another.
        """
        from vtscore.security.path_validation import (  # noqa: PLC0415
            get_file_access_base_dir,
            sanitize_template_value,
            validate_server_filepath,
        )

        slug = sanitize_template_value(detector_name)
        if "{detector_name}" in template:
            resolved = template.replace("{detector_name}", slug)
        elif disambiguate:
            p = Path(template)
            resolved = str(p.with_name(f"{p.stem}-{slug}{p.suffix}"))
        else:
            resolved = template

        # Re-validate the concrete path (defence in depth): the substituted slug
        # carries no path separators, so this can't escape the base dir, but the
        # check keeps this ingress identical to every other server_path write.
        return validate_server_filepath(resolved, base_dir=get_file_access_base_dir())

    @staticmethod
    def _status(written: list[str], skipped: list[tuple[str, str]]) -> dict[str, Any]:
        """Build the confirmation status dict from the write/skip tallies."""
        if not written:
            if not skipped:
                return {"message": "No portable detector bundles written.", "filepaths": []}
            reasons = "; ".join(f"{name} ({reason})" for name, reason in skipped)
            return {
                "message": f"No portable detector bundles written ({len(skipped)} skipped): {reasons}.",
                "filepaths": [],
            }

        parts = [f"Wrote {len(written)} portable detector bundle(s)."]
        if skipped:
            names = ", ".join(name for name, _reason in skipped)
            parts.append(f"Skipped {len(skipped)} non-exportable detector(s): {names}.")
        return {"message": " ".join(parts), "filepaths": written}


def _exported_by() -> str:
    """Provenance string for the bundle: ``vtsearch <version>`` when available."""
    try:
        import vtsearch  # noqa: PLC0415

        return f"vtsearch {vtsearch.__version__}"
    except Exception:  # noqa: BLE001 - version is cosmetic; never block an export on it.
        return "vtsearch"


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 ``Z``-terminated string."""
    from datetime import datetime, timezone  # noqa: PLC0415

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


EXPORTER = PortableDetectorLabelsetExporter()
