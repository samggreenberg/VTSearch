"""Portable detector export route.

Streams a standalone, transferable scoring bundle (ONNX model + manifest +
README) for a saved detector.  See :mod:`vtscore.detectors.portable_bundle` and
``docs/plans/detector-standalone-export.md``.

This is the sanctioned exception to the "No Persisted Vectors or MLPs" rule
(``CLAUDE.md``): the bundle persists the trained MLP - never embeddings or raw
media - so other parties can score their own media without VTSearch.

Migrated to ``flask_smorest`` so the route is described in
``/api/openapi.json``.  The success body is a raw ``application/zip`` download,
so (like the dataset-export route) it is left undescribed in the spec and only
the error responses carry schemas.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from flask import send_file
from flask_smorest import Blueprint, abort

from vtsearch.routes._shared import require_dataset_header
from vtsearch.state import snapshot_medias

logger = logging.getLogger(__name__)

detectors_export_bp = Blueprint(
    "detectors_export",
    __name__,
    description="Export a saved detector as a standalone, portable scoring bundle.",
)


@detectors_export_bp.route("/api/detectors/<detector_id>/portable-bundle", methods=["POST"])
@detectors_export_bp.alt_response(400, description="No medias loaded, or the detector has no labels for scoring.")
@detectors_export_bp.alt_response(404, description="Detector not found.")
@detectors_export_bp.alt_response(409, description="The active dataset can't supply the detector's embedder type.")
@require_dataset_header
def export_portable_bundle(detector_id: str):
    """Train the detector against the active dataset and stream its portable bundle.

    The detector's MLP is (re)trained from its on-disk labelset in the active
    dataset's embedder space - exactly as Find does - then serialised to an ONNX
    scorer and zipped with a manifest and README.  No embeddings or raw media
    are written; only the trained classifier, the embedder name, and the
    threshold travel in the bundle.
    """
    import vtsearch  # noqa: PLC0415
    from vtscore.datasets.labelset import LabelSet  # noqa: PLC0415
    from vtscore.detectors import portable_bundle as pb  # noqa: PLC0415
    from vtscore.detectors.model_loading import resolve_or_train_detector  # noqa: PLC0415
    from vtscore.detectors.registry import get_detector as reg_get_detector  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _read_detector, _slug  # noqa: PLC0415
    from vtscore.detectors.training import serialize_weights  # noqa: PLC0415
    from vtscore.embedding.binding import keying_embedder_for_snap  # noqa: PLC0415
    from vtscore.media import get_embedder  # noqa: PLC0415
    from vtsearch.routes.detectors.scoring import (  # noqa: PLC0415
        _dataset_supplies_detector_type,
        _detector_type,
        _type_incompatible_message,
    )

    d = reg_get_detector(detector_id)
    if d is None:
        abort(404, message=f"Detector '{detector_id}' not found")

    snap = snapshot_medias()
    if not snap:
        abort(400, message="No medias loaded")

    media_type = d.get("media_type", "") or next(iter(snap.values())).get("media_type", "image")
    det_data = _read_detector(_detector_path(d["name"]))

    # Type gate: the active dataset must bind an embedder of the detector's
    # locked type, else the labels would train/score in a foreign space.
    if not _dataset_supplies_detector_type(det_data, snap):
        abort(409, message=_type_incompatible_message(det_data))

    mlp, threshold, diagnostic = resolve_or_train_detector(detector_id, det_data, media_type, snap)
    if mlp is None:
        if diagnostic is not None:
            failed = diagnostic["failed_resolution"]
            total = diagnostic["total_labels"]
            abort(
                400,
                message=(
                    f"Detector '{d['name']}' could not be trained: {failed} of {total} "
                    "labeled items could not be resolved from their original files."
                ),
                resolution_diagnostic=diagnostic,
            )
        abort(400, message=f"Detector '{d['name']}' has no labels for scoring")

    # Score-space embedder: the concrete embedder of the detector's locked type
    # this dataset supplies (matches Find / resolve_or_train_detector's space).
    det_type = _detector_type(det_data)
    score_emb = keying_embedder_for_snap(SimpleNamespace(embedder_type=det_type), snap)
    embedder_display = score_emb or ""
    if score_emb:
        try:
            embedder_display = get_embedder(score_emb).display_name
        except Exception:  # noqa: BLE001 - cosmetic only; fall back to the raw name.
            embedder_display = score_emb

    weights = serialize_weights(mlp)
    labelset = LabelSet.from_dict((det_data or {}).get("labelset") or {})
    good_count = sum(1 for el in labelset.elements if el.label == "good")
    bad_count = sum(1 for el in labelset.elements if el.label == "bad")

    manifest = pb.build_manifest(
        detector_name=d.get("name", ""),
        media_type=media_type,
        embedder=score_emb or "",
        embedder_display_name=embedder_display,
        embedder_type=det_type,
        embedding_dim=pb.embedding_dim_from_weights(weights),
        threshold=threshold,
        good_count=good_count,
        bad_count=bad_count,
        exported_by=f"vtsearch {vtsearch.__version__}",
        exported_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    bundle = pb.build_bundle(weights=weights, manifest=manifest)

    return send_file(
        io.BytesIO(bundle),
        mimetype="application/zip",
        download_name=f"{_slug(d['name'])}-detector.zip",
        as_attachment=True,
    )
