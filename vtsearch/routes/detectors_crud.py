"""CRUD routes for autorun detectors, extractors, localizers, and pregen processors."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, jsonify, request

from vtsearch.auth import get_current_user
from vtsearch.config import DATA_DIR
from vtsearch.routes.helpers import get_json_or_400
from vtsearch.utils import (
    add_autorun_detector,
    add_autorun_extractor,
    add_autorun_localizer,
    get_autorun_detectors,
    get_autorun_extractors,
    get_autorun_localizers,
    remove_autorun_detector,
    remove_autorun_extractor,
    remove_autorun_localizer,
    rename_autorun_detector,
    rename_autorun_extractor,
    rename_autorun_localizer,
    set_autorun_detector_autodetect,
    snapshot_medias,
)

detectors_crud_bp = Blueprint("detectors_crud", __name__)


def get_detectors_dir() -> Path:
    """Return the per-user detectors directory.

    In multi-user mode each user gets their own ``detectors/`` subdirectory
    inside their data dir, preventing cross-user file access.  In single-user
    mode the directory from settings is used (defaulting to ``data/detectors/``).
    """
    from vtsearch.auth import DefaultLoginProvider, get_login_provider, get_user_data_dir

    provider = get_login_provider()
    if isinstance(provider, DefaultLoginProvider):
        from vtsearch.settings import get_detectors_dir as _get

        return _get()
    return get_user_data_dir() / "detectors"


#: Backward-compat alias — prefer :func:`get_detectors_dir` for live value.
SERVER_DETECTOR_DIR = DATA_DIR / "detectors"


# ---------------------------------------------------------------------------
# Autorun detectors
# ---------------------------------------------------------------------------


@detectors_crud_bp.route("/api/autorun-detectors")
def get_autorun_detectors_route():
    """Get all autorun detectors."""
    detectors = get_autorun_detectors()
    return jsonify({"detectors": list(detectors.values())})


@detectors_crud_bp.route("/api/autorun-detectors", methods=["POST"])
def add_autorun_detector_route():
    """Add a new autorun detector.

    Weights are optional: when omitted the detector is created as an untrained
    stub that can be trained later through labeling.
    """
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    name = data.get("name", "").strip()
    media_type = data.get("media_type", "").strip()
    weights = data.get("weights")
    raw_threshold = data.get("threshold", 0.5)
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        return jsonify({"error": "threshold must be a number"}), 400
    autodetect = data.get("autodetect", False)

    examples = data.get("examples")
    try:
        num_labels = int(data.get("num_labels", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "num_labels must be a number"}), 400

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not media_type:
        return jsonify({"error": "media_type is required"}), 400

    # Capture origins from request or from current votes
    good_origins = data.get("good_origins")
    bad_origins = data.get("bad_origins")
    det_inclusion = data.get("inclusion", 0)
    if weights and not good_origins:
        # Best-effort: capture origins from current votes if available
        from vtsearch.models import collect_media_origins
        from vtsearch.utils import bad_votes, good_votes

        if good_votes and bad_votes:
            snap = snapshot_medias()
            good_origins = collect_media_origins(good_votes, snap)
            bad_origins = collect_media_origins(bad_votes, snap)
            from vtsearch.utils import get_inclusion

            det_inclusion = get_inclusion()

    add_autorun_detector(
        name,
        media_type,
        weights,
        threshold,
        autodetect=autodetect,
        examples=examples,
        num_labels=num_labels,
        created_by=get_current_user(),
        good_origins=good_origins,
        bad_origins=bad_origins,
        inclusion=det_inclusion,
    )

    # Extract text_query from examples (first text example) for the registry
    text_query = ""
    if examples:
        for ex in examples:
            if isinstance(ex, dict) and ex.get("type") == "text" and ex.get("value"):
                text_query = ex["value"]
                break

    # Also register in the persistent model registry so the dashboard grid
    # picks up the new model immediately.
    from vtsearch.models.registry import find_by_detector_name, register_model

    if not find_by_detector_name(name):
        # When trainable (no pre-existing weights), also create a trainable
        # model file so that labels and examples persist across restarts.
        trainable_model_name = ""
        if not weights:
            from vtsearch.models.trainable_model_store import _model_path, _write_model
            import time as _time

            trainable_model_name = name
            tm_path = _model_path(name)
            if not tm_path.exists():
                model_data = {
                    "name": name,
                    "text_query": text_query,
                    "media_type": media_type,
                    "examples": examples or [],
                    "created_at": _time.time(),
                    "labelset": {"labels": []},
                }
                _write_model(tm_path, model_data)

        register_model(
            name=name,
            media_type=media_type,
            trainable=not weights,
            num_training=num_labels,
            detector_name=name,
            text_query=text_query,
            trainable_model_name=trainable_model_name,
            created_by=get_current_user(),
        )

    return jsonify({"success": True, "name": name})


@detectors_crud_bp.route("/api/autorun-detectors/<name>", methods=["DELETE"])
def delete_autorun_detector_route(name):
    """Delete a autorun detector."""
    if remove_autorun_detector(name):
        # Also remove from the persistent model registry.
        from vtsearch.models.registry import find_by_detector_name, unregister_model

        entry = find_by_detector_name(name)
        if entry:
            unregister_model(entry["id"])
        return jsonify({"success": True})
    return jsonify({"error": "Detector not found"}), 404


@detectors_crud_bp.route("/api/autorun-detectors/<name>/rename", methods=["PUT"])
def rename_autorun_detector_route(name):
    """Rename a autorun detector."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400

    if rename_autorun_detector(name, new_name):
        # Keep the model registry in sync.
        from vtsearch.models.registry import find_by_detector_name, update_model

        entry = find_by_detector_name(name)
        if entry:
            update_model(entry["id"], name=new_name, detector_name=new_name)
        return jsonify({"success": True, "new_name": new_name})
    return jsonify({"error": "Detector not found or new name already exists"}), 400


@detectors_crud_bp.route("/api/autorun-detectors/<name>/autodetect", methods=["PUT"])
def set_autorun_detector_autodetect_route(name):
    """Set the autodetect flag on a autorun detector."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    autodetect = data.get("autodetect")
    if autodetect is None:
        return jsonify({"error": "autodetect is required"}), 400

    from vtsearch.settings import add_autorun_detector_name, remove_autorun_detector_name

    found_in_memory = set_autorun_detector_autodetect(name, bool(autodetect))
    # Even if the detector isn't loaded in memory, persist the setting so it
    # takes effect on next load / CLI autorun.
    if autodetect:
        add_autorun_detector_name(name)
    else:
        remove_autorun_detector_name(name)
        if not found_in_memory:
            return jsonify({"error": "Detector not found"}), 404
    return jsonify({"success": True, "autodetect": bool(autodetect)})


@detectors_crud_bp.route("/api/autorun-detectors/<name>/export", methods=["GET"])
def export_autorun_detector_route(name):
    """Return the stored weights for a named autorun detector (in-memory only)."""
    detectors = get_autorun_detectors()
    det = detectors.get(name)
    if det is None:
        return jsonify({"error": "Detector not found"}), 404
    if not det.get("weights"):
        return jsonify({"error": "Detector has no trained weights"}), 400
    return jsonify(
        {
            "weights": det["weights"],
            "threshold": det.get("threshold", 0.5),
            "media_type": det.get("media_type", ""),
            "name": det["name"],
        }
    )


@detectors_crud_bp.route("/api/autorun-detectors/<name>/examples", methods=["GET"])
def get_detector_examples_route(name):
    """Get the examples for a autorun detector."""
    from vtsearch.utils import get_autorun_detector_examples

    examples = get_autorun_detector_examples(name)
    return jsonify({"name": name, "examples": examples})


@detectors_crud_bp.route("/api/autorun-detectors/<name>/examples", methods=["PUT"])
def set_detector_examples_route(name):
    """Set/replace the examples for a autorun detector."""
    from vtsearch.utils import set_autorun_detector_examples

    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    examples = data.get("examples")
    if examples is None:
        return jsonify({"error": "examples is required"}), 400

    if set_autorun_detector_examples(name, examples):
        return jsonify({"success": True, "name": name, "examples": examples})
    return jsonify({"error": "Detector not found"}), 404


@detectors_crud_bp.route("/api/autorun-detectors/import-pkl", methods=["POST"])
def import_detector_pkl():
    """Import a autorun detector from a PKL file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    name = request.form.get("name", "").strip()
    if not name:
        # Use filename without extension as default name
        name = Path(file.filename).stem

    try:
        # Read the JSON detector file
        text = file.read().decode("utf-8")
        detector_data = json.loads(text)

        # Prefer media_type stored in the file; fall back to current medias, then "audio"
        media_type = detector_data.get("media_type", "")
        if not media_type:
            snap = snapshot_medias()
            if snap:
                media_type = next(iter(snap.values())).get("type", "audio")
            else:
                media_type = "audio"

        from vtsearch.models.weights_compat import normalize_detector_weights

        try:
            nw = normalize_detector_weights(detector_data, media_type=media_type)
        except ValueError:
            return jsonify({"error": "Invalid detector file format"}), 400

        add_autorun_detector(
            name,
            media_type,
            nw.weights,
            nw.threshold,
            created_by=get_current_user(),
            good_origins=nw.good_origins,
            bad_origins=nw.bad_origins,
            inclusion=nw.inclusion,
        )

        return jsonify({"success": True, "name": name, "media_type": media_type})

    except Exception:
        import logging

        logging.getLogger(__name__).exception("Failed to import detector")
        return jsonify({"error": "Failed to import detector file"}), 500


@detectors_crud_bp.route("/api/detector/server-files", methods=["GET"])
def list_server_detector_files():
    """List detector JSON files saved on the server."""
    det_dir = get_detectors_dir()
    if not det_dir.is_dir():
        return jsonify({"files": []})

    files = []
    for p in sorted(det_dir.glob("*.json")):
        files.append(
            {
                "name": p.stem,
                "filename": p.name,
                "size_bytes": p.stat().st_size,
            }
        )
    return jsonify({"files": files})


@detectors_crud_bp.route("/api/detector/server-files/<name>", methods=["GET"])
def get_server_detector_file(name: str):
    """Return the contents of a server-side detector file by name."""
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ")
    if not safe_name:
        return jsonify({"error": "Invalid name"}), 400

    det_dir = get_detectors_dir()
    filepath = det_dir / f"{safe_name}.json"
    if not filepath.is_file():
        return jsonify({"error": "Detector file not found"}), 404

    try:
        filepath.resolve().relative_to(det_dir.resolve())
    except ValueError:
        return jsonify({"error": "Invalid name"}), 400

    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return jsonify(data)
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Failed to read detector file: %s", safe_name)
        return jsonify({"error": "Failed to read detector file"}), 500


# ---------------------------------------------------------------------------
# Extractor CRUD
# ---------------------------------------------------------------------------

# Registry of extractor type constructors.
# Each entry maps an extractor_type string to a callable(name, config) -> Extractor.
_EXTRACTOR_FACTORIES: dict = {}


def _ensure_extractor_factories():
    """Populate the factory registry on first use (lazy to avoid import cycles)."""
    if _EXTRACTOR_FACTORIES:
        return
    from vtsearch.media.image.extractor import ImageClassExtractor

    _EXTRACTOR_FACTORIES["image_class"] = ImageClassExtractor.from_config

    from vtsearch.media.image.ocr_extractor import OCRExtractor

    _EXTRACTOR_FACTORIES["ocr"] = OCRExtractor.from_config

    from vtsearch.media.audio.speech_extractor import SpeechExtractor

    _EXTRACTOR_FACTORIES["speech"] = SpeechExtractor.from_config


def _build_extractor(name: str, extractor_type: str, config: dict):
    """Instantiate an Extractor from its serialised form."""
    _ensure_extractor_factories()
    factory = _EXTRACTOR_FACTORIES.get(extractor_type)
    if factory is None:
        raise ValueError(f"Unknown extractor_type: {extractor_type!r}")
    return factory(name, config)


@detectors_crud_bp.route("/api/autorun-extractors")
def get_autorun_extractors_route():
    """Get all autorun extractors."""
    extractors = get_autorun_extractors()
    return jsonify({"extractors": list(extractors.values())})


@detectors_crud_bp.route("/api/autorun-extractors", methods=["POST"])
def add_autorun_extractor_route():
    """Add a new autorun extractor."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    name = data.get("name", "").strip()
    extractor_type = data.get("extractor_type", "").strip()
    media_type = data.get("media_type", "").strip()
    config = data.get("config")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not extractor_type:
        return jsonify({"error": "extractor_type is required"}), 400
    if not media_type:
        return jsonify({"error": "media_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    # Validate that the extractor can be built from this config
    try:
        _build_extractor(name, extractor_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid extractor config: {e}"}), 400

    add_autorun_extractor(name, extractor_type, media_type, config)
    return jsonify({"success": True, "name": name})


@detectors_crud_bp.route("/api/autorun-extractors/<name>", methods=["DELETE"])
def delete_autorun_extractor_route(name):
    """Delete a autorun extractor."""
    if remove_autorun_extractor(name):
        return jsonify({"success": True})
    return jsonify({"error": "Extractor not found"}), 404


@detectors_crud_bp.route("/api/autorun-extractors/<name>/rename", methods=["PUT"])
def rename_autorun_extractor_route(name):
    """Rename a autorun extractor."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400

    if rename_autorun_extractor(name, new_name):
        return jsonify({"success": True, "new_name": new_name})
    return jsonify({"error": "Extractor not found or new name already exists"}), 400


# ---------------------------------------------------------------------------
# Localizer CRUD
# ---------------------------------------------------------------------------

# Registry of localizer type constructors.
_LOCALIZER_FACTORIES: dict = {}


def _ensure_localizer_factories():
    """Populate the localizer factory registry on first use."""
    if _LOCALIZER_FACTORIES:
        return
    from vtsearch.media.image.face_localizer import FaceLocalizer

    _LOCALIZER_FACTORIES["face"] = FaceLocalizer.from_config


def _build_localizer(name: str, localizer_type: str, config: dict):
    """Instantiate a Localizer from its serialised form."""
    _ensure_localizer_factories()
    factory = _LOCALIZER_FACTORIES.get(localizer_type)
    if factory is None:
        raise ValueError(f"Unknown localizer_type: {localizer_type!r}")
    return factory(name, config)


@detectors_crud_bp.route("/api/autorun-localizers")
def get_autorun_localizers_route():
    """Get all autorun localizers."""
    localizers = get_autorun_localizers()
    return jsonify({"localizers": list(localizers.values())})


@detectors_crud_bp.route("/api/autorun-localizers", methods=["POST"])
def add_autorun_localizer_route():
    """Add a new autorun localizer."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    name = data.get("name", "").strip()
    localizer_type = data.get("localizer_type", "").strip()
    media_type = data.get("media_type", "").strip()
    config = data.get("config")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not localizer_type:
        return jsonify({"error": "localizer_type is required"}), 400
    if not media_type:
        return jsonify({"error": "media_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    try:
        _build_localizer(name, localizer_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid localizer config: {e}"}), 400

    add_autorun_localizer(name, localizer_type, media_type, config)
    return jsonify({"success": True, "name": name})


@detectors_crud_bp.route("/api/autorun-localizers/<name>", methods=["DELETE"])
def delete_autorun_localizer_route(name):
    """Delete a autorun localizer."""
    if remove_autorun_localizer(name):
        return jsonify({"success": True})
    return jsonify({"error": "Localizer not found"}), 404


@detectors_crud_bp.route("/api/autorun-localizers/<name>/rename", methods=["PUT"])
def rename_autorun_localizer_route(name):
    """Rename a autorun localizer."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400

    if rename_autorun_localizer(name, new_name):
        return jsonify({"success": True, "new_name": new_name})
    return jsonify({"error": "Localizer not found or new name already exists"}), 400


# ---------------------------------------------------------------------------
# Pregen Processors
# ---------------------------------------------------------------------------

# Default pregen processor definitions: (name, processor_type, kind, media_type, config)
# kind is "extractor" or "localizer"
_PREGEN_PROCESSORS = [
    {
        "name": "OCR (PaddleOCR)",
        "kind": "extractor",
        "processor_type": "ocr",
        "media_type": "image",
        "config": {"language": "en", "threshold": 0.5},
    },
    {
        "name": "Speech (Whisper Tiny)",
        "kind": "extractor",
        "processor_type": "speech",
        "media_type": "audio",
        "config": {"model_size": "tiny", "language": None},
    },
    {
        "name": "Face (MediaPipe)",
        "kind": "localizer",
        "processor_type": "face",
        "media_type": "image",
        "config": {"threshold": 0.5, "model_selection": 1},
    },
]


@detectors_crud_bp.route("/api/pregen-processors", methods=["GET"])
def list_pregen_processors():
    """Return the list of available pregen processors."""
    return jsonify({"processors": _PREGEN_PROCESSORS})


@detectors_crud_bp.route("/api/pregen-processors/add", methods=["POST"])
def add_pregen_processors():
    """Add all pregen processors as autorun entries.

    Registers the OCR extractor, Speech extractor, and Face localizer
    into the autorun extractors and localizers stores.
    """
    added = []
    for proc in _PREGEN_PROCESSORS:
        name = proc["name"]
        kind = proc["kind"]
        media_type = proc["media_type"]
        config = proc["config"]
        processor_type = proc["processor_type"]

        if kind == "extractor":
            add_autorun_extractor(name, processor_type, media_type, config)
        elif kind == "localizer":
            add_autorun_localizer(name, processor_type, media_type, config)
        added.append(name)

    return jsonify({"success": True, "added": added})
