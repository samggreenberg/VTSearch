"""CRUD routes for autorun extractors, localizers, and pregen processors."""

from __future__ import annotations

from flask import Blueprint, jsonify

from vtsearch.routes._shared import get_json_or_400
from vtsearch.state import (
    add_autorun_extractor,
    add_autorun_localizer,
    get_autorun_extractors,
    get_autorun_localizers,
    remove_autorun_extractor,
    remove_autorun_localizer,
    rename_autorun_extractor,
    rename_autorun_localizer,
)

processors_crud_bp = Blueprint("processors_crud", __name__)


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


@processors_crud_bp.route("/api/autorun-extractors")
def get_autorun_extractors_route():
    """Get all autorun extractors."""
    extractors = get_autorun_extractors()
    return jsonify({"extractors": list(extractors.values())})


@processors_crud_bp.route("/api/autorun-extractors", methods=["POST"])
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


@processors_crud_bp.route("/api/autorun-extractors/<name>", methods=["DELETE"])
def delete_autorun_extractor_route(name):
    """Delete a autorun extractor."""
    if remove_autorun_extractor(name):
        return jsonify({"success": True})
    return jsonify({"error": "Extractor not found"}), 404


@processors_crud_bp.route("/api/autorun-extractors/<name>/rename", methods=["PUT"])
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


@processors_crud_bp.route("/api/autorun-localizers")
def get_autorun_localizers_route():
    """Get all autorun localizers."""
    localizers = get_autorun_localizers()
    return jsonify({"localizers": list(localizers.values())})


@processors_crud_bp.route("/api/autorun-localizers", methods=["POST"])
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


@processors_crud_bp.route("/api/autorun-localizers/<name>", methods=["DELETE"])
def delete_autorun_localizer_route(name):
    """Delete a autorun localizer."""
    if remove_autorun_localizer(name):
        return jsonify({"success": True})
    return jsonify({"error": "Localizer not found"}), 404


@processors_crud_bp.route("/api/autorun-localizers/<name>/rename", methods=["PUT"])
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


@processors_crud_bp.route("/api/pregen-processors", methods=["GET"])
def list_pregen_processors():
    """Return the list of available pregen processors."""
    return jsonify({"processors": _PREGEN_PROCESSORS})


@processors_crud_bp.route("/api/pregen-processors/add", methods=["POST"])
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
