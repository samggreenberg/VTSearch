"""CRUD routes for autorun extractors, localizers, and pregen processors.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.
"""

from __future__ import annotations

from flask_smorest import Blueprint, abort

from vtsearch.schemas.processors import (
    AutorunExtractorCreateRequestSchema,
    AutorunExtractorsListResponseSchema,
    AutorunLocalizerCreateRequestSchema,
    AutorunLocalizersListResponseSchema,
    AutorunProcessorCreateResponseSchema,
    AutorunProcessorDeleteResponseSchema,
    AutorunProcessorRenameRequestSchema,
    AutorunProcessorRenameResponseSchema,
    PregenProcessorsAddResponseSchema,
    PregenProcessorsListResponseSchema,
)
from vtsearch.autorun_processors import (
    add_autorun_extractor,
    add_autorun_localizer,
    get_autorun_extractors,
    get_autorun_localizers,
    remove_autorun_extractor,
    remove_autorun_localizer,
    rename_autorun_extractor,
    rename_autorun_localizer,
)

processors_crud_bp = Blueprint(
    "processors_crud",
    __name__,
    description="Manage autorun extractors / localizers and the bundled pregen-processor list.",
)


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
    from vtscore.media.image.extractor import ImageClassExtractor

    _EXTRACTOR_FACTORIES["image_class"] = ImageClassExtractor.from_config

    from vtscore.media.image.ocr_extractor import OCRExtractor

    _EXTRACTOR_FACTORIES["ocr"] = OCRExtractor.from_config

    from vtscore.media.audio.speech_extractor import SpeechExtractor

    _EXTRACTOR_FACTORIES["speech"] = SpeechExtractor.from_config


def _build_extractor(name: str, extractor_type: str, config: dict):
    """Instantiate an Extractor from its serialised form."""
    _ensure_extractor_factories()
    factory = _EXTRACTOR_FACTORIES.get(extractor_type)
    if factory is None:
        raise ValueError(f"Unknown extractor_type: {extractor_type!r}")
    return factory(name, config)


@processors_crud_bp.route("/api/autorun-extractors")
@processors_crud_bp.response(200, AutorunExtractorsListResponseSchema)
def get_autorun_extractors_route():
    """Get all autorun extractors."""
    extractors = get_autorun_extractors()
    return {"extractors": list(extractors.values())}


@processors_crud_bp.route("/api/autorun-extractors", methods=["POST"])
@processors_crud_bp.arguments(AutorunExtractorCreateRequestSchema)
@processors_crud_bp.response(200, AutorunProcessorCreateResponseSchema)
@processors_crud_bp.alt_response(400, description="Unbuildable extractor config.")
def add_autorun_extractor_route(body: dict):
    """Add a new autorun extractor."""
    name = body["name"].strip()
    extractor_type = body["extractor_type"].strip()
    media_type = body["media_type"].strip()
    config = body["config"]

    try:
        _build_extractor(name, extractor_type, config)
    except Exception as e:
        abort(400, message=f"Invalid extractor config: {e}")

    add_autorun_extractor(name, extractor_type, media_type, config)
    return {"success": True, "name": name}


@processors_crud_bp.route("/api/autorun-extractors/<name>", methods=["DELETE"])
@processors_crud_bp.response(200, AutorunProcessorDeleteResponseSchema)
@processors_crud_bp.alt_response(404, description="Extractor not found.")
def delete_autorun_extractor_route(name: str):
    """Delete a autorun extractor."""
    if remove_autorun_extractor(name):
        return {"success": True}
    abort(404, message="Extractor not found")


@processors_crud_bp.route("/api/autorun-extractors/<name>/rename", methods=["PUT"])
@processors_crud_bp.arguments(AutorunProcessorRenameRequestSchema)
@processors_crud_bp.response(200, AutorunProcessorRenameResponseSchema)
@processors_crud_bp.alt_response(400, description="Extractor not found, or new name already exists.")
def rename_autorun_extractor_route(body: dict, name: str):
    """Rename a autorun extractor."""
    new_name = body["new_name"].strip()
    if rename_autorun_extractor(name, new_name):
        return {"success": True, "new_name": new_name}
    abort(400, message="Extractor not found or new name already exists")


# ---------------------------------------------------------------------------
# Localizer CRUD
# ---------------------------------------------------------------------------

# Registry of localizer type constructors.
_LOCALIZER_FACTORIES: dict = {}


def _ensure_localizer_factories():
    """Populate the localizer factory registry on first use."""
    if _LOCALIZER_FACTORIES:
        return
    from vtscore.media.image.face_localizer import FaceLocalizer

    _LOCALIZER_FACTORIES["face"] = FaceLocalizer.from_config


def _build_localizer(name: str, localizer_type: str, config: dict):
    """Instantiate a Localizer from its serialised form."""
    _ensure_localizer_factories()
    factory = _LOCALIZER_FACTORIES.get(localizer_type)
    if factory is None:
        raise ValueError(f"Unknown localizer_type: {localizer_type!r}")
    return factory(name, config)


@processors_crud_bp.route("/api/autorun-localizers")
@processors_crud_bp.response(200, AutorunLocalizersListResponseSchema)
def get_autorun_localizers_route():
    """Get all autorun localizers."""
    localizers = get_autorun_localizers()
    return {"localizers": list(localizers.values())}


@processors_crud_bp.route("/api/autorun-localizers", methods=["POST"])
@processors_crud_bp.arguments(AutorunLocalizerCreateRequestSchema)
@processors_crud_bp.response(200, AutorunProcessorCreateResponseSchema)
@processors_crud_bp.alt_response(400, description="Unbuildable localizer config.")
def add_autorun_localizer_route(body: dict):
    """Add a new autorun localizer."""
    name = body["name"].strip()
    localizer_type = body["localizer_type"].strip()
    media_type = body["media_type"].strip()
    config = body["config"]

    try:
        _build_localizer(name, localizer_type, config)
    except Exception as e:
        abort(400, message=f"Invalid localizer config: {e}")

    add_autorun_localizer(name, localizer_type, media_type, config)
    return {"success": True, "name": name}


@processors_crud_bp.route("/api/autorun-localizers/<name>", methods=["DELETE"])
@processors_crud_bp.response(200, AutorunProcessorDeleteResponseSchema)
@processors_crud_bp.alt_response(404, description="Localizer not found.")
def delete_autorun_localizer_route(name: str):
    """Delete a autorun localizer."""
    if remove_autorun_localizer(name):
        return {"success": True}
    abort(404, message="Localizer not found")


@processors_crud_bp.route("/api/autorun-localizers/<name>/rename", methods=["PUT"])
@processors_crud_bp.arguments(AutorunProcessorRenameRequestSchema)
@processors_crud_bp.response(200, AutorunProcessorRenameResponseSchema)
@processors_crud_bp.alt_response(400, description="Localizer not found, or new name already exists.")
def rename_autorun_localizer_route(body: dict, name: str):
    """Rename a autorun localizer."""
    new_name = body["new_name"].strip()
    if rename_autorun_localizer(name, new_name):
        return {"success": True, "new_name": new_name}
    abort(400, message="Localizer not found or new name already exists")


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
@processors_crud_bp.response(200, PregenProcessorsListResponseSchema)
def list_pregen_processors():
    """Return the list of available pregen processors."""
    return {"processors": _PREGEN_PROCESSORS}


@processors_crud_bp.route("/api/pregen-processors/add", methods=["POST"])
@processors_crud_bp.response(200, PregenProcessorsAddResponseSchema)
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

    return {"success": True, "added": added}
