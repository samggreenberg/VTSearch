"""Blueprint for on-demand media/text embedding.

Exposes a single endpoint, ``POST /api/embed``, that turns a media file
or a text snippet into a vector using a caller-chosen embedder.  Two
input modes share the route:

* ``multipart/form-data`` with ``embedder=<name>`` and ``file=<binary>``
  — runs :meth:`MediaEmbedder.embed_media` on the upload.
* ``application/json`` with ``{"embedder": "<name>", "text": "..."}``
  — runs :meth:`MediaEmbedder.embed_text`.

The user does not pass a ``media_type``: every embedder declares its
``media_type_id``, so picking the embedder already determines the
expected modality.  Wrong-modality requests (e.g. an audio file sent to
an image embedder) are caught by a fast extension pre-check before any
model weights are loaded.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from vtsearch.config import DATA_DIR
from vtsearch.media import all_embedders, get_by_extension, get_embedder
from vtsearch.media.embedder import media_from_path
from vtsearch.routes.helpers import get_json_or_400

embed_bp = Blueprint("embed", __name__)
logger = logging.getLogger(__name__)


def _unknown_embedder_response(name: str):
    available = sorted(e.name for e in all_embedders())
    return (
        jsonify({"error": f"Unknown embedder '{name}'. Available: {available}"}),
        404,
    )


def _vector_response(vec, embedder):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return jsonify(
        {
            "embedding": arr.tolist(),
            "dim": int(arr.shape[0]),
            "norm": float(np.linalg.norm(arr)),
            "embedder": embedder.name,
            "media_type": embedder.media_type_id,
        }
    )


def _is_multipart_request() -> bool:
    """True when the request carries a file upload or form fields.

    Falls back to inspecting ``Content-Type`` so callers that send
    multipart with no plain form fields (just the file) are still routed
    to the media path.
    """
    if request.files or request.form:
        return True
    content_type = (request.content_type or "").lower()
    return "multipart/form-data" in content_type


@embed_bp.route("/api/embed", methods=["POST"])
def embed():
    """Embed a media file (multipart) or text snippet (JSON) on demand."""
    if _is_multipart_request():
        return _embed_media_upload()
    return _embed_text_json()


def _embed_text_json():
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    name = str(data.get("embedder") or "").strip()
    if not name:
        return jsonify({"error": "embedder is required"}), 400

    try:
        embedder = get_embedder(name)
    except KeyError:
        return _unknown_embedder_response(name)

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return (
            jsonify(
                {
                    "error": (
                        "text is required for JSON input. "
                        "To embed a media file instead, send a multipart/form-data "
                        "request with a 'file' field."
                    )
                }
            ),
            400,
        )

    if not embedder.supports_text:
        return (
            jsonify(
                {
                    "error": f"Embedder '{embedder.name}' does not support text input.",
                    "supports_text": False,
                }
            ),
            400,
        )

    try:
        embedder.load_models()
        vec = embedder.embed_text(text)
    except Exception as exc:
        logger.exception("embed text failed for '%s'", name)
        return jsonify({"error": f"Embedding failed: {exc}"}), 500

    if vec is None:
        return jsonify({"error": f"Embedder '{embedder.name}' returned no vector for the text"}), 500

    return _vector_response(vec, embedder)


def _embed_media_upload():
    name = str(request.form.get("embedder") or "").strip()
    if not name:
        return jsonify({"error": "embedder is required"}), 400

    try:
        embedder = get_embedder(name)
    except KeyError:
        return _unknown_embedder_response(name)

    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "file is required"}), 400

    suffix = Path(file.filename).suffix
    detected = get_by_extension(suffix.lower()) if suffix else None
    if detected is not None and detected.type_id != embedder.media_type_id:
        return (
            jsonify(
                {
                    "error": (
                        f"Embedder '{embedder.name}' expects {embedder.media_type_id} files, "
                        f"but the uploaded file looks like {detected.type_id} (extension '{suffix}')."
                    ),
                    "expected_media_type": embedder.media_type_id,
                    "detected_media_type": detected.type_id,
                }
            ),
            400,
        )

    DATA_DIR.mkdir(exist_ok=True)
    temp_path = DATA_DIR / f"temp_embed_{uuid.uuid4().hex}{suffix or '.bin'}"
    try:
        file.save(temp_path)
        try:
            embedder.load_models()
            vec = embedder.embed_media(media_from_path(temp_path))
        except Exception as exc:
            logger.exception("embed media failed for '%s'", name)
            return jsonify({"error": f"Embedding failed: {exc}"}), 500

        if vec is None:
            return (
                jsonify(
                    {
                        "error": (
                            f"Embedder '{embedder.name}' (media_type={embedder.media_type_id}) "
                            f"could not embed the uploaded file."
                        ),
                        "media_type": embedder.media_type_id,
                    }
                ),
                400,
            )
        return _vector_response(vec, embedder)
    finally:
        temp_path.unlink(missing_ok=True)
