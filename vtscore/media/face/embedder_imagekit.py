"""The face embedder: face identity over the ``imagekit`` provider seam (the server's chosen backend).

One ``FacePipeline`` (:mod:`vtscore.media.face.imagekit_bridge`) sits behind
it, and the server picks the backend - the open SCRFD + ArcFace/MagFace ONNX
weights, or an external backend module - so the app never sees which.  Like the ``image2face`` converter, it runs only on a server with face
models (``imagekit_bridge.face_models``); without them it declines every item.

**Most faces never reach this embedder.**  The converter embeds each face from
the *full photo* in the same pass that detects it, and files the vector under
this embedder's name on the crop it emits; the embed stage then finds nothing
missing.  What does reach it is a face crop with no vector of its own - a reload
that lost its vectors, a crop from somewhere else - and for those the pipeline
runs on the crop itself: the provider's detector finds the face inside the crop
(every provider embeds from a detection, the external one included), and the
best detection's embedding is the crop's.  A crop in which the provider finds no
face gets no vector and is dropped at the end of the load, with the reason in
the drop toast.

**One embedding space per dataset.**  Every face the converter embeds carries
its space's tag in ``custom_metadata["Face Compat"]``.  Before this embedder
adds a vector to a crop that carries a *different* tag it raises
:class:`imagekit.face.compat.CompatibilityError`, which fails the batch with
that message rather than quietly mixing two providers' vectors under one
name.  A face dataset embedded by the retired FaceNet embedder (#5320) carries
no tag; its stored vectors still load, but they are FaceNet's, so re-import
the photos to search it in this embedder's space.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from vtscore.media.embedder import MediaEmbedder, timed_progress
from vtscore.media.face import imagekit_bridge as bridge

logger = logging.getLogger(__name__)


class ImagekitFaceEmbedder(MediaEmbedder):
    """Embeds face crops through the server's face provider."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    @property
    def name(self) -> str:
        return bridge.EMBEDDER_NAME

    @property
    def display_name(self) -> str:
        return "Face identity (512d)"

    @property
    def model_id(self) -> Optional[str]:
        # The embedding space is the provider's compat tag; it is only known
        # once the pipeline is built, and reading this must never build it.
        return bridge.loaded_compat() or None

    @property
    def media_type_id(self) -> str:
        return "face"

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def is_default(self) -> bool:
        return True

    @property
    def supports_text(self) -> bool:
        return False

    @property
    def supports_patch_regions(self) -> bool:
        return False

    @property
    def license_notice(self) -> Optional[str]:
        return (
            "Embeds through the face provider this server is configured with (IMAGEKIT_FACE_PROVIDER). "
            "Open-weights face models from the InsightFace model zoo are licensed for non-commercial research "
            "only; a deployment supplies weights it may use."
        )

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return
        with timed_progress(self._on_progress, "loading", f"Loading the {bridge.provider_name()} face provider…", 2, 2):
            self._model = bridge.get_face_pipeline()

    def _decode(self, media: dict) -> np.ndarray:
        """The crop as an image tensor, from inline bytes or the media path."""
        from pathlib import Path  # noqa: PLC0415

        from imagekit.util import decode_image_buffer, read_image_tensor  # noqa: PLC0415

        blob = media.get("media_bytes")
        if isinstance(blob, (bytes, bytearray)) and blob:
            return decode_image_buffer(bytes(blob))
        return read_image_tensor(Path(media["media_path"]))

    def _check_compat(self, media: dict) -> None:
        """Refuse to add this provider's vector to a crop stamped with another space."""
        from imagekit.face.compat import CompatibilityError  # noqa: PLC0415

        stamped = (media.get("custom_metadata") or {}).get(bridge.COMPAT_METADATA_KEY)
        current = bridge.loaded_compat()
        if stamped and current and str(stamped) != current:
            raise CompatibilityError(
                f"face {media.get('origin_name') or media.get('filename') or media.get('id')!r} carries "
                f"{bridge.COMPAT_METADATA_KEY} {stamped!r} but this server's face provider embeds {current!r}; "
                "one dataset holds one embedding space"
            )

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        self._check_compat(media)
        try:
            image = self._decode(media)
        except Exception:
            logger.exception("Error decoding face crop")
            return None
        records = self._model.process(image, media_id=str(media.get("id", "")))
        if not records:
            logger.warning(
                "imagekit face provider found no face in crop %r", media.get("origin_name") or media.get("id")
            )
            return None
        return records[0].unit_embedding().astype(np.float32)

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return [None] * len(medias)
        total = len(medias)
        results: list[Optional[np.ndarray]] = []
        with self._embed_lock:
            for i, media in enumerate(medias):
                self._on_progress("embedding", f"Embedding faces ({bridge.provider_name()})…", i + 1, total)
                results.append(self._embed_media_impl(media))
        return results


EMBEDDER = ImagekitFaceEmbedder()
