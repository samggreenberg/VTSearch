"""Face media type - face crops localised out of images (no native import).

``face`` is the mirror image of ``document`` in the half-media-type model:

* ``document`` is a **convert-out** half type — importable but *not*
  embeddable; a PDF must be converted to image/text before it can be searched.
* ``face`` is a **convert-in** half type — embeddable but *not* importable; a
  face never arrives from a ``.face`` file, it is cropped out of an image by
  the ``image2face`` converter
  (:class:`~vtscore.converters.image2face.Image2FaceMediaConverter`) and then
  embedded in FaceNet identity space by
  :class:`~vtscore.media.face.embedder_facenet.FaceEmbedder`.

Because a face crop *is* a small image, this type serves and thumbnails its
media exactly like the image type; it just declares no importable file
extensions, so it never appears as a folder-import / demo ingestion category.
"""

from __future__ import annotations

from pathlib import Path

from vtscore.media.base import (
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
)

_FACE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class FaceMediaType(MediaType):
    """Handles face crops - HTTP serving, thumbnails, and identity embedding.

    Embedding is handled by
    :class:`~vtscore.media.face.embedder_facenet.FaceEmbedder`.
    """

    #: A face crop is a small still image: a browsable-thumbnail type.
    has_thumbnail = True

    def __init__(self) -> None:
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "face"

    @property
    def name(self) -> str:
        return "Face"

    @property
    def icon(self) -> str:
        return "face"

    @property
    def importable(self) -> bool:
        # A convert-in half type: faces only ever arise from converting an
        # image (image2face), never from a native import, so this type is
        # hidden from every import surface.
        return False

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        # No native import: faces are produced by the image2face converter.
        return []

    @property
    def dir_key(self) -> str:
        return "face_dir"

    @property
    def pickle_extra_fields(self) -> list[str]:
        return ["width", "height", "thumbnail_bytes"]

    # ------------------------------------------------------------------
    # Display metadata
    # ------------------------------------------------------------------

    def display_metadata(self, media: dict) -> dict:
        result: dict = {}
        w, h = media.get("width"), media.get("height")
        if w and h:
            result["Dimensions"] = f"{w}×{h}"
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        result.update({k: v for k, v in super().display_metadata(media).items() if k not in result})
        return result

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    @property
    def demo_datasets(self) -> list:
        return []

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_media_data(self, file_path: Path, media_bytes: bytes | None = None) -> dict:
        # Faces are not imported from files in the normal flow, but implement
        # this defensively (image-like) so a face crop loaded from disk still
        # gets its dimensions + thumbnail populated.
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        from vtscore.media.image.thumbnail import make_image_thumbnail  # noqa: PLC0415

        if media_bytes is None:
            with open(file_path, "rb") as f:
                media_bytes = f.read()
        try:
            with Image.open(io.BytesIO(media_bytes)) as img:
                width, height = img.width, img.height
        except Exception:
            width, height = None, None
        thumb = make_image_thumbnail(media_bytes)
        return {
            "media_bytes": media_bytes,
            "duration": 0,
            "width": width,
            "height": height,
            "thumbnail_bytes": thumb[0] if thumb is not None else None,
        }

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def media_response(self, media: dict) -> MediaResponse:
        filename = media.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".png"
        mimetype = _FACE_MIME_TYPES.get(ext, "image/png")
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype=mimetype, download_name=f"media_{media['id']}{ext}")
        return MediaResponse(
            data=data,
            mimetype=mimetype,
            download_name=f"media_{media['id']}{ext}",
        )

    def image_response(self, media: dict) -> MediaResponse | None:
        """Return the face crop as an image so the ``/image`` and
        ``/thumbnail`` routes (which delegate for every ``media_type != image``)
        paint the crop instead of aborting."""
        resp = self.media_response(media)
        if not resp.data:
            return None
        return resp
