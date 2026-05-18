"""Image embedder — FaceNet face-identity space (facenet-pytorch).

Embeds (cropped) face images into a 512-dim identity space using
``facenet-pytorch``'s ``InceptionResnetV1`` with VGGFace2 weights. The
output vector lives in **face-identity space**, not generic content
space — two photos of the same person score close together regardless
of pose / lighting / clothing, while two photos of different people in
similar scenes score far apart.

Typically paired with
:class:`~vtsearch.media.image.clipper.ImageFaceClipper` so that each
input image is first split into one crop per detected face. The
embedder also works on un-clipped images (the model just sees a 160×160
resize without explicit detection), but results are much better when
the input is already a face-centred crop.

``facenet-pytorch`` is an opt-in dependency (declared in pyproject's
``DEP001`` ignore-list, same pattern as ``mediapipe``). If it is not
installed, :meth:`load_models` raises an :class:`ImportError` that the
:class:`MediaEmbedder` base class re-wraps with an install hint.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtsearch.media.embedder import MediaEmbedder, timed_progress
from vtsearch.media.image._image_bulk import bulk_embed_image_files

if TYPE_CHECKING:
    from PIL import Image


_INPUT_SIZE = 160  # facenet-pytorch's InceptionResnetV1 input resolution.


class ImageFaceEmbedder(MediaEmbedder):
    """Embeds face images using FaceNet (InceptionResnetV1, VGGFace2)."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    @property
    def name(self) -> str:
        return "face"

    @property
    def display_name(self) -> str:
        return "FaceNet (face identity, 512d)"

    @property
    def media_type_id(self) -> str:
        return "image"

    @property
    def is_default(self) -> bool:
        return False

    @property
    def supports_text(self) -> bool:
        return False

    @property
    def supports_patch_regions(self) -> bool:
        return False

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return
        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 2):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Loading FaceNet weights…", 2, 2):
            # facenet-pytorch lazy-downloads weights on first instantiation.
            from facenet_pytorch import InceptionResnetV1  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

            model = InceptionResnetV1(pretrained="vggface2")
            model.eval()
            self._model = model.to("cpu")

    def _preprocess(self, image: "Image.Image") -> np.ndarray:
        """Resize to 160×160 RGB and normalise to ``(x − 127.5) / 128``.

        Returns a ``(3, 160, 160)`` ``float32`` array — caller stacks into
        a batch dimension before passing to the model.
        """
        rgb = image.convert("RGB").resize((_INPUT_SIZE, _INPUT_SIZE))
        arr = np.asarray(rgb, dtype=np.float32)
        arr = (arr - 127.5) / 128.0
        return arr.transpose(2, 0, 1)

    def _l2_normalise(self, out: np.ndarray) -> np.ndarray:
        if out.ndim == 1:
            norm = float(np.linalg.norm(out))
            return out if norm == 0.0 else (out / norm)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return out / norms

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            import torch  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415

            blob = media.get("media_bytes")
            if isinstance(blob, (bytes, bytearray)) and blob:
                with Image.open(io.BytesIO(bytes(blob))) as _img:
                    image = _img.convert("RGB")
            else:
                file_path = Path(media["media_path"])
                with Image.open(file_path) as _img:
                    image = _img.convert("RGB")

            arr = self._preprocess(image)
            tensor = torch.from_numpy(arr).unsqueeze(0)
            device = next(self._model.parameters()).device
            tensor = tensor.to(device)
            with torch.no_grad():
                out = self._model(tensor).detach().cpu().numpy()[0]
            return self._l2_normalise(out).astype(np.float32)
        except Exception:
            logging.getLogger(__name__).exception("Error embedding face image")
            return None

    def _forward_pil_batch(self, images: list["Image.Image"]) -> np.ndarray:
        import torch  # noqa: PLC0415

        stacked = np.stack([self._preprocess(im) for im in images], axis=0)
        tensor = torch.from_numpy(stacked)
        device = next(self._model.parameters()).device
        tensor = tensor.to(device)
        with torch.no_grad():
            out = self._model(tensor).detach().cpu().numpy()
        return self._l2_normalise(out).astype(np.float32)

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return [None] * len(medias)
        with self._embed_lock:
            return bulk_embed_image_files(
                medias,
                forward_pil_batch=self._forward_pil_batch,
                batch_size=self.embed_batch_size,
                on_progress=self._on_progress,
                label="FaceNet",
            )


EMBEDDER = ImageFaceEmbedder()
