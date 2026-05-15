"""Shared base class for the two EUPE embedder variants.

EUPE (Efficient Universal Perception Encoder, `facebookresearch/EUPE`)
is Meta's distillation-based "universal" ViT-B/16.  It's exposed as two
embedders that share the same backbone but differ in what they expose:

- ``eupe_single`` — CLS-pooled single vector per image; fast, small
  storage, no region search.
- ``eupe_patch``  — same CLS vector plus a per-patch grid + HAC region
  tree; ~30× slower per image and ~100× more storage, but enables region
  similarity and region-aware MLP scoring.

EUPE's weights mirror at ``facebook/EUPE-ViT-B`` are **ungated** (no HF
token needed), but the **outputs** are bound by Meta's FAIR Noncommercial
Research Licence — embeddings, datasets and detectors produced via EUPE
inherit that restriction.  Both variants surface that via
:attr:`license_notice` so the embedder picker shows a warning chip
before users opt in.

EUPE's attention path uses ``torch.nn.functional.scaled_dot_product_attention``
which **does not return weights**, so we don't have a real CLS→patch
attention map — :attr:`patch_saliency` falls back to a CLS-cosine-similarity
proxy (softmax of each patch's cosine similarity to the CLS vector).
See :func:`vtsearch.media.patch_embed.eupe_features_to_patch_output`
for the adapter.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from vtsearch.config import EUPE_MODEL_ID
from vtsearch.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    timed_progress,
)
from vtsearch.media.image._image_bulk import bulk_embed_image_files
from vtsearch.media.patch_embed import PatchEmbedOutput, eupe_features_to_patch_output


# ImageNet normalisation, matching EUPE's eval transform.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


LICENSE_NOTICE = (
    "EUPE is released under Meta's FAIR Noncommercial Research Licence. "
    "Embeddings, datasets, and detectors produced with EUPE inherit that "
    "restriction — they may only be used for noncommercial research. "
    "Pick a different image embedder if your use case is commercial."
)


class _EupeBase(MediaEmbedder):
    """Backbone loader + CLS / patch forward passes for EUPE.

    Subclasses set :attr:`name` and :attr:`supports_patch_regions`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._preprocess = None

    @property
    def media_type_id(self) -> str:
        return "image"

    @property
    def supports_text(self) -> bool:
        return False

    @property
    def license_notice(self) -> Optional[str]:
        return LICENSE_NOTICE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 3):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing torchvision…", 2, 3):
            from torchvision import transforms  # noqa: PLC0415

        with timed_progress(self._on_progress, "loading", "Importing torch.hub…", 3, 3):
            import torch.hub  # noqa: F401, PLC0415

        # Point torch.hub at our shared model cache so that pre-baked
        # Docker images can ship the cloned repo + downloaded weights
        # without re-fetching at first run.
        cache_dir = embedder_load_setup(self._on_progress, "Loading EUPE weights…")
        os.environ.setdefault("TORCH_HOME", str(Path(cache_dir).expanduser()))

        with intercept_tqdm_progress(self._on_progress):
            self._model = torch.hub.load(
                "facebookresearch/EUPE",
                "eupe_vitb16",
                source="github",
                pretrained=True,
                weights=EUPE_MODEL_ID,
                trust_repo=True,
            )
        self._model = self._model.to("cpu")
        self._model.eval()

        # EUPE doesn't ship a HF AutoImageProcessor, so build the
        # standard ImageNet eval transform inline.
        self._preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
            ]
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        features = self._forward_features(media)
        if features is None:
            return None
        cls = features["x_norm_clstoken"][0].detach().cpu().float().numpy()
        norm = float(np.linalg.norm(cls))
        if norm > 0:
            cls = cls / norm
        return cls.astype(np.float32)

    def _compute_patch_output(self, media: dict) -> Optional[PatchEmbedOutput]:
        features = self._forward_features(media)
        if features is None:
            return None
        return eupe_features_to_patch_output(features)

    def _forward_pil_batch(self, images: list) -> np.ndarray:
        """Run EUPE on a batch of PIL images, returning ``(N, D)`` CLS vectors."""
        features = self._forward_features_batch(images)
        if features is None:
            raise RuntimeError("EUPE bulk forward returned no features")
        cls = features["x_norm_clstoken"]  # (N, D)
        arr = cls.detach().cpu().float().numpy()
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        return (arr / np.maximum(norms, 1e-12)).astype(np.float32)

    def _patch_forward_pil_batch(self, images: list) -> list[Optional[PatchEmbedOutput]]:
        features = self._forward_features_batch(images)
        if features is None:
            raise RuntimeError("EUPE bulk forward returned no features")
        return [eupe_features_to_patch_output(features, batch_index=i) for i in range(len(images))]

    def _forward_features_batch(self, images: list):
        """Run ``model.forward_features`` on a list of PIL images.

        Returns the raw EUPE feature dict (with a leading batch dim) or
        ``None`` on any failure.
        """
        if self._model is None:
            self.load_models()
        if self._model is None or self._preprocess is None:
            return None
        try:
            import torch  # noqa: PLC0415

            tensors = [self._preprocess(im.convert("RGB")) for im in images]
            batch = torch.stack(tensors, dim=0)
            device = next(self._model.parameters()).device
            batch = batch.to(device)
            with torch.no_grad():
                features = self._model.forward_features(batch)
            return features
        except Exception:
            logging.getLogger(__name__).exception("Error running EUPE bulk forward")
            return None

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._preprocess is None:
            return [None] * len(medias)
        with self._embed_lock:
            return bulk_embed_image_files(
                medias,
                forward_pil_batch=self._forward_pil_batch,
                batch_size=self.embed_batch_size,
                on_progress=self._on_progress,
                label="EUPE",
            )

    def _forward_features(self, media: dict):
        """Run one ``model.forward_features`` pass on the media's image file.

        Returns the raw EUPE feature dict (CLS / storage / patch tokens)
        or ``None`` on any failure.  Both :meth:`_embed_media_impl` and
        the patch-region path go through here so the model is never
        evaluated twice for the same image.
        """
        if self._model is None:
            self.load_models()
        if self._model is None or self._preprocess is None:
            return None
        file_path = Path(media["media_path"])
        try:
            import torch  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415

            with Image.open(file_path) as _img:
                image = _img.convert("RGB")
            tensor = self._preprocess(image).unsqueeze(0)
            device = next(self._model.parameters()).device
            tensor = tensor.to(device)
            with torch.no_grad():
                features = self._model.forward_features(tensor)
            return features
        except Exception:
            logging.getLogger(__name__).exception("Error running EUPE on %s", file_path)
            return None
