"""Image embedder — EUPE / Efficient Universal Perception Encoder.

EUPE (`facebookresearch/EUPE`) is Meta's distillation-based "universal"
vision encoder: a ViT-B/16 backbone trained to balance multiple downstream
tasks (classification, segmentation, depth, vision-language).  We expose
the ViT-B/16 LVD-1689M variant — 768-dim outputs, 14 × 14 patch grid at
224 ², four "storage" / register tokens between CLS and patches.

This embedder is not the same as ``facebook/PE-Core-B16-224`` (Meta's
Perception Encoder Core) — that was the previous, misleading "eupe" slug,
which never actually loaded because PE-Core's HF repo has no ``config.json``
and the dev embedder relied on ``AutoModel.from_pretrained``.  This file is
a complete rewrite onto the real EUPE model the slug claims to be.

Loading goes through :func:`torch.hub.load`:

    model = torch.hub.load(
        "facebookresearch/EUPE", "eupe_vitb16",
        source="github",
        pretrained=True,
        weights=EUPE_MODEL_ID,   # HF mirror URL, see vtsearch/config.py
    )

The weights URL on `facebook/EUPE-ViT-B` is ungated, so users don't need an
HF token to fetch them — but the **outputs** are bound by the FAIR
Noncommercial Research Licence (no commercial use of any embedding /
dataset / detector produced via EUPE).  We surface that via
:attr:`license_notice` so the embedder picker shows a warning chip
before users opt in.

Image preprocessing is bog-standard ImageNet (resize-256-bicubic →
center-crop-224 → ImageNet mean/std normalisation), matching the
``make_classification_eval_transform`` recipe in EUPE's own training code.

Per-patch features and the patch_grid come straight from
``model.forward_features(x)``.  EUPE's attention path uses
``torch.nn.functional.scaled_dot_product_attention`` which **does not
return weights**, so we don't have a real CLS→patch attention map —
:attr:`patch_saliency` falls back to a CLS-cosine-similarity proxy
(softmax of each patch's cosine similarity to the CLS vector).  See
:func:`vtsearch.models.patch_regions.eupe_features_to_patch_output` for the
adapter.

There is no text encoder, so :attr:`supports_text` is ``False`` and the UI
greys text-search affordances for datasets embedded with EUPE.
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
from vtsearch.models.patch_regions import PatchEmbedOutput, eupe_features_to_patch_output


# ImageNet normalisation, matching EUPE's eval transform.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


_LICENSE_NOTICE = (
    "EUPE is released under Meta's FAIR Noncommercial Research Licence. "
    "Embeddings, datasets, and detectors produced with EUPE inherit that "
    "restriction — they may only be used for noncommercial research. "
    "Pick a different image embedder if your use case is commercial."
)


class ImageEupeEmbedder(MediaEmbedder):
    """Embeds images using facebookresearch/EUPE ViT-B/16.

    Output dimension: 768.  Patch grid: 14 × 14 at 224 ² input.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._preprocess = None

    @property
    def name(self) -> str:
        return "eupe"

    @property
    def media_type_id(self) -> str:
        return "image"

    @property
    def supports_text(self) -> bool:
        return False

    @property
    def supports_patch_regions(self) -> bool:
        return True

    @property
    def license_notice(self) -> Optional[str]:
        return _LICENSE_NOTICE

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
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
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

    def _patch_forward_impl(self, media: dict) -> Optional[PatchEmbedOutput]:
        features = self._forward_features(media)
        if features is None:
            return None
        return eupe_features_to_patch_output(features)

    def _forward_features(self, media: dict):
        """Run one ``model.forward_features`` pass on the media's image file.

        Returns the raw EUPE feature dict (CLS / storage / patch tokens)
        or ``None`` on any failure.  Both :meth:`_embed_media_impl` and
        :meth:`_patch_forward_impl` go through here so the model is never
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

            image = Image.open(file_path).convert("RGB")
            tensor = self._preprocess(image).unsqueeze(0)
            device = next(self._model.parameters()).device
            tensor = tensor.to(device)
            with torch.no_grad():
                features = self._model.forward_features(tensor)
            return features
        except Exception:
            logging.getLogger(__name__).exception("Error running EUPE on %s", file_path)
            return None


EMBEDDER = ImageEupeEmbedder()
