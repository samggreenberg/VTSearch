"""Shared base for image structural embedders (SIFT/VLAD and learned variants).

A structural embedder produces *two* per-image artifacts (design:
``docs/plans/structural-embedder.md``):

* a fixed-D **VLAD vector** that rides the existing single-vector pipeline as
  ``media["embedding"]`` (Stage 1 retrieval), and
* a set of **local features** (keypoints + descriptors) stored as
  ``media["local_features"]`` for the geometric re-rank + match-stat classifier
  (Stage 2 verification).

This base owns everything that is independent of the matching backend - PIL
decode → grayscale, VLAD aggregation against the shipped codebook, and the bulk
plumbing - so a concrete embedder only supplies a
:class:`~vtscore.media.structural.StructuralMatcher` via :meth:`_make_matcher`.
A future ``embedder_superpoint_lightglue`` reuses this verbatim and swaps only
the matcher.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtscore.media.embedder import MediaEmbedder
from vtscore.media.image._image_bulk import (
    _load_pil,
    _pil_source_for,
    bulk_embed_image_files,
    bulk_patch_forward_image_files,
)
from vtscore.media.structural import (
    DEFAULT_MAX_FEATURES,
    StructuralFeatures,
    StructuralMatcher,
    aggregate_vlad,
    load_vlad_codebook,
)

if TYPE_CHECKING:
    from PIL import Image

_log = logging.getLogger(__name__)


class _StructuralImageBase(MediaEmbedder):
    """Backbone-agnostic base for image structural embedders.

    Subclasses override :meth:`_make_matcher` (and the ``name`` /
    ``display_name`` identity properties).  ``max_features`` caps keypoints per
    image (top-M by response) to bound ``media["local_features"]`` size.
    """

    def __init__(self) -> None:
        super().__init__()
        self._matcher: Optional[StructuralMatcher] = None
        self._codebook: Optional[np.ndarray] = None

    # --- identity / capabilities -------------------------------------------

    @property
    def media_type_id(self) -> str:
        return "image"

    @property
    def supports_text(self) -> bool:
        # No text encoder maps into VLAD space; the text sort greys out via the
        # already-shipped supports_text gate.
        return False

    @property
    def is_default(self) -> bool:
        # A specialist instance-matching tool, never a media-type default.
        return False

    @property
    def supports_geometric_verification(self) -> bool:
        return True

    @property
    def max_features(self) -> int:
        return DEFAULT_MAX_FEATURES

    # --- backend hook ------------------------------------------------------

    def _make_matcher(self) -> StructuralMatcher:
        """Return the matching backend for this embedder.  Subclasses override."""
        raise NotImplementedError

    # --- model lifecycle ---------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._matcher is not None:
            return
        self._on_progress("loading", "Loading structural matcher…", 0, 0)
        self._matcher = self._make_matcher()
        self._codebook = load_vlad_codebook()
        # Sentinel so MediaEmbedder.load_models() treats this embedder as loaded
        # (it checks ``self._model is not None``); there is no torch module here.
        self._model = self._matcher

    # --- decode helper -----------------------------------------------------

    def _gray_for(self, media: dict) -> Optional[np.ndarray]:
        source = _pil_source_for(media)
        if source is None:
            return None
        img = _load_pil(source)
        if img is None:
            return None
        return self._pil_to_gray(img)

    @staticmethod
    def _pil_to_gray(image: "Image.Image") -> np.ndarray:
        return np.asarray(image.convert("L"), dtype=np.uint8)

    # --- Stage 1: VLAD embedding -------------------------------------------

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._matcher is None or self._codebook is None:
            self.load_models()
        if self._matcher is None or self._codebook is None:
            return None
        gray = self._gray_for(media)
        if gray is None:
            return None
        feat = self._matcher.detect_and_describe(gray, max_features=self.max_features)
        return aggregate_vlad(feat.descriptors, self._codebook)

    def _forward_vlad_batch(self, images: list["Image.Image"]) -> np.ndarray:
        assert self._matcher is not None and self._codebook is not None
        vecs = [
            aggregate_vlad(
                self._matcher.detect_and_describe(self._pil_to_gray(im), max_features=self.max_features).descriptors,
                self._codebook,
            )
            for im in images
        ]
        return np.stack(vecs, axis=0)

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        if self._matcher is None or self._codebook is None:
            self.load_models()
        if self._matcher is None or self._codebook is None:
            return [None] * len(medias)
        with self._embed_lock:
            return bulk_embed_image_files(
                medias,
                forward_pil_batch=self._forward_vlad_batch,
                batch_size=self.embed_batch_size,
                on_progress=self._on_progress,
                label=self.display_name,
            )

    # --- Stage 2: local features -------------------------------------------

    def _local_features_forward_impl(self, media: dict) -> Optional[StructuralFeatures]:
        if self._matcher is None:
            self.load_models()
        if self._matcher is None:
            return None
        gray = self._gray_for(media)
        if gray is None:
            return None
        return self._matcher.detect_and_describe(gray, max_features=self.max_features)

    def _detect_batch(self, images: list["Image.Image"]) -> list[StructuralFeatures]:
        assert self._matcher is not None
        return [
            self._matcher.detect_and_describe(self._pil_to_gray(im), max_features=self.max_features) for im in images
        ]

    def _local_features_forward_bulk_impl(self, medias: list[dict]) -> list[Optional[StructuralFeatures]]:
        if self._matcher is None:
            self.load_models()
        if self._matcher is None:
            return [None] * len(medias)
        with self._embed_lock:
            return bulk_patch_forward_image_files(
                medias,
                forward_pil_batch=self._detect_batch,
                batch_size=self.embed_batch_size,
                on_progress=self._on_progress,
                label=f"{self.display_name} features",
            )
