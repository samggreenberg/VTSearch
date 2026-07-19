"""Structural (instance-matching) features and geometric verification.

This is the library-tier, **media-agnostic** core of the structural-embedder
pipeline (design: ``docs/plans/structural-embedder.md``).  It reconciles the
set-of-descriptors + geometric-fit world of instance retrieval with VTSearch's
fixed-vector + cosine world via the two-stage architecture:

* **Stage 1 (retrieval).** :func:`aggregate_vlad` folds an image's local
  descriptor set into a single fixed-D L2-normalised VLAD vector that *is* a
  metric-space embedding - it rides the existing ``media["embeddings"]``
  pipeline (diversity tree, cosine/example sort, ``train_model``) unchanged.

* **Stage 2 (verification).** A :class:`StructuralMatcher` matches the raw
  keypoints/descriptors of a template against a candidate and RANSAC-fits a
  **similarity transform** (4-DoF: translation, rotation, uniform scale - no
  shear, no perspective), emitting a :class:`MatchStats` bundle.
  :func:`match_stats_to_features` turns that bundle into a fixed-D vector so the
  verification classifier reuses ``train_model`` verbatim.

The :class:`StructuralMatcher` protocol and the data structures are kept free of
any image-specific assumption (keypoints are just ``(x, y, scale, orientation)``
rows in normalised coordinates) so an audio constellation-fingerprint backend -
the next media target - drops in without reshaping the interface.

No Flask, no app-tier imports: this module lives under ``vtscore`` and stays
import-clean.  Heavy optional deps (``cv2``) are imported lazily inside the
functions that need them so the dataclasses/protocol import without OpenCV.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------
# Constants (pinned by the pre-impl spike; see the design doc's open questions)
# --------------------------------------------------------------------------

SIFT_DESCRIPTOR_DIM = 128
"""Dimensionality of a classic SIFT descriptor."""

DEFAULT_VLAD_CENTROIDS = 64
"""Visual-vocabulary size for the shipped v1 codebook.

VLAD dimensionality is ``DEFAULT_VLAD_CENTROIDS * SIFT_DESCRIPTOR_DIM`` (8192 for
the v1 default).  Bigger = more discriminative but larger vectors; the spike on a
demo dataset pins the final size (the design doc floats 128/256 as candidates).
"""

DEFAULT_MAX_FEATURES = 1024
"""Cap on keypoints kept per image (top-M by SIFT response).

Bounds worst-case ``media["local_features"]`` size; a typical image yields ~1-2k
SIFT keypoints and we keep the strongest ``DEFAULT_MAX_FEATURES`` of them.
"""

DEFAULT_MIN_INLIERS = 8
"""Cold-start inlier-count threshold for ``MatchStats.is_match``.

Used as the default geometric-consistency gate before the verification
classifier has votes to train on (mirrors the safe-threshold fallback the
detector MLP uses below 6 labels).
"""

_CODEBOOK_ASSET = Path(__file__).resolve().parent / "assets" / "vlad_codebook_v1.npy"
"""Shipped, fixed VLAD vocabulary (a pre-computed k-means codebook).

A code asset, not per-dataset state: it is loaded like model weights and never
written to a detector JSON, ``settings.json``, or the dataset pickle.  Rebuild
with ``scripts/build_vlad_codebook.py``.
"""


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class StructuralFeatures:
    """Per-media local features for instance matching.

    Stored as ``media["local_features"]`` (dataset pickle + RAM only - never a
    detector JSON or ``settings.json``).  In the pickle the arrays are kept
    compact (``keypoints`` fp16, ``descriptors`` uint8 via :meth:`compact`);
    the matcher casts back to float32 when matching.
    """

    keypoints: np.ndarray
    """``(M, 4)`` array: ``x, y, scale, orientation``.

    ``x``/``y`` are **normalised** image coordinates in ``[0, 1]`` so the
    features are resolution-independent and the geometric model is fit in a
    common frame.  ``orientation`` is in radians.
    """

    descriptors: np.ndarray
    """``(M, d)`` descriptor matrix (``d == SIFT_DESCRIPTOR_DIM`` for SIFT)."""

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints)
        self.descriptors = np.asarray(self.descriptors)

    @property
    def count(self) -> int:
        """Number of keypoints/descriptors."""
        return int(self.keypoints.shape[0])

    def compact(self) -> "StructuralFeatures":
        """Return a storage-compact copy (fp16 keypoints, uint8 descriptors).

        SIFT descriptors are already integer-valued in ``[0, 255]``, so the
        uint8 cast is near-lossless; keypoint coordinates tolerate fp16.  This
        is the form written into the dataset pickle, mirroring patch
        ``to_fp16``.
        """
        kp = np.asarray(self.keypoints, dtype=np.float16)
        desc = np.clip(np.rint(np.asarray(self.descriptors, dtype=np.float32)), 0, 255).astype(np.uint8)
        return StructuralFeatures(keypoints=kp, descriptors=desc)

    def keypoints_f32(self) -> np.ndarray:
        """Keypoints cast to float32 (compact storage may be fp16)."""
        return np.asarray(self.keypoints, dtype=np.float32)

    def descriptors_f32(self) -> np.ndarray:
        """Descriptors cast to float32 (compact storage may be uint8)."""
        return np.asarray(self.descriptors, dtype=np.float32)


# Ordered list of the match-statistic feature names.  ``match_stats_to_features``
# emits these in order; the verification classifier trains on the resulting
# fixed-D vector.  Keep the order stable - it defines the feature layout.
MATCH_STAT_FEATURES: tuple[str, ...] = (
    "inlier_count",
    "inlier_ratio",
    "tentative_count",
    "mean_reproj_error",
    "median_reproj_error",
    "scale",
    "reflection",
    "inlier_spread",
    "model_ok",
)

MATCH_STAT_DIM = len(MATCH_STAT_FEATURES)
"""Dimensionality of the match-statistic feature vector."""


@dataclass
class MatchStats:
    """Statistics emitted by one geometric fit of a template against a candidate.

    These are the genuinely-structural learnable signal: stacked into a fixed-D
    vector by :func:`match_stats_to_features`, they train the verification
    classifier whose decision boundary *is* the calibrated match threshold.
    """

    inlier_count: int = 0
    """Number of correspondences consistent with the fitted similarity model."""

    inlier_ratio: float = 0.0
    """``inlier_count / tentative_count`` (0 when there are no tentative matches)."""

    tentative_count: int = 0
    """Number of pre-RANSAC descriptor matches (after the Lowe ratio test)."""

    mean_reproj_error: float = 0.0
    """Mean reprojection error of inliers, in normalised image units."""

    median_reproj_error: float = 0.0
    """Median reprojection error of inliers, in normalised image units."""

    scale: float = 1.0
    """Uniform scale of the fitted similarity transform (1.0 when no fit)."""

    reflection: bool = False
    """Whether the fit includes a reflection (a degenerate similarity)."""

    inlier_spread: float = 0.0
    """Spatial spread of the inliers in the candidate, in normalised units.

    The RMS distance of inlier locations from their centroid - a proxy for how
    much of the candidate image the matched region covers.
    """

    model_ok: bool = False
    """Whether a plausible (non-degenerate, sane-scale) model was fit at all."""

    inlier_box: Optional[tuple[float, float, float, float]] = None
    """Axis-aligned bounds ``(x0, y0, x1, y1)`` of the inliers in the candidate.

    Normalised coordinates; ``None`` when there is no usable fit.  Drives the
    matched-region overlay (reuses patch's best-region machinery).  Not part of
    the match-statistic feature vector.
    """

    def is_match(self, min_inliers: int = DEFAULT_MIN_INLIERS) -> bool:
        """Cold-start geometric gate before the verification classifier exists."""
        return self.model_ok and self.inlier_count >= min_inliers


def match_stats_to_features(stats: MatchStats) -> np.ndarray:
    """Stack a :class:`MatchStats` into the fixed-D verification feature vector.

    Counts are ``log1p``-compressed so a candidate with 500 inliers does not
    dwarf one with 30 in the linear classifier.  The layout follows
    :data:`MATCH_STAT_FEATURES`.
    """
    return np.array(
        [
            np.log1p(float(stats.inlier_count)),
            float(stats.inlier_ratio),
            np.log1p(float(stats.tentative_count)),
            float(stats.mean_reproj_error),
            float(stats.median_reproj_error),
            float(stats.scale),
            1.0 if stats.reflection else 0.0,
            float(stats.inlier_spread),
            1.0 if stats.model_ok else 0.0,
        ],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------
# Matcher protocol
# --------------------------------------------------------------------------


@runtime_checkable
class StructuralMatcher(Protocol):
    """Pluggable feature-detection + geometric-verification backend.

    The v1 implementation is :class:`SiftMatcher` (classic SIFT + RANSAC,
    CPU-only, ungated).  A learned-feature backend (SuperPoint + LightGlue) or
    an audio constellation-fingerprint backend slots in behind this same
    protocol without touching Stage 1, the classifier, or the UI.
    """

    def detect_and_describe(self, image_gray: np.ndarray, *, max_features: int = ...) -> StructuralFeatures:
        """Detect keypoints and compute descriptors for one grayscale image."""
        ...

    def verify(self, template: StructuralFeatures, candidate: StructuralFeatures) -> MatchStats:
        """Match *template* against *candidate* and geometrically verify the fit."""
        ...


# --------------------------------------------------------------------------
# VLAD aggregation (Stage 1)
# --------------------------------------------------------------------------


def rootsift(descriptors: np.ndarray) -> np.ndarray:
    """RootSIFT-normalise a descriptor matrix.

    L1-normalise each descriptor then take the element-wise square root.  This
    Hellinger-kernel trick (Arandjelovic & Zisserman) measurably improves SIFT
    matching/aggregation at zero cost, and is applied to both the descriptors
    and the codebook so they live in the same space.
    """
    d = np.asarray(descriptors, dtype=np.float32)
    if d.size == 0:
        return d.reshape(0, d.shape[-1] if d.ndim == 2 else 0)
    l1 = np.abs(d).sum(axis=1, keepdims=True)
    l1[l1 == 0] = 1.0
    return np.sqrt(d / l1)


def aggregate_vlad(descriptors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Fold a descriptor set into a single fixed-D L2-normalised VLAD vector.

    VLAD assigns each (rootSIFT-normalised) descriptor to its nearest codebook
    centroid, accumulates the residual ``descriptor - centroid`` per centroid,
    intra-normalises each centroid block (signed-square-root power law + L2),
    flattens, and L2-normalises the whole vector.  The output dimensionality is
    ``codebook.shape[0] * codebook.shape[1]``.

    An empty descriptor set yields the zero vector (a valid, if uninformative,
    embedding - ``score_against_query`` returns 0 for it).
    """
    cb = np.asarray(codebook, dtype=np.float32)
    k, d = cb.shape
    out_dim = k * d
    desc = np.asarray(descriptors, dtype=np.float32)
    if desc.size == 0:
        return np.zeros(out_dim, dtype=np.float32)
    if desc.ndim != 2 or desc.shape[1] != d:
        raise ValueError(f"descriptors must be (M, {d}); got {desc.shape}")

    desc = rootsift(desc)
    cb_root = rootsift(cb)

    # Nearest centroid per descriptor (squared-L2; the constant ||desc||^2 term
    # is shared across centroids so we can drop it and compare -2*desc·c + ||c||^2).
    cb_sq = (cb_root**2).sum(axis=1)  # (K,)
    sims = desc @ cb_root.T  # (M, K)
    dist = cb_sq[None, :] - 2.0 * sims  # (M, K), up to the shared ||desc||^2
    assign = np.argmin(dist, axis=1)  # (M,)

    vlad = np.zeros((k, d), dtype=np.float32)
    for j in range(k):
        members = desc[assign == j]
        if members.shape[0]:
            vlad[j] = (members - cb_root[j]).sum(axis=0)

    # Intra-normalisation: power-law (signed sqrt) per block then global L2.
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    flat = vlad.reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm > 0:
        flat = flat / norm
    return flat.astype(np.float32)


@functools.lru_cache(maxsize=1)
def load_vlad_codebook() -> np.ndarray:
    """Load and cache the shipped fixed VLAD vocabulary, shape ``(K, d)`` float32.

    The codebook is a code asset (like model weights), not per-dataset state:
    it introduces no new persisted artifact and no ingest-time fit pass.
    """
    if not _CODEBOOK_ASSET.exists():
        raise FileNotFoundError(
            f"VLAD codebook asset missing: {_CODEBOOK_ASSET}. Build it with `python scripts/build_vlad_codebook.py`."
        )
    return np.load(_CODEBOOK_ASSET).astype(np.float32)


# --------------------------------------------------------------------------
# SIFT matcher (v1 backend)
# --------------------------------------------------------------------------

# Lowe's ratio-test threshold for accepting a tentative descriptor match.
_LOWE_RATIO = 0.75
# RANSAC reprojection tolerance, in normalised image units.
_RANSAC_REPROJ_THRESHOLD = 0.02
# Plausible-scale window for the fitted similarity model.
_MIN_SANE_SCALE = 0.1
_MAX_SANE_SCALE = 10.0
# Minimum inlier support for a model to count as plausible.  A similarity
# transform is fit from a 2-point minimal sample, so a 2-inlier fit has *zero*
# redundancy and is trivially consistent; requiring more guards against random
# descriptor coincidences between unrelated images declaring a spurious match.
_MIN_MODEL_INLIERS = 4


class SiftMatcher:
    """Classic SIFT + similarity-transform RANSAC, the v1 structural backend.

    CPU-friendly, deterministic enough for clean planar logos, patent-expired
    (SIFT is in mainline OpenCV since 4.4), and light on dependencies.  Conforms
    to :class:`StructuralMatcher`.
    """

    def __init__(self, *, ransac_threshold: float = _RANSAC_REPROJ_THRESHOLD) -> None:
        self._ransac_threshold = float(ransac_threshold)
        self._sift = None  # lazily created cv2.SIFT instance

    def _get_sift(self, max_features: int):
        import cv2  # noqa: PLC0415

        # SIFT's nfeatures cap is baked into the detector instance, so make a
        # fresh one whenever the cap changes (cheap; just a config object).
        if self._sift is None or getattr(self, "_sift_nfeatures", None) != max_features:
            # SIFT_create is missing from the cv2 type stubs (present at runtime
            # since OpenCV 4.4, when SIFT moved into the mainline module).
            self._sift = cv2.SIFT_create(nfeatures=int(max_features))  # pyright: ignore[reportAttributeAccessIssue]
            self._sift_nfeatures = int(max_features)
        return self._sift

    def detect_and_describe(
        self, image_gray: np.ndarray, *, max_features: int = DEFAULT_MAX_FEATURES
    ) -> StructuralFeatures:
        """Detect SIFT keypoints + descriptors; return them in normalised coords.

        *image_gray* is an ``(H, W)`` uint8 grayscale array.  Keypoint ``x``/``y``
        are normalised by image width/height so downstream geometry is
        resolution-independent.
        """
        gray = np.asarray(image_gray)
        if gray.ndim != 2:
            raise ValueError(f"image_gray must be 2-D (H, W); got shape {gray.shape}")
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        h, w = gray.shape
        sift = self._get_sift(max_features)
        kps, desc = sift.detectAndCompute(gray, None)
        if not kps or desc is None or len(kps) == 0:
            return StructuralFeatures(
                keypoints=np.zeros((0, 4), dtype=np.float32),
                descriptors=np.zeros((0, SIFT_DESCRIPTOR_DIM), dtype=np.float32),
            )
        inv_w = 1.0 / max(w, 1)
        inv_h = 1.0 / max(h, 1)
        kp_arr = np.array(
            [(kp.pt[0] * inv_w, kp.pt[1] * inv_h, kp.size, np.deg2rad(kp.angle)) for kp in kps],
            dtype=np.float32,
        )
        return StructuralFeatures(keypoints=kp_arr, descriptors=np.asarray(desc, dtype=np.float32))

    def verify(self, template: StructuralFeatures, candidate: StructuralFeatures) -> MatchStats:
        """Match *template* keypoints against *candidate* and fit a similarity transform.

        Returns a :class:`MatchStats` describing the geometric consistency of
        the best similarity fit.  A degenerate input (too few matches, no model)
        returns a zero-inlier, ``model_ok=False`` result.
        """
        import cv2  # noqa: PLC0415

        t_desc = template.descriptors_f32()
        c_desc = candidate.descriptors_f32()
        # knnMatch needs at least 2 candidate descriptors; a fit needs >= 2
        # correspondences (a similarity transform has 2 DoF pairs).
        if t_desc.shape[0] < 2 or c_desc.shape[0] < 2:
            return MatchStats()

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn = matcher.knnMatch(t_desc, c_desc, k=2)
        good: list = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < _LOWE_RATIO * n.distance:
                good.append(m)

        tentative = len(good)
        if tentative < 2:
            return MatchStats(tentative_count=tentative)

        t_kp = template.keypoints_f32()
        c_kp = candidate.keypoints_f32()
        src = np.array([t_kp[m.queryIdx, :2] for m in good], dtype=np.float32)
        dst = np.array([c_kp[m.trainIdx, :2] for m in good], dtype=np.float32)

        # estimateAffinePartial2D fits exactly a 4-DoF similarity (translation,
        # rotation, uniform scale - no shear, no anisotropic scale), which is the
        # geometric model the design locks in.  It needs only 2 correspondences.
        model, inlier_mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._ransac_threshold,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if model is None or inlier_mask is None or not np.isfinite(model).all():
            # RANSAC can return a non-finite (degenerate) model; treat it as no
            # fit rather than letting NaNs flow into the scale/determinant maths
            # (which raises numpy "invalid value" warnings and yields garbage stats).
            return MatchStats(tentative_count=tentative)

        mask = inlier_mask.ravel().astype(bool)
        inlier_count = int(mask.sum())
        a, b = float(model[0, 0]), float(model[1, 0])
        scale = float(np.hypot(a, b))
        # estimateAffinePartial2D never introduces a reflection (det = a^2+b^2 >= 0),
        # but keep the determinant-sign check so a future full-affine backend that
        # reuses this code path is covered.
        reflection = bool(np.linalg.det(np.asarray(model[:, :2], dtype=np.float64)) < 0)
        model_ok = inlier_count >= _MIN_MODEL_INLIERS and _MIN_SANE_SCALE <= scale <= _MAX_SANE_SCALE and not reflection

        # Reprojection error + spatial spread over the inlier set.
        mean_err = median_err = 0.0
        spread = 0.0
        inlier_box: Optional[tuple[float, float, float, float]] = None
        if inlier_count:
            src_in = src[mask]
            dst_in = dst[mask]
            proj = (src_in @ model[:, :2].T) + model[:, 2]
            errs = np.linalg.norm(proj - dst_in, axis=1)
            mean_err = float(errs.mean())
            median_err = float(np.median(errs))
            centroid = dst_in.mean(axis=0)
            spread = float(np.sqrt(((dst_in - centroid) ** 2).sum(axis=1).mean()))
            x0, y0 = dst_in.min(axis=0)
            x1, y1 = dst_in.max(axis=0)
            inlier_box = (float(x0), float(y0), float(x1), float(y1))

        return MatchStats(
            inlier_count=inlier_count,
            inlier_ratio=inlier_count / tentative if tentative else 0.0,
            tentative_count=tentative,
            mean_reproj_error=mean_err,
            median_reproj_error=median_err,
            scale=scale,
            reflection=reflection,
            inlier_spread=spread,
            model_ok=model_ok,
            inlier_box=inlier_box if model_ok else None,
        )
