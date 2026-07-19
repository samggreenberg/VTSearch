"""Parametrised geometric verification models for structural matching.

Extends the 4-DoF similarity fit baked into :class:`SiftMatcher.verify` with a
**3-DoF scale+translation model** (isotropic scale + translation, no rotation)
— the natural geometry for screenshots and scanned documents, where a target
is a digital overlay that never rotates.  The 2026-07-13 screenshot-iconography
study (docs/reports) measured the 3-DoF model against production 4-DoF on
shared correspondences.

Both models emit the production :class:`MatchStats` bundle and share the
production thresholds and sanity gates, so a backend can swap geometry without
touching Stage 1, the verification classifier, or the UI.
"""

from __future__ import annotations

import numpy as np

from vtscore.media.structural import (
    _MAX_SANE_SCALE,
    _MIN_MODEL_INLIERS,
    _MIN_SANE_SCALE,
    _RANSAC_REPROJ_THRESHOLD,
    MatchStats,
)

MODELS = ("similarity", "scale_translation")
MIN_SAMPLE = {"similarity": 2, "scale_translation": 2}

_RNG_SEED = 42
_MAX_ITERS = 2000
_REFINE_ITERS = 10


def _ls_scale_translation(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Least-squares fit of ``dst = s * src + t`` (isotropic s, no rotation)."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - ms, dst - md
    denom = float((sc * sc).sum())
    if denom < 1e-12:
        return None
    s = float((sc * dc).sum() / denom)
    return s, md - s * ms


def _st_inliers(src: np.ndarray, dst: np.ndarray, s: float, t: np.ndarray, thr: float) -> np.ndarray:
    return np.linalg.norm(src * s + t - dst, axis=1) < thr


def fit_scale_translation(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    ransac_threshold: float = _RANSAC_REPROJ_THRESHOLD,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """RANSAC fit of the 3-DoF model; returns ``(2x3 model matrix, inlier mask)``.

    2-point minimal samples with vectorised hypothesis evaluation: when the
    correspondence set is small enough we enumerate ALL C(n, 2) minimal
    samples (exhaustive — strictly better than random sampling), otherwise a
    seeded random draw of ``_MAX_ITERS`` pairs.  Least-squares refinement on
    the winning inlier set.  Deterministic.
    """
    n = src.shape[0]
    if n < 2:
        return None, None
    src64 = src.astype(np.float64)
    dst64 = dst.astype(np.float64)
    if n * (n - 1) // 2 <= _MAX_ITERS:
        ii, jj = np.triu_indices(n, k=1)
    else:
        rng = np.random.RandomState(_RNG_SEED)
        ii = rng.randint(0, n, size=_MAX_ITERS)
        jj = rng.randint(0, n, size=_MAX_ITERS)
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
    ds = src64[jj] - src64[ii]
    dd = dst64[jj] - dst64[ii]
    denom = (ds * ds).sum(axis=1)
    ok = denom > 1e-12
    if not ok.any():
        return None, None
    ii, jj, ds, dd, denom = ii[ok], jj[ok], ds[ok], dd[ok], denom[ok]
    s_hyp = (ds * dd).sum(axis=1) / denom
    t_hyp = 0.5 * (dst64[ii] + dst64[jj]) - s_hyp[:, None] * 0.5 * (src64[ii] + src64[jj])
    proj = s_hyp[:, None, None] * src64[None, :, :] + t_hyp[:, None, :]
    err = np.linalg.norm(proj - dst64[None, :, :], axis=2)
    inl = err < ransac_threshold
    counts = inl.sum(axis=1)
    best = int(np.argmax(counts))
    if int(counts[best]) < 2:
        return None, None
    mask = inl[best]
    s_t = _ls_scale_translation(src64[mask], dst64[mask])
    for _ in range(_REFINE_ITERS):
        if s_t is None:
            break
        s, t = s_t
        new_mask = _st_inliers(src64, dst64, s, t, ransac_threshold)
        if new_mask.sum() < 2 or bool(np.array_equal(new_mask, mask)):
            mask = new_mask if new_mask.sum() >= 2 else mask
            break
        mask = new_mask
        s_t = _ls_scale_translation(src64[mask], dst64[mask])
    if s_t is None:
        return None, None
    s, t = s_t
    model = np.array([[s, 0.0, t[0]], [0.0, s, t[1]]], dtype=np.float64)
    return model, _st_inliers(src64, dst64, s, t, ransac_threshold)


def fit_model(
    src: np.ndarray,
    dst: np.ndarray,
    tentative: int,
    model_name: str,
    *,
    ransac_threshold: float = _RANSAC_REPROJ_THRESHOLD,
) -> MatchStats:
    """Fit *model_name* to correspondences; emit the production MatchStats.

    ``similarity`` reproduces SiftMatcher.verify's cv2.estimateAffinePartial2D
    fit exactly; ``scale_translation`` is the 3-DoF variant.  Both share the
    production reprojection threshold, inlier floor, and sane-scale window.
    """
    import cv2  # noqa: PLC0415

    if model_name not in MODELS:
        raise ValueError(f"unknown geometric model {model_name!r}; expected one of {MODELS}")
    if src.shape[0] < MIN_SAMPLE[model_name]:
        return MatchStats(tentative_count=tentative)

    if model_name == "similarity":
        model, inlier_mask = cv2.estimateAffinePartial2D(
            src.astype(np.float32),
            dst.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=_MAX_ITERS,
            confidence=0.99,
            refineIters=_REFINE_ITERS,
        )
        if model is None or inlier_mask is None or not np.isfinite(model).all():
            return MatchStats(tentative_count=tentative)
        mask = inlier_mask.ravel().astype(bool)
        a, b = float(model[0, 0]), float(model[1, 0])
        scale = float(np.hypot(a, b))
        reflection = bool(np.linalg.det(np.asarray(model[:, :2], dtype=np.float64)) < 0)
    elif model_name == "scale_translation":
        model, mask = fit_scale_translation(src, dst, ransac_threshold=ransac_threshold)
        if model is None or mask is None:
            return MatchStats(tentative_count=tentative)
        s = float(model[0, 0])
        scale = abs(s)
        # a negative isotropic scale is a 180-degree rotation, which the
        # no-rotation model must reject; flag it via the reflection gate
        reflection = s < 0

    inlier_count = int(mask.sum())
    model_ok = inlier_count >= _MIN_MODEL_INLIERS and _MIN_SANE_SCALE <= scale <= _MAX_SANE_SCALE and not reflection

    mean_err = median_err = spread = 0.0
    inlier_box = None
    if inlier_count:
        src_in, dst_in = src[mask].astype(np.float64), dst[mask].astype(np.float64)
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
        reflection=bool(reflection),
        inlier_spread=spread,
        model_ok=model_ok,
        inlier_box=inlier_box if model_ok else None,
    )
