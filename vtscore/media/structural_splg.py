"""SuperPoint + LightGlue structural backend (StructuralMatcher-conformant).

The learned-feature backend the structural-embedder design doc reserves: same
two-stage architecture, but Stage-2 keypoints/descriptors come from SuperPoint
and correspondences from LightGlue instead of SIFT + Lowe-ratio matching.
Evaluated against the SIFT backend in the 2026-07-13 screenshot-iconography
study (docs/experiments).

Integration notes
-----------------
* ``detect_and_describe`` returns keypoints in the production normalised
  convention ``(x/W, y/H, detection_score, 0.0)`` and float 256-d SuperPoint
  descriptors.  **Caveat:** :meth:`StructuralFeatures.compact` casts
  descriptors to uint8, which is near-lossless for integer-valued SIFT but
  destroys unit-scale float descriptors — persist these as fp16 (a
  ``compact_fp16()`` variant or a dtype flag) before wiring this backend into
  the ingest path.
* ``verify`` needs only StructuralFeatures (no image), like the SIFT path:
  LightGlue's positional encoding is fed the stored normalised coordinates
  with a nominal unit image size, so every image lives in the same [-1, 1]
  frame.  Non-square aspect anisotropy is absorbed the same way the SIFT
  path absorbs it (both axes normalised independently).
* ``geometry_model`` picks the RANSAC fit: ``similarity`` (production 4-DoF)
  or ``scale_translation`` (3-DoF, no rotation — screenshots/documents; see
  vtscore.media.structural_geometry).
* Extraction resizes the long side to ``extract_long_side`` (default 1536)
  before SuperPoint; normalised coordinates make this transparent.
* Weights: first use downloads SuperPoint (~5 MB) + LightGlue (~45 MB)
  checkpoints via torch.hub into ``TORCH_HOME``.
* Extra deps (not in the default install):
  ``pip install --no-deps git+https://github.com/cvg/LightGlue.git`` plus
  ``kornia`` (lightglue's ALIKED module imports it at package import time).
  GPU strongly recommended for bulk matching; CPU works for single pairs.
"""

from __future__ import annotations

import numpy as np

from vtscore.media.structural import MatchStats, StructuralFeatures
from vtscore.media.structural_geometry import fit_model

DEFAULT_MAX_FEATURES = 2048
DEFAULT_LONG_SIDE = 1536
_LG_FILTER_THRESHOLD = 0.1  # LightGlue's default match-confidence cutoff


class SplgMatcher:
    """SuperPoint + LightGlue + RANSAC geometric verification."""

    def __init__(
        self,
        geometry_model: str = "similarity",
        *,
        device: str | None = None,
        extract_long_side: int = DEFAULT_LONG_SIDE,
    ) -> None:
        self.geometry_model = geometry_model
        self._device = device
        self._extractor = None
        self._extractor_cap: int | None = None
        self._matcher = None
        self._long_side = int(extract_long_side)

    @property
    def device(self) -> str:
        if self._device is None:
            import torch  # noqa: PLC0415

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _get_extractor(self, max_features: int):
        from lightglue import SuperPoint  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        if self._extractor is None or self._extractor_cap != max_features:
            self._extractor = SuperPoint(max_num_keypoints=int(max_features)).eval().to(self.device)
            self._extractor_cap = int(max_features)
        return self._extractor

    def _get_matcher(self):
        from lightglue import LightGlue  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        if self._matcher is None:
            self._matcher = (
                LightGlue(features="superpoint", filter_threshold=_LG_FILTER_THRESHOLD).eval().to(self.device)
            )
        return self._matcher

    def detect_and_describe(
        self, image_gray: np.ndarray, *, max_features: int = DEFAULT_MAX_FEATURES
    ) -> StructuralFeatures:
        """SuperPoint keypoints + descriptors in normalised coordinates."""
        import torch  # noqa: PLC0415

        gray = np.asarray(image_gray)
        if gray.ndim != 2:
            raise ValueError(f"image_gray must be 2-D (H, W); got shape {gray.shape}")
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        h, w = gray.shape
        scale = min(1.0, self._long_side / max(h, w))
        if scale < 1.0:
            import cv2  # noqa: PLC0415

            gray = cv2.resize(
                gray,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rh, rw = gray.shape
        img = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].to(self.device)
        extractor = self._get_extractor(max_features)
        with torch.no_grad():
            out = extractor({"image": img})
        kpts = out["keypoints"][0].cpu().numpy()
        scores = out["keypoint_scores"][0].cpu().numpy()
        desc = out["descriptors"][0].cpu().numpy()
        if kpts.shape[0] == 0:
            return StructuralFeatures(
                keypoints=np.zeros((0, 4), dtype=np.float32),
                descriptors=np.zeros((0, 256), dtype=np.float32),
            )
        kp_arr = np.stack(
            [kpts[:, 0] / max(rw, 1), kpts[:, 1] / max(rh, 1), scores, np.zeros(len(kpts))],
            axis=1,
        ).astype(np.float32)
        return StructuralFeatures(keypoints=kp_arr, descriptors=desc.astype(np.float32))

    def match(self, template: StructuralFeatures, candidate: StructuralFeatures) -> tuple[np.ndarray, np.ndarray, int]:
        """LightGlue correspondences ``(src, dst, tentative_count)``.

        The tentative count (matches above LightGlue's confidence filter) is
        the analogue of the SIFT path's Lowe-ratio survivor count.
        """
        import torch  # noqa: PLC0415

        t_kp = template.keypoints_f32()
        c_kp = candidate.keypoints_f32()
        if t_kp.shape[0] < 2 or c_kp.shape[0] < 2:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), 0
        dev = self.device
        unit = torch.tensor([[1.0, 1.0]], device=dev)
        data = {
            "image0": {
                "keypoints": torch.from_numpy(t_kp[:, :2]).to(dev)[None],
                "descriptors": torch.from_numpy(template.descriptors_f32()).to(dev)[None],
                "image_size": unit,
            },
            "image1": {
                "keypoints": torch.from_numpy(c_kp[:, :2]).to(dev)[None],
                "descriptors": torch.from_numpy(candidate.descriptors_f32()).to(dev)[None],
                "image_size": unit,
            },
        }
        matcher = self._get_matcher()
        with torch.no_grad():
            out = matcher(data)
        matches = out["matches"][0].cpu().numpy()
        if matches.size == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), 0
        src = t_kp[matches[:, 0], :2].astype(np.float32)
        dst = c_kp[matches[:, 1], :2].astype(np.float32)
        return src, dst, int(matches.shape[0])

    def verify(self, template: StructuralFeatures, candidate: StructuralFeatures) -> MatchStats:
        """Match with LightGlue and RANSAC-fit the configured geometric model."""
        src, dst, tentative = self.match(template, candidate)
        return fit_model(src, dst, tentative, self.geometry_model)
