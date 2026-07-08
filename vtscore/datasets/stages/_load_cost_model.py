"""Affine per-phase load-cost coefficients, fit from measured GRID timings.

The dataset-load progress bar paces itself with a per-phase weight vector
(download, model, embed, finalize). A single static vector can't be right at
both small and large ``n``, because the phases scale with ``n`` differently:
model-load is a fixed cost, embed and finalize grow with ``n``, and download
tracks archive bytes. This module holds the measured affine coefficients so
:func:`vtscore.datasets.stages._common.load_step_weights` can compute an
``n``-aware weight vector.

    T_download ≈ download_size_mb / DOWNLOAD_MB_PER_S
    T_model    ≈ a_model                         (warm steady-state; cold noted in the plan)
    T_embed    ≈ a_embed + b_embed · n
    T_finalize ≈ a_fin   + b_fin   · n
    weight_phase = T_phase / Σ T_phase

The numbers below are produced by ``scripts/profiling/fit_load_weights.py`` from
the calibration harness (``scripts/profiling/calibrate_load_weights.py``); see
``docs/plans/progress-weight-calibration.md`` (Results) for the runs and fit
diagnostics. Re-run that harness to refresh — do not hand-tune. Cells with no
measured row fall back to the static per-device/media profiles in ``_common``.
"""

from __future__ import annotations

from typing import Optional

# key: (device, media_type, embedder) -> affine coefficients (seconds).
# ``device`` is normalized to "cuda" / "cpu"; ``embedder`` is the encoder name.
# POPULATED FROM CALIBRATION (HLTCOE Grid, a100; see plan Results). Cells with no
# row fall back to the static per-(device, media) profiles in ``_common``.
LOAD_COST_MODEL: dict[tuple[str, str, str], dict[str, float]] = {
    ("cpu", "image", "siglip"): {
        "a_model": 0.5,
        "a_embed": 1.971,
        "b_embed": 0.292716,
        "a_fin": 0.4663,
        "b_fin": 0.002124,
    },
    ("cpu", "audio", "clap"): {"a_model": 0.5, "a_embed": 0.0, "b_embed": 0.184566, "a_fin": 0.0, "b_fin": 0.002525},
    ("cuda", "image", "siglip"): {
        "a_model": 0.5,
        "a_embed": 2.1194,
        "b_embed": 0.006784,
        "a_fin": 8.0066,
        "b_fin": 0.000195,
    },
    ("cuda", "audio", "clap"): {
        "a_model": 0.5,
        "a_embed": 0.6868,
        "b_embed": 0.036626,
        "a_fin": 0.0,
        "b_fin": 0.003618,
    },
}

# Cold-download bandwidth (archive MB per second) over the measured hosts
# (device-pooled; the two devices agreed within ~10%). 0 disables the download
# term (weights then reflect only model+embed+finalize).
DOWNLOAD_MB_PER_S: float = 10.05


def normalize_device(device: str) -> str:
    """Collapse ``resolve_device()`` output ("cuda:0", "cuda", "cpu", "mps"…) to
    the coarse key used by :data:`LOAD_COST_MODEL` ("cuda" / "cpu")."""
    return "cuda" if device.startswith("cuda") else "cpu"


def cost_model_weights(
    device: str,
    media_type: str,
    embedder: str,
    n: int,
    download_size_mb: Optional[float] = None,
) -> Optional[list[float]]:
    """Return normalized ``[download, model, embed, finalize]`` weights from the
    affine cost model, or ``None`` when no coefficient row matches (so the caller
    can fall back to the static profile).

    ``download_size_mb`` of 0/``None`` collapses the download slice — which is
    exactly right for local-folder imports and cache-backed re-adds, where no
    archive is fetched.
    """
    row = LOAD_COST_MODEL.get((normalize_device(device), media_type, embedder))
    if row is None or n <= 0:
        return None
    t_download = 0.0
    if download_size_mb and DOWNLOAD_MB_PER_S > 0:
        t_download = download_size_mb / DOWNLOAD_MB_PER_S
    t_model = row.get("a_model", 0.0)
    t_embed = row.get("a_embed", 0.0) + row.get("b_embed", 0.0) * n
    t_finalize = row.get("a_fin", 0.0) + row.get("b_fin", 0.0) * n
    parts = [max(0.0, t_download), max(0.0, t_model), max(0.0, t_embed), max(0.0, t_finalize)]
    total = sum(parts)
    if total <= 0:
        return None
    return [p / total for p in parts]
