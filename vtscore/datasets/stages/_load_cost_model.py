"""Affine per-phase load-cost coefficients, fit from measured GRID timings.

The dataset-load progress bar paces itself with a per-phase weight vector
(download+extract, load, embed, finalize). A single static vector can't be
right at both small and large ``n``, because the phases scale with ``n``
differently: model-load is a fixed cost, per-item source decode, embed and
finalize grow with ``n``, and download/extraction track archive bytes. This
module holds the measured affine coefficients so
:func:`vtscore.datasets.stages._common.load_step_weights` (and the runtime
:class:`~vtscore.datasets.stages._common.AdaptiveLoadPacer`) can compute an
``n``-aware phase-time model.

    T_download ≈ download_size_mb / DOWNLOAD_MB_PER_S
    T_extract  ≈ download_size_mb / EXTRACT_MB_PER_S
    T_load     ≈ a_model + b_load · n            (warm model load + per-item source decode)
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
# POPULATED FROM CALIBRATION (HLTCOE Grid rack8n06, 2026-07-18 run under
# /exp/…/calib-2556; see plan Results). Cells with no row fall back to the
# static per-(device, media) profiles in ``_common``.
# ``b_load`` is the per-item source read/decode cost inside the "loading" step
# (step 2 covers warm model load *plus* reading every source file into medias —
# for a 1000-file audio demo over NFS that decode is tens of seconds, so it
# must scale with ``n`` rather than hide inside the fixed ``a_model``).
# NB the calibration demos (caltech101/esc50) decode inside the embed loop, so
# no warm step-2 rows exist to fit ``b_load`` from — the audio values below are
# measured from a live profiled GTZAN load (per-file decode path): 49.6s for
# 455 files ≈ 0.11 s/item pure decode+clip, on both devices (decode is
# CPU-bound regardless of the embed device). Still an underestimate for loads
# that also unpickle a large embeddings cache in step 2, which is why the
# AdaptiveLoadPacer re-estimates every phase's term from its observed pace.
LOAD_COST_MODEL: dict[tuple[str, str, str], dict[str, float]] = {
    ("cpu", "image", "siglip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 5.0828,
        "b_embed": 0.583843,
        "a_fin": 0.2884,
        "b_fin": 0.003179,
    },
    ("cpu", "audio", "clap"): {
        "a_model": 0.5,
        "b_load": 0.11,
        "a_embed": 1.1159,
        "b_embed": 0.317579,
        "a_fin": 0.0587,
        "b_fin": 0.00237,
    },
    ("cuda", "image", "siglip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.8928,
        "b_embed": 0.01233,
        "a_fin": 0.8098,
        "b_fin": 0.003719,
    },
    ("cuda", "audio", "clap"): {
        "a_model": 0.5,
        "b_load": 0.11,
        "a_embed": 0.0,
        "b_embed": 0.063223,
        "a_fin": 0.0796,
        "b_fin": 0.003154,
    },
}

# Cold-download bandwidth (archive MB per second) over the measured hosts
# (device-pooled median; observed 3.3–4.4 across the two device runs — the
# cluster's shared egress swings by hours-of-day far more than by host). 0
# disables the download term (weights then reflect only model+embed+finalize).
# This is only the *prior*: the AdaptiveLoadPacer re-estimates the download
# term from the observed transfer rate once bytes start flowing.
DOWNLOAD_MB_PER_S: float = 3.836

# Archive extraction rate (archive MB per second) on the measured hosts.
# Extraction runs after the download inside the same acquire step, but its cost
# scales with archive bytes at a very different rate than the network transfer,
# so it gets its own term (and its own "extracting" status / bar slice).
EXTRACT_MB_PER_S: float = 61.477


# Per-(device, media) finalize sub-slot shares — the measured counterpart to the
# static ``FinalizeProgress._SLOTS`` ballpark. The finalize phase (step 4) is
# split into ordered sub-stages (cleanup, dedup, coverage-atlas, signpost-texts,
# registry, projection); this maps each ``(device, media_type)`` to that phase's
# measured wall-clock share per sub-stage, so the finalize slice paces against
# the real per-device mix rather than one static guess. The shares are raw
# weights (need not sum to 1 — :func:`finalize_slot_shares` is normalized by its
# consumer). Slot order here is the execution order; a slot omitted from a row
# simply isn't paced separately for that cell.
#
# EMPTY until a calibration run populates it. The env-gated load profiler already
# records ``finalize:<slot>`` rows (see ``_load_profiler`` /
# ``FinalizeProgress.begin``); ``scripts/profiling/fit_load_weights.py`` fits
# them into this table's body. Re-run that harness to refresh — do not hand-tune.
# Cells with no measured row fall back to the static ``FinalizeProgress._SLOTS``
# ballpark (which is why the finalize sub-stage motivating this table — a
# non-cuML GPU host where the coverage k-means outweighs the registry save — is
# uncalibrated here and awaits a GPU calibration run; see issue #2624).
FINALIZE_SLOT_SHARES: dict[tuple[str, str], tuple[tuple[str, float], ...]] = {}


def finalize_slot_shares(device: str, media_type: str) -> Optional[tuple[tuple[str, float], ...]]:
    """Return the measured finalize sub-slot ``(slot, share)`` tuples for
    ``(device, media_type)``, or ``None`` when no calibrated row exists (so the
    caller falls back to the static ``FinalizeProgress._SLOTS`` ballpark).

    ``device`` is normalized to "cuda" / "cpu"; the shares are returned verbatim
    (raw weights) — the consumer normalizes them into ordered sub-ranges.
    """
    row = FINALIZE_SLOT_SHARES.get((normalize_device(device), media_type))
    if not row:
        return None
    return row


def normalize_device(device: str) -> str:
    """Collapse ``resolve_device()`` output ("cuda:0", "cuda", "cpu", "mps"…) to
    the coarse key used by :data:`LOAD_COST_MODEL` ("cuda" / "cpu")."""
    return "cuda" if device.startswith("cuda") else "cpu"


def cost_model_terms(
    device: str,
    media_type: str,
    embedder: str,
    n: int,
    download_size_mb: Optional[float] = None,
) -> Optional[dict[str, float]]:
    """Return predicted per-phase seconds from the affine cost model, or
    ``None`` when no coefficient row matches (so the caller can fall back to
    the static profile).

    Keys: ``download``, ``extract``, ``load``, ``embed``, ``finalize`` —
    ``download``/``extract`` are the two sub-phases of the acquire step (step
    1), ``load`` is warm model load plus per-item source decode (step 2).

    ``download_size_mb`` of 0/``None`` collapses the download **and** extract
    terms — which is exactly right for local-folder imports and cache-backed
    re-adds, where no archive is fetched or unpacked.
    """
    row = LOAD_COST_MODEL.get((normalize_device(device), media_type, embedder))
    if row is None or n <= 0:
        return None
    t_download = 0.0
    t_extract = 0.0
    if download_size_mb:
        if DOWNLOAD_MB_PER_S > 0:
            t_download = download_size_mb / DOWNLOAD_MB_PER_S
        if EXTRACT_MB_PER_S > 0:
            t_extract = download_size_mb / EXTRACT_MB_PER_S
    terms = {
        "download": max(0.0, t_download),
        "extract": max(0.0, t_extract),
        "load": max(0.0, row.get("a_model", 0.0) + row.get("b_load", 0.0) * n),
        "embed": max(0.0, row.get("a_embed", 0.0) + row.get("b_embed", 0.0) * n),
        "finalize": max(0.0, row.get("a_fin", 0.0) + row.get("b_fin", 0.0) * n),
    }
    if sum(terms.values()) <= 0:
        return None
    return terms


def cost_model_weights(
    device: str,
    media_type: str,
    embedder: str,
    n: int,
    download_size_mb: Optional[float] = None,
) -> Optional[list[float]]:
    """Return normalized ``[acquire, load, embed, finalize]`` step weights from
    the affine cost model (acquire = download + extract), or ``None`` when no
    coefficient row matches (so the caller can fall back to the static
    profile)."""
    terms = cost_model_terms(device, media_type, embedder, n, download_size_mb)
    if terms is None:
        return None
    parts = [
        terms["download"] + terms["extract"],
        terms["load"],
        terms["embed"],
        terms["finalize"],
    ]
    total = sum(parts)
    return [p / total for p in parts]
