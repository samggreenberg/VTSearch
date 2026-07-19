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
# ``device`` is "cpu", "cuda", or "cuda+cuml"; ``embedder`` is the encoder
# name (empty for convert-out media like document, which embed via a
# converter target). The "cuda+cuml" variant covers GPU hosts where cuML
# serves the coverage-atlas k-means / UMAP (see ``vtscore/gpu_backends.py``)
# — that moves the dominant finalize work to the GPU, so the finalize
# coefficients differ materially from the CPU-clustering "cuda" rows. Lookup
# tries the variant matching the live cuML state first and falls back to the
# other, so a host missing one variant still gets the closer measured row.
# POPULATED FROM CALIBRATION (HLTCOE Grid rack8n06 v100 node, 2026-07-18/19
# sweep under /exp/…/calib-2623, issue #2623; see plan Results). Covers every
# demo-backed media type × every loadable registered embedder; the "cuda"
# (cuML-off) variant was measured for the default embedders only, the
# "cuda+cuml" sweep for all of them. Cells with no row fall back to the
# static per-(device, media) profiles in ``_common``. Not calibratable on the
# measurement host (and equally unloadable in a served app on it):
# video/videomae + video/languagebind (transformers version skew) and
# audio/paraspeechclap (upstream HF weights file removed).
# ``b_load`` is the per-item source read/decode cost inside the "loading" step
# (step 2 covers warm model load *plus* reading every source file into medias —
# for a 1000-file audio demo over NFS that decode is tens of seconds, so it
# must scale with ``n`` rather than hide inside the fixed ``a_model``).
# NB the calibration demos (caltech101/esc50) decode inside the embed loop, so
# no warm step-2 rows exist to fit ``b_load`` from — the audio value below is
# measured from a live profiled GTZAN load (per-file decode path): 49.6s for
# 455 files ≈ 0.11 s/item pure decode+clip, applied to every audio row since
# the decode is CPU-bound and embedder-independent. Still an underestimate for
# loads that also unpickle a large embeddings cache in step 2, which is why the
# AdaptiveLoadPacer re-estimates every phase's term from its observed pace.
# The ``document`` rows (empty embedder — convert-out type) are single-size
# constants: the load term is dominated by the document2image page
# rasterisation, measured at n=240 pages from 150 PDFs.
_AUDIO_B_LOAD = 0.11

LOAD_COST_MODEL: dict[tuple[str, str, str], dict[str, float]] = {
    ("cpu", "audio", "ast"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 5.6749,
        "b_embed": 1.213088,
        "a_fin": 0.1727,
        "b_fin": 0.003869,
    },
    ("cpu", "audio", "clap"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 3.5416,
        "b_embed": 0.288775,
        "a_fin": 0.0,
        "b_fin": 0.004185,
    },
    ("cpu", "audio", "clap_general"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 16.1027,
        "b_embed": 0.40223,
        "a_fin": 0.0639,
        "b_fin": 0.003868,
    },
    ("cpu", "audio", "clap_music"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 16.9172,
        "b_embed": 0.402425,
        "a_fin": 0.1893,
        "b_fin": 0.003363,
    },
    ("cpu", "audio", "whisper_encoder"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 22.1213,
        "b_embed": 0.409268,
        "a_fin": 0.0322,
        "b_fin": 0.003757,
    },
    ("cpu", "document", ""): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 44.7291,
        "b_embed": 0.0,
        "a_fin": 15.2846,
        "b_fin": 0.0,
    },
    ("cpu", "image", "clip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 16.715,
        "b_embed": 0.051,
        "a_fin": 0.1349,
        "b_fin": 0.005717,
    },
    ("cpu", "image", "dinov2_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 18.8802,
        "b_embed": 0.387,
        "a_fin": 0.0,
        "b_fin": 0.008272,
    },
    ("cpu", "image", "dinov2_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 14.2485,
        "b_embed": 0.20084,
        "a_fin": 0.0,
        "b_fin": 0.00861,
    },
    ("cpu", "image", "dinov3_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 22.0937,
        "b_embed": 0.312758,
        "a_fin": 0.0,
        "b_fin": 0.008906,
    },
    ("cpu", "image", "dinov3_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 23.4089,
        "b_embed": 0.153089,
        "a_fin": 0.0,
        "b_fin": 0.007971,
    },
    ("cpu", "image", "eupe_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 32.4078,
        "b_embed": 0.284594,
        "a_fin": 1.4028,
        "b_fin": 0.005827,
    },
    ("cpu", "image", "eupe_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 25.2026,
        "b_embed": 0.150847,
        "a_fin": 1.0131,
        "b_fin": 0.006429,
    },
    ("cpu", "image", "sift_vlad"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.0,
        "b_embed": 0.108494,
        "a_fin": 0.3056,
        "b_fin": 0.022271,
    },
    ("cpu", "image", "siglip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 7.5042,
        "b_embed": 0.17947,
        "a_fin": 0.0,
        "b_fin": 0.007637,
    },
    ("cpu", "image", "siglip2"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 20.5357,
        "b_embed": 0.160309,
        "a_fin": 0.0,
        "b_fin": 0.008356,
    },
    ("cpu", "image", "siglip_l"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.0,
        "b_embed": 2.6201,
        "a_fin": 0.2769,
        "b_fin": 0.007371,
    },
    ("cpu", "text", "bge"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 17.2372,
        "b_embed": 0.11965,
        "a_fin": 0.0,
        "b_fin": 0.003473,
    },
    ("cpu", "text", "e5"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 6.8054,
        "b_embed": 0.137529,
        "a_fin": 0.0,
        "b_fin": 0.004273,
    },
    ("cpu", "video", "xclip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.488,
        "b_embed": 0.44465,
        "a_fin": 0.0351,
        "b_fin": 0.003186,
    },
    ("cuda", "audio", "clap"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 9.8547,
        "b_embed": 0.095121,
        "a_fin": 0.0,
        "b_fin": 0.004233,
    },
    ("cuda", "document", ""): {
        "a_model": 53.6342,
        "b_load": 0.0,
        "a_embed": 17.8238,
        "b_embed": 0.0,
        "a_fin": 18.4145,
        "b_fin": 0.0,
    },
    ("cuda", "image", "siglip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 4.534,
        "b_embed": 0.014688,
        "a_fin": 0.0,
        "b_fin": 0.008609,
    },
    ("cuda", "text", "e5"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 2.3957,
        "b_embed": 0.01576,
        "a_fin": 0.0,
        "b_fin": 0.003939,
    },
    ("cuda", "video", "xclip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.952,
        "b_embed": 0.147662,
        "a_fin": 0.0,
        "b_fin": 0.003823,
    },
    ("cuda+cuml", "audio", "ast"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 2.0354,
        "b_embed": 0.047166,
        "a_fin": 0.4869,
        "b_fin": 0.003928,
    },
    ("cuda+cuml", "audio", "clap"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 2.6679,
        "b_embed": 0.109959,
        "a_fin": 0.5875,
        "b_fin": 0.003837,
    },
    ("cuda+cuml", "audio", "clap_general"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 0.6566,
        "b_embed": 0.113987,
        "a_fin": 0.7722,
        "b_fin": 0.003842,
    },
    ("cuda+cuml", "audio", "clap_music"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 2.5482,
        "b_embed": 0.110015,
        "a_fin": 0.685,
        "b_fin": 0.003779,
    },
    ("cuda+cuml", "audio", "whisper_encoder"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 2.4531,
        "b_embed": 0.033555,
        "a_fin": 0.3943,
        "b_fin": 0.003789,
    },
    ("cuda+cuml", "document", ""): {
        "a_model": 45.4673,
        "b_load": 0.0,
        "a_embed": 13.6691,
        "b_embed": 0.0,
        "a_fin": 15.8971,
        "b_fin": 0.0,
    },
    ("cuda+cuml", "image", "clip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.9236,
        "b_embed": 0.014377,
        "a_fin": 0.0086,
        "b_fin": 0.007303,
    },
    ("cuda+cuml", "image", "dinov2_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.934,
        "b_embed": 0.033091,
        "a_fin": 0.0,
        "b_fin": 0.007588,
    },
    ("cuda+cuml", "image", "dinov2_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.8451,
        "b_embed": 0.015571,
        "a_fin": 0.0,
        "b_fin": 0.007173,
    },
    ("cuda+cuml", "image", "dinov3_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 2.6292,
        "b_embed": 0.034172,
        "a_fin": 0.0,
        "b_fin": 0.007623,
    },
    ("cuda+cuml", "image", "dinov3_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 2.5573,
        "b_embed": 0.019125,
        "a_fin": 0.0,
        "b_fin": 0.007347,
    },
    ("cuda+cuml", "image", "eupe_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 2.6362,
        "b_embed": 0.040342,
        "a_fin": 0.6297,
        "b_fin": 0.007429,
    },
    ("cuda+cuml", "image", "eupe_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 3.0167,
        "b_embed": 0.026009,
        "a_fin": 0.326,
        "b_fin": 0.007193,
    },
    ("cuda+cuml", "image", "sift_vlad"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 4.2984,
        "b_embed": 0.081432,
        "a_fin": 0.0,
        "b_fin": 0.014836,
    },
    ("cuda+cuml", "image", "siglip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.4109,
        "b_embed": 0.016008,
        "a_fin": 0.0,
        "b_fin": 0.007316,
    },
    ("cuda+cuml", "image", "siglip2"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 3.6047,
        "b_embed": 0.014941,
        "a_fin": 0.4438,
        "b_fin": 0.007312,
    },
    ("cuda+cuml", "image", "siglip_l"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 2.884,
        "b_embed": 0.088089,
        "a_fin": 0.105,
        "b_fin": 0.007667,
    },
    ("cuda+cuml", "text", "bge"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 6.0879,
        "b_embed": 0.012067,
        "a_fin": 0.4114,
        "b_fin": 0.002217,
    },
    ("cuda+cuml", "text", "e5"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 3.8359,
        "b_embed": 0.012501,
        "a_fin": 0.1454,
        "b_fin": 0.002217,
    },
    ("cuda+cuml", "video", "xclip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.0,
        "b_embed": 0.14607,
        "a_fin": 0.2533,
        "b_fin": 0.003055,
    },
}

# Cold-download bandwidth (archive MB per second) over the measured hosts
# (device-pooled median; observed 8.8–9.1 across the 2026-07-18/19 sweep — the
# cluster's shared egress swings by hours-of-day far more than by host). 0
# disables the download term (weights then reflect only model+embed+finalize).
# This is only the *prior*: the AdaptiveLoadPacer re-estimates the download
# term from the observed transfer rate once bytes start flowing.
DOWNLOAD_MB_PER_S: float = 8.967

# Archive extraction rate (archive MB per second) on the measured hosts.
# Extraction runs after the download inside the same acquire step, but its cost
# scales with archive bytes at a very different rate than the network transfer,
# so it gets its own term (and its own "extracting" status / bar slice).
EXTRACT_MB_PER_S: float = 45.66


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


def _cuml_active() -> bool:
    """Whether cuML will serve this process's clustering (never raises)."""
    try:
        from vtscore.gpu_backends import cuml_enabled  # noqa: PLC0415

        return cuml_enabled()
    except Exception:
        return False


def _lookup_row(device: str, media_type: str, embedder: str) -> Optional[dict[str, float]]:
    """Find the closest measured coefficient row for the cell.

    CPU devices map straight to their row. CUDA devices have two measured
    variants — "cuda+cuml" (GPU clustering) and "cuda" (CPU clustering on a
    GPU host) — and the live cuML state picks which is tried first; the other
    is the fallback, since a same-device row with a different finalize cost
    beats falling back to the static profile entirely.
    """
    dev = normalize_device(device)
    if dev != "cuda":
        return LOAD_COST_MODEL.get((dev, media_type, embedder))
    variants = ("cuda+cuml", "cuda") if _cuml_active() else ("cuda", "cuda+cuml")
    for variant in variants:
        row = LOAD_COST_MODEL.get((variant, media_type, embedder))
        if row is not None:
            return row
    return None


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
    row = _lookup_row(device, media_type, embedder)
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
