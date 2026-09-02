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
``scripts/profiling/README.md`` for the measurement matrix and fitting
procedure. Re-run that harness to refresh — do not hand-tune. Cells with no
measured row fall back to the static per-device/media profiles in ``_common``.

**These are the shipped defaults, not the last word.** They were measured on one
GPU cluster; a deployment on different hardware, different storage, or a
different network can be several times faster or slower. An admin who runs
``scripts/profiling/tune_timing_profile.py`` on their own machines gets a
``VTSEARCH_TIMING_PROFILE`` JSON whose ``dataset_load`` cells override the table
below, per cell, without touching this file (see :mod:`vtscore.timing`). The
lookups here consult that profile first and fall back to these constants.
"""

from __future__ import annotations

from typing import Optional

from vtscore import timing
from vtscore.timing.profile import cuml_active as _cuml_enabled
from vtscore.timing.profile import normalize_device as _normalize_device

# key: (device, media_type, embedder) -> affine coefficients (seconds).
# ``device`` is "cpu", "cuda", or "cuda+cuml"; ``embedder`` is the encoder
# name (empty for convert-out media like document, which embed via a
# converter target). The "cuda+cuml" variant covers GPU hosts where cuML
# serves the coverage-atlas k-means / UMAP (see ``vtscore/gpu_backends.py``)
# — that moves the dominant finalize work to the GPU, so the finalize
# coefficients differ materially from the CPU-clustering "cuda" rows. Lookup
# tries the variant matching the live cuML state first and falls back to the
# other, so a host missing one variant still gets the closer measured row.
# POPULATED FROM CALIBRATION. Re-swept in full on 2026-09-01 (issue #3062
# follow-up) so every row comes from one measurement under the warm-rows-only
# fitter; the previous table was the 2026-07-18/19 sweep on rack8n06 (#2623).
# Covers every demo-backed media type x every loadable registered embedder; the
# "cuda" (cuML-off) variant is measured for the default embedders only, the
# "cuda+cuml" sweep for all of them. Cells with no row fall back to the static
# per-(device, media) profiles in ``_common``.
#
# Hardware (the v100 fleet is NOT homogeneous, so it is recorded per device key):
#   cpu rows        -- rack7n03 + rack7n04, Xeon Gold 5218R (both cpu jobs on the
#                      same CPU model, so the cpu rows stay mutually comparable)
#   cuda / cuda+cuml -- rack7n06, Xeon E5-2698 v4 + V100-SXM2. Both variants ran
#                      on the SAME node so the cuML on/off contrast isolates cuML
#                      rather than the machine.
#
# Not calibratable (static fallback stands):
#   video/videomae      -- ``'VideoMAEv2' object has no attribute
#                          'all_tied_weights_keys'`` (transformers version skew)
#   video/languagebind  -- checkpoint type ``LanguageBindVideo`` unknown to
#                          transformers
#   audio/paraspeechclap -- NOTE the old "upstream removed the weights file"
#                          reason is STALE: the checkpoint downloads fine now.
#                          It fails two other ways instead -- on CPU the
#                          ``load_models()`` text warmup raises ``expected mat1
#                          and mat2 to have the same dtype: c10::BFloat16 !=
#                          float`` (``_paraspeechclap_model.py``), and on GPU the
#                          first embed hits a cpu/cuda device mismatch while text
#                          queries still hit the dtype error. Deceptively, later
#                          loads then succeed, so it emits fittable-looking rows
#                          from a path no deployment reaches cleanly -- it is
#                          excluded from the fit deliberately. See #2635.
#   face                -- no demo datasets.
#
# ``b_fin`` tracks EMBEDDING DIMENSION, not the embedder: within one dimension it
# is near-constant (the six audio encoders, all 512/768-d, spread 1.09x), but
# across dimensions it scales -- cpu/image runs clip (512-d) 0.0059, the 768-d
# group 0.0071-0.0081, siglip_l/siglip2_l (1152-d) 0.0082/0.0087, and sift_vlad
# (8192-d) 0.0281. The finalize phase clusters the vectors, so wider vectors cost
# more. Use that as the sanity check on a new row, not "finalize is constant".
#
# DOWNLOAD_MB_PER_S / EXTRACT_MB_PER_S below are NOT from this sweep: it pre-
# linked every demo source to avoid re-downloading 31 GB, leaving no usable
# extract rows and a single download size, so they keep their #2623 values.
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
        "a_embed": 0.0,
        "b_embed": 0.673733,
        "a_fin": 0.0,
        "b_fin": 0.00425,
    },
    ("cpu", "audio", "beats"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 0.2116,
        "b_embed": 0.183158,
        "a_fin": 0.0,
        "b_fin": 0.004181,
    },
    ("cpu", "audio", "clap"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 0.0,
        "b_embed": 0.240967,
        "a_fin": 0.0,
        "b_fin": 0.003662,
    },
    ("cpu", "audio", "clap_general"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 0.0,
        "b_embed": 0.34062,
        "a_fin": 0.0,
        "b_fin": 0.003503,
    },
    ("cpu", "audio", "clap_music"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 0.0,
        "b_embed": 0.328306,
        "a_fin": 0.0,
        "b_fin": 0.003802,
    },
    ("cpu", "audio", "whisper_encoder"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 1.136,
        "b_embed": 0.262094,
        "a_fin": 0.0,
        "b_fin": 0.003285,
    },
    ("cpu", "document", ""): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 28.3593,
        "b_embed": 0.0,
        "a_fin": 15.8791,
        "b_fin": 0.0,
    },
    ("cpu", "image", "clip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.9067,
        "b_embed": 0.047896,
        "a_fin": 0.0,
        "b_fin": 0.005925,
    },
    ("cpu", "image", "dinov2_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.9612,
        "b_embed": 0.236732,
        "a_fin": 0.0,
        "b_fin": 0.007364,
    },
    ("cpu", "image", "dinov2_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.1309,
        "b_embed": 0.127984,
        "a_fin": 0.0,
        "b_fin": 0.007462,
    },
    ("cpu", "image", "dinov3_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.3353,
        "b_embed": 0.20223,
        "a_fin": 0.0071,
        "b_fin": 0.007094,
    },
    ("cpu", "image", "dinov3_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.3248,
        "b_embed": 0.118459,
        "a_fin": 0.0,
        "b_fin": 0.007116,
    },
    ("cpu", "image", "eupe_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.161,
        "b_embed": 0.214392,
        "a_fin": 0.0,
        "b_fin": 0.00786,
    },
    ("cpu", "image", "eupe_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.0,
        "b_embed": 0.134101,
        "a_fin": 0.0,
        "b_fin": 0.008102,
    },
    ("cpu", "image", "sift_vlad"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 6.2084,
        "b_embed": 0.047929,
        "a_fin": 0.0,
        "b_fin": 0.028128,
    },
    ("cpu", "image", "siglip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.0,
        "b_embed": 0.115176,
        "a_fin": 0.0,
        "b_fin": 0.007349,
    },
    ("cpu", "image", "siglip2"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 2.2953,
        "b_embed": 0.114403,
        "a_fin": 0.0,
        "b_fin": 0.007626,
    },
    ("cpu", "image", "siglip2_l"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.0,
        "b_embed": 1.687938,
        "a_fin": 0.0,
        "b_fin": 0.008696,
    },
    ("cpu", "image", "siglip_l"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 46.273,
        "b_embed": 1.512354,
        "a_fin": 0.0,
        "b_fin": 0.00819,
    },
    ("cpu", "text", "bge"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 4.7887,
        "b_embed": 0.083216,
        "a_fin": 0.0,
        "b_fin": 0.003418,
    },
    ("cpu", "text", "e5"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.8877,
        "b_embed": 0.085647,
        "a_fin": 0.0,
        "b_fin": 0.004136,
    },
    ("cpu", "video", "xclip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.0,
        "b_embed": 0.564064,
        "a_fin": 0.0283,
        "b_fin": 0.002696,
    },
    ("cuda", "audio", "clap"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 1.2889,
        "b_embed": 0.096506,
        "a_fin": 0.0,
        "b_fin": 0.003508,
    },
    ("cuda", "audio", "clap_general"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 0.0,
        "b_embed": 0.108974,
        "a_fin": 0.0,
        "b_fin": 0.003458,
    },
    ("cuda", "document", ""): {
        "a_model": 39.3245,
        "b_load": 0.0,
        "a_embed": 8.4854,
        "b_embed": 0.0,
        "a_fin": 12.9926,
        "b_fin": 0.0,
    },
    ("cuda", "image", "siglip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.0,
        "b_embed": 0.013266,
        "a_fin": 0.0,
        "b_fin": 0.0068,
    },
    ("cuda", "text", "e5"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.0077,
        "b_embed": 0.011266,
        "a_fin": 0.0,
        "b_fin": 0.003602,
    },
    ("cuda", "video", "xclip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.0,
        "b_embed": 0.424908,
        "a_fin": 0.0,
        "b_fin": 0.004067,
    },
    ("cuda+cuml", "audio", "ast"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 3.9087,
        "b_embed": 0.040243,
        "a_fin": 0.0,
        "b_fin": 0.003885,
    },
    ("cuda+cuml", "audio", "beats"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 6.238,
        "b_embed": 0.022472,
        "a_fin": 0.0582,
        "b_fin": 0.003606,
    },
    ("cuda+cuml", "audio", "clap"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 14.2685,
        "b_embed": 0.07973,
        "a_fin": 0.0134,
        "b_fin": 0.003566,
    },
    ("cuda+cuml", "audio", "clap_general"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 12.7615,
        "b_embed": 0.092852,
        "a_fin": 0.0935,
        "b_fin": 0.003786,
    },
    ("cuda+cuml", "audio", "clap_music"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 12.2451,
        "b_embed": 0.094286,
        "a_fin": 0.0783,
        "b_fin": 0.003556,
    },
    ("cuda+cuml", "audio", "whisper_encoder"): {
        "a_model": 0.5,
        "b_load": _AUDIO_B_LOAD,
        "a_embed": 5.6528,
        "b_embed": 0.020961,
        "a_fin": 0.0,
        "b_fin": 0.003678,
    },
    ("cuda+cuml", "document", ""): {
        "a_model": 39.3174,
        "b_load": 0.0,
        "a_embed": 12.1888,
        "b_embed": 0.0,
        "a_fin": 14.3655,
        "b_fin": 0.0,
    },
    ("cuda+cuml", "image", "clip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.0708,
        "b_embed": 0.012297,
        "a_fin": 0.0,
        "b_fin": 0.005334,
    },
    ("cuda+cuml", "image", "dinov2_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.173,
        "b_embed": 0.024177,
        "a_fin": 0.0284,
        "b_fin": 0.005542,
    },
    ("cuda+cuml", "image", "dinov2_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.5841,
        "b_embed": 0.013673,
        "a_fin": 0.0,
        "b_fin": 0.005436,
    },
    ("cuda+cuml", "image", "dinov3_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.7247,
        "b_embed": 0.024469,
        "a_fin": 0.0,
        "b_fin": 0.005411,
    },
    ("cuda+cuml", "image", "dinov3_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.5282,
        "b_embed": 0.015507,
        "a_fin": 0.0,
        "b_fin": 0.005414,
    },
    ("cuda+cuml", "image", "eupe_patch"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.8808,
        "b_embed": 0.029221,
        "a_fin": 0.0,
        "b_fin": 0.005494,
    },
    ("cuda+cuml", "image", "eupe_single"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.3147,
        "b_embed": 0.019916,
        "a_fin": 0.0,
        "b_fin": 0.005462,
    },
    ("cuda+cuml", "image", "sift_vlad"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 2.2266,
        "b_embed": 0.056347,
        "a_fin": 0.0,
        "b_fin": 0.010999,
    },
    ("cuda+cuml", "image", "siglip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 0.5176,
        "b_embed": 0.013271,
        "a_fin": 0.0,
        "b_fin": 0.005507,
    },
    ("cuda+cuml", "image", "siglip2"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.5025,
        "b_embed": 0.012108,
        "a_fin": 0.0,
        "b_fin": 0.005388,
    },
    ("cuda+cuml", "image", "siglip2_l"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 1.4977,
        "b_embed": 0.08835,
        "a_fin": 0.0,
        "b_fin": 0.005805,
    },
    ("cuda+cuml", "image", "siglip_l"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 3.024,
        "b_embed": 0.086311,
        "a_fin": 0.0,
        "b_fin": 0.005695,
    },
    ("cuda+cuml", "text", "bge"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 5.8472,
        "b_embed": 0.01048,
        "a_fin": 0.0619,
        "b_fin": 0.002094,
    },
    ("cuda+cuml", "text", "e5"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 5.6353,
        "b_embed": 0.010862,
        "a_fin": 0.4075,
        "b_fin": 0.001996,
    },
    ("cuda+cuml", "video", "xclip"): {
        "a_model": 0.5,
        "b_load": 0.0,
        "a_embed": 2.6709,
        "b_embed": 0.413409,
        "a_fin": 0.012,
        "b_fin": 0.004143,
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
# POPULATED FROM CALIBRATION (HLTCOE Grid rack8n06 v100 node, 2026-07-18/19
# sweep under /exp/…/calib-2623 — the same JSONL that sourced LOAD_COST_MODEL;
# the profiler was already stamping ``finalize:<slot>`` rows during that run,
# so no separate sweep was needed; issues #2623/#2624, see plan Results).
# ``scripts/profiling/fit_load_weights.py`` fits the rows into this table's
# body (median seconds per slot across loads, normalized per cell). Re-run
# that harness to refresh — do not hand-tune. Unlike LOAD_COST_MODEL there is
# no "cuda+cuml" variant: each "cuda" row pools cuML-on and cuML-off loads,
# because the measured split moved no slot's share by more than ~0.11 (e.g.
# image coverage 0.41 cuML vs 0.44 without) — far from the registry/coverage
# flip the static ballpark gets wrong, and not worth a third key. The
# opt-in ``projection`` / ``signpost_texts`` slots never ran during
# calibration, so they emit no row here and keep their static ballpark via
# the :func:`_finalize_slots` merge; ``dedup`` / ``cleanup`` did run and
# measure ~0 (the default fast-hash dedup — an opt-in near-dup merge would
# be undersold by these shares until a calibrated run covers it). Cells with
# no measured row fall back to the static ``FinalizeProgress._SLOTS``
# ballpark entirely.
FINALIZE_SLOT_SHARES: dict[tuple[str, str], tuple[tuple[str, float], ...]] = {
    ("cpu", "audio"): (("cleanup", 0.0001), ("dedup", 0.0002), ("coverage", 0.5751), ("registry", 0.4246)),
    ("cpu", "document"): (("cleanup", 0.0001), ("dedup", 0.0001), ("coverage", 0.0321), ("registry", 0.9679)),
    ("cpu", "image"): (("cleanup", 0.0001), ("dedup", 0.0001), ("coverage", 0.4475), ("registry", 0.5524)),
    ("cpu", "text"): (("cleanup", 0.0001), ("dedup", 0.0003), ("coverage", 0.9865), ("registry", 0.0132)),
    ("cpu", "video"): (("cleanup", 0.0003), ("dedup", 0.0003), ("coverage", 0.4478), ("registry", 0.5516)),
    ("cuda", "audio"): (("cleanup", 0.0001), ("dedup", 0.0002), ("coverage", 0.658), ("registry", 0.3416)),
    ("cuda", "document"): (("cleanup", 0.0001), ("dedup", 0.0001), ("coverage", 0.0468), ("registry", 0.9532)),
    ("cuda", "image"): (("cleanup", 0.0001), ("dedup", 0.0001), ("coverage", 0.5173), ("registry", 0.4825)),
    ("cuda", "text"): (("cleanup", 0.0001), ("dedup", 0.0003), ("coverage", 0.9814), ("registry", 0.0182)),
    ("cuda", "video"): (("cleanup", 0.0002), ("dedup", 0.0003), ("coverage", 0.5873), ("registry", 0.4123)),
}


def finalize_slot_shares(device: str, media_type: str, embedder: str = "") -> Optional[tuple[tuple[str, float], ...]]:
    """Return the measured finalize sub-slot ``(slot, share)`` tuples for this
    cell, or ``None`` when nothing measured covers it (so the caller falls back
    to the static ``FinalizeProgress._SLOTS`` ballpark).

    An admin ``VTSEARCH_TIMING_PROFILE`` wins over the checked-in table, since
    it was measured on the hardware actually serving the app. ``device`` is
    normalized to "cuda" / "cpu"; the shares are returned verbatim (raw
    weights) — the consumer normalizes them into ordered sub-ranges.
    """
    tuned = timing.slot_shares("dataset_load", "finalize", device=device, media_type=media_type, embedder=embedder)
    if tuned:
        return tuple(tuned.items())
    row = FINALIZE_SLOT_SHARES.get((normalize_device(device), media_type))
    if not row:
        return None
    return row


def normalize_device(device: str) -> str:
    """Collapse ``resolve_device()`` output ("cuda:0", "cuda", "cpu", "mps"…) to
    the coarse key used by :data:`LOAD_COST_MODEL` ("cuda" / "cpu")."""
    return _normalize_device(device)


def _cuml_active() -> bool:
    """Whether cuML will serve this process's clustering (never raises)."""
    return _cuml_enabled()


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

    An admin ``VTSEARCH_TIMING_PROFILE`` measured on this deployment's own
    hardware takes precedence over the checked-in table; only cells the profile
    does not cover fall through to the constants above.
    """
    if n > 0:
        tuned = timing.step_terms(
            "dataset_load",
            device=device,
            media_type=media_type,
            embedder=embedder,
            n=n,
            size_mb=download_size_mb or 0.0,
        )
        if tuned is not None:
            if not download_size_mb:
                # No archive to fetch or unpack (local folder, warm cache): zero
                # the byte-scaled phases even if the profile carries a fixed
                # intercept for them, so a cached re-add doesn't budget a
                # download slice for work that will never run.
                tuned = {**tuned, "download": 0.0, "extract": 0.0}
            if sum(tuned.values()) > 0:
                return tuned

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
