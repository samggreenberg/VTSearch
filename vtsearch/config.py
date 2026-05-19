"""Configuration and constants for VTSearch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Data paths are anchored to the repository root, NOT to the current working
# directory.  Without this, starting the app from a different CWD (systemd,
# cron, dev shell) would create a fresh empty `data/` and silently lose the
# user's existing datasets, settings, and embeddings.  Override with the
# ``VTSEARCH_DATA_DIR`` env var if you need to relocate state outside the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ["VTSEARCH_DATA_DIR"]) if "VTSEARCH_DATA_DIR" in os.environ else _REPO_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_CACHE_DIR = (
    Path(os.environ["VTSEARCH_MODELS_DIR"]) if "VTSEARCH_MODELS_DIR" in os.environ else DATA_DIR / "models"
)

# ---------------------------------------------------------------------------
# Runtime tunables
# ---------------------------------------------------------------------------

# Thread count for native math libraries (OpenMP / MKL) and ``torch``.  The
# default of 1 keeps memory overhead low in constrained environments — each
# additional thread allocates its own scratch buffers.  Override with
# ``VTSEARCH_TORCH_THREADS`` on bigger boxes where embedding throughput
# matters more than RSS.  Consumed by ``app.py`` (OMP/MKL env vars set
# before torch import) and ``vtsearch.media.torch_setup.ensure_torch_configured``
# (``torch.set_num_threads``).
TORCH_THREADS = max(1, int(os.environ.get("VTSEARCH_TORCH_THREADS", "1")))

# Preferred compute device for embedding and training.  ``"auto"`` resolves
# to ``"cuda"`` when a GPU is visible to PyTorch and ``"cpu"`` otherwise;
# explicit values like ``"cuda"``, ``"cuda:0"``, ``"cpu"``, or ``"mps"`` are
# passed through unchanged.  Resolution is lazy — the env var stores the
# user's intent, ``resolve_device()`` actually imports torch when called.
# Currently advisory: every embedder still loads on CPU.  Reserved for the
# upcoming device-aware embedder refactor.
DEVICE = os.environ.get("VTSEARCH_DEVICE", "auto").lower()


def resolve_device() -> str:
    """Resolve :data:`DEVICE` to a concrete ``torch.device`` string.

    Imports torch lazily so that simply importing this module does not pull
    torch in.  Returns ``"cpu"`` if torch is unavailable.
    """
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def _parse_server_roots(value: str | None) -> tuple[Path, ...]:
    """Parse ``VTSEARCH_SERVER_ROOTS`` into a tuple of resolved Paths.

    Splits on :data:`os.pathsep` (``:`` on Unix, ``;`` on Windows).  Empty
    segments are ignored.  When the env var is unset or empty the tuple
    contains a single entry — ``Path.cwd()`` at import time — which
    reproduces the historical "anything under CWD" behaviour exactly.
    """
    if not value:
        return (Path.cwd().resolve(),)
    roots: list[Path] = []
    for segment in value.split(os.pathsep):
        segment = segment.strip()
        if segment:
            roots.append(Path(segment).resolve())
    return tuple(roots) if roots else (Path.cwd().resolve(),)


# Allowed roots for the server file browser and server-path importers/exporters.
# In single-user mode these define the directories the user is permitted to
# read from and write to via the API.  The first entry is the default browse
# root (what ``/api/browse`` shows when no ``path`` parameter is provided).
# Multi-user mode is unaffected — each user is still confined to their own
# ``data/<username>/`` subtree regardless of this setting.
SERVER_ROOTS: tuple[Path, ...] = _parse_server_roots(os.environ.get("VTSEARCH_SERVER_ROOTS"))

# Maximum size (in megabytes) accepted for a single HTTP request body.  Wired
# into Flask's ``MAX_CONTENT_LENGTH`` config in ``app.py``.  ``0`` (the
# default) disables the cap, preserving Flask's out-of-the-box "no limit"
# behaviour so existing large-archive uploads keep working.  Set
# ``VTSEARCH_MAX_UPLOAD_MB`` to a positive integer to reject oversized
# uploads with HTTP 413 before they consume disk.
MAX_UPLOAD_MB = max(0, int(os.environ.get("VTSEARCH_MAX_UPLOAD_MB", "0")))

# Training
#
# ``TRAIN_EPOCHS`` is an *upper bound* — :func:`vtsearch.training.mlp.train_model`
# also short-circuits on a loss plateau (see ``TRAIN_PATIENCE``).  Override with
# ``VTSEARCH_TRAIN_EPOCHS`` for benchmarking or to disable early-stop entirely
# by pairing with ``VTSEARCH_TRAIN_PATIENCE=0``.
TRAIN_EPOCHS = int(os.environ.get("VTSEARCH_TRAIN_EPOCHS", "200"))
# Number of epochs the training loss must fail to improve before training
# stops early.  Set to 0 to disable early-stop and always run ``TRAIN_EPOCHS``.
TRAIN_PATIENCE = int(os.environ.get("VTSEARCH_TRAIN_PATIENCE", "10"))
# Default ``calibrate_count`` baked into ``data/settings.json`` on first run.
# Each unit adds one full fold-training pass per learned-sort; lower it to
# trade calibration quality for latency.  Min 1 (clamped in
# :mod:`vtsearch.settings`).  The default is 1: with
# ``calibration_fraction=0.5`` a single fold already trains on half the
# labels, and a second fold mostly averages out per-split noise — bumping
# back up is a one-setting change when the noise actually matters.
DEFAULT_CALIBRATE_COUNT = max(1, int(os.environ.get("VTSEARCH_CALIBRATE_COUNT", "1")))
MLP_HIDDEN_MIN = 4
MLP_HIDDEN_MAX = 32
MLP_DROPOUT = 0.5

# Model IDs
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000  # CLAP model expected input sample rate
XCLIP_MODEL_ID = "microsoft/xclip-base-patch32"
E5_MODEL_ID = "intfloat/e5-base-v2"
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DINOV2_MODEL_ID = "facebook/dinov2-base"
DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
EUPE_MODEL_ID = "https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt"
"""Direct URL to the real EUPE ViT-B/16 weights on Hugging Face.

Loaded via :func:`torch.hub.load` from the ``facebookresearch/EUPE`` GitHub
repo with this URL passed as the ``weights`` kwarg.  The HF repo
``facebook/EUPE-ViT-B`` is ungated; the underlying weights are released
under Meta's FAIR Noncommercial Research Licence (surfaced to users via
``MediaEmbedder.license_notice`` on the EUPE embedder).

Not the same model as ``facebook/PE-Core-B16-224`` — that was Meta's
Perception Encoder Core, which the dev "eupe" slug was confusingly
aliased to via a broken ``AutoModel.from_pretrained`` path (the PE-Core
HF repo has no ``config.json`` so ``AutoModel`` could never load it).
"""
CLAP_MUSIC_MODEL_ID = "laion/larger_clap_music_and_speech"
BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"
LANGUAGEBIND_VIDEO_MODEL_ID = "LanguageBind/LanguageBind_Video_V1.5_FT"
VIDEOMAE_MODEL_ID = "OpenGVLab/VideoMAEv2-Base"
"""Hugging Face repo for VideoMAE v2 Base weights.

Loaded via ``AutoModel.from_pretrained(..., trust_remote_code=True)``.
Vision-only encoder — there is no paired text tower, so the embedder
sets ``supports_text=False`` and :meth:`embed_text` returns ``None``.
The masked-autoencoder objective produces unusually strong action /
motion features compared to image-only encoders applied per frame.
"""


# ---------------------------------------------------------------------------
# CoreConfig — runtime config bundle the (future) ``vtscore`` library consumes
# ---------------------------------------------------------------------------
#
# Today every library-candidate package reaches into ``vtsearch.settings``
# directly for tunables like ``saved_datasets_dir``, ``detectors_dir``,
# ``calibrate_count``, etc.  That couples the library to the app's settings
# layer and makes it impossible to vendor the library as ``vtscore`` (see
# ``docs/plans/extract-library.md`` — Phase 2).
#
# ``CoreConfig`` is the seam: a frozen value object that bundles every knob
# library code reads.  Follow-up PRs convert each call site to accept (or
# look up) a ``CoreConfig`` instead of importing ``vtsearch.settings``.
# Until those land this class is unused at runtime — the scaffold just
# defines the type so the conversions can happen one file at a time.
#
# The app side will build a fresh ``CoreConfig`` at each request boundary
# via :meth:`CoreConfig.from_settings`; library callers can construct one
# directly with whatever values they want.


@dataclass(frozen=True)
class CoreConfig:
    """Runtime configuration bundle the ``vtscore`` library consumes.

    Field set is intentionally narrow — only knobs that library code (loaders,
    detectors, training, embedding) reads.  User-pref concerns like theme or
    grid-icon size are app-tier and stay in ``vtsearch.settings``.
    """

    # Server-tier settings (shared across users, stored in data/settings.json)
    saved_datasets_dir: Path
    detectors_dir: Path
    max_concurrent_dataset_downloads: int
    max_concurrent_dataset_embeddings: int
    autorun_detectors: tuple[str, ...]

    # Per-user settings (stored under each user's data dir)
    safe_thresholds: bool
    calibrate_count: int
    calibration_fraction: float
    enrich_descriptions: bool
    autopilot_goal_diversity: int
    inclusion: int

    # Filesystem root for caches, embeddings, model downloads.  Phase 4 will
    # route every hardcoded ``data/`` path through this field.
    data_dir: Path

    @classmethod
    def from_settings(cls, settings_path: str | Path | None = None) -> CoreConfig:
        """Snapshot the current user's ``vtsearch.settings`` into a CoreConfig.

        Called by the Flask app at the request boundary (after auth resolves
        the current user) and by the CLI before kicking off autodetect.  The
        result is a frozen immutable value safe to hand to background
        threads — settings changes during a request will not retroactively
        rewrite a config already in flight.

        When *settings_path* is given, the server-tier settings file path is
        redirected to that location first.  The CLI uses this to point at a
        run-specific settings JSON without each call site importing
        :mod:`vtsearch.settings` directly.
        """
        from vtsearch import settings as _settings  # noqa: PLC0415

        if settings_path is not None:
            _settings.set_settings_path(settings_path)

        return cls(
            saved_datasets_dir=_settings.get_saved_datasets_dir(),
            detectors_dir=_settings.get_detectors_dir(),
            max_concurrent_dataset_downloads=_settings.get_max_concurrent_dataset_downloads(),
            max_concurrent_dataset_embeddings=_settings.get_max_concurrent_dataset_embeddings(),
            autorun_detectors=tuple(_settings.get_autorun_detectors()),
            safe_thresholds=_settings.get_safe_thresholds(),
            calibrate_count=_settings.get_calibrate_count(),
            calibration_fraction=_settings.get_calibration_fraction(),
            enrich_descriptions=_settings.get_enrich_descriptions(),
            autopilot_goal_diversity=_settings.get_autopilot_goal_diversity(),
            inclusion=_settings.get_inclusion(),
            data_dir=DATA_DIR,
        )
