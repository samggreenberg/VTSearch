"""Configuration and constants for VTSearch."""

import os
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
# ``TRAIN_EPOCHS`` is an *upper bound* — :func:`vtsearch.models.training.train_model`
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
