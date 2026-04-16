"""Configuration and constants for VTSearch."""

import os
from pathlib import Path

# Data paths are anchored to the repository root, NOT to the current working
# directory.  Without this, starting the app from a different CWD (systemd,
# cron, dev shell) would create a fresh empty `data/` and silently lose the
# user's existing datasets, settings, and embeddings.  Override with the
# ``VTSEARCH_DATA_DIR`` env var if you need to relocate state outside the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (
    Path(os.environ["VTSEARCH_DATA_DIR"]) if "VTSEARCH_DATA_DIR" in os.environ else _REPO_ROOT / "data"
)
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_CACHE_DIR = (
    Path(os.environ["VTSEARCH_MODELS_DIR"]) if "VTSEARCH_MODELS_DIR" in os.environ else DATA_DIR / "models"
)

# Training
TRAIN_EPOCHS = 200
MLP_HIDDEN_MIN = 4
MLP_HIDDEN_MAX = 32
MLP_DROPOUT = 0.5

# Model IDs
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000  # CLAP model expected input sample rate
XCLIP_MODEL_ID = "microsoft/xclip-base-patch32"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
E5_MODEL_ID = "intfloat/e5-base-v2"
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
CLAP_MUSIC_MODEL_ID = "laion/larger_clap_music_and_speech"
BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"
LANGUAGEBIND_VIDEO_MODEL_ID = "LanguageBind/LanguageBind_Video_V1.5_FT"
