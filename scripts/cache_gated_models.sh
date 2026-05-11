#!/usr/bin/env bash
# Cache gated Hugging Face model weights to a local directory **once**, so that
# Dockerfile.image-embedders can bake them into the image without ever needing
# an HF token during ``docker build``.
#
# The gated models are:
#   * DINOv3  (facebook/dinov3-vitb16-pretrain-lvd1689m) — Meta research licence
#   * EUPE    (facebook/PE-Core-B16-224)                 — Meta research licence
#
# Both require you to:
#   1. Have a Hugging Face account.
#   2. Visit each model page and click "Agree and access" to accept the licence.
#   3. Create a User Access Token at https://huggingface.co/settings/tokens
#      (a read-only token is enough).
#
# Run this script **once** on the host, with your token available either as an
# env var or via a prior ``huggingface-cli login``. The script downloads
# everything into ``./model_cache/`` (or any directory you pass as $1). After
# that, ``docker build`` reads from the cache and never needs your token again.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/cache_gated_models.sh
#   # or, after `huggingface-cli login`:
#   ./scripts/cache_gated_models.sh
#   # or, custom output dir:
#   ./scripts/cache_gated_models.sh /path/to/cache
#
# The cache directory is ``./model_cache/`` by default and is gitignored —
# weights never end up in the repo or in a Docker image layer except via the
# explicit COPY in Dockerfile.image-embedders.

set -euo pipefail

CACHE_DIR="${1:-./model_cache}"
mkdir -p "$CACHE_DIR"

echo "Caching gated VTSearch image-embedder weights to: $CACHE_DIR"
echo

# Resolve a token from the env, but don't echo it. If HF_TOKEN isn't set,
# huggingface_hub falls back to the credential saved by ``huggingface-cli
# login`` — both paths use the same underlying loader.
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Using HF token from \$HF_TOKEN env var."
elif [[ -f "$HOME/.cache/huggingface/token" ]] || [[ -f "$HOME/.huggingface/token" ]]; then
    echo "Using HF token from cached huggingface-cli login."
else
    cat <<'EOF' >&2
ERROR: No Hugging Face credentials found.

Either set HF_TOKEN in your environment, e.g.:
    export HF_TOKEN=hf_xxx
    ./scripts/cache_gated_models.sh

Or log in interactively first:
    huggingface-cli login
    ./scripts/cache_gated_models.sh

You also need to visit each model page once and click "Agree and access":
    https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
    https://huggingface.co/facebook/PE-Core-B16-224
EOF
    exit 1
fi
echo

python3 - "$CACHE_DIR" <<'PYEOF'
import os
import sys

cache_dir = sys.argv[1]
# ``token=True`` tells huggingface_hub to use whichever token the user
# previously set (HF_TOKEN env var or ``huggingface-cli login``), without
# us having to splice it into a string anywhere.
token = os.environ.get("HF_TOKEN") or True

print("[1/2] Downloading DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)...", flush=True)
from transformers import AutoImageProcessor, AutoModel

from vtsearch.config import DINOV3_MODEL_ID

AutoModel.from_pretrained(DINOV3_MODEL_ID, cache_dir=cache_dir, token=token)
AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID, cache_dir=cache_dir, token=token)
print("  DINOv3 cached.", flush=True)

print("[2/2] Downloading EUPE (facebook/PE-Core-B16-224)...", flush=True)
from vtsearch.config import EUPE_MODEL_ID

AutoModel.from_pretrained(
    EUPE_MODEL_ID, cache_dir=cache_dir, token=token, trust_remote_code=True
)
AutoImageProcessor.from_pretrained(
    EUPE_MODEL_ID, cache_dir=cache_dir, token=token, trust_remote_code=True
)
print("  EUPE cached.", flush=True)
PYEOF

echo
echo "Done. Now rebuild the image — no HF token needed:"
echo "    docker build -f Dockerfile.image-embedders -t vtsearch:image-embedders ."
