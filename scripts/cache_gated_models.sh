#!/usr/bin/env bash
# Cache image-embedder model weights to a local directory **once**, so that
# docker/Dockerfile.image-embedders can bake them into the image without ever
# needing an HF token during ``docker build``.
#
# Two embedders need a host-side cache step:
#
#   * DINOv3  (facebook/dinov3-vitb16-pretrain-lvd1689m) — gated under Meta's
#     research licence; you must accept it on HF and provide an HF token here.
#   * EUPE    (facebookresearch/EUPE, ViT-B/16 LVD-1689M weights mirrored on
#     facebook/EUPE-ViT-B) — the HF mirror is ungated and needs no token, but
#     loading still requires cloning the EUPE repo via ``torch.hub`` and
#     downloading the .pt weights file. Doing that once on the host means
#     ``docker build`` can run fully offline.
#
# For DINOv3 only, you must:
#   1. Have a Hugging Face account.
#   2. Visit https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
#      and click "Agree and access" to accept the licence.
#   3. Create a User Access Token at https://huggingface.co/settings/tokens
#      (a read-only token is enough).
#
# Run this script **once** on the host. The script downloads everything into
# ``./model_cache/`` (or any directory you pass as $1):
#   * DINOv3 lands under ``$CACHE/models--facebook--dinov3-...``
#     (the standard HuggingFace cache layout).
#   * EUPE lands under ``$CACHE/hub/`` (torch.hub layout: cloned repo +
#     ``checkpoints/EUPE-ViT-B.pt``).
# After that, ``docker build`` reads from the cache and never needs your
# token (or the network) again.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/cache_gated_models.sh
#   # or, after `huggingface-cli login`:
#   ./scripts/cache_gated_models.sh
#   # or, custom output dir:
#   ./scripts/cache_gated_models.sh /path/to/cache
#
# If you only want EUPE (no HF token, no licence acceptance), run with
# ``SKIP_DINOV3=1``:
#   SKIP_DINOV3=1 ./scripts/cache_gated_models.sh
#
# The cache directory is ``./model_cache/`` by default and is gitignored —
# weights never end up in the repo or in a Docker image layer except via the
# explicit COPY in docker/Dockerfile.image-embedders.

set -euo pipefail

CACHE_DIR="${1:-./model_cache}"
mkdir -p "$CACHE_DIR"

echo "Caching VTSearch image-embedder weights to: $CACHE_DIR"
echo

SKIP_DINOV3="${SKIP_DINOV3:-0}"

if [[ "$SKIP_DINOV3" != "1" ]]; then
    # Resolve a token from the env, but don't echo it. If HF_TOKEN isn't
    # set, huggingface_hub falls back to the credential saved by
    # ``huggingface-cli login`` — both paths use the same underlying loader.
    if [[ -n "${HF_TOKEN:-}" ]]; then
        echo "Using HF token from \$HF_TOKEN env var (DINOv3)."
    elif [[ -f "$HOME/.cache/huggingface/token" ]] || [[ -f "$HOME/.huggingface/token" ]]; then
        echo "Using HF token from cached huggingface-cli login (DINOv3)."
    else
        cat <<'EOF' >&2
ERROR: No Hugging Face credentials found (needed for DINOv3).

Either set HF_TOKEN in your environment, e.g.:
    export HF_TOKEN=hf_xxx
    ./scripts/cache_gated_models.sh

Or log in interactively first:
    huggingface-cli login
    ./scripts/cache_gated_models.sh

You also need to visit the DINOv3 model page once and click "Agree and access":
    https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

If you don't want DINOv3, re-run with SKIP_DINOV3=1 to cache only EUPE:
    SKIP_DINOV3=1 ./scripts/cache_gated_models.sh
EOF
        exit 1
    fi
    echo
fi

export VTSEARCH_CACHE_DIR="$CACHE_DIR"
export VTSEARCH_SKIP_DINOV3="$SKIP_DINOV3"

python3 - <<'PYEOF'
import os
import sys

cache_dir = os.environ["VTSEARCH_CACHE_DIR"]
skip_dinov3 = os.environ.get("VTSEARCH_SKIP_DINOV3", "0") == "1"

steps = []
if not skip_dinov3:
    steps.append("DINOv3")
steps.append("EUPE")
total = len(steps)
step_no = 0

if not skip_dinov3:
    step_no += 1
    print(
        f"[{step_no}/{total}] Downloading DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)...",
        flush=True,
    )
    # ``token=True`` tells huggingface_hub to use whichever token the user
    # previously set (HF_TOKEN env var or ``huggingface-cli login``),
    # without us having to splice it into a string anywhere.
    token = os.environ.get("HF_TOKEN") or True
    from transformers import AutoImageProcessor, AutoModel

    from vtscore.config import DINOV3_MODEL_ID

    AutoModel.from_pretrained(DINOV3_MODEL_ID, cache_dir=cache_dir, token=token)
    AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID, cache_dir=cache_dir, token=token)
    print("  DINOv3 cached.", flush=True)

step_no += 1
print(
    f"[{step_no}/{total}] Downloading EUPE (facebookresearch/EUPE, ViT-B/16)...",
    flush=True,
)
# EUPE loads via torch.hub: a cloned GitHub repo plus a downloaded .pt
# weights file. Both land under $TORCH_HOME/hub/, so we point TORCH_HOME
# at the same directory the Dockerfile will COPY into the image.
os.environ["TORCH_HOME"] = cache_dir

import torch  # noqa: E402

from vtscore.config import EUPE_MODEL_ID  # noqa: E402

torch.hub.load(
    "facebookresearch/EUPE",
    "eupe_vitb16",
    source="github",
    pretrained=True,
    weights=EUPE_MODEL_ID,
    trust_repo=True,
)
print("  EUPE cached.", flush=True)
PYEOF

echo
echo "Done. Now rebuild the image — no HF token needed:"
echo "    docker build -f docker/Dockerfile.image-embedders -t vtsearch:image-embedders ."
