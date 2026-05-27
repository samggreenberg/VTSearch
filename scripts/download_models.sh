#!/usr/bin/env bash
# Pre-download all VTSearch embedding models for offline use.
#
# Usage:
#   ./scripts/download_models.sh [CACHE_DIR]
#
# CACHE_DIR defaults to $VTSEARCH_MODELS_DIR if set, otherwise data/models.
#
# After downloading, point the app at the cache folder:
#   export VTSEARCH_MODELS_DIR=/path/to/cache
#   python app.py
#
# Or just run this script with no arguments and the default data/models
# folder will be used (same place the app looks by default).

set -euo pipefail

CACHE_DIR="${1:-${VTSEARCH_MODELS_DIR:-data/models}}"

echo "Downloading all VTSearch embedding models to: $CACHE_DIR"
mkdir -p "$CACHE_DIR"

python3 - "$CACHE_DIR" <<'PYEOF'
import sys

cache_dir = sys.argv[1]

# ------------------------------------------------------------------
# 1. CLAP  (audio):  laion/clap-htsat-unfused
# ------------------------------------------------------------------
print("\n[1/5] Downloading CLAP model (laion/clap-htsat-unfused) ...")
from transformers import ClapModel, ClapProcessor

ClapModel.from_pretrained("laion/clap-htsat-unfused", cache_dir=cache_dir)
ClapProcessor.from_pretrained("laion/clap-htsat-unfused", cache_dir=cache_dir)
print("  CLAP done.")

# ------------------------------------------------------------------
# 2. SigLIP  (image, default):  google/siglip-base-patch16-224
# ------------------------------------------------------------------
print("\n[2/5] Downloading SigLIP model (google/siglip-base-patch16-224) ...")
from transformers import SiglipModel, SiglipImageProcessor, SiglipTokenizer

SiglipModel.from_pretrained("google/siglip-base-patch16-224", cache_dir=cache_dir)
SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224", cache_dir=cache_dir)
SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224", cache_dir=cache_dir)
print("  SigLIP done.")

# ------------------------------------------------------------------
# 3. CLIP  (image, alternative):  openai/clip-vit-base-patch32
# ------------------------------------------------------------------
print("\n[3/5] Downloading CLIP model (openai/clip-vit-base-patch32) ...")
from transformers import CLIPModel, CLIPProcessor

CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
print("  CLIP done.")

# ------------------------------------------------------------------
# 4. X-CLIP  (video):  microsoft/xclip-base-patch32
# ------------------------------------------------------------------
print("\n[4/5] Downloading X-CLIP model (microsoft/xclip-base-patch32) ...")
from transformers import XCLIPModel, XCLIPProcessor

XCLIPModel.from_pretrained("microsoft/xclip-base-patch32", cache_dir=cache_dir)
XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32", cache_dir=cache_dir)
print("  X-CLIP done.")

# ------------------------------------------------------------------
# 5. E5  (text):  intfloat/e5-base-v2
# ------------------------------------------------------------------
print("\n[5/5] Downloading E5 model (intfloat/e5-base-v2) ...")
from sentence_transformers import SentenceTransformer

SentenceTransformer("intfloat/e5-base-v2", cache_folder=cache_dir)
print("  E5 done.")

print("\nAll models downloaded successfully to:", cache_dir)
print("To use offline, set HF_HUB_OFFLINE=1 and (if needed) VTSEARCH_MODELS_DIR=" + cache_dir)
PYEOF
