"""Pre-download SigLIP weights into ``$VTSEARCH_MODELS_DIR``.

Used by ``docker/Dockerfile.labbench`` to bake weights into the image. Uses
``huggingface_hub.snapshot_download`` rather than
``SiglipModel.from_pretrained`` so the build step only downloads files -
it does not import torch, does not construct the model, and does not
allocate a CPU copy of the weights. That makes the step substantially
faster, uses far less RAM (the build no longer needs to hold the full
model in memory), and surfaces per-file tqdm progress so the BuildKit
log never looks like a 30-minute hang.

Run on the host to debug:
    VTSEARCH_MODELS_DIR=/tmp/m python -u scripts/preload_siglip.py
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download

from vtscore.config import SIGLIP_MODEL_ID


def main() -> int:
    cache_dir = os.environ.get("VTSEARCH_MODELS_DIR")
    if not cache_dir:
        print("VTSEARCH_MODELS_DIR is not set", file=sys.stderr, flush=True)
        return 1

    print(f"[preload-siglip] repo={SIGLIP_MODEL_ID} cache={cache_dir}", flush=True)
    # SigLIP ships safetensors AND pytorch_model.bin; transformers prefers
    # safetensors, so skip the duplicate .bin (~370 MB) plus other framework
    # formats we never load.
    path = snapshot_download(
        repo_id=SIGLIP_MODEL_ID,
        cache_dir=cache_dir,
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.ot", "*.onnx"],
        max_workers=4,
    )
    print(f"[preload-siglip] cached at {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
