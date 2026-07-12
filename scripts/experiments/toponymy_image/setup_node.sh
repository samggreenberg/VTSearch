#!/bin/bash
# Build the experiment venv on node-local scratch (run inside the GPU job).
#
#   srun --jobid=<JOBID> --overlap bash scripts/experiments/toponymy_image/setup_node.sh
#
# Heavy state (venv, HF models, datasets) lives on /scratch/$USER — node-local,
# so later steps must run on the same node. Small durable outputs go to
# $TOPO_RESULTS (default /exp/$USER/experiments/toponymy-image/results).
set -euo pipefail

WORK=${TOPO_WORK:-/scratch/jobs/$USER/topo-image}
VENV=$WORK/venv
UV=${UV_BIN:-$HOME/.local/bin/uv}
export UV_CACHE_DIR=$WORK/uv-cache
export UV_PYTHON_INSTALL_DIR=$WORK/uv-python

mkdir -p "$WORK" "$WORK/vts-models" "$WORK/hf"

# Reuse the SigLIP checkpoint already cached by the VTSearch deployment.
for src in /exp/sgreenberg/projects/VTSearch/data/models/models--google--siglip*; do
    [ -e "$src" ] || continue
    [ -e "$WORK/vts-models/$(basename "$src")" ] || ln -s "$src" "$WORK/vts-models/" 2>/dev/null || true
done

if [ ! -x "$VENV/bin/python" ]; then
    "$UV" venv --python 3.12 "$VENV"
fi

# transformers is left to toponymy's resolution (it pins <5, currently 4.57.x
# — which natively supports BLIP and Qwen2.5-VL; Florence-2 runs via
# trust_remote_code). timm + einops are Florence-2 remote-code deps;
# qwen-vl-utils is the Qwen2.5-VL vision helper; sentence-transformers is the
# neutral name-quality encoder in evaluate.py.
"$UV" pip install --python "$VENV/bin/python" \
    toponymy==0.5.2 \
    torch \
    accelerate \
    numpy requests tqdm scikit-learn threadpoolctl umap-learn \
    huggingface_hub pandas pyarrow pyyaml matplotlib scipy \
    sentencepiece protobuf Pillow pydantic \
    sentence-transformers qwen-vl-utils timm einops

"$VENV/bin/python" - <<'EOF'
import torch, transformers, toponymy, umap, PIL  # noqa
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("toponymy ok, pillow", PIL.__version__)
EOF
echo "venv ready at $VENV"
