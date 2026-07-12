#!/bin/bash
# Build the experiment venv on node-local scratch (run inside the GPU job).
#
#   srun --jobid=<JOBID> --overlap bash scripts/experiments/toponymy_audio/setup_node.sh
#
# Heavy state (venv, HF models, datasets) lives on /scratch/$USER — node-local,
# so later steps must run on the same node. Small durable outputs go to
# $TOPO_RESULTS (default /exp/$USER/experiments/toponymy-audio/results).
set -euo pipefail

WORK=${TOPO_WORK:-/scratch/jobs/$USER/topo-audio}
VENV=$WORK/venv
UV=${UV_BIN:-$HOME/.local/bin/uv}
export UV_CACHE_DIR=$WORK/uv-cache
export UV_PYTHON_INSTALL_DIR=$WORK/uv-python

mkdir -p "$WORK" "$WORK/vts-models" "$WORK/hf"

# Reuse the CLAP checkpoint already cached by the VTSearch deployment.
CLAP_SRC=/exp/sgreenberg/projects/VTSearch/data/models/models--laion--clap-htsat-unfused
[ -e "$WORK/vts-models/models--laion--clap-htsat-unfused" ] || ln -s "$CLAP_SRC" "$WORK/vts-models/" 2>/dev/null || true

if [ ! -x "$VENV/bin/python" ]; then
    "$UV" venv --python 3.12 "$VENV"
fi

# transformers is left to toponymy's resolution (it pins <5, currently 4.57.x —
# one reason NOT to install toponymy into the app venv; see the report).
"$UV" pip install --python "$VENV/bin/python" \
    toponymy==0.5.2 \
    torch \
    accelerate \
    openai-whisper \
    numpy requests tqdm scikit-learn threadpoolctl umap-learn \
    huggingface_hub pandas pyarrow pyyaml matplotlib scipy \
    librosa soundfile sentencepiece protobuf Pillow py7zr pydantic

"$VENV/bin/python" - <<'EOF'
import torch, transformers, toponymy, umap, whisper, librosa  # noqa
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("toponymy", toponymy.__version__ if hasattr(toponymy, "__version__") else "0.5.2")
EOF
echo "venv ready at $VENV"
