#!/bin/bash
set -euo pipefail

# Unified dependency installer for VTSearch.
#
# ONE script for both CPU and GPU installs. With no argument it AUTO-DETECTS
# whether this host has a usable NVIDIA GPU (via nvidia-smi) and installs the
# matching dependency set, so you don't have to know -- or tell it -- what
# hardware you have:
#
#   - GPU present -> CUDA torch, with the right cuXYZ wheel tag auto-selected
#     from the GPU's compute capability (scripts/detect_cuda_tag.py).
#   - No GPU      -> the smaller CPU-only torch wheel (~200 MB vs ~2 GB).
#
# Override the auto-detection by passing an argument:
#   bash scripts/install.sh            # auto-detect CPU vs GPU (recommended)
#   bash scripts/install.sh cpu        # force the CPU install
#   bash scripts/install.sh gpu        # force the GPU install (auto-detect CUDA tag)
#   bash scripts/install.sh cu118      # force the GPU install with an explicit CUDA tag
#   bash scripts/install.sh cu121      # ...
#   bash scripts/install.sh cu124      # for CUDA 12.4 (V100/Volta, A100, H100)
#   bash scripts/install.sh cu128      # for CUDA 12.8 (Blackwell; drops Volta)
#
# About the CUDA tag (only relevant to GPU installs): it selects which prebuilt
# torch wheel you get, and each wheel only ships kernel images for a fixed set
# of GPU architectures. There is a floor AND a ceiling. Newer GPUs need newer
# tags: Ampere/Ada work on cu118+, Hopper (H100) on cu121+, Blackwell (B100/B200,
# RTX 50xx) on cu128+. But the newest wheels also DROP the oldest architectures,
# so "just use the latest tag" is wrong for old hardware: cu128 dropped Volta
# (sm_70), so a Tesla V100 needs cu124 (or cu121/cu118), NOT cu128. Rule of
# thumb: use the oldest tag your driver supports that still covers your GPU.
# cu124 is a safe default that spans Volta through Hopper, and what the
# auto-detect picks for those cards.
#
# (VTSearch also smoke-tests CUDA at runtime and falls back to CPU if the
# installed wheel can't run on the GPU, so a mismatch degrades instead of
# crashing - but you only get GPU acceleration with a matching wheel.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Shared progress-printing helpers.
# shellcheck source=_progress.sh
source "$SCRIPT_DIR/_progress.sh"

# --- GPU presence ------------------------------------------------------------
# True (exit 0) when nvidia-smi exists and lists at least one GPU. This is the
# CPU-vs-GPU decision; the finer cuXYZ tag choice is detect_cuda_tag.py's job.
vts_have_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi -L >/dev/null 2>&1 || return 1
    [ -n "$(nvidia-smi -L 2>/dev/null)" ]
}

# --- CUDA tag resolution -----------------------------------------------------
# Echo a cuXYZ tag for the GPU. An explicit arg wins; otherwise auto-detect via
# detect_cuda_tag.py (which prints its reasoning to stderr -> the terminal), and
# fall back to the safe Volta..Hopper default if it can't tell.
vts_resolve_cuda_tag() {
    local explicit="${1:-}"
    if [ -n "$explicit" ]; then
        echo "$explicit"
        return 0
    fi
    local pybin detected
    pybin="$(command -v python || command -v python3 || true)"
    if [ -n "$pybin" ] && detected="$("$pybin" "$SCRIPT_DIR/detect_cuda_tag.py")"; then
        echo "$detected"
    else
        echo "  (auto-detect failed; falling back to default tag cu124)" >&2
        echo "cu124"
    fi
}

# --- pre-commit wiring (shared) ----------------------------------------------
vts_install_precommit() {
    if [ -d "$REPO_ROOT/.git" ] && [ -f "$REPO_ROOT/.pre-commit-config.yaml" ]; then
        (cd "$REPO_ROOT" && pre-commit install --install-hooks) || \
            echo "warning: pre-commit install failed; run it manually to enable git hooks"
    else
        echo "  (skipped: no git checkout or .pre-commit-config.yaml)"
    fi
}

# --- CPU install -------------------------------------------------------------
# Installs runtime + dev dependencies and the vtsearch package itself (editable)
# by forwarding to pyproject.toml via requirements/base.txt, which is just
# `--extra-index-url <cpu wheel index>` + `-e .[dev]`.
vts_install_cpu() {
    vts_progress_init 4 "Installing VTSearch CPU dependencies"

    vts_progress_step "Checking Python version (>= 3.10)"
    # shellcheck source=_check-python.sh
    source "$SCRIPT_DIR/_check-python.sh"

    vts_progress_step "Upgrading pip / setuptools / wheel"
    pip install --upgrade pip "setuptools<82" wheel --progress-bar on

    vts_progress_step "Installing runtime + dev dependencies (this may take several minutes)"
    pip install -r "$REPO_ROOT/requirements/base.txt" --progress-bar on

    vts_progress_step "Wiring up pre-commit git hook"
    vts_install_precommit

    vts_progress_done "CPU dependencies installed successfully"
}

# --- GPU install -------------------------------------------------------------
# The PyTorch extra-index (download.pytorch.org/whl/cu*) sometimes serves source
# tarballs for packages like numpy and scipy, so we pre-install them as
# binary-only wheels before the full requirements pass, avoiding the need for a
# C++ compiler.
vts_install_gpu() {
    local cuda_tag="$1"
    local extra_index="https://download.pytorch.org/whl/${cuda_tag}"

    vts_progress_init 6 "Installing VTSearch GPU dependencies (CUDA tag: ${cuda_tag})"

    vts_progress_step "Checking Python version (>= 3.10)"
    # shellcheck source=_check-python.sh
    source "$SCRIPT_DIR/_check-python.sh"

    vts_progress_step "Upgrading pip / setuptools / wheel"
    pip install --upgrade pip "setuptools<82" wheel --progress-bar on

    vts_progress_step "Pre-installing binary-only wheels (numpy, scipy) from PyPI"
    pip install --only-binary :all: \
      --index-url https://pypi.org/simple \
      "numpy" \
      "scipy" \
      --progress-bar on

    # Pin torch/torchvision/torchaudio to the chosen CUDA index with --index-url
    # (NOT --extra-index-url). With --extra-index-url, PyPI stays in the candidate
    # set, and when PyPI ships a *newer* torch than this CUDA index tops out at
    # (e.g. cu124 caps at 2.6.0 while PyPI has 2.7.x, a cu126 build), pip prefers
    # the higher version and silently installs the wrong-arch wheel -- which then
    # fails with cudaErrorNoKernelImageForDevice on older GPUs. Installing from the
    # CUDA index alone forces the matching +${cuda_tag} build; the PyTorch index
    # mirrors torch's dependency closure, so a sole --index-url resolves cleanly.
    vts_progress_step "Installing CUDA torch from ${cuda_tag} index (pinned so PyPI can't substitute a mismatched build)"
    pip install --index-url "$extra_index" \
      --prefer-binary \
      torch torchvision torchaudio \
      --progress-bar on

    # Install everything else. torch is already satisfied by the pinned build above,
    # so this --extra-index-url pass won't replace it (no --upgrade).
    vts_progress_step "Installing remaining dependencies via ${extra_index} (this may take several minutes)"
    pip install --extra-index-url "$extra_index" \
      --prefer-binary \
      -r "$REPO_ROOT/requirements/gpu.txt" \
      --progress-bar on

    vts_progress_step "Wiring up pre-commit git hook"
    vts_install_precommit

    vts_progress_done "GPU dependencies installed successfully"
}

# --- dispatch ----------------------------------------------------------------
MODE="${1:-auto}"
case "$MODE" in
    cpu)
        vts_install_cpu
        ;;
    gpu)
        vts_install_gpu "$(vts_resolve_cuda_tag)"
        ;;
    cu*)
        vts_install_gpu "$(vts_resolve_cuda_tag "$MODE")"
        ;;
    auto|"")
        if vts_have_gpu; then
            echo "Detected an NVIDIA GPU -> installing the GPU dependency set." >&2
            vts_install_gpu "$(vts_resolve_cuda_tag)"
        else
            echo "No NVIDIA GPU detected (nvidia-smi absent or lists no device) -> installing the CPU dependency set." >&2
            vts_install_cpu
        fi
        ;;
    *)
        cat >&2 <<EOF
ERROR: unknown argument '$MODE'.

Usage:
  bash scripts/install.sh            # auto-detect CPU vs GPU
  bash scripts/install.sh cpu        # force CPU install
  bash scripts/install.sh gpu        # force GPU install (auto-detect CUDA tag)
  bash scripts/install.sh cu124      # force GPU install with an explicit CUDA tag
EOF
        exit 2
        ;;
esac
