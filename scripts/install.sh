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
# "GPU present" means nvidia-smi can actually see the card. If the GPU is
# physically present (an NVIDIA PCI device) but nvidia-smi can't -- the usual
# state on a fresh cloud GPU box like an AWS g4dn whose NVIDIA driver isn't
# installed yet -- auto mode PAUSES and asks whether to install CPU-only torch
# now or stop to install the driver, instead of silently landing on CPU. On a
# non-interactive shell it stops (set VTSEARCH_ASSUME_CPU=1 to proceed with CPU
# unattended). An explicit 'cpu'/'gpu'/'cuXYZ' argument skips the check.
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

# True (exit 0) when an NVIDIA GPU is PHYSICALLY present, regardless of whether
# the driver is installed. This is the key distinction nvidia-smi can't make:
# on a fresh cloud GPU instance (e.g. an AWS g4dn with a Tesla T4) the card is
# in the box but the kernel driver isn't installed yet, so nvidia-smi is absent
# and vts_have_gpu() above returns false -- which would silently install the
# CPU wheels even though the user wants GPU. We detect the hardware itself via
# the PCI vendor ID 0x10de (NVIDIA) on a display/3D controller (class 0x03xxxx).
# sysfs is read directly because it's always present on Linux, needs no driver,
# and needs no lspci; lspci is only a last-resort fallback when sysfs is empty.
vts_nvidia_hardware_present() {
    local dev vendor class
    for dev in /sys/bus/pci/devices/*; do
        [ -r "$dev/vendor" ] && [ -r "$dev/class" ] || continue
        read -r vendor < "$dev/vendor" || continue
        [ "$vendor" = "0x10de" ] || continue
        read -r class < "$dev/class" || continue
        case "$class" in 0x03*) return 0 ;; esac  # 0x03xxxx = display controller
    done
    # Fallback for non-Linux / unusual sysfs layouts: ask lspci if it exists.
    if command -v lspci >/dev/null 2>&1; then
        lspci 2>/dev/null | grep -iE 'NVIDIA.*(VGA|3D|Display)' >/dev/null 2>&1 && return 0
    fi
    return 1
}

# Print actionable guidance when the GPU is physically present but unusable
# because the driver is missing (nvidia-smi absent / not listing a device).
# Installing the kernel driver is a system-level action pip can't do.
vts_report_driver_missing() {
    cat >&2 <<'EOF'
NOTICE: An NVIDIA GPU is physically present, but no usable driver was found
        (nvidia-smi is absent or lists no device), so the GPU cannot be used yet.

This is the common state on a fresh cloud GPU instance (e.g. an AWS g4dn with a
Tesla T4) booted from a base AMI: the card is attached but the NVIDIA kernel
driver isn't installed. pip cannot fix this -- the driver is a system package.

To get GPU acceleration, install the NVIDIA driver, then re-run this script:

  # Ubuntu / Debian:
  sudo apt-get update && sudo apt-get install -y nvidia-driver-535   # or newer
  sudo reboot   # if nvidia-smi still fails after install

  # Amazon Linux / RHEL: install the driver per AWS's GPU-instance guide:
  #   https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
  # Or use the AWS Deep Learning AMI, which ships the driver preinstalled.

Verify the driver is live (lists your GPU and a CUDA version) with:

  nvidia-smi
EOF
}

# The GPU is in the box but the driver is missing. Rather than silently install
# CPU wheels (the user has GPU hardware and probably wants it) OR hard-abort
# (forcing a re-run for someone who's fine with CPU), pause and ask: continue
# with a CPU-only install now, or stop to install the driver and re-run.
#
# Returns 0 to proceed with the CPU install, non-zero to stop. On a
# non-interactive shell (no TTY: CI, `curl ... | bash`, Docker build) there's no
# one to prompt, so we default to STOP -- the notice above explains how to fix
# the driver or force CPU explicitly with `bash scripts/install.sh cpu`. Set
# VTSEARCH_ASSUME_CPU=1 to skip the prompt and proceed with CPU unattended.
vts_prompt_driver_missing() {
    vts_report_driver_missing

    if [ "${VTSEARCH_ASSUME_CPU:-0}" = "1" ]; then
        echo "" >&2
        echo "VTSEARCH_ASSUME_CPU=1 set -> proceeding with a CPU-only install." >&2
        return 0
    fi

    if [ ! -t 0 ]; then
        echo "" >&2
        echo "Non-interactive shell; not prompting. Stopping so this doesn't" >&2
        echo "silently land on CPU. Re-run with 'cpu' to force a CPU install," >&2
        echo "or set VTSEARCH_ASSUME_CPU=1 to proceed with CPU unattended." >&2
        return 1
    fi

    local reply
    printf '\nInstall CPU-only torch now instead? You will NOT get GPU acceleration. [y/N] ' >&2
    read -r reply
    case "$reply" in
        [yY] | [yY][eE][sS]) return 0 ;;  # proceed with the CPU install
        *) return 1 ;;                    # stop so the user can fix the driver
    esac
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

# --- optional cuML / RAPIDS accelerator (best-effort) ------------------------
# cuML powers GPU UMAP (VTSBrowse projection) and GPU k-means (diversity tree);
# VTSearch auto-detects it at runtime and falls back to the CPU umap-learn /
# scikit-learn paths when it's absent (see vtscore/gpu_backends.py). We install
# it by default on GPU hosts, but as an ISOLATED, NON-FATAL step:
#
#   - It's a multi-gigabyte RAPIDS stack (cudf, rmm, cupy, ...) and lives on a
#     separate index (pypi.nvidia.com), so a slow/unreachable index or a
#     resolver hiccup must NOT abort an otherwise-good GPU install.
#   - The wheel is CUDA-major-pinned: cu11* tags -> cuml-cu11, cu12* -> cuml-cu12.
#   - The cu12 wheel is ALSO capped below RAPIDS 26 (see VTS_CUML_CU12_SPEC).
#     RAPIDS 26.x raised its CUDA floor to require nvidia-nvjitlink-cu12>=12.9,
#     but the torch CUDA wheels we pin (cu124 = CUDA 12.4 .. cu128 = CUDA 12.8)
#     top out at 12.8 and no torch wheel ships 12.9 yet, so an UNpinned
#     cuml-cu12 floats to 26.x and pip dies with an nvjitlink conflict (and
#     cupy then JIT-compiles cuVS/raft kernels against mismatched CUDA-13 fp8
#     headers -> "cuda_fp8.hpp: this declaration has no storage class" at fit
#     time). 25.x declares cuda-toolkit==12.* and resolves cleanly against the
#     pinned torch. Bump the cap once a torch wheel ships the CUDA minor 26.x
#     needs. See docs/DEPLOYMENT.md "cuML crashes compiling a kernel".
#   - RAPIDS ships linux-only wheels for a fixed Python range; on an unsupported
#     platform/Python this step just warns and the app uses the CPU fallback.
#
# Override/skip with VTSEARCH_SKIP_CUML=1 (e.g. air-gapped installs that can't
# reach the NVIDIA index, or when you simply don't want the download).
vts_install_cuml() {
    local cuda_tag="$1"

    if [ "${VTSEARCH_SKIP_CUML:-0}" = "1" ]; then
        echo "  (skipped: VTSEARCH_SKIP_CUML=1; GPU UMAP/k-means will use the CPU fallback)"
        return 0
    fi

    # The cu12 spec is capped below RAPIDS 26 (whose CUDA-12.9 floor outruns the
    # torch CUDA wheels we pin); see the header comment above. Override the cap
    # via VTS_CUML_CU12_SPEC, e.g. once a matching torch wheel exists:
    #   VTS_CUML_CU12_SPEC='cuml-cu12' bash scripts/install.sh cu128
    local cuml_pkg
    case "$cuda_tag" in
        cu11*) cuml_pkg="cuml-cu11" ;;
        *) cuml_pkg="${VTS_CUML_CU12_SPEC:-cuml-cu12<26}" ;;  # cu12x + anything else
    esac

    # Heads-up: cuML depends on NEWER nvidia-*-cu12 runtime wheels than the torch
    # build pins exactly (e.g. cu124 torch pins ==12.4.x), so installing it upgrades
    # those libs and pip prints a red "dependency conflicts" report naming torch's
    # now-unsatisfied pins. This is distinct from the FATAL RAPIDS-26 nvjitlink/fp8
    # break the cuml-cu12<26 cap above prevents -- this one is cosmetic. That is
    # EXPECTED and non-fatal -- pip still completes the install and does not roll
    # anything back, and CUDA 12.x runtimes are compatible across minor versions.
    # The GPU smoke test at the end of this install confirms torch + cuML actually
    # work together, so you don't have to guess whether the red text mattered.
    echo "  Note: pip may print a red 'dependency conflicts' report below (torch pins"
    echo "        older CUDA runtime libs than cuML wants). It is expected and"
    echo "        non-fatal; the final GPU smoke test verifies they coexist."

    # Isolated pass against the NVIDIA index. Guarded so `set -e` can't abort the
    # install on failure; the runtime cuML detection degrades to CPU either way.
    if pip install --extra-index-url https://pypi.nvidia.com \
        --prefer-binary \
        "$cuml_pkg" \
        --progress-bar on; then
        echo "  cuML installed: GPU UMAP + k-means enabled."
    else
        echo "warning: cuML (${cuml_pkg}) install failed; GPU UMAP/k-means will fall" >&2
        echo "         back to the CPU umap-learn / scikit-learn paths. This is" >&2
        echo "         non-fatal. Re-run later with a reachable pypi.nvidia.com, or" >&2
        echo "         set VTSEARCH_SKIP_CUML=1 to silence this step." >&2
    fi
}

# --- post-install GPU smoke test (best-effort) -------------------------------
# After the GPU install, actually exercise the stack so the red pip "dependency
# conflicts" report from the cuML step turns into a definitive pass/fail instead
# of an ambiguous wall of text: import torch, run a tiny CUDA matmul, and import
# cuML. This catches the one case where the conflict WOULD have mattered -- a
# runtime-lib bump that breaks torch's GPU path -- at install time rather than
# leaving a silent landmine. Diagnostic only and non-fatal: VTSearch smoke-tests
# CUDA at runtime and falls back to CPU, so a failure here warns but never aborts.
vts_smoke_test_gpu() {
    local pybin
    pybin="$(command -v python || command -v python3 || true)"
    if [ -z "$pybin" ]; then
        echo "  (skipped: no python on PATH to run the smoke test)"
        return 0
    fi

    local rc=0
    "$pybin" - <<'PY' || rc=$?
import sys

try:
    import torch
except Exception as e:  # noqa: BLE001 - report any import failure verbatim
    print(f"  torch: IMPORT FAILED: {type(e).__name__}: {e}")
    sys.exit(2)

print(f"  torch {torch.__version__}")
gpu_ok = False
if not torch.cuda.is_available():
    print("  torch.cuda.is_available() = False -> VTSearch will run on CPU.")
    print("  (a driver/CUDA-tag mismatch, NOT the cuML dependency warning.)")
else:
    try:
        x = torch.randn(256, 256, device="cuda")
        checksum = (x @ x).sum().item()
        print(f"  GPU op OK on {torch.cuda.get_device_name(0)} (matmul checksum {checksum:.1f})")
        gpu_ok = True
    except Exception as e:  # noqa: BLE001 - report any runtime failure verbatim
        print(f"  GPU op FAILED: {type(e).__name__}: {e}")
        print("  The installed CUDA runtime may be incompatible with this torch build.")
        sys.exit(3)

try:
    import cuml

    print(f"  cuML {getattr(cuml, '__version__', '?')} import OK -> GPU UMAP/k-means enabled.")
except Exception:  # noqa: BLE001 - absence is fine; app uses the CPU fallback
    print("  cuML not importable -> GPU UMAP/k-means will use the CPU fallback.")

sys.exit(0 if gpu_ok else 1)
PY

    case "$rc" in
        0) ;;  # GPU verified end-to-end; the python output already said so.
        1) ;;  # torch fine but no usable CUDA -> CPU mode; python explained why.
        *)
            echo "warning: GPU smoke test did not pass (see above). The install may" >&2
            echo "         still be usable -- VTSearch falls back to CPU automatically --" >&2
            echo "         but you will NOT get GPU acceleration. If you expected it," >&2
            echo "         re-check the driver and the CUDA tag (bash scripts/install.sh cuXYZ)." >&2
            ;;
    esac
    return 0
}

# --- GPU install -------------------------------------------------------------
# The PyTorch extra-index (download.pytorch.org/whl/cu*) sometimes serves source
# tarballs for packages like numpy and scipy, so we pre-install them as
# binary-only wheels before the full requirements pass, avoiding the need for a
# C++ compiler.
vts_install_gpu() {
    local cuda_tag="$1"
    local extra_index="https://download.pytorch.org/whl/${cuda_tag}"

    vts_progress_init 8 "Installing VTSearch GPU dependencies (CUDA tag: ${cuda_tag})"

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

    vts_progress_step "Installing optional cuML/RAPIDS accelerator (best-effort; CUDA tag: ${cuda_tag})"
    vts_install_cuml "$cuda_tag"

    vts_progress_step "Verifying the GPU stack (torch CUDA op + cuML import)"
    vts_smoke_test_gpu

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
        elif vts_nvidia_hardware_present; then
            # Card is in the box but nvidia-smi can't see it -> driver missing.
            # Pause and ask rather than silently installing CPU: continue with a
            # CPU-only install now, or stop to install the driver and re-run.
            if vts_prompt_driver_missing; then
                echo "Proceeding with a CPU-only install." >&2
                vts_install_cpu
            else
                echo "Stopping. Install the NVIDIA driver (see above) and re-run, or" >&2
                echo "run 'bash scripts/install.sh cpu' to install CPU-only torch." >&2
                exit 1
            fi
        else
            echo "No NVIDIA GPU detected (no NVIDIA PCI device found) -> installing the CPU dependency set." >&2
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
