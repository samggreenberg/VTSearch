"""Compute-device resolution and embedding precision.

Two knobs that both answer "how does the forward pass run on this host":
:data:`DEVICE` / :func:`resolve_device` pick *where*, and
:data:`EMBED_PRECISION` / :func:`embed_precision` pick *in what dtype*.  They
share a module because the precision resolver consults the resolved device (bf16
needs sm_80+, half is CUDA-only), and because both import torch lazily so that
importing :mod:`vtscore.config` stays torch-free.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is imported lazily everywhere below, never at module scope
    import torch

# Preferred compute device for embedding and training.  ``"auto"`` resolves
# to ``"cuda"`` when a GPU is visible to PyTorch and ``"cpu"`` otherwise;
# explicit values like ``"cuda"``, ``"cuda:0"``, ``"cpu"``, or ``"mps"`` are
# passed through unchanged.  Resolution is lazy; the env var stores the
# user's intent, ``resolve_device()`` actually imports torch when called.
# Embedders (via ``to_compute_device``/``resolve_device``) and MLP
# training/scoring both honour this, so a visible, usable GPU is used for
# embedding and learned-sort automatically; CPU remains the safe fallback.
DEVICE = os.environ.get("VTSEARCH_DEVICE", "auto").lower()


# Per-device cache for the CUDA smoke-test below.  ``None`` until probed;
# the probe runs once per device string per process and is cheap thereafter.
_cuda_runnable: dict[str, bool] = {}


def _describe_cuda_mismatch(device: str) -> str:
    """Best-effort diagnostic suffix naming the GPU and the build's arch list.

    Returns a string like ``" (Tesla V100S-PCIE-32GB, compute capability 7.0;
    this torch build ships kernels for sm_75, sm_80, ...)"`` so the kernel-image
    warning pins the exact mismatch: what the GPU *is* versus what the installed
    wheel was compiled *for*.  Returns ``""`` if torch can't be queried (the
    warning still fires, just without the extra detail).
    """
    try:
        import torch  # noqa: PLC0415

        idx = torch.device(device).index or 0
        name = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        archs = ", ".join(torch.cuda.get_arch_list()) or "none"
        return f" ({name}, compute capability {major}.{minor}; this torch build ships kernels for {archs})"
    except Exception:
        return ""


def _cuda_can_run(device: str = "cuda") -> bool:
    """Return ``True`` only if torch can actually launch a kernel on *device*.

    ``torch.cuda.is_available()`` is necessary but **not** sufficient: it
    reports ``True`` whenever a driver and a CUDA device are visible, even when
    the installed torch wheel was compiled without a kernel image for that GPU's
    compute capability.  In that case every real op raises
    ``cudaErrorNoKernelImageForDevice`` (``torch.AcceleratorError``), which is
    exactly what makes a "switched GPU" host 500 on every MLP train.

    We force one tiny kernel launch + synchronize so the failure surfaces here,
    once, instead of deep inside training.  The result is cached per device so
    the rest of the app can fall back to CPU and keep working rather than
    crash-looping on a GPU it cannot use.  This keeps VTSearch resilient across
    heterogeneous fleets: the same install runs on whatever node it lands on,
    using the GPU when the wheel supports it and CPU when it doesn't.
    """
    import logging  # noqa: PLC0415

    cached = _cuda_runnable.get(device)
    if cached is not None:
        return cached

    ok = False
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            # A real launch + sync: allocation alone can succeed lazily, so we
            # add and reduce to force the kernel and then block on the result.
            probe = torch.zeros(1, device=device)
            _ = (probe + 1).sum().item()
            torch.cuda.synchronize(probe.device)
            ok = True
    except Exception:
        logging.getLogger(__name__).warning(
            "CUDA reports a usable device but torch cannot run a kernel on %s%s "
            "(the installed torch build lacks a kernel image for this GPU's "
            "compute capability); falling back to CPU. Reinstall a torch build "
            "whose CUDA tag covers this GPU's compute capability (see "
            "scripts/install.sh). Note the right tag is not simply the "
            "newest one: the newest wheels DROP the oldest architectures, so an "
            "older GPU needs an OLDER tag (e.g. cu128 dropped Volta/sm_70, so a "
            "V100 needs cu124, not cu128). Or set VTSEARCH_DEVICE=cpu to silence "
            "this warning.",
            device,
            _describe_cuda_mismatch(device),
            exc_info=True,
        )
        ok = False

    _cuda_runnable[device] = ok
    return ok


@functools.lru_cache(maxsize=None)
def resolve_device() -> str:
    """Resolve :data:`DEVICE` to a concrete ``torch.device`` string.

    Imports torch lazily so that simply importing this module does not pull
    torch in.  Returns ``"cpu"`` if torch is unavailable.

    The result is cached for the life of the process (device identity is
    fixed once resolved): settings getters and embedder helpers call this on
    hot request paths, and the un-memoized availability/backends probes are
    not free. Tests that need a different resolution reload this module or
    monkeypatch ``resolve_device`` itself, both of which bypass the cache.

    Whether ``DEVICE`` is ``"auto"`` or an explicit ``"cuda"``/``"cuda:N"``,
    the chosen CUDA device is smoke-tested via :func:`_cuda_can_run` before it
    is returned; a device that cannot actually execute a kernel (wrong/missing
    kernel image for the GPU) falls back to ``"cpu"`` so the app degrades
    gracefully instead of crashing on every train.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"

    if DEVICE != "auto":
        # Honour an explicit pin, but still refuse a CUDA device the wheel
        # can't run on - a hard crash helps nobody, and CPU keeps the app up.
        if DEVICE.startswith("cuda") and not _cuda_can_run(DEVICE):
            return "cpu"
        return DEVICE

    if torch.cuda.is_available() and _cuda_can_run("cuda"):
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Compute precision for the *embedding* forward pass.  Every image embedder used
# to load fp32 and run with no autocast, which on a V100 means 15.7 TFLOPS of
# fp32 instead of 125 TFLOPS on the fp16 tensor cores (issue #3143).
#
# The default is ``"fp32"`` - exactly the old behaviour - and #3143 closed on the
# decision to *keep* it there rather than flip it.  Two reasons, both measured:
#
# * **The speedup is not where the issue put it.**  Its 4.2x was the ``siglip2_l``
#   forward in isolation, before #3151 overlapped decode with the forward.  End to
#   end, half precision is 2.0-2.5x on ``siglip2_l`` and **0.99x on the shipped
#   default ``siglip``**, whose forward is no longer the bottleneck.  A global flip
#   would buy the default embedder nothing.
# * **It still changes the vectors**, by 2.9e-06 (``siglip2_l``) / 1.3e-06
#   (``siglip``) median 1-cos on a fixed node.  Retrieval order survives intact
#   (Spearman 1 +/- 4e-07), but the whole pre-embedded pile and every published
#   result (#3129) are fp32, and a pile with some cells fp16 and some fp32 is a
#   confound that would surface months later as an unexplained arm difference.
#   It is all cells or none - and two of the three columns would gain nothing.
#
# So this stays an opt-in: the escape hatch in both directions, and worth setting
# for a bulk build on one of the heavy encoders.  ``bf16`` is disqualified on the
# numbers (1.3e-04 drift, enough to change the top-1 result on 2-3% of categories).
# Full study: docs/experiments/2026-08-17-embed-precision-3143/REPORT.md.
#
# Modes:
#
# * ``fp32``   - full precision (default; unchanged behaviour).
# * ``fp16``   - cast the weights.  Fastest: no per-op cast overhead, and the
#                weights themselves halve, which is what lets a bigger batch fit.
# * ``bf16``   - cast the weights to bfloat16.  Wider exponent, fewer mantissa
#                bits than fp16; needs sm_80+ (a V100 is sm_70, so fp16 only).
# * ``autocast_fp16`` / ``autocast_bf16`` - keep fp32 weights and wrap the
#                forward in ``torch.autocast``, which keeps the reduction-heavy
#                ops (softmax, layer norm) in fp32.  Numerically the safer half,
#                slower than a weight cast.
# * ``auto``   - ``bf16`` where supported, else ``fp16``, else ``fp32``.
EMBED_PRECISION = os.environ.get("VTSEARCH_EMBED_PRECISION", "fp32").strip().lower()

#: Modes that keep fp32 weights and cast per-op inside ``torch.autocast``.
_AUTOCAST_MODES = {"autocast_fp16": "fp16", "autocast_bf16": "bf16"}
_PRECISION_MODES = {"fp32", "fp16", "bf16", "auto", *_AUTOCAST_MODES}


def _bf16_supported() -> bool:
    """True when the resolved device can actually run bfloat16 kernels."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    if not resolve_device().startswith("cuda"):
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def embed_precision() -> str:
    """The effective embedding precision: one of :data:`_PRECISION_MODES` minus ``auto``.

    Half precision is a **CUDA-only** win here, so every half mode degrades to
    ``"fp32"`` off CUDA rather than being honoured: fp16 on a CPU tensor is
    emulated and slower than the fp32 it replaced, which would make the escape
    hatch a performance trap on exactly the hosts that need the most help.  An
    unknown mode also degrades to ``"fp32"`` with a warning - a typo in an env
    var must not silently change what gets embedded.
    """
    mode = EMBED_PRECISION
    if mode not in _PRECISION_MODES:
        logging.getLogger(__name__).warning(
            "VTSEARCH_EMBED_PRECISION=%r is not one of %s; using fp32",
            mode,
            ", ".join(sorted(_PRECISION_MODES)),
        )
        return "fp32"
    if mode == "fp32":
        return "fp32"
    if not resolve_device().startswith("cuda"):
        return "fp32"
    if mode == "auto":
        return "bf16" if _bf16_supported() else "fp16"
    if mode in ("bf16", "autocast_bf16") and not _bf16_supported():
        # Say so: silently substituting fp16 for the bf16 someone asked for
        # would put a different numeric format in a study that named one.
        logging.getLogger(__name__).warning(
            "VTSEARCH_EMBED_PRECISION=%s but this device has no bfloat16 support; using fp32", mode
        )
        return "fp32"
    return mode


def embed_weight_dtype() -> "torch.dtype | None":
    """The dtype to *cast the weights* to, or ``None`` to leave them fp32.

    ``None`` for both ``fp32`` and the autocast modes - the latter keep fp32
    master weights on purpose and cast per-op instead.
    """
    mode = embed_precision()
    if mode not in ("fp16", "bf16"):
        return None
    import torch  # noqa: PLC0415

    return torch.float16 if mode == "fp16" else torch.bfloat16


def embed_autocast_dtype() -> "torch.dtype | None":
    """The dtype for a ``torch.autocast`` block, or ``None`` for no autocast."""
    half = _AUTOCAST_MODES.get(embed_precision())
    if half is None:
        return None
    import torch  # noqa: PLC0415

    return torch.float16 if half == "fp16" else torch.bfloat16
