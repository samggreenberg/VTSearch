"""Configuration and constants for VTSearch."""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch is imported lazily everywhere below, never at module scope
    import torch

# Data paths are anchored to the repository root, NOT to the current working
# directory.  Without this, starting the app from a different CWD (systemd,
# cron, dev shell) would create a fresh empty `data/` and silently lose the
# user's existing datasets, settings, and embeddings.  Override with the
# ``VTSEARCH_DATA_DIR`` env var if you need to relocate state outside the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ["VTSEARCH_DATA_DIR"]) if "VTSEARCH_DATA_DIR" in os.environ else _REPO_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_CACHE_DIR = (
    Path(os.environ["VTSEARCH_MODELS_DIR"]) if "VTSEARCH_MODELS_DIR" in os.environ else DATA_DIR / "models"
)

# ---------------------------------------------------------------------------
# Runtime tunables
# ---------------------------------------------------------------------------

# Thread count for native math libraries (OpenMP / MKL) and ``torch``.  The
# default of 1 keeps memory overhead low in constrained environments; each
# additional thread allocates its own scratch buffers.  Override with
# ``VTSEARCH_TORCH_THREADS`` on bigger boxes where embedding throughput
# matters more than RSS.  Consumed by ``app.py`` (OMP/MKL env vars set
# before torch import) and ``vtscore.media.torch_setup.ensure_torch_configured``
# (``torch.set_num_threads``).
TORCH_THREADS = max(1, int(os.environ.get("VTSEARCH_TORCH_THREADS", "1")))

# Upper bound on the image-decode prefetch pool (see
# :func:`resolve_decode_workers`).  Decode throughput saturates well before a
# fat node's core count, and every extra worker holds one more decoded bitmap,
# so the pool is capped rather than scaled to whatever the box happens to have.
DEFAULT_DECODE_WORKER_CAP = 8


def allocated_cpus() -> int:
    """How many CPUs this process may actually run on.

    ``os.cpu_count()`` reports the *machine's* cores, which is the wrong number
    on a shared box: a SLURM job asking for ``--cpus-per-task=8`` on a 96-core
    node is entitled to 8, and sizing a pool from 96 would oversubscribe the
    allocation.  ``os.sched_getaffinity`` reflects the cgroup/affinity mask the
    scheduler actually imposed, so it is the allocation.  Falls back to
    ``cpu_count()`` on platforms without it (macOS, Windows).
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def resolve_decode_workers(cap: int = DEFAULT_DECODE_WORKER_CAP) -> int:
    """Threads used to decode images ahead of the GPU forward during bulk embed.

    Bulk image embedding used to decode a batch on the calling thread, run the
    forward, then decode the next batch, so nothing overlapped and the GPU sat
    idle through every decode.  On a small model that is most of the wall clock
    (measured: 82% idle for base SigLIP on a V100).  Pillow's decode releases
    the GIL in C, so a plain thread pool both parallelises a batch's decode and
    overlaps it with the previous batch's forward — see
    :mod:`vtscore.media.image._image_bulk`.

    Sized from :func:`allocated_cpus`, minus one for the calling thread (which
    runs the processor and tensor marshalling while the pool decodes), capped at
    *cap*.  Override with ``VTSEARCH_DECODE_WORKERS``; ``0`` disables the
    prefetch entirely and decodes inline on the calling thread.
    """
    raw = os.environ.get("VTSEARCH_DECODE_WORKERS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return max(1, min(cap, allocated_cpus() - 1))


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
# fp32 instead of 125 TFLOPS on the fp16 tensor cores - measured at 4.2x for
# ``siglip2_l`` (issue #3143).
#
# The default is deliberately ``"fp32"``, i.e. exactly the old behaviour.  Half
# precision *changes the vectors* (cosine similarities shift by ~1e-3, against a
# ~1e-7 fp32 kernel-selection noise floor), and the calibration studies resolve
# effects of 0.005, so it cannot be turned on silently: the whole pre-embedded
# pile and every published result (#3129) are fp32, and a pile with some cells
# fp16 and some fp32 is a confound that would surface months later as an
# unexplained arm difference.  Flipping this default is gated on the #3143
# measurement; until then it is the experiment's knob and the user's escape
# hatch, in both directions.
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


# Image-processor resize/normalise: which backend runs it, and on which device.
#
# Issue #3146 assumed every image embedder was on the slow PIL/numpy path
# because none of them passes ``use_fast``.  Measured against the installed
# transformers (5.12.x) that is **false**: v5 removed the ``Fast`` suffix, so
# ``SiglipImageProcessor`` *is* the torchvision implementation and the slow one
# was renamed ``SiglipImageProcessorPil``.  Passing nothing already selects
# torchvision, and ``use_fast`` is itself deprecated in favour of ``backend=``.
#
# What that leaves is two knobs worth having, for two different reasons.
#
# ``VTSEARCH_IMAGE_PROCESSOR_BACKEND`` exists to *pin* the thing that silently
# changed under us.  ``transformers>=4.49`` is the only pin in
# ``requirements/image-embedders.txt``, and the two sides of that range
# disagree about which backend is the default — so the same code, same weights
# and same image produce different pixels depending on which transformers a
# host resolved.  #3146 measured how much: PIL and torchvision differ on
# **53-59% of pixel elements**, and downstream by a median ``1 - cos`` of
# 1.5e-04 on ``siglip2_l`` — 50x the fp16 perturbation #3143 measured, and
# larger than several effects the calibration studies exist to resolve.
#
# It therefore defaults to ``"torchvision"`` rather than to ``"auto"`` (#3173).
# That is a deliberate change of posture, not a tuning choice: ``auto`` means
# "whatever this host's resolver picked", which is not a property of this
# repository at all.  Naming the backend costs nothing if the difference does
# not matter and is essential if it does, and it is the only half of the
# reproducibility story that survives someone's local environment — the
# requirements pin controls what *installs*, this controls what *runs*.
#
# The default is chosen to match the pre-embedded pile rather than to change
# it: the pile is torchvision-built (#3146 reproduced its cells to 7.6e-13
# under torchvision against 2.4e-06 under PIL), and on a ``transformers`` 5
# host naming the backend is a no-op.  What it *does* change is a
# ``transformers`` 4.x host, which resolves the bare class name to PIL and so
# has been quietly producing vectors inconsistent with the pile.  Such a host
# now agrees with everyone else.  That is the point of the change, and it is
# the reason it is a behaviour change worth announcing.
#
# ``VTSEARCH_IMAGE_PROCESSOR_DEVICE`` is the actual performance knob, and it is
# the half of #3146's proposed fix that is still live: the torchvision backend
# accepts ``device=``, which runs resize/normalise on the GPU instead of the
# calling thread.  It is *not* free numerically — GPU resampling differs from
# CPU torchvision by more than CPU torchvision differs from PIL — so like the
# precision knob it defaults to the old behaviour and its adoption is gated on
# the #3146 measurement.  It stays ``"auto"``.
#
# * ``torchvision`` - the fast, tensor-based implementation.  The default.
# * ``pil``         - the legacy PIL/numpy implementation.  Not every
#                     architecture ships one (DINOv3 does not), and transformers
#                     falls back to torchvision with a *warning* when it is asked
#                     for one that does not exist — so this is a request, not a
#                     guarantee.  :func:`verify_image_processor_backend` turns
#                     that warning into one of our own, naming the embedder.
# * ``auto``        - pass nothing; whatever the installed transformers picks.
#                     The escape hatch back to pre-#3173 behaviour, and the only
#                     mode whose meaning depends on the host.
#: The shipped backend.  Named once so the env read and the typo-degrade path
#: below cannot drift apart: a misspelled mode must land on the *default*, not
#: on ``auto`` — degrading to "pass nothing" would answer a typo with the one
#: mode whose meaning depends on the host, which is exactly what #3173 removes.
_DEFAULT_PROCESSOR_BACKEND = "torchvision"

IMAGE_PROCESSOR_BACKEND = os.environ.get("VTSEARCH_IMAGE_PROCESSOR_BACKEND", _DEFAULT_PROCESSOR_BACKEND).strip().lower()

#: * ``auto`` - pass nothing; the processor returns CPU tensors (today's behaviour).
#: * ``cpu``  - explicit CPU.
#: * ``cuda`` - resize/normalise on the GPU.  Degrades to ``auto`` off CUDA
#:              rather than raising: an escape hatch that crashes on a laptop is
#:              not an escape hatch.
IMAGE_PROCESSOR_DEVICE = os.environ.get("VTSEARCH_IMAGE_PROCESSOR_DEVICE", "auto").strip().lower()

_PROCESSOR_BACKENDS = {"auto", "pil", "torchvision"}
_PROCESSOR_DEVICES = {"auto", "cpu", "cuda"}


def _transformers_major() -> int | None:
    """Major version of the installed ``transformers``, or ``None`` if it is absent.

    An *unparseable* version string resolves to 5 rather than to ``None``: a
    version we cannot read is far likelier to be a newer one than a pre-4.49 one
    we no longer support, and guessing legacy would silently pick the legacy
    spelling of every decision below.
    """
    try:
        import transformers  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return int(str(transformers.__version__).split(".")[0])
    except ValueError:
        return 5


def _transformers_backend_kwarg() -> str:
    """``"backend"`` on transformers>=5, ``"use_fast"`` on 4.x, ``""`` if neither.

    The spelling of this argument changed *and* its default flipped across the
    ``>=4.49`` range we pin, which is precisely why the backend is worth naming
    explicitly.  Resolved at call time from the installed version rather than
    assumed, because guessing wrong here fails silently: an unknown kwarg is
    swallowed into ``**kwargs`` by several processor classes.
    """
    major = _transformers_major()
    if major is None:
        return ""
    return "backend" if major >= 5 else "use_fast"


def image_processor_load_kwargs() -> dict[str, Any]:
    """Kwargs for ``*ImageProcessor.from_pretrained`` selecting the backend.

    Non-empty by default: since #3173 the backend is *named* rather than
    inherited from whatever the installed transformers happens to default to.
    Empty only for ``auto`` (the explicit opt-out) and on a transformers so old
    it has neither spelling of the argument.

    ``Any`` rather than the more honest ``str | bool``: splatting a
    ``dict[str, X]`` into a call makes a type checker treat *every* remaining
    parameter as possibly receiving ``X``, so a narrower annotation here reports
    six errors at the call sites about ``on_progress`` — a parameter this dict
    never supplies.
    """
    mode = IMAGE_PROCESSOR_BACKEND
    if mode not in _PROCESSOR_BACKENDS:
        logging.getLogger(__name__).warning(
            "VTSEARCH_IMAGE_PROCESSOR_BACKEND=%r is not one of %s; using %s",
            mode,
            ", ".join(sorted(_PROCESSOR_BACKENDS)),
            _DEFAULT_PROCESSOR_BACKEND,
        )
        mode = _DEFAULT_PROCESSOR_BACKEND
    if mode == "auto":
        return {}
    kwarg = _transformers_backend_kwarg()
    if not kwarg:
        return {}
    if kwarg == "backend":
        return {"backend": mode}
    return {"use_fast": mode == "torchvision"}


def image_processor_call_kwargs() -> dict[str, Any]:
    """Kwargs for the processor *call* placing resize/normalise on a device.

    Empty for ``auto`` and off CUDA, so the default path is unchanged and a
    CPU-only host never asks for a device it does not have.
    """
    mode = IMAGE_PROCESSOR_DEVICE
    if mode not in _PROCESSOR_DEVICES:
        logging.getLogger(__name__).warning(
            "VTSEARCH_IMAGE_PROCESSOR_DEVICE=%r is not one of %s; passing nothing",
            mode,
            ", ".join(sorted(_PROCESSOR_DEVICES)),
        )
        return {}
    if mode == "auto":
        return {}
    if mode == "cuda" and not resolve_device().startswith("cuda"):
        return {}
    return {"device": mode}


#: Class-name suffixes that name the implementation outright.  ``…Pil`` is the
#: transformers v5 spelling of the legacy path; ``…Fast`` is the v4 spelling of
#: the torchvision one.  A *bare* name means neither, and that is the whole
#: confusion this module exists to pin down: bare is torchvision on v5 and PIL
#: on v4, so it can only be read with the installed version in hand.
_PIL_CLASS_SUFFIX = "Pil"
_FAST_CLASS_SUFFIX = "Fast"


def processor_backend_from_class_name(class_name: str, transformers_major: int | None) -> str | None:
    """Which backend a processor class *name* denotes, or ``None`` if it denotes none.

    Split out from :func:`resolved_processor_backend` so that the one place this
    rename is decoded serves both a live object and a recorded name.  The #3146
    experiment harness reads class names back out of a provenance JSON on a
    machine that may not have the model loaded at all
    (``scripts/experiments/fastproc/check_arms.py``), so it has a name and no
    object; keeping a second copy of the rule there is how the harness would
    drift from the app it exists to measure.

    ``None`` means "no answer", not "neither backend": a name that is not an
    image processor's, and a *bare* name with no version to read it against,
    both land here.  A guess would be worse than an abstention — a wrong backend
    label is exactly the failure this whole path exists to prevent.
    """
    if "ImageProcessor" not in class_name:
        return None
    if class_name.endswith(_PIL_CLASS_SUFFIX):
        return "pil"
    if class_name.endswith(_FAST_CLASS_SUFFIX):
        return "torchvision"
    if transformers_major is None:
        return None
    return "torchvision" if transformers_major >= 5 else "pil"


def resolved_processor_backend(processor: Any) -> str | None:
    """Which backend a *loaded* processor actually is, or ``None`` if unknowable.

    Reads the class that was constructed rather than the request that was made,
    because those are allowed to differ — see
    :func:`verify_image_processor_backend`.  Composite wrappers (``CLIPProcessor``
    and the ``AutoProcessor``-built SigLIP 2 ones) hold the image half as
    ``.image_processor``; a bare ``*ImageProcessor`` is its own image half.
    """
    inner = getattr(processor, "image_processor", processor)
    return processor_backend_from_class_name(type(inner).__name__, _transformers_major())


def verify_image_processor_backend(processor: Any, *, embedder: str) -> str | None:
    """Warn when the processor that loaded is not the backend that was asked for.

    Requesting a backend is a *request*, not a guarantee: transformers **warns
    and continues** when the one you named is unavailable for an architecture,
    so the default outcome of an impossible request is a silently mislabelled
    processor rather than an error.  DINOv3 is the concrete case — it ships no
    PIL implementation at all, so ``backend="pil"`` there hands back torchvision.
    Since #3173 names a backend by default, every embedder now makes such a
    request on every load, which is precisely why the result has to be read back
    instead of assumed.

    A **warning**, not a raise: the app has to keep running on whatever host it
    landed on, and the honest outcome of "you cannot have PIL here" is degraded
    reproducibility rather than a dead embedder.  The experiment harness makes
    the opposite call and refuses to build a cell
    (``scripts/experiments/fastproc/build_arm.py``), because a mislabelled arm
    poisons a study in a way a running app is not poisoned.

    Returns the backend actually resolved, or ``None`` when it cannot be told.
    """
    requested = IMAGE_PROCESSOR_BACKEND
    if requested not in _PROCESSOR_BACKENDS:
        requested = _DEFAULT_PROCESSOR_BACKEND
    got = resolved_processor_backend(processor)
    if requested == "auto" or got is None or got == requested:
        # ``auto`` asked for nothing, so nothing can contradict it.
        return got
    inner = getattr(processor, "image_processor", processor)
    logging.getLogger(__name__).warning(
        "%s: asked transformers for the %r image-processor backend but loaded %s (= %r). "
        "transformers warns and falls back rather than raising when a backend is "
        "unavailable for an architecture, so this is a fallback, not a crash. The two "
        "backends resize and normalise differently — they disagree on 53-59%% of pixel "
        "elements and by a median 1-cos of ~1.5e-04 downstream (#3146) — so vectors "
        "embedded here are not interchangeable with vectors embedded under %r. Set "
        "VTSEARCH_IMAGE_PROCESSOR_BACKEND=%s to make this host's behaviour the intended "
        "one, or install a transformers that provides %r for this model.",
        embedder,
        requested,
        type(inner).__name__,
        got,
        requested,
        got,
        requested,
    )
    return got


# Server-side file access is governed by the active login provider, not a
# configurable allow-list.  In single-user / no-auth mode the lone trusted user
# may read from and write to any server-readable path (server-path importers,
# exporters, and the file browser are unrestricted).  In multi-user mode each
# user is confined to their own ``data/<username>/`` subtree.  See
# :func:`vtscore.security.path_validation.get_file_access_base_dir`.

# Maximum size (in megabytes) accepted for a single HTTP request body.  Wired
# into Flask's ``MAX_CONTENT_LENGTH`` config in ``app.py``.  Defaults to
# ``2048`` (2 GiB): generous enough for typical media archives and dataset
# pickles while still rejecting pathological uploads with HTTP 413 before they
# consume disk.  Set ``VTSEARCH_MAX_UPLOAD_MB`` to a different positive integer
# to raise or lower the cap, or to ``0`` to disable it entirely (Flask's
# out-of-the-box "no limit" behaviour) for genuinely large-archive uploads.
MAX_UPLOAD_MB = max(0, int(os.environ.get("VTSEARCH_MAX_UPLOAD_MB", "2048")))

# Pixel budget for a single image *decode*.  Pillow's own decompression-bomb
# ceiling is lifted at startup (see :mod:`vtscore.media.image.decode`) so a
# merely-large photo — a gigapixel panorama, a whole-slide scan — is imported
# rather than refused; this budget replaces it with the protection that
# actually matters, capping how big a bitmap any one decode may materialise.
# Sources above it are downsampled (aspect preserved) before the pixels reach
# a thumbnail, embedder, extractor or converter, all of which resize to a few
# hundred pixels anyway.  64 MP is ~192 MB as RGB — comfortably above any
# real photograph, so ordinary media is never touched.  Crop/clip paths
# deliberately bypass this and decode at native size.  Set
# ``VTSEARCH_MAX_DECODE_PIXELS=0`` to disable bounding entirely.
MAX_DECODE_PIXELS = max(0, int(os.environ.get("VTSEARCH_MAX_DECODE_PIXELS", str(64_000_000))))

# Training
#
# ``TRAIN_EPOCHS`` is an *upper bound*; :func:`vtscore.training.mlp.train_model`
# also short-circuits on a loss plateau (see ``TRAIN_PATIENCE``).  Override with
# ``VTSEARCH_TRAIN_EPOCHS`` for benchmarking or to disable early-stop entirely
# by pairing with ``VTSEARCH_TRAIN_PATIENCE=0``.
TRAIN_EPOCHS = int(os.environ.get("VTSEARCH_TRAIN_EPOCHS", "200"))
# Number of epochs the training loss must fail to improve before training
# stops early.  Set to 0 to disable early-stop and always run ``TRAIN_EPOCHS``.
TRAIN_PATIENCE = int(os.environ.get("VTSEARCH_TRAIN_PATIENCE", "10"))
# Default ``calibrate_count`` baked into ``data/settings.json`` on first run.
# Each unit adds one full fold-training pass per learned-sort; lower it to
# trade calibration quality for latency.  Min 1 (clamped in
# :mod:`vtsearch.settings`).  The default is 2: the Inclusion knob is a
# quantile rule over the *pooled* held-out fold scores (see
# ``vtscore.training.thresholds.conformal_threshold``), so its resolution is
# bounded by how many calibration scores the pool holds - at ~12 votes a
# single fold yields only ~4 positive scores, i.e. ~4 usable knob positions;
# a second fold doubles that for one extra fold fit.
DEFAULT_CALIBRATE_COUNT = max(1, int(os.environ.get("VTSEARCH_CALIBRATE_COUNT", "2")))
MLP_HIDDEN_MIN = 8
MLP_HIDDEN_MAX = 32
MLP_DROPOUT = 0.5
# Label-smoothing epsilon for MLP training targets (Good trains toward
# ``1 - eps``, Bad toward ``eps``).  Not a knob-mover: it exists as tie
# insurance for the conformal inclusion rule, which needs distinct calibration
# score values - smoothing bounds the optimal logit (~ +/-2.9 at 0.05), so a
# strongly-fit fold model can't collapse every score to exact 0.0/1.0 sigmoids
# where all quantiles degenerate to the same cut.
MLP_LABEL_SMOOTHING = 0.05

# Inverse regularisation strength of the production **linear SVM head**
# (``LINEAR_SVM_HEAD`` in ``vtscore.training.mlp``, fitted by
# ``vtscore.training.svm.fit_linear_svm_head``).  1.0 is sklearn's own default
# and the value the eval harness's ``svm_linear`` arm was measured at, so the
# shipped head and the measured arm are the same fit.  Lower it to regularise
# harder on very few votes; raise it to trust the labels more.
SVM_HEAD_C = float(os.environ.get("VTSEARCH_SVM_HEAD_C", "1.0"))

# --- Browse projection (UMAP Stage 1) ---------------------------------------
# Default UMAP knobs for the VTSBrowse 2-D projection.  Overridable per
# deployment via the ``projection_n_neighbors`` / ``projection_min_dist``
# server settings; these constants are the fallback and the values the
# ingest-time pre-build stamps.  The persisted projection is keyed on the
# effective values so changing a setting forces a recompute instead of
# serving a layout fit under the old params.  See
# docs/plans/vtsbrowse-empirical-tuning.md.
PROJECTION_N_NEIGHBORS = 15
PROJECTION_MIN_DIST = 0.1

# Per-embedder UMAP projection defaults, from the empirical sweep in
# ``docs/plans/vtsbrowse-empirical-tuning.md`` (§Results). Keyed off the dataset's
# *primary* embedder and consulted when the corresponding ``ServerSettings`` knob
# is left at the global default above (an explicit operator override still wins).
# Untuned embedders fall back to the globals. Image embedders peak at a smaller
# neighbourhood than the old global 15; large ``n_neighbors`` hurt every embedder.
PROJECTION_DEFAULTS_BY_EMBEDDER: dict[str, tuple[int, float]] = {
    "clap": (15, 0.10),  # audio: flat separability peak across 10-30
    "clap_general": (15, 0.10),  # same family, same flat audio peak
    "clip": (10, 0.05),  # image: the most n_neighbors-sensitive embedder
    "siglip": (10, 0.05),
    "siglip_l": (10, 0.05),
}

# Compaction (``compact_layout``) default. The sweep found compaction consistently
# costs ~2% taxonomy separability and ~5-6% neighbourhood structure (trustworthiness
# / continuity / recall) on every dataset and embedder, so it is off and not
# exposed as a user-facing setting; the ``compact`` param on ``fit_projection``
# remains for future experimentation.
PROJECTION_COMPACT_DEFAULT = False

# Model IDs
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000  # CLAP model expected input sample rate
XCLIP_MODEL_ID = "microsoft/xclip-base-patch32"
E5_MODEL_ID = "intfloat/e5-base-v2"
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
# SigLIP2-L: the SigLIP 2 SO400M/384 checkpoint, loaded through ``transformers``
# like its base sibling (SigLIP 2 has a first-party HF port, so there is no
# reason to route it through open_clip the way ``SIGLIP_L_MODEL_ID`` is).  The
# fixed-resolution ``patch14-384`` variant, not the NaFlex one, so the standard
# ``AutoProcessor`` image pipeline applies.  Emits 1152-d vectors, so its
# galleries are *not* interchangeable with the 768-d base SigLIP 2.
SIGLIP2_L_MODEL_ID = "google/siglip2-so400m-patch14-384"
# SigLIP-L: the SO400M/384 checkpoint, loaded via ``open_clip`` (not
# transformers) so its 1152-d vectors match galleries produced by open_clip's
# own ``ViT-SO400M-14-SigLIP-384`` model.  The arch name is the open_clip
# model key; the ``webli`` tag selects the WebLI-pretrained weights.
SIGLIP_L_MODEL_ID = "ViT-SO400M-14-SigLIP-384"
SIGLIP_L_PRETRAINED = "webli"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
# CLIP ViT-L/14, **evaluation only** (#3292): a single-vector, language-aligned
# encoder from a different lineage than SigLIP, used to test whether #3287's
# `calibration_fraction` optimum follows single-vector geometry or just the
# SigLIP family.  Chosen over the base/32 checkpoint above because it emits
# **768-d** vectors, exactly like `SIGLIP_MODEL_ID`, so a difference between the
# two cannot be "CLIP's vectors are narrower".  Not offered in the app -- see
# `ImageClipLargeEmbedder.eval_only`.
CLIP_L_MODEL_ID = "openai/clip-vit-large-patch14"
DINOV2_MODEL_ID = "facebook/dinov2-base"
DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
EUPE_MODEL_ID = "https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt"
"""Direct URL to the real EUPE ViT-B/16 weights on Hugging Face.

Loaded via :func:`torch.hub.load` from the ``facebookresearch/EUPE`` GitHub
repo with this URL passed as the ``weights`` kwarg.  The HF repo
``facebook/EUPE-ViT-B`` is ungated; the underlying weights are released
under Meta's FAIR Noncommercial Research Licence (surfaced to users via
``MediaEmbedder.license_notice`` on the EUPE embedder).

Not the same model as ``facebook/PE-Core-B16-224``; that was Meta's
Perception Encoder Core, which the dev "eupe" slug was confusingly
aliased to via a broken ``AutoModel.from_pretrained`` path (the PE-Core
HF repo has no ``config.json`` so ``AutoModel`` could never load it).
"""
CLAP_MUSIC_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_GENERAL_MODEL_ID = "laion/larger_clap_general"
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16000  # AST expects 16 kHz mono
# BEATs: Microsoft's self-supervised audio encoder (MIT, part of ``microsoft/unilm``).
# The official weights are published as loose ``.pt`` files on Azure blob storage
# rather than on the Hub, so we pull the ``iter3+`` AudioSet-2M checkpoint from an
# MIT-licensed Hub mirror of that release. ``iter3_plus_AS2M`` is the
# self-supervised encoder, *not* one of the AudioSet-finetuned classifier
# variants: it has no prediction head, which is what we want for embeddings.
BEATS_CHECKPOINT_REPO = "lpepino/beats_ckpts"
BEATS_CHECKPOINT_FILE = "BEATs_iter3_plus_AS2M.pt"
BEATS_SAMPLE_RATE = 16000  # BEATs expects 16 kHz mono
BEATS_EMBED_DIM = 768
BEATS_MAX_SAMPLES = 16000 * 10  # cap clips at the 10 s AudioSet window BEATs was trained on
BEATS_MIN_SAMPLES = 16000  # zero-pad anything shorter, so short clips still yield patches
# Global fbank normalisation constants baked into the released BEATs
# checkpoints; the encoder expects ``(fbank - mean) / (2 * std)``.
BEATS_FBANK_MEAN = 15.41663
BEATS_FBANK_STD = 6.55582
WHISPER_MODEL_ID = "openai/whisper-base"
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono
# ParaSpeechCLAP: dual-encoder speech↔text "style" CLAP (MIT-licensed).
# Unlike the AST / Whisper speech embedders, it has a paired text tower, so
# text queries like "a deep, raspy voice" or "a whispered, anxious style" land
# in the same space as the speech embeddings.  Reconstructed from the upstream
# checkpoint via ``_paraspeechclap_model.py`` (WavLM speech + Granite text +
# projection heads); the ``combined`` variant covers both speaker-level
# (pitch/texture/clarity) and utterance-level (emotion/speaking-style) attributes.
PARASPEECHCLAP_SPEECH_MODEL_ID = "microsoft/wavlm-large"
PARASPEECHCLAP_TEXT_MODEL_ID = "ibm-granite/granite-embedding-278m-multilingual"
PARASPEECHCLAP_CHECKPOINT_REPO = "ajd12342/paraspeechclap-combined"
# Upstream renamed the released weights to ``slap-combined.pth.tar``; the old
# ``paraspeechclap-combined.pth.tar`` was removed and now 404s (issue #2635).
PARASPEECHCLAP_CHECKPOINT_FILE = "slap-combined.pth.tar"
PARASPEECHCLAP_EMBED_DIM = 768
PARASPEECHCLAP_SAMPLE_RATE = 16000  # WavLM expects 16 kHz mono
PARASPEECHCLAP_MAX_SAMPLES = 16000 * 30  # cap clips at 30 s to bound CPU memory/latency
BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"
LANGUAGEBIND_VIDEO_MODEL_ID = "LanguageBind/LanguageBind_Video_V1.5_FT"
VIDEOMAE_MODEL_ID = "OpenGVLab/VideoMAEv2-Base"
"""Hugging Face repo for VideoMAE v2 Base weights.

Loaded via ``AutoModel.from_pretrained(..., trust_remote_code=True)``.
Vision-only encoder with no paired text tower, so the embedder
sets ``supports_text=False`` and :meth:`embed_text` returns ``None``.
The masked-autoencoder objective produces unusually strong action /
motion features compared to image-only encoders applied per frame.
"""


# ---------------------------------------------------------------------------
# CoreConfig: runtime config bundle the (future) ``vtscore`` library consumes
# ---------------------------------------------------------------------------
#
# Today every library-candidate package reaches into ``vtsearch.settings``
# directly for tunables like ``saved_datasets_dir``, ``detectors_dir``,
# ``calibrate_count``, etc.  That couples the library to the app's settings
# layer and makes it impossible to vendor the library as ``vtscore`` (see
# ``docs/architecture.md``, Phase 2).
#
# ``CoreConfig`` is the seam: a frozen value object that bundles every knob
# library code reads.  Follow-up PRs convert each call site to accept (or
# look up) a ``CoreConfig`` instead of importing ``vtsearch.settings``.
# Until those land this class is unused at runtime; the scaffold just
# defines the type so the conversions can happen one file at a time.
#
# The app side will build a fresh ``CoreConfig`` at each request boundary
# via :meth:`CoreConfig.from_settings`; library callers can construct one
# directly with whatever values they want.
#
# The implementation of :meth:`from_settings` is installed by the app via
# :func:`register_core_config_builder`; see ``vtsearch/shim`` for the
# concrete builder that snapshots ``vtsearch.settings``.  This keeps the
# library import-clean: ``vtscore.config`` itself never imports
# ``vtsearch.settings`` (Phase 8 of ``docs/architecture.md``).
# Library-only consumers without an app skip ``from_settings()`` entirely
# and construct ``CoreConfig`` directly.


_core_config_builder: Callable[[str | Path | None], CoreConfig] | None = None


def register_core_config_builder(fn: Callable[[str | Path | None], CoreConfig]) -> None:
    """Install the app-side builder that reads ``vtsearch.settings``.

    The Flask app wires this at startup via
    :func:`vtsearch.shim.register_app_config_builder`.  Once registered,
    :meth:`CoreConfig.from_settings` delegates to *fn*; the library file
    itself stays settings-import-free.
    """
    global _core_config_builder
    _core_config_builder = fn


@dataclass(frozen=True)
class CoreConfig:
    """Runtime configuration bundle the ``vtscore`` library consumes.

    Field set is intentionally narrow; only knobs that library code (loaders,
    detectors, training, embedding) reads.  User-pref concerns like theme or
    grid-icon size are app-tier and stay in ``vtsearch.settings``.
    """

    # Server-tier settings (shared across users, stored in data/settings.json)
    saved_datasets_dir: Path
    detectors_dir: Path
    max_concurrent_dataset_downloads: int
    max_concurrent_dataset_embeddings: int
    autofind_detectors: tuple[str, ...]

    dataset_max_age_days: int | None

    # Per-user settings (stored under each user's data dir)
    calibrate_count: int
    # The user's *explicit* Train/Calibrate split, or ``None`` when unset.
    # ``None`` is resolved at training time to the per-space production
    # default for the detector's embedder (0.3 single-vector / 0.5 patch;
    # issue #3287) - see
    # :func:`vtscore.detectors.training.resolve_calibration_fraction`.
    calibration_fraction: float | None
    enrich_descriptions: bool
    autopilot_goal_diversity: int
    inclusion: int

    # Filesystem root for caches, embeddings, model downloads.  Phase 4 will
    # route every hardcoded ``data/`` path through this field.
    data_dir: Path

    # Auto-Find results exporter (server-tier). When an autodetect run has no
    # explicit ``--exporter``, the CLI falls back to this exporter +
    # field-value map. ``""`` means "no configured exporter" (CLI defaults to
    # ``gui``). Defaulted here so library-only ``CoreConfig(...)`` constructions
    # without the app shim keep working unchanged.
    autofind_exporter: str = ""
    autofind_exporter_field_values: dict[str, dict[str, str]] = field(default_factory=dict)

    # Operator overrides for the Browse projection's UMAP knobs (server-tier
    # ``projection_n_neighbors`` / ``projection_min_dist``).  Mirrored onto the
    # library tier because *both* fit paths — the on-demand route and the
    # ingest-time pre-build, which cannot import ``vtsearch.settings`` — resolve
    # their params through :func:`vtscore.projection.params.resolve_projection_params`.
    # A value equal to the global constant above means "no override", which is
    # what lets ``PROJECTION_DEFAULTS_BY_EMBEDDER`` apply.  Defaulted here so
    # library-only ``CoreConfig(...)`` constructions without the app shim keep
    # working unchanged.
    projection_n_neighbors: int = PROJECTION_N_NEIGHBORS
    projection_min_dist: float = PROJECTION_MIN_DIST

    # Per-media-type opt-in to the generative signpost captioner (image VLM /
    # audio captioner) instead of the default zero-shot tag texts.  ``{}`` (the
    # default) means tags for every type.  Read by
    # :func:`vtscore.projection.signpost_texts.provider_for`.  Defaulted here so
    # library-only ``CoreConfig(...)`` constructions without the app shim keep
    # working unchanged.
    signpost_captioner: dict[str, bool] = field(default_factory=dict)

    # Per-media-type operator-supplied zero-shot tag vocabulary for signpost
    # region names, replacing the built-in AudioSet-527 / OpenImages-600 lists
    # for the whole deployment (the app populates it from the server-tier
    # ``browse_signpost_vocab`` setting, not from a per-user one).  ``{}``
    # (the default) means the shipped vocabulary for every type.  Read by
    # :func:`vtscore.projection.signpost_texts.provider_for`.  Defaulted here so
    # library-only ``CoreConfig(...)`` constructions without the app shim keep
    # working unchanged.
    signpost_vocab: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings_path: str | Path | None = None) -> CoreConfig:
        """Snapshot the current user's ``vtsearch.settings`` into a CoreConfig.

        Called by the Flask app at the request boundary (after auth resolves
        the current user) and by the CLI before kicking off autodetect.  The
        result is a frozen immutable value safe to hand to background
        threads; settings changes during a request will not retroactively
        rewrite a config already in flight.

        When *settings_path* is given, the server-tier settings file path is
        redirected to that location first.  The CLI uses this to point at a
        run-specific settings JSON without each call site importing
        :mod:`vtsearch.settings` directly.

        Implementation note: the body of this classmethod lives in
        ``vtsearch/shim/`` and is installed at app startup via
        :func:`register_core_config_builder`.  Library-only consumers
        without the shim should construct :class:`CoreConfig` directly.
        """
        if _core_config_builder is None:
            raise RuntimeError(
                "CoreConfig.from_settings() requires the app-side builder to be "
                "registered.  The Flask app installs it during startup via "
                "vtsearch.shim.register_app_config_builder().  Library-only "
                "callers should construct CoreConfig(...) directly instead."
            )
        return _core_config_builder(settings_path)
