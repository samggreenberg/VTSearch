"""Which transformers image-processor implementation runs, and on what device.

``transformers`` renamed and re-defaulted its image-processor backends between
4.x and 5.x, so the *same* code, weights and image produce different pixels
depending on which version a host resolved.  This module pins that choice
(:data:`IMAGE_PROCESSOR_BACKEND`), turns the library's silent fallbacks into
warnings that name the embedder (:func:`verify_image_processor_backend`), and
carries the GPU-resize knob (:data:`IMAGE_PROCESSOR_DEVICE`).  The long comment
below is the measurement that justifies the default; read it before changing it.

Depends on :mod:`vtscore.config.device` for the resolved device only, and
reaches it through the module rather than by importing the function, so that
stubbing ``vtscore.config.device.resolve_device`` reaches this module too.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from vtscore.config import device as _device


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
    if mode == "cuda" and not _device.resolve_device().startswith("cuda"):
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
