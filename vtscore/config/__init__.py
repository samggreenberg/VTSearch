"""Configuration and constants for VTSearch.

This is the most-imported module in ``vtscore``, and it used to be one 933-line
file holding six unrelated things.  It is now a package, split along the seams
that were already there, layered so each module only reads from the ones above
it:

* :mod:`~vtscore.config.paths` - the filesystem roots (``DATA_DIR`` and
  friends).  Depends on nothing.
* :mod:`~vtscore.config.runtime` - scalar tunables: thread and decode-worker
  sizing, the upload and decode caps, the training/MLP/SVM knobs, and the UMAP
  projection defaults.  Pure numbers, no torch.  Depends on nothing.
* :mod:`~vtscore.config.models` - Hugging Face identifiers and per-checkpoint
  constants for every embedder.  Identifiers only; nothing loads at import time.
  Depends on nothing.
* :mod:`~vtscore.config.device` - ``DEVICE``/:func:`resolve_device` (including
  the CUDA kernel smoke-test) and ``EMBED_PRECISION``/:func:`embed_precision`.
  Torch is imported lazily inside the functions, never at module scope.
* :mod:`~vtscore.config.processor_backend` - which ``transformers``
  image-processor implementation runs, and on what device.  Reads
  :mod:`~vtscore.config.device`.
* :mod:`~vtscore.config.core_config` - :class:`CoreConfig` and the app-side
  builder hook.  Reads the projection defaults from
  :mod:`~vtscore.config.runtime`.

Everything public is re-exported here, so ``vtscore.config.X`` resolves exactly
as it did when this was one module.  **Two things the split does change**, both
of which only matter to tests:

* **Patch targets.**  A caller outside this package still resolves
  ``vtscore.config.X`` (whether it imports lazily inside a function or was
  reloaded after this package was), so stubbing ``vtscore.config.resolve_device``
  reaches it.  A caller *inside* a submodule resolves its own module global,
  which this package's attribute is only a copy of - so stubbing
  ``resolve_device`` for :func:`embed_precision`, or ``allocated_cpus`` for
  :func:`resolve_decode_workers`, means patching ``vtscore.config.device`` /
  ``vtscore.config.runtime``, not this package, where the rebind is silently
  ignored.  :mod:`~vtscore.config.processor_backend` deliberately calls through
  the :mod:`~vtscore.config.device` *module* so that one target covers it too.
* **Reloading.**  Almost every constant here is read from the environment at
  import time, and tests re-read them with ``importlib.reload``.
  ``importlib.reload(vtscore.config)`` now only re-runs the re-exports below -
  the submodules are already in ``sys.modules`` and do not re-execute, so the
  env vars are *not* re-read.  Use :func:`_reload_all`, which reloads the
  submodules in dependency order and then this package.

Private names are deliberately **not** re-exported: a stub installed on a copy
of ``_cuda_can_run`` or ``_core_config_builder`` would be silently ignored, so
the copy should not exist.  Reach for the submodule instead.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from vtscore.config.core_config import CoreConfig, register_core_config_builder
from vtscore.config.device import (
    DEVICE,
    EMBED_PRECISION,
    embed_autocast_dtype,
    embed_precision,
    embed_weight_dtype,
    resolve_device,
)
from vtscore.config.models import (
    AST_MODEL_ID,
    AST_SAMPLE_RATE,
    BEATS_CHECKPOINT_FILE,
    BEATS_CHECKPOINT_REPO,
    BEATS_EMBED_DIM,
    BEATS_FBANK_MEAN,
    BEATS_FBANK_STD,
    BEATS_MAX_SAMPLES,
    BEATS_MIN_SAMPLES,
    BEATS_SAMPLE_RATE,
    BGE_MODEL_ID,
    CLAP_GENERAL_MODEL_ID,
    CLAP_MODEL_ID,
    CLAP_MUSIC_MODEL_ID,
    CLAP_SAMPLE_RATE,
    CLIP_L_MODEL_ID,
    CLIP_MODEL_ID,
    DINOV2_MODEL_ID,
    DINOV3_MODEL_ID,
    E5_MODEL_ID,
    EUPE_MODEL_ID,
    LANGUAGEBIND_VIDEO_MODEL_ID,
    PARASPEECHCLAP_CHECKPOINT_FILE,
    PARASPEECHCLAP_CHECKPOINT_REPO,
    PARASPEECHCLAP_EMBED_DIM,
    PARASPEECHCLAP_MAX_SAMPLES,
    PARASPEECHCLAP_SAMPLE_RATE,
    PARASPEECHCLAP_SPEECH_MODEL_ID,
    PARASPEECHCLAP_TEXT_MODEL_ID,
    SIGLIP2_L_MODEL_ID,
    SIGLIP2_MODEL_ID,
    SIGLIP_L_MODEL_ID,
    SIGLIP_L_PRETRAINED,
    SIGLIP_MODEL_ID,
    VIDEOMAE_MODEL_ID,
    WHISPER_MODEL_ID,
    WHISPER_SAMPLE_RATE,
    XCLIP_MODEL_ID,
)
from vtscore.config.paths import DATA_DIR, EMBEDDINGS_DIR, MODELS_CACHE_DIR
from vtscore.config.processor_backend import (
    IMAGE_PROCESSOR_BACKEND,
    IMAGE_PROCESSOR_DEVICE,
    image_processor_call_kwargs,
    image_processor_load_kwargs,
    processor_backend_from_class_name,
    resolved_processor_backend,
    verify_image_processor_backend,
)
from vtscore.config.runtime import (
    DEFAULT_CALIBRATE_COUNT,
    DEFAULT_DECODE_WORKER_CAP,
    MAX_DECODE_PIXELS,
    MAX_UPLOAD_MB,
    MLP_DROPOUT,
    MLP_HIDDEN_MAX,
    MLP_HIDDEN_MIN,
    MLP_LABEL_SMOOTHING,
    PROJECTION_COMPACT_DEFAULT,
    PROJECTION_DEFAULTS_BY_EMBEDDER,
    PROJECTION_MIN_DIST,
    PROJECTION_N_NEIGHBORS,
    SVM_HEAD_C,
    TORCH_THREADS,
    TRAIN_EPOCHS,
    TRAIN_PATIENCE,
    allocated_cpus,
    resolve_decode_workers,
)


#: Every public name the pre-split module exposed, so ``vtscore.config.X``
#: and ``from vtscore.config import X`` keep resolving unchanged.  Grouped by
#: submodule, in the order they are imported above.
__all__ = [
    "CoreConfig",
    "register_core_config_builder",
    "DEVICE",
    "EMBED_PRECISION",
    "embed_autocast_dtype",
    "embed_precision",
    "embed_weight_dtype",
    "resolve_device",
    "AST_MODEL_ID",
    "AST_SAMPLE_RATE",
    "BEATS_CHECKPOINT_FILE",
    "BEATS_CHECKPOINT_REPO",
    "BEATS_EMBED_DIM",
    "BEATS_FBANK_MEAN",
    "BEATS_FBANK_STD",
    "BEATS_MAX_SAMPLES",
    "BEATS_MIN_SAMPLES",
    "BEATS_SAMPLE_RATE",
    "BGE_MODEL_ID",
    "CLAP_GENERAL_MODEL_ID",
    "CLAP_MODEL_ID",
    "CLAP_MUSIC_MODEL_ID",
    "CLAP_SAMPLE_RATE",
    "CLIP_L_MODEL_ID",
    "CLIP_MODEL_ID",
    "DINOV2_MODEL_ID",
    "DINOV3_MODEL_ID",
    "E5_MODEL_ID",
    "EUPE_MODEL_ID",
    "LANGUAGEBIND_VIDEO_MODEL_ID",
    "PARASPEECHCLAP_CHECKPOINT_FILE",
    "PARASPEECHCLAP_CHECKPOINT_REPO",
    "PARASPEECHCLAP_EMBED_DIM",
    "PARASPEECHCLAP_MAX_SAMPLES",
    "PARASPEECHCLAP_SAMPLE_RATE",
    "PARASPEECHCLAP_SPEECH_MODEL_ID",
    "PARASPEECHCLAP_TEXT_MODEL_ID",
    "SIGLIP2_L_MODEL_ID",
    "SIGLIP2_MODEL_ID",
    "SIGLIP_L_MODEL_ID",
    "SIGLIP_L_PRETRAINED",
    "SIGLIP_MODEL_ID",
    "VIDEOMAE_MODEL_ID",
    "WHISPER_MODEL_ID",
    "WHISPER_SAMPLE_RATE",
    "XCLIP_MODEL_ID",
    "DATA_DIR",
    "EMBEDDINGS_DIR",
    "MODELS_CACHE_DIR",
    "IMAGE_PROCESSOR_BACKEND",
    "IMAGE_PROCESSOR_DEVICE",
    "image_processor_call_kwargs",
    "image_processor_load_kwargs",
    "processor_backend_from_class_name",
    "resolved_processor_backend",
    "verify_image_processor_backend",
    "DEFAULT_CALIBRATE_COUNT",
    "DEFAULT_DECODE_WORKER_CAP",
    "MAX_DECODE_PIXELS",
    "MAX_UPLOAD_MB",
    "MLP_DROPOUT",
    "MLP_HIDDEN_MAX",
    "MLP_HIDDEN_MIN",
    "MLP_LABEL_SMOOTHING",
    "PROJECTION_COMPACT_DEFAULT",
    "PROJECTION_DEFAULTS_BY_EMBEDDER",
    "PROJECTION_MIN_DIST",
    "PROJECTION_N_NEIGHBORS",
    "SVM_HEAD_C",
    "TORCH_THREADS",
    "TRAIN_EPOCHS",
    "TRAIN_PATIENCE",
    "allocated_cpus",
    "resolve_decode_workers",
]

#: Every submodule of this package, in dependency order (a module only reads
#: from ones earlier in the tuple).  ``tests_lib/core/test_config_package.py``
#: pins it against the files on disk, so a seventh module cannot be added
#: without landing here.
_RELOAD_ORDER = ("paths", "runtime", "models", "device", "processor_backend", "core_config")


def _reload_all() -> ModuleType:
    """Re-execute every submodule, then this package, and return it.

    What ``importlib.reload(vtscore.config)`` did before the split: re-read every
    environment variable this package consults at import time.  Reloading the
    package alone no longer does that (the submodules are cached in
    ``sys.modules`` and the re-exports above just rebind the stale values), so
    tests that exercise an env var go through here instead.

    Reloading resets module state as well as constants - notably
    ``core_config._core_config_builder`` drops back to ``None``, exactly as a
    whole-module reload always has.  Callers that need the session's state back
    afterwards must snapshot and restore each submodule's ``__dict__``, not just
    this package's.
    """
    for name in _RELOAD_ORDER:
        submodule = sys.modules.get(f"{__name__}.{name}")
        if submodule is not None:
            importlib.reload(submodule)
    return importlib.reload(sys.modules[__name__])
