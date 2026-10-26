"""How the app reaches the ``imagekit`` face-provider seam.

The face path has two implementations side by side.  The shipped default is
MTCNN + FaceNet (``facenet-pytorch``), measured on the FHIBE face bench and
untouched by this module.  The second is the ``imagekit`` package's provider
seam (:mod:`imagekit.face.pipeline`): one ``FacePipeline`` that detects, aligns
and embeds every face in an image, behind which a server swaps the backend
with an environment variable - ``oss`` (open SCRFD + ArcFace/MagFace ONNX
weights) or ``vista`` (the proprietary ``vistaimage`` package) - and the app
never sees which.

Two environment variables, read here and nowhere else in the app:

* ``IMAGEKIT_FACE_PROVIDER`` picks the provider (``oss`` when unset; the
  package's own default, :data:`imagekit.face.PROVIDER_ENV_VAR`).
* ``IMAGEKIT_FACE_CONFIG`` names the pipeline config JSON (model paths, runtime
  block; the schema is in ``imagekit/README.md``).  Without it the seam is
  *unconfigured*: the converter's ``imagekit`` provider produces no faces and
  the embedder declines every item, both with a log line naming the variable,
  rather than guessing a model path.

The pipeline is built once per process (:func:`get_face_pipeline`) and shared
by the ``image2face`` converter and the ``imagekit_face`` embedder, the way
every other model here is a process singleton.  Runtime options come from the
app's own device and thread settings.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imagekit.face.pipeline import FacePipeline
    from imagekit.runtime import RuntimeOptions

logger = logging.getLogger(__name__)

#: Names the pipeline config JSON the server's provider loads from.
CONFIG_ENV_VAR = "IMAGEKIT_FACE_CONFIG"

#: The registry name of the embedder that fronts the seam, and the key the
#: converter files a face's vector under.
EMBEDDER_NAME = "imagekit_face"

#: ``custom_metadata`` keys the converter stamps on every face it embeds
#: through the seam.  The compat tag is what the embedder checks before it
#: will embed anything further into the same dataset.
COMPAT_METADATA_KEY = "Face Compat"
QUALITY_METADATA_KEY = "Face Quality"


class FaceProviderUnconfigured(RuntimeError):
    """``IMAGEKIT_FACE_CONFIG`` is unset, so no provider can be built."""


_lock = threading.Lock()
_pipeline: Any = None
_compat = ""


def runtime_options() -> "RuntimeOptions":
    """The app's device and thread settings as :class:`imagekit.runtime.RuntimeOptions`."""
    from imagekit.runtime import RuntimeOptions  # noqa: PLC0415

    from vtscore.config import resolve_device  # noqa: PLC0415
    from vtscore.config.runtime import TORCH_THREADS  # noqa: PLC0415

    device = resolve_device()
    use_gpu = device.startswith("cuda")
    _, _, index = device.partition(":")
    return RuntimeOptions(
        use_gpu=use_gpu, gpu_device_id=int(index) if index.isdigit() else 0, num_threads=TORCH_THREADS
    )


def provider_name() -> str:
    """The provider the server selects (``IMAGEKIT_FACE_PROVIDER``, else ``oss``)."""
    from imagekit.face.pipeline import default_provider_name  # noqa: PLC0415

    return default_provider_name()


def pipeline_config() -> dict[str, Any]:
    """The pipeline config ``IMAGEKIT_FACE_CONFIG`` names; raises :class:`FaceProviderUnconfigured` when unset."""
    from imagekit.face.pipeline import read_pipeline_config  # noqa: PLC0415

    path = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if not path:
        raise FaceProviderUnconfigured(
            f"the imagekit face provider is not configured: set {CONFIG_ENV_VAR} to a pipeline config JSON "
            "(model paths and runtime; see imagekit/README.md), or use the mtcnn provider"
        )
    return read_pipeline_config(path)


def get_face_pipeline() -> "FacePipeline":
    """The process-wide, initialised pipeline for the server's provider; built on first use.

    Raises :class:`FaceProviderUnconfigured` when the config variable is
    unset, and whatever the provider raises when it cannot load (a missing
    ``vistaimage``, a bad model path).  A failure is not cached: the next call
    tries again, so a fixed environment needs no restart of the import.
    """
    global _pipeline, _compat
    with _lock:
        if _pipeline is not None:
            return _pipeline
        from imagekit.face.pipeline import load_pipeline  # noqa: PLC0415

        config = pipeline_config()
        pipeline = load_pipeline(config, runtime_options(), provider_name())
        _compat = pipeline.compatibility()
        _pipeline = pipeline
        logger.info("imagekit face provider %r ready (compat %s)", provider_name(), _compat)
        return pipeline


def loaded_compat() -> str:
    """The compat tag of the pipeline already built, or ``""`` before :func:`get_face_pipeline` ran."""
    return _compat


def reset_for_tests() -> None:
    """Drop the cached pipeline so a test can install a fake or re-run the build."""
    global _pipeline, _compat
    with _lock:
        _pipeline = None
        _compat = ""
