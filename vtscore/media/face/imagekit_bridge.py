"""How the app reaches the ``imagekit`` face-provider seam, and whether it can.

Every face in the app goes through the ``imagekit`` package's provider seam
(:mod:`imagekit.face.pipeline`): one ``FacePipeline`` that detects, aligns and
embeds every face in an image, behind which a server swaps the backend - ``oss``
(open SCRFD + ArcFace/MagFace ONNX weights) or ``ext`` (an external backend
module) - and the app never sees which.  The ``image2face`` converter and the
``face`` embedder both reach it through this module.

**The app downloads no face models.**  It uses the ones a deployment puts on
disk, and Face is on exactly when they are there (:func:`face_models`):

* ``IMAGEKIT_FACE_CONFIG`` names the pipeline config JSON (model paths, runtime
  block; the schema is in ``imagekit/README.md``).  Unset, the app looks for
  one at :func:`default_config_path`, ``imagekit/face.json`` in the models
  folder (``VTSEARCH_MODELS_DIR``), so dropping a models directory there turns
  Face on with no environment to set.
* ``IMAGEKIT_FACE_PROVIDER`` picks the provider; unset, the config's own
  ``provider`` key, else ``oss``.  The ``ext`` provider imports the backend
  module ``IMAGEKIT_EXT_BACKEND`` names, else the config's ``ext_backend``,
  else ``ext_backend``.

Without a config whose models exist, Face is *off*: the ``image2face``
converter hides itself from every picker and refuses to run, naming what is
missing, and the app's listings drop the ``face`` media type and embedder
(``vtsearch.settings.unavailable_plugins``).  Face datasets already built still
open: their vectors are in the pickle.

The pipeline is built once per process (:func:`get_face_pipeline`) and shared
by the converter and the embedder, the way every other model here is a process
singleton.  Runtime options come from the app's own device and thread settings.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from importlib import util as importlib_util
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Sequence

if TYPE_CHECKING:
    from imagekit.face.pipeline import FacePipeline
    from imagekit.runtime import RuntimeOptions

logger = logging.getLogger(__name__)

#: Names the pipeline config JSON the server's provider loads from.
CONFIG_ENV_VAR = "IMAGEKIT_FACE_CONFIG"

#: Where, under the models folder, the app looks for a pipeline config when
#: :data:`CONFIG_ENV_VAR` is unset.
DEFAULT_CONFIG_RELPATH = Path("imagekit") / "face.json"

#: The registry name of the face embedder, and the key the converter files a
#: face's vector under.
EMBEDDER_NAME = "face"

#: ``custom_metadata`` keys the converter stamps on every face it embeds.  The
#: compat tag is what the embedder checks before it will embed anything further
#: into the same dataset.
COMPAT_METADATA_KEY = "Face Compat"
QUALITY_METADATA_KEY = "Face Quality"

#: The Face Quality below which a face is a *hard view* (turned, covered, small)
#: and a search started from it alone tends to miss the person (#5372), per
#: embedding space: a raw embedding's norm means nothing across models, so a
#: space with no entry gets no verdict.  Read from the FHIBE one-photo sessions
#: on buffalo_s, at 1024 px and 640 px: a starting face under 14 flags 1.2% of
#: sessions and 11 of the 12 that fail at every beta.  A session whose best
#: starting face clears it did not fail, four-photo sessions included.
HARD_VIEW_QUALITY: dict[str, float] = {"oss-arcface-mbf:1": 14.0}

#: The weight a hard-view starting face takes in the example centroid when at
#: least one other starting face is clear (#5389); a clear face weighs 1.  It
#: still counts as a Good vote.  On FHIBE's four-photo sessions one hard face
#: among clear ones cost F 0.10-0.11 at 50 votes at beta 4, and leaving it out
#: of the session recovered all of it (#5376).
HARD_VIEW_EXAMPLE_WEIGHT = 0.25

#: How long a :func:`face_models` answer is reused.  The listing routes ask once
#: per converter and per demo, so a page load asks dozens of times; a model
#: dropped in still shows up within a couple of seconds, with no restart.
_STATUS_TTL_S = 2.0


class FaceProviderUnconfigured(RuntimeError):
    """This server has no usable face models, so no provider can be built."""


class FaceModels(NamedTuple):
    """Whether this server has face models, and where they are."""

    available: bool
    #: Why Face is off, naming what is missing; ``""`` when it is on.
    reason: str
    #: The pipeline config in use (or the one that failed), ``None`` when none was found.
    config_path: Optional[Path]
    #: The provider that config selects.
    provider: str


_lock = threading.Lock()
_pipeline: Any = None
_compat = ""
_provider = ""
_status: Optional[tuple[float, tuple, FaceModels]] = None


def default_config_path() -> Path:
    """Where the app looks for a pipeline config when :data:`CONFIG_ENV_VAR` is unset."""
    from vtscore.config import paths  # noqa: PLC0415

    return Path(paths.MODELS_CACHE_DIR).expanduser() / DEFAULT_CONFIG_RELPATH


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


def _env_provider() -> str:
    from imagekit.face.pipeline import PROVIDER_ENV_VAR  # noqa: PLC0415

    return os.environ.get(PROVIDER_ENV_VAR, "").strip()


def _config_provider(config: dict[str, Any]) -> str:
    """The provider *config* runs under: the environment's, else its own ``provider``, else ``oss``."""
    from imagekit.face.pipeline import DEFAULT_PROVIDER, PROVIDER_KEY  # noqa: PLC0415

    return _env_provider() or str(config.get(PROVIDER_KEY) or "").strip() or DEFAULT_PROVIDER


def _missing_paths(config: dict[str, Any], provider: str) -> list[str]:
    """The model files *config* names that are not on disk, each as ``"<what> <path>"``."""
    from imagekit.face.pipeline import BUILTIN_PROVIDERS, DETECTOR_KEY, EMBEDDER_KEY  # noqa: PLC0415

    detector = config.get(DETECTOR_KEY)
    embedder = config.get(EMBEDDER_KEY)
    detector = detector if isinstance(detector, dict) else {}
    embedder = embedder if isinstance(embedder, dict) else {}
    wanted = {
        "detector model": detector.get("model_bundle") or detector.get("model"),
        "detector config": detector.get("config"),
        "embedder model": embedder.get("model"),
        "embedder config": embedder.get("config"),
    }
    missing: list[str] = []
    for what, path in wanted.items():
        if path and not Path(str(path)).exists():
            missing.append(f"{what} {path}")
        elif not path and what.endswith("model") and provider in BUILTIN_PROVIDERS:
            missing.append(f"{what} (the config names none)")
    return missing


def _importable(module: str) -> bool:
    """Whether *module* is installed here, found without importing it (a dotted path's parents aside)."""
    try:
        return importlib_util.find_spec(module) is not None
    except ValueError:  # already imported, but built in memory with no spec
        return True
    except ImportError:  # a dotted path whose parent package is missing
        return False


def _check() -> FaceModels:
    """:func:`face_models` without the cache: the config, its provider and its model files, on disk."""
    from imagekit.face.pipeline import DEFAULT_PROVIDER, available_providers, read_pipeline_config  # noqa: PLC0415

    named = os.environ.get(CONFIG_ENV_VAR, "").strip()
    path = Path(named).expanduser() if named else default_config_path()
    provider = _env_provider() or DEFAULT_PROVIDER
    if not named and not path.is_file():
        return FaceModels(
            False,
            f"no face models: put a pipeline config at {path}, or set {CONFIG_ENV_VAR} to one "
            "(imagekit/README.md says where models come from)",
            None,
            provider,
        )
    if not path.is_file():
        return FaceModels(False, f"{CONFIG_ENV_VAR} names {path}, which is not a file", path, provider)
    try:
        config = read_pipeline_config(path)
    except Exception as exc:  # a broken JSON file is a deployment error, reported, not raised
        return FaceModels(False, f"{path} is not a readable pipeline config: {exc}", path, provider)
    provider = _config_provider(config)
    if provider not in available_providers():
        return FaceModels(
            False, f"{path} selects face provider {provider!r}; available: {available_providers()}", path, provider
        )
    if provider == "ext":
        from imagekit.face.providers.ext import backend_module_name  # noqa: PLC0415

        backend = backend_module_name(config)
        if not _importable(backend):
            return FaceModels(
                False,
                f"the 'ext' face provider needs its backend module {backend!r}, not installed here",
                path,
                provider,
            )
    missing = _missing_paths(config, provider)
    if missing:
        return FaceModels(False, f"{path} names face models that are missing: {'; '.join(missing)}", path, provider)
    return FaceModels(True, "", path, provider)


def face_models() -> FaceModels:
    """Whether this server has face models, checked on disk without loading any.

    Cheap enough for a listing route to call: a small JSON read and a few
    ``stat`` calls, reused for :data:`_STATUS_TTL_S`.  A model that is present
    but cannot load (a corrupt file) still reads as available here; it fails at
    :func:`get_face_pipeline`, loudly.

    So does a server that cannot import ``imagekit`` at all: a slim install
    leaves out the face path's own dependencies (the LabBench images have no
    ``cv2``), and that makes a server without Face, not one that cannot start.
    """
    global _status
    try:
        from imagekit.face.providers.ext import BACKEND_ENV_VAR  # noqa: PLC0415
    except ImportError as exc:
        return FaceModels(False, f"the face path's dependencies are not installed here ({exc})", None, "")

    key = (
        os.environ.get(CONFIG_ENV_VAR, ""),
        _env_provider(),
        os.environ.get(BACKEND_ENV_VAR, ""),
        str(default_config_path()),
    )
    now = time.monotonic()
    cached = _status
    if cached is not None and cached[1] == key and now - cached[0] < _STATUS_TTL_S:
        return cached[2]
    status = _check()
    _status = (now, key, status)
    return status


def provider_name() -> str:
    """The provider this server runs: the built pipeline's, else the one its config selects."""
    return _provider or face_models().provider


def get_face_pipeline() -> "FacePipeline":
    """The process-wide, initialised pipeline for the server's provider; built on first use.

    Raises :class:`FaceProviderUnconfigured`, naming what is missing, when the
    server has no face models (:func:`face_models`), and whatever the provider
    raises when they cannot load (a missing ``ext`` backend module, a corrupt model).
    A failure is not cached: the next call tries again, so models put in place
    later need no restart.
    """
    global _pipeline, _compat, _provider
    with _lock:
        if _pipeline is not None:
            return _pipeline
        from imagekit.face.pipeline import load_pipeline, read_pipeline_config  # noqa: PLC0415

        status = _check()
        if not status.available or status.config_path is None:
            raise FaceProviderUnconfigured(status.reason)
        pipeline = load_pipeline(read_pipeline_config(status.config_path), runtime_options(), status.provider)
        _compat = pipeline.compatibility()
        _provider = status.provider
        _pipeline = pipeline
        logger.info("face provider %r ready from %s (compat %s)", status.provider, status.config_path, _compat)
        return pipeline


def loaded_compat() -> str:
    """The compat tag of the pipeline already built, or ``""`` before :func:`get_face_pipeline` ran."""
    return _compat


def reset_for_tests() -> None:
    """Drop the cached pipeline and availability answer so a test can install a fake or re-run the build."""
    global _pipeline, _compat, _provider, _status
    with _lock:
        _pipeline = None
        _compat = ""
        _provider = ""
        _status = None


def is_hard_view(quality: Any, compat: Any) -> Optional[bool]:
    """Whether a face of this *quality* is a hard view (:data:`HARD_VIEW_QUALITY`).

    ``None`` when there is no verdict: no quality, or a *compat* space with no cut.
    """
    cut = HARD_VIEW_QUALITY.get(str(compat or ""))
    if cut is None or not isinstance(quality, (int, float)) or isinstance(quality, bool):
        return None
    return float(quality) < cut


def hard_view_of_file(path: Path) -> Optional[bool]:
    """:func:`is_hard_view` for a face picture on disk, read by the server's provider.

    The face is the one the face embedder would embed (the provider's first
    detection).  ``None`` when Face is off, nothing is detected, or the provider
    fails: the verdict is advice, so it never stands in the way of the picture.
    """
    if not face_models().available:
        return None
    try:
        from imagekit.util import read_image_tensor  # noqa: PLC0415

        records = get_face_pipeline().process(read_image_tensor(Path(path)), media_id=Path(path).name)
    except Exception:
        logger.warning("could not read the Face Quality of %s", path, exc_info=True)
        return None
    if not records:
        return None
    return is_hard_view(records[0].quality, loaded_compat())


def example_weights(hard: Sequence[Optional[bool]]) -> Optional[list[float]]:
    """Each starting face's weight in the example centroid, one per :func:`is_hard_view` verdict in *hard*.

    A hard view weighs :data:`HARD_VIEW_EXAMPLE_WEIGHT` and every other face 1,
    but only while some face is clear (``False``): with none clear, there is no
    better face to lean on and every face weighs the same.  ``None`` when every
    weight would be 1, so a caller keeps its plain mean.
    """
    if HARD_VIEW_EXAMPLE_WEIGHT == 1.0 or not any(h is False for h in hard) or not any(h is True for h in hard):
        return None
    return [HARD_VIEW_EXAMPLE_WEIGHT if h is True else 1.0 for h in hard]


def media_hard_view(media: dict) -> Optional[bool]:
    """:func:`is_hard_view` for a loaded face, from the quality and space its converter stamped on it."""
    meta = media.get("custom_metadata") or {}
    return is_hard_view(meta.get(QUALITY_METADATA_KEY), meta.get(COMPAT_METADATA_KEY))
