"""The provider seam: the ``FacePipeline`` protocol and the registry that hands one out.

An application asks :func:`get_pipeline` for a pipeline by provider name -
``"oss"`` (this package's detector + embedder), ``"ext"`` (an external backend
module behind the same interface), or any name a third party registered - and
from then on sees only :class:`FacePipeline` and
:class:`~imagekit.face.records.FaceRecord`.  Swapping the backend is one
environment variable (:data:`PROVIDER_ENV_VAR`) on the server that has it.

Third parties register in two ways: :func:`register_provider` at runtime, or
an ``importlib.metadata`` entry point in the :data:`ENTRY_POINT_GROUP` group
whose value is a zero-argument factory returning a pipeline::

    [project.entry-points."imagekit.face_providers"]
    acme = "acme_faces:AcmeFacePipeline"

Every pipeline is handed the same ``config`` mapping at ``initialize()``
(:func:`read_pipeline_config` reads one from JSON); the keys the built-in
providers read are documented on :func:`resolve_config_paths`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from imagekit.face.records import FaceRecord
from imagekit.runtime import RuntimeOptions

logger = logging.getLogger(__name__)

#: Environment variable naming the default provider.
PROVIDER_ENV_VAR = "IMAGEKIT_FACE_PROVIDER"

#: The provider used when the environment names none.
DEFAULT_PROVIDER = "oss"

#: Entry-point group third-party providers register into.
ENTRY_POINT_GROUP = "imagekit.face_providers"

#: Provider names this package itself supplies.
BUILTIN_PROVIDERS: tuple[str, ...] = ("oss", "ext")


@runtime_checkable
class FacePipeline(Protocol):
    """Detect, align and embed every face in an image, as :class:`FaceRecord` s.

    ``initialize`` is called once with the runtime options and a config
    mapping; ``process`` / ``process_buffer`` then return one record per
    detected face (every detection is embedded, not just the first), highest
    confidence first; ``compatibility`` names the embedding space the records
    live in.  ``media_id`` is stamped onto each record so a caller can key the
    results; it is keyword-only and optional.
    """

    def initialize(self, opts: RuntimeOptions, config: Mapping[str, Any]) -> None: ...

    def process(
        self, image: Any, *, confidence_threshold: float | None = None, media_id: str = ""
    ) -> list[FaceRecord]: ...

    def process_buffer(
        self, buffer: bytes, *, confidence_threshold: float | None = None, media_id: str = ""
    ) -> list[FaceRecord]: ...

    def compatibility(self) -> str: ...


PipelineFactory = Callable[[], FacePipeline]

_registry: dict[str, PipelineFactory] = {}
_builtins_registered = False
_entry_points_loaded = False


def register_provider(name: str, factory: PipelineFactory, *, replace: bool = False) -> None:
    """Make :func:`get_pipeline` build *name* with *factory* (a zero-argument callable).

    Registering a name that is already taken raises unless *replace* is set,
    so two plugins cannot silently shadow each other.
    """
    key = str(name).strip()
    if not key:
        raise ValueError("provider name must be non-empty")
    _ensure_builtins()
    if key in _registry and not replace:
        raise ValueError(f"face provider {key!r} is already registered; pass replace=True to override it")
    _registry[key] = factory


def unregister_provider(name: str) -> bool:
    """Forget *name*; returns whether it was registered."""
    return _registry.pop(str(name).strip(), None) is not None


def reset_for_tests() -> None:
    """Forget every registration, built-ins included; the next lookup re-registers them.

    The registry is process-global, so a test suite calls this before each test
    to keep a fake provider from leaking into the next one.
    """
    global _builtins_registered, _entry_points_loaded
    _registry.clear()
    _builtins_registered = False
    _entry_points_loaded = False


def _ensure_builtins() -> None:
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True
    from imagekit.face.providers.ext import ExtFacePipeline  # noqa: PLC0415
    from imagekit.face.providers.oss import OssFacePipeline  # noqa: PLC0415

    _registry.setdefault("oss", OssFacePipeline)
    _registry.setdefault("ext", ExtFacePipeline)


def _load_entry_points() -> None:
    """Register every :data:`ENTRY_POINT_GROUP` entry point not already registered; once."""
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in _registry:
            logger.warning("face provider entry point %r (%s) ignored: the name is taken", ep.name, ep.value)
            continue
        try:
            _registry[ep.name] = ep.load()
        except Exception:
            logger.exception("face provider entry point %r (%s) failed to load", ep.name, ep.value)


def available_providers() -> list[str]:
    """Every provider name :func:`get_pipeline` accepts right now, sorted."""
    _ensure_builtins()
    _load_entry_points()
    return sorted(_registry)


def default_provider_name() -> str:
    """The provider :func:`get_pipeline` picks when given none: the environment's, else ``"oss"``."""
    return os.environ.get(PROVIDER_ENV_VAR, "").strip() or DEFAULT_PROVIDER


def get_pipeline(name: str | None = None) -> FacePipeline:
    """A **new, uninitialised** pipeline for provider *name* (default: :func:`default_provider_name`).

    Raises :class:`KeyError` naming the available providers when *name* is unknown.
    """
    key = (name or default_provider_name()).strip()
    _ensure_builtins()
    if key not in _registry:
        _load_entry_points()
    if key not in _registry:
        raise KeyError(f"unknown face provider {key!r}; available: {available_providers()}")
    return _registry[key]()


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

#: Config keys the built-in providers read; see :func:`resolve_config_paths`.
PROVIDER_KEY = "provider"
RUNTIME_KEY = "runtime"
DETECTOR_KEY = "detector"
EMBEDDER_KEY = "embedder"
_PATH_KEYS: tuple[tuple[str, str], ...] = (
    (DETECTOR_KEY, "model_bundle"),
    (DETECTOR_KEY, "config"),
    (EMBEDDER_KEY, "model"),
    (EMBEDDER_KEY, "config"),
)


def runtime_options_from_config(config: Mapping[str, Any]) -> RuntimeOptions:
    """The ``runtime`` block of a pipeline config as :class:`RuntimeOptions` (defaults when absent)."""
    block = config.get(RUNTIME_KEY) or {}
    if not isinstance(block, Mapping):
        raise ValueError("pipeline config: 'runtime' must be a JSON object")
    return RuntimeOptions(
        use_gpu=bool(block.get("use_gpu", False)),
        gpu_device_id=int(block.get("gpu_device_id", 0)),
        num_threads=int(block.get("num_threads", 0)),
    )


def resolve_config_paths(config: Mapping[str, Any], base_dir: str | Path) -> dict[str, Any]:
    """A copy of *config* with its model paths made absolute against *base_dir*.

    The pipeline config is one JSON object::

        {
          "provider": "oss",                       # optional; the env var / "oss" otherwise
          "runtime":  {"use_gpu": false, "gpu_device_id": 0, "num_threads": 0},
          "detector": {"model_bundle": "models/det_10g.onnx",
                       "config": null,            # optional <model>.json override
                       "confidence_threshold": 0.5},
          "embedder": {"model": "models/w600k_r50.onnx",
                       "config": "models/arcface_r50.json"}
        }

    Relative paths are relative to the file the config was read from, which
    is what lets a models directory move as one unit.
    """
    out: dict[str, Any] = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in config.items()}
    base = Path(base_dir)
    for block, key in _PATH_KEYS:
        section = out.get(block)
        if isinstance(section, dict) and section.get(key):
            path = Path(str(section[key]))
            section[key] = str(path if path.is_absolute() else (base / path).resolve())
    return out


def read_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Read a pipeline config JSON file, resolving its paths against the file's directory."""
    import json  # noqa: PLC0415

    config_path = Path(path)
    with open(config_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: pipeline config must be a JSON object")
    return resolve_config_paths(raw, config_path.parent)


def load_pipeline(
    config: Mapping[str, Any],
    opts: RuntimeOptions | None = None,
    name: str | None = None,
) -> FacePipeline:
    """:func:`get_pipeline` plus ``initialize``: the one-call form.

    The provider is *name*, else the config's ``provider``, else the
    environment's; the runtime options are *opts*, else the config's
    ``runtime`` block.
    """
    provider = name or str(config.get(PROVIDER_KEY) or "") or None
    pipeline = get_pipeline(provider)
    pipeline.initialize(opts if opts is not None else runtime_options_from_config(config), config)
    return pipeline
