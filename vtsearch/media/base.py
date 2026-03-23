"""Abstract base classes for media types and processors.

To add a new media type:

1. Create a subdirectory under ``vtsearch/media/`` (e.g. ``vtsearch/media/code/``).
2. Add a ``requirements.txt`` listing any pip packages your embedder needs.
3. Implement a subclass of :class:`MediaType` in ``media_type.py``.
4. Register it in ``vtsearch/media/__init__.py``::

       from vtsearch.media.code.media_type import CodeMediaType
       register(CodeMediaType())

That is all.  The rest of the application (routing, dataset loading, model
initialisation, demo listing) picks up your new type automatically through
the registry.
"""

from __future__ import annotations

import contextlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Type alias for progress callbacks.  Modules that accept an ``on_progress``
# parameter use this signature so callers can report status without depending
# on ``vtsearch.utils.progress``.
ProgressCallback = Callable[[str, str, int, int], None]

__all__ = [
    "DemoDataset",
    "Detector",
    "Extractor",
    "MediaClipper",
    "MediaEmbedder",
    "MediaResponse",
    "MediaType",
    "Processor",
    "ProgressCallback",
    "intercept_tqdm_progress",
    "intercept_weight_loading_progress",
]


def _noop_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
    """Default no-op progress callback used when no real reporter is set."""


@contextlib.contextmanager
def intercept_tqdm_progress(callback: ProgressCallback) -> Any:
    """Temporarily hook tqdm progress bars to forward updates to *callback*.

    HuggingFace ``transformers`` and ``huggingface_hub`` use :mod:`tqdm` for
    download and weight-loading progress bars.  Those bars write to stderr,
    which the GUI never sees.  This context manager monkey-patches the base
    ``tqdm.std.tqdm`` class so that every ``update()`` call also pushes
    ``(status, message, current, total)`` to *callback*.

    All tqdm subclasses (``tqdm.auto.tqdm``, ``huggingface_hub.utils.tqdm``,
    etc.) resolve ``update`` through MRO to ``tqdm.std.tqdm.update``, so a
    single patch covers the entire hierarchy.

    Only bars with a known *total* are forwarded; indeterminate spinners are
    silently ignored.
    """
    import tqdm.std

    _orig_init = tqdm.std.tqdm.__init__
    _orig_update = tqdm.std.tqdm.update
    _orig_close = tqdm.std.tqdm.close

    # Track all active bars.  We report progress from the bar with the
    # largest ``total`` (typically the model weight file download).
    _bars: list[tqdm.std.tqdm] = []

    def _primary_bar() -> tqdm.std.tqdm | None:
        if not _bars:
            return None
        return max(_bars, key=lambda b: getattr(b, "total", 0) or 0)

    def _report(bar: tqdm.std.tqdm) -> None:
        total = getattr(bar, "total", None)
        if not total or total <= 0:
            return
        current = int(getattr(bar, "n", 0))
        desc = (getattr(bar, "desc", "") or "Loading…").rstrip(": ")
        callback("loading", desc, current, int(total))

    # Redirect intercepted bars' output to devnull so they don't print
    # to the console.  The callback receives all progress updates instead.
    _devnull = open(os.devnull, "w")  # noqa: SIM115

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Inject file=devnull so tqdm wraps it in its DisableOnWriteError
        # wrapper during init — suppresses all console output from the bar.
        if "file" not in kwargs:
            kwargs["file"] = _devnull
        _orig_init(self, *args, **kwargs)
        total = getattr(self, "total", None)
        if total and total > 0 and not getattr(self, "disable", False):
            _bars.append(self)
            if _primary_bar() is self:
                _report(self)

    def _patched_update(self: Any, n: int = 1) -> None:
        _orig_update(self, n)
        if _primary_bar() is self:
            _report(self)

    def _patched_close(self: Any) -> None:
        _orig_close(self)
        if self in _bars:
            _bars.remove(self)

    tqdm.std.tqdm.__init__ = _patched_init  # type: ignore[assignment]
    tqdm.std.tqdm.update = _patched_update  # type: ignore[assignment]
    tqdm.std.tqdm.close = _patched_close  # type: ignore[assignment]
    try:
        yield
    finally:
        tqdm.std.tqdm.__init__ = _orig_init  # type: ignore[assignment]
        tqdm.std.tqdm.update = _orig_update  # type: ignore[assignment]
        tqdm.std.tqdm.close = _orig_close  # type: ignore[assignment]
        _devnull.close()


@contextlib.contextmanager
def intercept_weight_loading_progress(callback: ProgressCallback, label: str = "Loading model weights…") -> Any:
    """Track tensor-level progress during model weight loading.

    HuggingFace ``transformers`` with ``low_cpu_mem_usage=True`` dispatches
    tensors one-by-one via ``set_module_tensor_to_device`` from ``accelerate``.
    PyTorch's ``load_state_dict`` (used by ``sentence-transformers``) loads
    tensors via ``__getitem__`` on the state dict.

    This context manager monkey-patches both paths to count tensor operations
    and report ``(current, total)`` progress via *callback*.  The total is
    discovered by also intercepting ``safetensors.torch.load_file`` and
    ``torch.load`` to count keys in loaded state dicts.
    """
    _counter = [0]
    _total = [0]
    _patches: list[tuple] = []

    def _report() -> None:
        if _total[0] > 0:
            callback("loading", label, min(_counter[0], _total[0]), _total[0])

    # --- Intercept safetensors.torch.load_file to learn total tensor count ---
    try:
        import safetensors.torch as _st  # noqa: PLC0415

        _orig_lf = _st.load_file

        def _tracked_lf(*a: Any, **kw: Any) -> Any:
            r = _orig_lf(*a, **kw)
            _total[0] += len(r)
            return r

        _st.load_file = _tracked_lf
        _patches.append((_st, "load_file", _orig_lf))
    except ImportError:
        pass

    # --- Intercept torch.load for .bin weight files ---
    try:
        import torch as _torch  # noqa: PLC0415

        _orig_tl = _torch.load

        def _tracked_tl(*a: Any, **kw: Any) -> Any:
            r = _orig_tl(*a, **kw)
            if isinstance(r, dict) and r:
                sample = next(iter(r.values()))
                if isinstance(sample, _torch.Tensor):
                    _total[0] += len(r)
            return r

        _torch.load = _tracked_tl
        _patches.append((_torch, "load", _orig_tl))
    except ImportError:
        pass

    # --- Intercept set_module_tensor_to_device (HF with low_cpu_mem_usage) ---
    try:
        import transformers.modeling_utils as _tm  # noqa: PLC0415

        _orig_smttd = _tm.set_module_tensor_to_device

        def _tracked_smttd(*a: Any, **kw: Any) -> Any:
            r = _orig_smttd(*a, **kw)
            _counter[0] += 1
            _report()
            return r

        _tm.set_module_tensor_to_device = _tracked_smttd
        _patches.append((_tm, "set_module_tensor_to_device", _orig_smttd))
    except (ImportError, AttributeError):
        pass

    # --- Intercept Module.load_state_dict (PyTorch / SentenceTransformers) ---
    try:
        import torch.nn as _nn  # noqa: PLC0415

        _orig_lsd = _nn.Module.load_state_dict

        class _CountingStateDict(dict):
            """Dict wrapper that counts unique key accesses for progress."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._seen: set = set()

            def __getitem__(self, key: Any) -> Any:
                val = super().__getitem__(key)
                if key not in self._seen:
                    self._seen.add(key)
                    _counter[0] += 1
                    _report()
                return val

        def _tracked_lsd(self_model: Any, state_dict: Any, *a: Any, **kw: Any) -> Any:
            if isinstance(state_dict, dict) and not isinstance(state_dict, _CountingStateDict):
                if _total[0] == 0:
                    _total[0] = len(state_dict)
                state_dict = _CountingStateDict(state_dict)
            return _orig_lsd(self_model, state_dict, *a, **kw)

        _nn.Module.load_state_dict = _tracked_lsd  # type: ignore[assignment]
        _patches.append((_nn.Module, "load_state_dict", _orig_lsd))
    except ImportError:
        pass

    try:
        yield
    finally:
        for obj, attr, orig in _patches:
            setattr(obj, attr, orig)


class MediaEmbedder(ABC):
    """Abstract base class for media embedders.

    A *media embedder* takes a media file (or a text description) and produces
    a fixed-size vector embedding.  Each embedder is associated with exactly one
    :class:`MediaType` (via :attr:`media_type_id`), but a single media type may
    have multiple embedders (e.g. different CLIP variants for images).

    Subclasses must implement:

    * :attr:`name` — unique human-readable identifier (also used as the
      registry key).
    * :attr:`media_type_id` — which media type this embedder works with.
    * :meth:`load_models` — load (and cache) the embedding model.
    * :meth:`embed_media` — embed a media file from disk.
    * :meth:`embed_text` — embed a text query in the same vector space.
    """

    def __init__(self) -> None:
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this embedder, e.g. ``"clap"``, ``"clip"``."""

    @property
    @abstractmethod
    def media_type_id(self) -> str:
        """The ``type_id`` of the media type this embedder works with."""

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load_models(self) -> None:
        """Load (and cache) the embedding model.

        Called lazily the first time this embedder needs to produce a vector.
        Implementations must be idempotent — a second call should be a no-op.
        """

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @abstractmethod
    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        """Return a fixed-size embedding vector for the media file at *file_path*.

        Returns ``None`` if the file cannot be embedded.
        """

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Return an embedding of *text* in the **same vector space** as :meth:`embed_media`.

        The default implementation returns ``None`` (text sorting unavailable).
        """
        return None

    @property
    def description_wrappers(self) -> list[str]:
        """Wrapper templates for enriching sort descriptions.

        Each template is a format string containing ``{text}``.  Override in
        subclasses to provide media-specific wrappers that improve embedding quality.
        """
        return []

    def embed_text_enriched(self, text: str) -> Optional[np.ndarray]:
        """Embed *text* using the average over all description wrappers.

        Falls back to :meth:`embed_text` if no wrappers are defined or all fail.
        """
        wrappers = self.description_wrappers
        if not wrappers:
            return self.embed_text(text)

        embeddings = []
        for wrapper in wrappers:
            wrapped = wrapper.format(text=text)
            vec = self.embed_text(wrapped)
            if vec is not None:
                embeddings.append(vec)

        if not embeddings:
            return self.embed_text(text)

        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary of this embedder."""
        return {
            "name": self.name,
            "media_type_id": self.media_type_id,
        }


@dataclass
class MediaResponse:
    """Framework-agnostic representation of media content for HTTP serving.

    This decouples media type implementations from Flask so they can be used
    as standalone libraries.  The Flask route layer converts this into a real
    ``flask.Response`` via :func:`media_response_to_flask`.

    Attributes:
        data: The payload — ``bytes`` for binary media, ``dict`` for JSON.
        mimetype: MIME type string (e.g. ``"audio/wav"``, ``"application/json"``).
        download_name: Suggested filename for the ``Content-Disposition`` header.
    """

    data: bytes | dict
    mimetype: str
    download_name: str = ""


@dataclass
class DemoDataset:
    """Metadata describing one demo dataset that belongs to a media type."""

    id: str
    """Unique key used throughout the app (e.g. ``"nature_sounds"``)."""

    label: str
    """Human-readable display name (e.g. ``"Animal & Nature Sounds"``)."""

    description: str
    """Long-form description shown in the UI."""

    categories: list
    """Category names used to filter the raw source data."""

    source: str = ""
    """Identifier for the raw data source (e.g. ``"cifar10_sample"``, ``"ucf101"``).
    Leave empty for sources that don't require an explicit identifier."""

    required_folder: Optional[Path] = None
    """Local directory containing the extracted source files for this dataset.

    Used both as a staleness check (a cached ``.pkl`` is only considered valid
    when this directory still exists on disk) and as the browsable root for
    the *Select Media Example* file picker.  Set this to the directory that the
    downloader extracts source files into (e.g.
    ``DATA_DIR / "ESC-50-master" / "audio"`` or
    ``DATA_DIR / "caltech-101" / "101_ObjectCategories"``).
    Leave ``None`` only for datasets that have no on-disk source directory
    (e.g. text datasets generated from in-memory data)."""

    slice_start: int = 0
    """Per-category start index for element slicing (inclusive).

    When multiple datasets share the same categories, this allows them to
    use disjoint subsets of elements within each category."""

    slice_end: Optional[int] = None
    """Per-category end index for element slicing (exclusive).

    ``None`` means take all remaining elements after ``slice_start``."""

    download_size_mb: float = 0
    """Estimated download size in megabytes for this demo dataset's raw data.

    Used by the frontend to display the expected download size before the
    user starts loading.  Set to ``0`` for datasets that don't require a
    network download (e.g. scikit-learn datasets that download automatically)."""


class MediaType(ABC):
    """Abstract base class that every media type must implement.

    A *media type* bundles together everything the application needs to work
    with a particular kind of media:

    * Human-readable identity: :attr:`name` and :attr:`icon`.
    * Which file extensions to scan when importing a folder (:attr:`file_extensions`).
    * Whether the viewer should loop (:attr:`loops`).
    * Which demo datasets are available (:attr:`demo_datasets`).
    * How to serve a media over HTTP (:meth:`media_response`).
    * How to load media-specific media fields from a file (:meth:`load_media_data`).
    * An optional folder-import alias (:attr:`folder_import_name`) for the
      ``/api/dataset/load-folder`` endpoint.

    Embedding is handled by :class:`MediaEmbedder` objects, which are registered
    separately.  A media type may have zero, one, or many embedders.

    Adding a new media type
    -----------------------
    See the module docstring above for the four-step process.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def type_id(self) -> str:
        """Unique internal identifier, e.g. ``"audio"``, ``"video"``."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name shown in the UI, e.g. ``"Audio"``."""

    @property
    @abstractmethod
    def icon(self) -> str:
        """Icon for the UI (emoji or icon name), e.g. ``"🔊"``."""

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def file_extensions(self) -> list:
        """Glob patterns for importable files, e.g. ``["*.wav", "*.mp3"]``."""

    @property
    def folder_import_name(self) -> str:
        """Alias used by the ``/api/dataset/load-folder`` endpoint.

        Defaults to :attr:`type_id`.  Override if your type uses a legacy
        plural name (e.g. ``"sounds"``, ``"videos"``).
        """
        return self.type_id

    @property
    def tab_title(self) -> str:
        """Plural display name used for UI tabs (e.g. ``"Videos"``, ``"Sounds"``).

        Defaults to :attr:`name` + ``"s"``.  Override for irregular plurals
        or custom labels.
        """
        return self.name + "s"

    @property
    def dir_key(self) -> str:
        """Key used in pickle files to store the external media directory path.

        For example ``"audio_dir"``, ``"video_dir"``.  When a pickle file
        contains this key, its value is a path to a directory where media
        files for this type are stored externally.

        Defaults to ``type_id + "_dir"``.  Override for legacy naming
        (e.g. ``"text_dir"`` for the paragraph type).
        """
        return self.type_id + "_dir"

    @property
    def legacy_bytes_keys(self) -> list[str]:
        """Legacy key names for inline bytes in old pickle files.

        Old pickle formats stored media bytes under type-specific keys
        (e.g. ``"wav_bytes"``, ``"video_bytes"``).  New pickles use the
        generic ``"media_bytes"`` / ``"media_string"`` keys.  When loading
        old pickles, these keys are tried in order as fallbacks.

        Defaults to an empty list (no legacy keys).
        """
        return []

    @property
    def pickle_extra_fields(self) -> list[str]:
        """Extra field names to preserve when loading media from pickle files.

        When a pickle stores type-specific metadata beyond the standard
        fields (id, type, duration, file_size, md5, embedding, etc.),
        list the field names here so the loader copies them automatically.

        For example, the image type returns ``["width", "height"]`` and
        the text type returns ``["word_count", "character_count"]``.

        Defaults to an empty list (no extra fields).
        """
        return []

    # ------------------------------------------------------------------
    # Display metadata
    # ------------------------------------------------------------------

    def display_metadata(self, media: dict) -> dict[str, Any]:
        """Build an ordered dict of display-worthy metadata for *media*.

        The returned dict maps human-readable labels to values.  These are
        shown in the labeling UI's metadata grid.  The base implementation
        includes **Category** and **File Size** when present.  Subclasses
        override this to add type-specific fields (Duration, Dimensions,
        Word Count, etc.).

        Importers can additionally set ``media["custom_metadata"]`` to
        supply per-item fields (e.g. ``{"Uploaded By": "alice"}``).  The
        API route merges both sources before sending the response.
        """
        result: dict[str, Any] = {}
        cat = media.get("category")
        if cat and cat not in ("unknown", "custom"):
            result["Category"] = cat
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        return result

    # ------------------------------------------------------------------
    # Viewer behaviour
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def loops(self) -> bool:
        """``True`` if the viewer should loop (audio/video); ``False`` otherwise."""

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def demo_datasets(self) -> list:
        """List of :class:`DemoDataset` objects available for this media type."""

    # ------------------------------------------------------------------
    # Embeddings (delegated to MediaEmbedder)
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load (and cache) the embedding models for this media type.

        Default implementation is a no-op.  Media types that still carry
        their own model logic (legacy) can override this.
        """

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        """Return a fixed-size embedding vector for the media file at *file_path*.

        Returns ``None`` by default.  Overridden by media types that have
        inline embedding logic (legacy) or that delegate to an embedder.
        """
        return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Return an embedding of *text* in the **same vector space** as :meth:`embed_media`.

        Returns ``None`` by default.
        """
        return None

    def embed_text_enriched(self, text: str) -> Optional[np.ndarray]:
        """Embed *text* using the average over all description wrappers.

        Returns ``None`` by default.
        """
        return None

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def load_demo_source(
        self,
        source: str,
        categories: list,
        slice_start: int,
        slice_end: int | None,
        clips: dict[int, dict],
        on_progress: "ProgressCallback | None" = None,
    ) -> str | None:
        """Download and embed a demo dataset source, populating *clips* in-place.

        Each media type overrides this to handle its own demo sources (e.g.
        the audio type handles ESC-50, the image type handles CIFAR-10 /
        Caltech-101/256, etc.).

        Args:
            source: The ``source`` identifier from the :class:`DemoDataset`.
            categories: List of category names to include.
            slice_start: Per-category start index for element slicing.
            slice_end: Per-category end index for element slicing (``None``
                means take all remaining).
            clips: Dict to populate in-place.  The caller has already cleared
                it.  Keys should be sequential integer clip IDs starting at 1.
            on_progress: Optional progress callback.

        Returns:
            An optional external media directory path (absolute string) to
            embed in the pickle cache (e.g. ``"audio_dir"`` value).  Return
            ``None`` when all media bytes are stored inline in the clips.

        Raises:
            ValueError: If *source* is not recognised by this media type.
        """
        raise ValueError(f"Media type {self.type_id!r} does not support demo source {source!r}")

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    @abstractmethod
    def load_media_data(self, file_path: Path) -> dict:
        """Load and return media-specific fields for a media dict.

        The returned dict is merged into the *base* media dict (which already
        contains ``id``, ``type``, ``file_size``, ``md5``, ``embedding``,
        ``filename``, and ``category``).  You must include at minimum a
        ``"duration"`` key.

        Example return value for audio::

            {"media_bytes": b"...", "duration": 3.2}

        Example return value for images::

            {"media_bytes": b"...", "duration": 0, "width": 32, "height": 32}
        """

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def _resolve_media_bytes(self, media: dict) -> bytes | None:
        """Return binary media data, lazy-loading from ``media_path`` if needed.

        In thin mode, ``media_bytes`` is ``None`` but ``media_path`` points to
        the source file on disk.  This helper transparently loads the bytes on
        demand so that :meth:`media_response` works regardless of how the media
        was loaded.
        """
        media_bytes = media.get("media_bytes")
        if media_bytes is not None:
            return media_bytes
        media_path = media.get("media_path")
        if media_path:
            path = Path(media_path)
            if path.exists():
                with open(path, "rb") as f:
                    return f.read()
        return None

    def _resolve_media_string(self, media: dict) -> str:
        """Return text content, lazy-loading from ``media_path`` if needed.

        Same lazy-loading pattern as :meth:`_resolve_media_bytes` but for
        text media types that store ``media_string`` instead of ``media_bytes``.
        """
        media_string = media.get("media_string")
        if media_string is not None:
            return media_string
        media_path = media.get("media_path")
        if media_path:
            path = Path(media_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        return ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary of this media type's metadata.

        Used by the ``/api/media-types`` endpoint to let the frontend render
        media type UI dynamically without hardcoding.
        """
        return {
            "type_id": self.type_id,
            "name": self.name,
            "icon": self.icon,
            "tab_title": self.tab_title,
            "folder_import_name": self.folder_import_name,
            "loops": self.loops,
            "file_extensions": self.file_extensions,
        }

    @abstractmethod
    def media_response(self, media: dict) -> MediaResponse:
        """Return a :class:`MediaResponse` with the media's media content.

        For binary media, set ``data`` to raw bytes with an appropriate
        ``mimetype``.  For structured data (e.g. text paragraphs), set
        ``data`` to a JSON-serialisable dict with ``mimetype="application/json"``.

        Implementations should use :meth:`_resolve_media_bytes` or
        :meth:`_resolve_media_string` to transparently support both preloaded
        medias and thin (lazy-loaded) medias.
        """


class Processor(ABC):
    """Abstract base class for all processors (detectors, localizers, extractors, etc.).

    A *Processor* takes a single media media and produces an answer.  The
    exact type of the answer depends on the subclass:

    * A :class:`Detector` returns ``bool`` — "does this media match?"
    * A :class:`Localizer` returns ``list[dict]`` — "where in this media
      is the item of interest?" (bounding boxes with confidence scores).
    * An :class:`Extractor` returns ``list[dict]`` — "what details are
      inside this media?" (bounding boxes, labels, and other metadata).

    Every processor knows its :attr:`name` (a unique human-readable
    identifier) and the :attr:`media_type` it operates on (e.g.
    ``"audio"``, ``"image"``).

    Subclasses must implement:

    * :attr:`name`
    * :attr:`media_type`
    * :meth:`process` — run the processor on a single media dict.

    Subclasses *may* override:

    * :meth:`load_model` — called once before first use to load heavy
      resources (model weights, etc.).  Default is a no-op.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this processor, e.g. ``"dog_barks"``."""

    @property
    @abstractmethod
    def media_type(self) -> str:
        """The media ``type_id`` this processor operates on (e.g. ``"image"``)."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load any heavyweight resources (model weights, etc.).

        Called lazily before the first :meth:`process` call.  The default
        implementation is a no-op — override in subclasses that need
        one-time model loading.
        """

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, media: dict[str, Any]) -> Any:
        """Run this processor on *media* and return the result.

        The return type depends on the subclass:

        * :class:`Detector` → ``bool``
        * :class:`Localizer` → ``list[dict[str, Any]]``
        * :class:`Extractor` → ``list[dict[str, Any]]``
        """

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of this processor's metadata."""
        return {
            "name": self.name,
            "media_type": self.media_type,
        }


class Detector(Processor):
    """Abstract base class for detectors.

    A *Detector* answers "is this media Good?" with a boolean.  Each
    concrete ``Detector`` operates on exactly **one** media type (declared
    via :attr:`media_type`).

    Subclasses must implement:

    * :attr:`name` — unique identifier for this detector.
    * :attr:`media_type` — which media type it works on.
    * :meth:`detect` — run detection on a single media dict and return
      ``True`` if the media matches, ``False`` otherwise.

    The generic :meth:`process` method delegates to :meth:`detect`.
    """

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @abstractmethod
    def detect(self, media: dict[str, Any]) -> bool:
        """Run detection on *media* and return whether it matches.

        Returns ``True`` if the media is a positive match for this detector,
        ``False`` otherwise.
        """

    def process(self, media: dict[str, Any]) -> bool:
        """Run detection on *media* (delegates to :meth:`detect`)."""
        return self.detect(media)


class Localizer(Processor):
    """Abstract base class for localizers.

    A *Localizer* sits between a :class:`Detector` and an :class:`Extractor`
    in the processor hierarchy.  While a Detector answers "does this media
    match?" (bool) and an Extractor answers "what details are inside this
    media?" (bounding boxes, labels, metadata, etc.), a Localizer answers
    "**where** in this media is the item of interest?" by returning bounding
    boxes with confidence scores — but no further classification or
    extraction metadata.

    For example an image localizer might return bounding boxes around
    regions of interest without identifying what class each region belongs
    to; a video localizer might return temporal intervals where something
    noteworthy happens.

    A Localizer can be composed with other processors: one could build a
    Localizer by running a Detector and then a follow-up localization step,
    or build an Extractor by running a Localizer followed by a
    classification/extraction step.  However, none of these compositions
    are required — each processor type can be implemented independently.

    Each concrete ``Localizer`` operates on exactly **one** media type
    (declared via :attr:`media_type`), just like Detectors and Extractors.

    Subclasses must implement:

    * :attr:`name` — unique identifier for this localizer.
    * :attr:`media_type` — which media type it works on.
    * :meth:`localize` — run localization on a single media dict and return
      a list of bounding-box dicts.

    The generic :meth:`process` method delegates to :meth:`localize`.
    """

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------

    @abstractmethod
    def localize(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Run localization on *media* and return a list of bounding-box dicts.

        Each dict in the returned list describes **one region** where the
        item of interest was found.  Every dict **must** include:

        * ``"confidence"`` — a float in ``[0, 1]``.
        * ``"bbox"`` — the bounding box (format is media-specific, e.g.
          ``[x1, y1, x2, y2]`` for images).

        Returns an empty list when nothing is found.

        Example return value for an image localizer::

            [
                {"confidence": 0.95, "bbox": [10, 20, 200, 300]},
                {"confidence": 0.73, "bbox": [400, 50, 600, 250]},
            ]

        Example return value for a video temporal localizer::

            [
                {"confidence": 0.88, "bbox": [1.2, 3.4]},
            ]
        """

    def process(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Run localization on *media* (delegates to :meth:`localize`)."""
        return self.localize(media)


class Extractor(Processor):
    """Abstract base class for extractors.

    While a *Detector* answers "is this media Good?" (True/False), an
    *Extractor* answers "what Good things are inside this media, and where?"
    by returning structured details for each occurrence found.

    For example an image extractor might return bounding boxes and class
    labels; a video extractor might return start/stop timestamps of events.

    Each concrete ``Extractor`` operates on exactly **one** media type
    (declared via :attr:`media_type`), just like Detectors.

    Subclasses must implement:

    * :attr:`name` — unique identifier for this extractor.
    * :attr:`media_type` — which media type it works on.
    * :meth:`extract` — run extraction on a single media dict and return a
      list of result dicts.

    The generic :meth:`process` method delegates to :meth:`extract`.
    """

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    @abstractmethod
    def extract(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Run extraction on *media* and return a list of result dicts.

        Each dict in the returned list describes **one occurrence** of the
        thing the extractor is looking for.  The schema of these dicts is
        extractor-specific, but every dict **must** include a ``"confidence"``
        key with a float in ``[0, 1]``.

        Returns an empty list when nothing is found.

        Example return value for an image bounding-box extractor::

            [
                {"confidence": 0.92, "bbox": [x1, y1, x2, y2], "label": "car"},
                {"confidence": 0.87, "bbox": [x1, y1, x2, y2], "label": "car"},
            ]

        Example return value for a video timestamp extractor::

            [
                {"confidence": 0.85, "start": 1.2, "end": 3.4, "label": "explosion"},
            ]
        """

    def process(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Run extraction on *media* (delegates to :meth:`extract`)."""
        return self.extract(media)


class MediaClipper(ABC):
    """Abstract base class for media clippers.

    A *MediaClipper* takes a single media item and returns one or more media
    items of the **same** type.  Each concrete clipper operates on exactly one
    media type (declared via :attr:`media_type`).

    Unlike :class:`Processor` subclasses which return metadata *about* a media
    (booleans, bounding boxes, labels), a ``MediaClipper`` returns **new media
    dicts** that can be used directly in place of the original.

    Examples:

    * ``SoundTilingClipper(2)`` — tiles a 9.5 s audio clip into five 2 s
      clips equally spaced across the original duration.
    * ``ImageTilingClipper()`` — covers a tall image with equidistant square
      tiles using the shorter dimension as the tile size.
    * ``TextSentenceClipper()`` — splits a paragraph into individual sentences.
    * Any ``*DefaultClipper`` — returns the media unchanged (single-element
      list).

    Subclasses must implement:

    * :attr:`name` — unique identifier for this clipper.
    * :attr:`media_type` — which media ``type_id`` it works on.
    * :meth:`clip` — split a single media dict into a list of media dicts.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this clipper, e.g. ``"video_tiling"``."""

    @property
    @abstractmethod
    def media_type(self) -> str:
        """The media ``type_id`` this clipper operates on (e.g. ``"audio"``)."""

    @property
    def display_name(self) -> str:
        """Human-readable name for UI dropdowns.

        Defaults to title-casing the :attr:`name` with underscores replaced
        by spaces (e.g. ``"sound_tiling"`` → ``"Sound Tiling"``).  Subclasses
        may override for a custom label.
        """
        return self.name.replace("_", " ").title()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> list[dict[str, Any]]:
        """Return a list of user-configurable parameter descriptors.

        Each descriptor is a dict with keys:

        * ``key`` — parameter name (used in :meth:`with_params`).
        * ``label`` — human-readable label for the UI.
        * ``type`` — ``"number"`` or ``"string"``.
        * ``default`` — current/default value.
        * ``min`` / ``max`` / ``step`` — optional numeric constraints.

        Clippers with no configurable parameters return ``[]`` (the default).
        """
        return []

    def with_params(self, params: dict[str, Any]) -> "MediaClipper":
        """Return a **new** clipper of the same type with overridden parameters.

        *params* is a dict mapping parameter keys (as declared by
        :attr:`parameters`) to their new values.  Unknown keys are ignored.

        The default implementation returns ``self`` unchanged (suitable for
        clippers that have no parameters).
        """
        return self

    # ------------------------------------------------------------------
    # Clipping
    # ------------------------------------------------------------------

    @abstractmethod
    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Split *media* into one or more media dicts of the same type.

        Each dict in the returned list is a **new** media dict that preserves
        the structure of the original (``id``, ``type``, ``category``,
        ``origin``, ``origin_name``, etc.) but contains the clipped content
        (updated ``media_bytes`` / ``media_string``, ``duration``, and any
        type-specific fields).

        Returns a list with at least one element.  Default clippers return
        ``[media]`` unchanged.
        """

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of this clipper's metadata."""
        d: dict[str, Any] = {
            "name": self.name,
            "display_name": self.display_name,
            "media_type": self.media_type,
        }
        params = self.parameters
        if params:
            d["parameters"] = params
        return d
