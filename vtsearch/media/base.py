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
    "MediaResponse",
    "MediaType",
    "Processor",
    "ProgressCallback",
    "intercept_tqdm_progress",
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

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
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
    """Local directory that must exist for a cached ``.pkl`` to be usable.

    Audio and video datasets store references to external media files rather
    than inlining the bytes, so a stale ``.pkl`` left behind after the source
    directory was removed would incorrectly appear ready.  Set this to the
    directory that the importer places the source files into (e.g.
    ``DATA_DIR / "ESC-50-master" / "audio"``).  Leave ``None`` for datasets
    whose ``.pkl`` is entirely self-contained (images, text)."""

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

    * How to embed a file into a fixed-size vector (:meth:`embed_media`).
    * How to embed a text query into the **same** vector space (:meth:`embed_text`).
    * Human-readable identity: :attr:`name` and :attr:`icon`.
    * Which file extensions to scan when importing a folder (:attr:`file_extensions`).
    * Whether the viewer should loop (:attr:`loops`).
    * Which demo datasets are available (:attr:`demo_datasets`).
    * How to serve a media over HTTP (:meth:`media_response`).
    * How to load media-specific media fields from a file (:meth:`load_media_data`).
    * An optional folder-import alias (:attr:`folder_import_name`) for the
      ``/api/dataset/load-folder`` endpoint.

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
    # Embeddings
    # ------------------------------------------------------------------

    @abstractmethod
    def load_models(self) -> None:
        """Load (and cache) the embedding models for this media type.

        Called lazily the first time this media type needs to embed something
        (i.e. on the first ``embed_media``, ``embed_text``, or getter call).
        Implementations must be idempotent — a second call should be a no-op.
        """

    @abstractmethod
    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        """Return a fixed-size embedding vector for the media file at *file_path*.

        Returns ``None`` if the file cannot be embedded (model not loaded,
        corrupt file, etc.).
        """

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Return an embedding of *text* in the **same vector space** as :meth:`embed_media`.

        This is used for text-query sorting: the resulting vector is compared
        against media embeddings via cosine similarity.

        The default implementation returns ``None`` (null embedder), which
        means text-query sorting is unavailable for this media type.  Media
        types that support text sorting should override this method.

        Returns ``None`` if the model is not loaded, encoding fails, or the
        media type does not support text embedding.
        """
        return None

    @property
    def description_wrappers(self) -> list[str]:
        """Return wrapper templates for enriching sort descriptions.

        Each template is a format string containing ``{text}`` where the
        user's description will be inserted.  Override in subclasses to
        provide media-specific wrappers that improve embedding quality.

        The default returns an empty list (no wrappers — plain embedding only).
        """
        return []

    def embed_text_enriched(self, text: str) -> Optional[np.ndarray]:
        """Embed *text* using the average over all description wrappers.

        For each wrapper in :attr:`description_wrappers`, formats the wrapper
        with *text*, embeds the result, and returns the mean of all resulting
        vectors (L2-normalised).  Falls back to :meth:`embed_text` if no
        wrappers are defined or all wrapper embeddings fail.

        Returns ``None`` only if :meth:`embed_text` also returns ``None``.
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
        """Unique identifier for this clipper, e.g. ``"video_tiling_2s"``."""

    @property
    @abstractmethod
    def media_type(self) -> str:
        """The media ``type_id`` this clipper operates on (e.g. ``"audio"``)."""

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
        return {
            "name": self.name,
            "media_type": self.media_type,
        }
