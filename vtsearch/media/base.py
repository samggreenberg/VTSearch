"""Core media type abstractions: MediaType, MediaResponse, DemoDataset.

To add a new media type:

1. Create a subdirectory under ``vtsearch/media/`` (e.g. ``vtsearch/media/code/``).
2. Add a ``requirements.txt`` listing any pip packages your embedder needs
   (auto-discovered by ``install-plugin-deps.sh``).
3. Implement a subclass of :class:`MediaType` in ``media_type.py``.
4. Register it in ``vtsearch/media/__init__.py``::

       from vtsearch.media.code.media_type import CodeMediaType
       register(CodeMediaType())

That is all.  The rest of the application (routing, dataset loading, model
initialisation, demo listing) picks up your new type automatically through
the registry.
"""

from __future__ import annotations

import logging
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
    "MediaResponse",
    "MediaType",
    "ProgressCallback",
    "demo_slice",
]


def _fetch_media_url(url: str) -> bytes | None:
    """Fetch binary content from a ``media_url``.

    Used by :meth:`MediaType._resolve_media_bytes` and
    :meth:`MediaType._resolve_media_string` as a last-resort fallback when
    neither ``media_bytes`` nor ``media_path`` are available (e.g. for
    URL-backed media from PullWrest).
    """
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            return resp.read()
    except Exception:
        logger = logging.getLogger(__name__)
        logger.warning("Failed to fetch media_url: %s", url, exc_info=True)
        return None


def _noop_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
    """Default no-op progress callback used when no real reporter is set."""


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
    use disjoint subsets of elements within each category.  Ignored when
    ``slice_frac_start``/``slice_frac_end`` are set."""

    slice_end: Optional[int] = None
    """Per-category end index for element slicing (exclusive).

    ``None`` means take all remaining elements after ``slice_start``.
    Ignored when ``slice_frac_start``/``slice_frac_end`` are set."""

    slice_frac_start: Optional[float] = None
    """Per-category fractional start (0.0–1.0).

    When set, each category is sliced proportionally rather than with
    fixed indices, so categories with different item counts each
    contribute the correct fraction of their items."""

    slice_frac_end: Optional[float] = None
    """Per-category fractional end (0.0–1.0).

    ``None`` means take all remaining elements after ``slice_frac_start``."""

    download_size_mb: float = 0
    """Estimated download size in megabytes for this demo dataset's raw data.

    Used by the frontend to display the expected download size before the
    user starts loading.  Set to ``0`` for datasets that don't require a
    network download (e.g. scikit-learn datasets that download automatically)."""


def demo_slice(items, slice_start, slice_end, slice_frac_start=None, slice_frac_end=None):
    """Slice a per-category item list using either absolute or fractional bounds."""
    if slice_frac_start is not None:
        n = len(items)
        start = int(n * slice_frac_start)
        end = int(n * slice_frac_end) if slice_frac_end is not None else n
        return items[start:end]
    return items[slice_start:slice_end]


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

    Embedding is handled by :class:`~vtsearch.media.embedder.MediaEmbedder`
    objects, which are registered separately.  A media type may have zero,
    one, or many embedders.

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
        """SVG icon type name for the UI, e.g. ``"audio"``."""

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

        Defaults to :attr:`type_id`.  Legacy plural names (e.g. ``"sounds"``,
        ``"videos"``) are accepted via the alias map in ``vtsearch.media``.
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
        # Clip boundary fields — present only on clipped sub-medias.
        cs = media.get("clip_start")
        if cs is not None:
            result["Clip Start"] = cs
        ce = media.get("clip_end")
        if ce is not None:
            result["Clip End"] = ce
        cb = media.get("clip_box")
        if cb is not None:
            result["Clip Box"] = ",".join(str(v) for v in cb)
        ci = media.get("clip_index")
        if ci is not None:
            result["Clip Index"] = ci
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
        **kwargs,
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
        """Return binary media data, lazy-loading from path or URL.

        Resolution order:

        1. ``media_bytes`` — already in memory.
        2. ``media_path`` — local file on disk (thin mode).
        3. ``media_url`` — remote URL (URL-backed media, e.g. PullWrest).
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
        media_url = media.get("media_url")
        if media_url:
            return _fetch_media_url(media_url)
        return None

    def _resolve_media_string(self, media: dict) -> str:
        """Return text content, lazy-loading from path or URL.

        Same resolution order as :meth:`_resolve_media_bytes` but for text
        media types that store ``media_string`` instead of ``media_bytes``.
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
        media_url = media.get("media_url")
        if media_url:
            data = _fetch_media_url(media_url)
            if data is not None:
                return data.decode("utf-8", errors="replace").strip()
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
