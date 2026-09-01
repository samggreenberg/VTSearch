"""Core media type abstractions: MediaType, MediaResponse, DemoDataset.

To add a new media type:

1. Create a subdirectory under ``vtsearch/media/`` (e.g. ``vtsearch/media/code/``).
2. Add any new pip packages your embedder needs to
   ``[project.dependencies]`` in the repo's ``pyproject.toml``, then
   re-run ``bash scripts/install.sh`` (or your editable install of
   choice) to pick them up.
3. Implement a subclass of :class:`MediaType` in ``media_type.py``.
4. Register it in ``vtsearch/media/__init__.py``::

       from vtscore.media.code.media_type import CodeMediaType
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

from vtscore.security.path_validation import resolve_media_file_path

# Type alias for progress callbacks.  Modules that accept an ``on_progress``
# parameter use this signature so callers can report status without depending
# on ``vtscore.concurrency.progress``.
ProgressCallback = Callable[[str, str, int, int], None]

__all__ = [
    "DemoDataset",
    "MediaResponse",
    "MediaType",
    "ProgressCallback",
    "demo_slice",
    "demo_slice_by_category",
]


def _fetch_media_url(url: str) -> bytes | None:
    """Fetch binary content from a ``media_url``, or ``None`` if it can't be had.

    Used by :meth:`MediaType._resolve_media_bytes` and
    :meth:`MediaType._resolve_media_string` as a last-resort fallback when
    neither ``media_bytes`` nor ``media_path`` are available (e.g. for
    URL-backed media from PullWrest).

    A ``media_url`` is **not** trusted input.  It rides along on a media dict
    that can arrive from a loaded pickle
    (``vtscore.datasets.loader_pickle._restore_media_url``), and whatever this
    returns is served straight back to the requester by the media routes.  It
    therefore goes through
    :func:`~vtscore.security.url_validation.fetch_validated_url`, the same SSRF
    guard the URL-backed dataset sources and the downloader use: only publicly
    routable ``http(s)`` URLs, with every redirect hop re-checked.  That is what
    keeps a ``file:///etc/passwd`` or ``http://169.254.169.254/…`` media_url
    from turning a media fetch into an arbitrary file read or an internal
    network probe — ``urllib.request.urlopen``, which this used to call, services
    ``file://`` and ``ftp://`` out of the box and would have obliged.
    """
    from vtscore.security.url_validation import fetch_validated_url  # noqa: PLC0415

    try:
        return fetch_validated_url(url)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.warning("Failed to fetch media_url: %s", url, exc_info=True)
        return None


def _resolve_archive_member_bytes(media: dict) -> bytes | None:
    """Return a media's bytes by streaming its archive member, or ``None``.

    For ``local_archive_member`` media (whose bytes live inside a tar/zip shard
    that we deliberately never extract) this reads the whole member on demand
    via :mod:`vtscore.datasets.archive_stream`.  Returns ``None`` for any media
    that is not archive-member-backed, or when the member can't be read, so the
    caller falls through to its remaining resolution order.

    Callers serving Range requests (video/audio playback) should prefer the
    route-level partial-read path so a large member is never fully buffered;
    this whole-member read backs the non-streaming callers (thumbnails, image,
    transcode of non-streamable containers).
    """
    from vtscore.datasets.archive_stream import ArchiveMemberError, archive_member_ref, read_member  # noqa: PLC0415

    ref = archive_member_ref(media)
    if ref is None:
        return None
    # The archive path rides on the media, so it is externally supplied for a
    # loaded pickle; confine it before opening (a no-op in single-user mode).
    if resolve_media_file_path(ref[0]) is None:
        return None
    try:
        return read_member(ref[0], ref[1])
    except (ArchiveMemberError, OSError):
        logging.getLogger(__name__).warning("Failed to read archive member %s::%s", ref[0], ref[1], exc_info=True)
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
        data: The payload - ``bytes`` for binary media, ``dict`` for JSON.
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

    items_per_category: int = 0
    """Approximate per-category item count of the raw source dataset.

    Used by the UI to estimate how many files the user will be loading
    *before* the dataset is downloaded.  The actual count after loading
    will match ``int(items_per_category * (slice_frac_end - slice_frac_start))``
    for fractionally sliced datasets.  Leave at ``0`` for datasets where
    the count is unknown or the full dataset is always loaded."""

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


def demo_slice_by_category(
    by_cat: dict,
    categories,
    slice_start,
    slice_end,
    slice_frac_start=None,
    slice_frac_end=None,
) -> list:
    """Flatten a ``{category: items}`` map into one list, slicing each category.

    The per-category loop every media type's demo loader runs after grouping a
    downloaded folder's metadata by label: walk *categories* in the caller's
    order (so the requested subset, and its ordering, is what comes back), and
    take the same :func:`demo_slice` window out of each bucket.  A category the
    map doesn't have contributes nothing rather than raising, since the
    downloaded source may legitimately be missing a requested label.
    """
    out: list = []
    for cat in categories:
        out.extend(demo_slice(by_cat.get(cat, []), slice_start, slice_end, slice_frac_start, slice_frac_end))
    return out


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

    Embedding is handled by :class:`~vtscore.media.embedder.MediaEmbedder`
    objects, which are registered separately.  A media type may have zero,
    one, or many embedders.

    Adding a new media type
    -----------------------
    See the module docstring above for the four-step process.
    """

    # Progress callback wired in by ``vtscore.media.set_progress_callback``.
    # Demo loaders / model loaders call it; defaults to a no-op so direct
    # instantiation (tests, scripts) doesn't crash before the registry is set up.
    _on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    #: Whether items of this media type have a *browsable thumbnail* — a small
    #: still image that stands in for the item in grids and on the VTSBrowse
    #: map.  Image/video/document render a real thumbnail; audio renders a
    #: waveform PNG (see ``vtscore.media.audio``'s ``generate_waveform_thumbnail``).
    #: Types with no meaningful still image (text) leave this ``False``.
    #:
    #: This flag is the single source of truth for the "thumbnail vs
    #: no-thumbnail" distinction.  It drives the VTSBrowse bin shape
    #: (:func:`~vtscore.projection.pyramid.bin_shape_for_media_type` — square
    #: for thumbnailed types so tiles pack edge-to-edge, hex otherwise) and is
    #: surfaced to the frontend via :meth:`to_dict` / ``GET /api/media-types``.
    has_thumbnail: bool = False

    #: Embeddable ``type_id``s a **non-embeddable** type can convert into
    #: (first entry = default).  This is one of the two orthogonal
    #: capabilities that ``MediaType`` used to conflate — see
    #: :attr:`importable` / :attr:`embeddable`.  ``document`` sets
    #: ``["image", "text"]`` (a PDF page → image, or its extracted text);
    #: a directly-embeddable type leaves this empty.  A *convert-out* "half
    #: type" is precisely ``embeddable is False and converts_to != []`` — a
    #: category that *mandates* a conversion step before it can be searched.
    converts_to: list[str] = []

    @property
    def importable(self) -> bool:
        """Whether this type is a first-class *ingestion* category the user
        picks when importing (folder scan, file upload, native demo tab).

        Every *full* type (image/audio/text/video) and every *convert-out*
        half type (``document``: importable but not embeddable) is importable.
        A *convert-in* half type — one that only ever arises from converting
        **another** type, like ``face`` (cropped out of images, never imported
        from a ``.face`` file) — overrides this to ``False``.  Import surfaces
        (folder-import media picker, importer media-type tabs) filter on this
        flag; browse / sort / demo surfaces show every type.
        """
        return True

    @property
    def embeddable(self) -> bool:
        """Whether this type can be embedded on its own (has ≥1 registered
        embedder), and is therefore directly sortable / browsable / text-
        queryable.

        Derived from the embedder registry
        (:func:`~vtscore.media.embedders_for_type`) so it stays ``True`` for
        image/audio/video/text and ``False`` for a *convert-out* half type
        like ``document`` automatically, with nothing to keep in sync.  A type
        with ``embeddable is False`` must be converted (see :attr:`converts_to`)
        before it enters the embedding space.
        """
        from vtscore.media import embedders_for_type  # noqa: PLC0415

        return bool(embedders_for_type(self.type_id))

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
        ``"videos"``) are accepted via the alias map in ``vtscore.media``.
        """
        return self.type_id

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

        A dataset prepped for Browse already carries a model-generated
        signpost text per media (the caption / tag list that letters the
        map); the base implementation surfaces it here under an explicitly
        hedged label so the work is visible in the labeling UI too.  See
        :func:`vtscore.projection.signpost_texts.signpost_metadata_entry`.
        """
        result: dict[str, Any] = {}
        cat = media.get("category")
        if cat and cat not in ("unknown", "custom"):
            result["Category"] = cat
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        # Clip boundary fields - present only on clipped sub-medias.
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
        # Converter / clipper provenance: which file this item was derived
        # from, and the chain that derived it.  Empty for a plainly imported
        # media (it is its own source).
        from vtscore.media.provenance import provenance_metadata  # noqa: PLC0415

        result.update(provenance_metadata(media))
        # Imported here rather than at module scope: the signpost text layer
        # reaches back into the media types to decode items, so an eager
        # import would close a cycle.
        from vtscore.projection.signpost_texts import signpost_metadata_entry  # noqa: PLC0415

        entry = signpost_metadata_entry(media)
        if entry is not None:
            result[entry[0]] = entry[1]
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

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Return an embedding of *text* in the same vector space as the embedder.

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
            skip_embedding: When ``True`` (passed via ``**kwargs``), populate
                each media with a deferred-embed placeholder (``embeddings={}``)
                instead of embedding it here.  The demo loader sets this when a
                clipper will split + re-embed every clip, so embedding the full
                parent would be wasted (and, for audio, can fail on parents
                longer than the embedder window before the clipper trims them).

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
    def load_media_data(self, file_path: Path, media_bytes: bytes | None = None) -> dict:
        """Load and return media-specific fields for a media dict.

        The returned dict is merged into the *base* media dict (which already
        contains ``id``, ``media_type``, ``file_size``, ``md5``, ``embedding``,
        ``filename``, and ``category``).  You must include at minimum a
        ``"duration"`` key.

        When *media_bytes* is provided, implementations should use it instead
        of re-reading the file from disk.  Callers that have already loaded
        the file contents (e.g. the folder loader, which needs them to
        compute MD5) pass them through to avoid a second read.

        Example return value for audio::

            {"media_bytes": b"...", "duration": 3.2}

        Example return value for images::

            {"media_bytes": b"...", "duration": 0, "width": 32, "height": 32}
        """

    #: Keys that carry a media's full payload.  A thin load stores a
    #: ``media_path`` reference instead of these, so
    #: :meth:`load_thin_media_data` strips them from whatever
    #: :meth:`load_media_data` returns.
    _PAYLOAD_KEYS = ("media_bytes", "media_string")

    def load_thin_media_data(self, file_path: Path) -> dict:
        """Return the *display* fields for a thin load: no payload, no re-read later.

        A thin load (the "Reference files in place" importer option, and every
        CLI ``--thin`` workflow) deliberately keeps a media's bytes out of
        memory.  It used to skip :meth:`load_media_data` wholesale, which also
        skipped the ingest-time ``thumbnail_bytes`` — so every grid / VTSBrowse
        tile fell back to decoding the full-resolution original on each cold
        request (~200 ms for a 12 MP photo, paid again after any reload).

        This hook produces the same display artifacts the full path does
        (``thumbnail_bytes``, ``width`` / ``height``, ``duration``) while
        leaving the payload behind.  The default reads the file once through
        :meth:`load_media_data` and drops :data:`_PAYLOAD_KEYS`; peak memory is
        therefore one file at a time rather than the whole dataset.  Override
        when the artifacts can be derived from the path without reading the
        bytes at all (see the video type), or return ``{}`` when the type has
        no ingest-time artifact worth the read.

        Only called for types whose items have a browsable thumbnail
        (:attr:`has_thumbnail`); everything else keeps the pure-reference load.
        """
        return {k: v for k, v in self.load_media_data(file_path).items() if k not in self._PAYLOAD_KEYS}

    def ensure_thumbnail_bytes(self, media: dict) -> bytes | None:
        """Populate and return *media*'s ``thumbnail_bytes``, generating if absent.

        :meth:`load_media_data` covers media that arrive from a file the loader
        can read at ingest.  Some media have no file at all: an
        **archive-member** media (see
        :mod:`vtscore.datasets.importers.local_archive_member`) carries only
        ``{archive path, member}`` and re-derives its bytes by streaming a
        single tar/zip member, so nothing is decodable until something asks.
        This hook is the type-agnostic way to ask, and it is what the
        background warm-up pass in
        :mod:`vtscore.datasets.thumbnail_warm` calls per media.

        The default is a pure read of whatever is already cached: a type with
        no cheap way to build a thumbnail from its resolvable bytes reports
        ``None`` and keeps the request-time fallback it already has.  Types
        whose thumbnail *is* derivable from the bytes override this and
        generate through the same helper their ``image_response`` fallback
        uses, so a warmed thumbnail is byte-identical to a lazily generated
        one.

        Implementations must memoise onto ``media["thumbnail_bytes"]`` and
        must **not** retain the resolved payload on the media — the whole
        point of the archive-member importer is that member bytes never stay
        resident.  In-memory only, like every other ``thumbnail_bytes``: these
        ride along on an explicit save and are otherwise regenerated.
        """
        return media.get("thumbnail_bytes")

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def _resolve_media_bytes(self, media: dict) -> bytes | None:
        """Return binary media data, lazy-loading from path or URL.

        Resolution order:

        1. ``media_bytes`` - already in memory.
        2. *lazy clip* - a derived sub-media that carries a clip recipe in
           ``origin.params`` instead of materialized bytes; its bytes are
           sliced/cropped from the source on demand (see
           :mod:`vtscore.media.lazy_clip`).
        3. ``media_path`` - local file on disk (thin mode).
        4. ``media_url`` - remote URL (URL-backed media, e.g. PullWrest),
           fetched only through the SSRF guard in :func:`_fetch_media_url`.
        """
        media_bytes = media.get("media_bytes")
        if media_bytes is not None:
            return media_bytes
        from vtscore.media.lazy_clip import lazy_clip_bytes  # noqa: PLC0415

        clipped = lazy_clip_bytes(media)
        if clipped is not None:
            return clipped
        archive_bytes = _resolve_archive_member_bytes(media)
        if archive_bytes is not None:
            return archive_bytes
        media_path = media.get("media_path")
        if media_path:
            path = resolve_media_file_path(media_path)
            if path is not None and path.exists():
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
            path = resolve_media_file_path(media_path)
            if path is not None and path.exists():
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
            "folder_import_name": self.folder_import_name,
            "loops": self.loops,
            "file_extensions": self.file_extensions,
            "has_thumbnail": self.has_thumbnail,
            "importable": self.importable,
            "embeddable": self.embeddable,
            "converts_to": list(self.converts_to),
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
