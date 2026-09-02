"""MediaClipper ABC for splitting media items into sub-items."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any


class MediaClipper(ABC):
    """Abstract base class for media clippers.

    A *MediaClipper* takes a single media item and returns one or more media
    items of the **same** type.  Each concrete clipper operates on exactly one
    media type (declared via :attr:`media_type`).

    Unlike :class:`~vtscore.media.processors.Processor` subclasses which
    return metadata *about* a media (booleans, bounding boxes, labels), a
    ``MediaClipper`` returns **new media dicts** that can be used directly in
    place of the original.

    Examples:

    * ``SoundTilingClipper(2)``: tiles a 9.5 s audio clip into five 2 s
      clips equally spaced across the original duration.
    * ``ImageTilingClipper()``: covers a tall image with equidistant square
      tiles using the shorter dimension as the tile size.
    * ``TextSentenceClipper()``: splits a paragraph into individual sentences.
    * Any ``*DefaultClipper``: returns the media unchanged (single-element
      list).

    Subclasses must implement:

    * :attr:`name`: unique identifier for this clipper.
    * :attr:`media_type`: which media ``type_id`` it works on.
    * :meth:`clip`: split a single media dict into a list of media dicts.
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

        Strips the media-type prefix before the first underscore and
        title-cases the remainder (e.g. ``"sound_tiling"`` → ``"Tiling"``).
        Subclasses may override for a custom label.
        """
        _, _, suffix = self.name.partition("_")
        return suffix.replace("_", " ").title() if suffix else self.name.title()

    @property
    def description(self) -> str:
        """Short tooltip description shown on hover in the clipper chooser.

        Subclasses should override to provide a brief explanation of what
        the clipper does.  Defaults to an empty string.
        """
        return ""

    @property
    def summary_template(self) -> str:
        """One-line preview of this clipper with ``{key}`` placeholders.

        The frontend substitutes each ``{key}`` with the current value of
        the parameter named ``key`` (see :attr:`parameters`) when rendering
        the import row preview.  Subclasses with configurable parameters
        should override to surface the active values, e.g.
        ``"Cut each audio file into {duration}s tiles."``.  Defaults to
        :attr:`description` so static clippers don't have to override.
        """
        return self.description

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> list[dict[str, Any]]:
        """Return a list of user-configurable parameter descriptors.

        Each descriptor is a dict with keys:

        * ``key``: parameter name (used in :meth:`with_params`).
        * ``label``: human-readable label for the UI.
        * ``type``: ``"number"`` or ``"string"``.
        * ``default``: current/default value.
        * ``min`` / ``max`` / ``step``: optional numeric constraints.

        Clippers with no configurable parameters return ``[]`` (the default).
        """
        return []

    @property
    def creation_questions(self) -> list[dict[str, Any]]:
        """Return a list of questions to present when the user chooses this clipper.

        Each question is a dict with the same schema as :attr:`parameters`
        descriptors:

        * ``key``: parameter name (used in :meth:`with_params`).
        * ``label``: human-readable label / question for the UI.
        * ``type``: ``"number"`` or ``"string"``.
        * ``default``: current/default value.
        * ``min`` / ``max`` / ``step``: optional numeric constraints.

        By default this returns :attr:`parameters`, so any clipper that
        already declares parameters automatically exposes them as creation
        questions.  Subclasses may override to present a different (richer
        or reduced) set of questions at creation time.
        """
        return self.parameters

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
        the structure of the original (``id``, ``media_type``, ``category``,
        ``origin``, ``origin_name``, etc.) but contains the clipped content
        (updated ``media_bytes`` / ``media_string``, ``duration``, and any
        type-specific fields).

        Returns a list with at least one element.  Default clippers return
        ``[media]`` unchanged.
        """

    # ------------------------------------------------------------------
    # Dataset-level resolution
    # ------------------------------------------------------------------

    def resolve_for_durations(self, durations: list[float]) -> "MediaClipper":
        """Reserved dataset-level resolution hook.  **Not currently called.**

        The load pipeline once resolved auto-routing per *dataset* from the
        full duration list; it now does so per *item* via
        :meth:`resolve_for_media`, and no code path invokes this method any
        more.  The name is kept because it is part of the published
        :class:`MediaClipper` contract and an out-of-tree clipper may
        already override it - but such an override is **inert**, so put the
        logic in :meth:`resolve_for_media` instead.

        If a dataset-level decision is ever needed again, wire the call site
        in ``vtscore/datasets/clipper_chain.py`` next to
        :meth:`resolve_for_media` and update this docstring, the three
        clipper guides, and ``tests/detectors/test_clippers.py``.
        """
        return self

    def resolve_for_media(self, media: dict[str, Any]) -> "MediaClipper":
        """Return a concrete clipper for a single media item.

        Called by the load pipeline once per media, before :meth:`clip`.
        This is the only resolution hook the pipeline invokes
        (:meth:`resolve_for_durations` is reserved and inert).  Most
        clippers ignore *media* and return ``self``.  Auto-selecting
        clippers (e.g. ``VideoAutoClipper``) override this to pick a
        different concrete clipper based on the item's own duration -
        e.g. pass-through for short clips, tiling for longer ones.

        The resolved clipper's :attr:`name` and parameter values are what
        get recorded in each clip's origin, so cross-dataset replay is
        deterministic regardless of the original auto policy.
        """
        return self

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
        desc = self.description
        if desc:
            d["description"] = desc
        template = self.summary_template
        if template and template != desc:
            d["summary_template"] = template
        params = self.parameters
        if params:
            d["parameters"] = params
        cq = self.creation_questions
        if cq:
            d["creation_questions"] = cq
        return d


# ----------------------------------------------------------------------
# Shared helpers for concrete clippers
# ----------------------------------------------------------------------


def clip_with_bounds(
    media: dict[str, Any],
    index: int,
    start: float,
    end: float,
) -> dict[str, Any]:
    """Return a shallow copy of *media* stamped with one clip's time bounds.

    Single-sources the ``duration`` / ``clip_index`` / ``clip_start`` /
    ``clip_end`` convention (including the 6-decimal rounding) that every
    time-based clipper - audio tiling/silence/speech/crop, video
    tiling/scene - writes onto each emitted clip.  Callers add their own
    type-specific fields (``media_bytes``, ``file_size``, ``scene_index``,
    ...) to the returned dict.
    """
    clip = dict(media)
    clip["duration"] = round(end - start, 6)
    clip["clip_index"] = index
    clip["clip_start"] = round(start, 6)
    clip["clip_end"] = round(end, 6)
    return clip


def validate_tiling_params(duration: float, min_overlap: float) -> None:
    """Raise :class:`ValueError` if a tiling clipper's constructor args are invalid.

    Shared by every fixed-length tiling clipper so the three rules -
    positive tile length, non-negative overlap, overlap strictly smaller
    than the tile - are stated once.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if min_overlap < 0:
        raise ValueError("min_overlap must be non-negative")
    if min_overlap >= duration:
        raise ValueError("min_overlap must be less than duration")


def tile_starts(total: float, duration: float, min_overlap: float = 0.0) -> list[float]:
    """Return the start times of equally-spaced tiles covering ``[0, total]``.

    Tiles are *duration* seconds long, the first starts at 0 and the last
    ends at *total*, and consecutive tiles overlap by at least *min_overlap*
    seconds (more tiles are emitted when the arithmetic requires it).  When
    *total* does not exceed *duration* a single tile at 0 covers everything.

    Callers that want to leave a short media untouched should test
    ``total <= duration`` themselves and return the original media rather
    than the one-element list this returns.
    """
    if total <= duration:
        return [0.0]
    max_stride = duration - min_overlap
    n_tiles = max(1, math.ceil((total - duration) / max_stride) + 1)
    if n_tiles == 1:
        return [0.0]
    return [i * (total - duration) / (n_tiles - 1) for i in range(n_tiles)]


def tiling_parameters(
    duration: float,
    min_overlap: float,
    *,
    item_label: str,
) -> list[dict[str, Any]]:
    """Return the standard ``duration`` / ``min_overlap`` parameter descriptors.

    *item_label* names the unit in the ``duration`` descriptor's help text
    (e.g. ``"audio segment"`` -> "Duration of each audio segment in
    seconds.").  *duration* and *min_overlap* become the descriptors'
    current defaults.
    """
    return [
        {
            "key": "duration",
            "label": "Clip length (seconds)",
            "description": f"Duration of each {item_label} in seconds.",
            "type": "number",
            "default": duration,
            "min": 0.1,
            "max": 300,
            "step": 0.1,
        },
        {
            "key": "min_overlap",
            "label": "Minimum overlap (seconds)",
            "description": "Minimum overlap between consecutive segments. Higher values produce more tiles.",
            "type": "number",
            "default": min_overlap,
            "min": 0,
            "max": 299.9,
            "step": 0.1,
        },
    ]


class DefaultClipper(MediaClipper):
    """Concrete no-op clipper: :meth:`clip` returns ``[media]`` unchanged.

    Every embeddable media type registers exactly one of these so the
    clipper chooser is never empty and "don't split anything" is an
    ordinary chain step rather than a special case.  Subclass it with a
    zero-argument constructor rather than instantiating it directly, so the
    type keeps a named class the registry, the docs and out-of-tree code
    can refer to::

        class SoundDefaultClipper(DefaultClipper):
            def __init__(self) -> None:
                super().__init__(
                    "sound_default", "audio", "Import each audio file as-is, without splitting."
                )

    The conventions this base fixes are the ``<type>_default`` name, the
    ``"None"`` display label used by the chooser, and the pass-through
    :meth:`clip`.
    """

    def __init__(self, name: str, media_type: str, description: str) -> None:
        self._name = name
        self._media_type = media_type
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def media_type(self) -> str:
        return self._media_type

    @property
    def display_name(self) -> str:
        return "None"

    @property
    def description(self) -> str:
        return self._description

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [media]
