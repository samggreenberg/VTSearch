"""MediaClipper ABC for splitting media items into sub-items."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MediaClipper(ABC):
    """Abstract base class for media clippers.

    A *MediaClipper* takes a single media item and returns one or more media
    items of the **same** type.  Each concrete clipper operates on exactly one
    media type (declared via :attr:`media_type`).

    Unlike :class:`~vtsearch.media.processors.Processor` subclasses which
    return metadata *about* a media (booleans, bounding boxes, labels), a
    ``MediaClipper`` returns **new media dicts** that can be used directly in
    place of the original.

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

    @property
    def creation_questions(self) -> list[dict[str, Any]]:
        """Return a list of questions to present when the user chooses this clipper.

        Each question is a dict with the same schema as :attr:`parameters`
        descriptors:

        * ``key`` — parameter name (used in :meth:`with_params`).
        * ``label`` — human-readable label / question for the UI.
        * ``type`` — ``"number"`` or ``"string"``.
        * ``default`` — current/default value.
        * ``min`` / ``max`` / ``step`` — optional numeric constraints.

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
        the structure of the original (``id``, ``type``, ``category``,
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
        """Return a concrete clipper for the dataset given its media durations.

        Called by the load pipeline once per dataset, before any
        :meth:`clip` calls.  Most clippers ignore *durations* and return
        ``self``.  Auto-selecting clippers (e.g. ``SoundAutoClipper``)
        override this to pick a different concrete clipper based on the
        dataset's typical duration — for example, pass-through for short
        clips, tiling for longer ones.

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
        params = self.parameters
        if params:
            d["parameters"] = params
        cq = self.creation_questions
        if cq:
            d["creation_questions"] = cq
        return d
