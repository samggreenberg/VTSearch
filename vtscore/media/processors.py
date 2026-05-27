"""Processor ABCs: Processor, Detector, Localizer, Extractor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Processor(ABC):
    """Abstract base class for all processors (detectors, localizers, extractors, etc.).

    A *Processor* takes a single media media and produces an answer.  The
    exact type of the answer depends on the subclass:

    * A :class:`Detector` returns ``bool``: "does this media match?"
    * A :class:`Localizer` returns ``list[dict]``: "where in this media
      is the item of interest?" (bounding boxes with confidence scores).
    * An :class:`Extractor` returns ``list[dict]``: "what details are
      inside this media?" (bounding boxes, labels, and other metadata).

    Every processor knows its :attr:`name` (a unique human-readable
    identifier) and the :attr:`media_type` it operates on (e.g.
    ``"audio"``, ``"image"``).

    Subclasses must implement:

    * :attr:`name`
    * :attr:`media_type`
    * :meth:`process`: run the processor on a single media dict.

    Subclasses *may* override:

    * :meth:`load_model`: called once before first use to load heavy
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
        implementation is a no-op; override in subclasses that need
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

    * :attr:`name`: unique identifier for this detector.
    * :attr:`media_type`: which media type it works on.
    * :meth:`detect`: run detection on a single media dict and return
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
    boxes with confidence scores but no further classification or
    extraction metadata.

    Each concrete ``Localizer`` operates on exactly **one** media type
    (declared via :attr:`media_type`), just like Detectors and Extractors.

    Subclasses must implement:

    * :attr:`name`: unique identifier for this localizer.
    * :attr:`media_type`: which media type it works on.
    * :meth:`localize`: run localization on a single media dict and return
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

        * ``"confidence"``: a float in ``[0, 1]``.
        * ``"bbox"``: the bounding box (format is media-specific, e.g.
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

    Each concrete ``Extractor`` operates on exactly **one** media type
    (declared via :attr:`media_type`), just like Detectors.

    Subclasses must implement:

    * :attr:`name`: unique identifier for this extractor.
    * :attr:`media_type`: which media type it works on.
    * :meth:`extract`: run extraction on a single media dict and return a
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
