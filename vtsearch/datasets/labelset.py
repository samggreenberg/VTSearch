"""LabelSet — a dataset of labeled elements with origin tracking.

A :class:`LabelSet` is conceptually an extension of a dataset: each element
knows its *origin* (where it came from), its *origin_name* (a unique
identifier within that origin), **and** its *label* (``"good"`` or
``"bad"``).

This structure is the canonical format for:

* Exporting labels (``GET /api/labels/export``)
* Importing labels (label importers)
* Passing labeled results to
  :class:`~vtsearch.exporters.base.ResultsExporter` instances

The serialised format is a strict superset of the legacy label-export
format.  Old consumers that only read ``md5`` and ``label`` keys still
work; new consumers get the additional provenance fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LabeledElement:
    """A single element in a :class:`LabelSet`.

    Attributes:
        md5: Content hash of the element's media bytes.
        label: ``"good"`` or ``"bad"``.
        origin: Serialised :class:`~vtsearch.datasets.origin.Origin` dict
            (with ``"importer"`` and ``"params"`` keys), or ``None`` when
            origin information is unavailable (e.g. imported from a legacy
            label file).
        origin_name: Name of the element within its origin (typically the
            filename, e.g. ``"clip_123.wav"``).
        filename: Original filename of the media file.
        category: Category or class label from the dataset structure.
    """

    md5: str
    label: str
    origin: dict[str, Any] | None = None
    origin_name: str = ""
    filename: str = ""
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict.

        Only non-empty optional fields are included so the output stays
        compact for legacy consumers.
        """
        d: dict[str, Any] = {"md5": self.md5, "label": self.label}
        if self.origin is not None:
            d["origin"] = self.origin
        if self.origin_name:
            d["origin_name"] = self.origin_name
        if self.filename:
            d["filename"] = self.filename
        if self.category:
            d["category"] = self.category
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabeledElement:
        """Reconstruct a :class:`LabeledElement` from a dict."""
        return cls(
            md5=d.get("md5", ""),
            label=d.get("label", ""),
            origin=d.get("origin"),
            origin_name=d.get("origin_name", ""),
            filename=d.get("filename", ""),
            category=d.get("category", ""),
        )


class LabelSet:
    """An ordered collection of :class:`LabeledElement` instances.

    A ``LabelSet`` extends the concept of a dataset: each element carries
    its provenance (origin + origin_name) and its label.

    Parameters:
        elements: Initial list of :class:`LabeledElement` instances.
    """

    def __init__(self, elements: list[LabeledElement] | None = None) -> None:
        self.elements: list[LabeledElement] = list(elements) if elements else []

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_clips_and_votes(
        cls,
        medias: dict[int, dict[str, Any]],
        good_votes: dict[int, None],
        bad_votes: dict[int, None],
    ) -> LabelSet:
        """Build a ``LabelSet`` from the current media and vote state.

        Args:
            medias: The global medias dict.
            good_votes: Dict of media IDs voted "good".
            bad_votes: Dict of media IDs voted "bad".

        Returns:
            A new ``LabelSet`` containing one :class:`LabeledElement` per
            voted media, in vote-insertion order (good votes first, then bad).
        """
        elements: list[LabeledElement] = []
        for cid in good_votes:
            media = medias.get(cid)
            if media:
                elements.append(_clip_to_element(media, "good"))
        for cid in bad_votes:
            media = medias.get(cid)
            if media:
                elements.append(_clip_to_element(media, "bad"))
        return cls(elements)

    @classmethod
    def from_results(
        cls,
        results: dict[str, Any],
        medias: dict[int, dict[str, Any]] | None = None,
    ) -> LabelSet:
        """Build a ``LabelSet`` from an auto-detect results dict.

        Each hit that scored at or above the detector's threshold is
        treated as a ``"good"`` label.

        Args:
            results: A results dict as produced by ``/api/auto-detect`` or
                :func:`~vtsearch.cli._build_results_dict`.
            medias: Optional medias dict for enriching hits with origin info.
                When provided, origin data is looked up from the media; when
                absent, origin data is taken from the hit dict itself (if
                present).

        Returns:
            A new ``LabelSet`` with one element per hit across all detectors.
        """
        elements: list[LabeledElement] = []
        for det_result in results.get("results", {}).values():
            for hit in det_result.get("hits", []):
                origin = hit.get("origin")
                origin_name = hit.get("origin_name", "")
                if medias and not origin:
                    media = medias.get(hit.get("id"))
                    if media:
                        origin = media.get("origin")
                        origin_name = origin_name or media.get(
                            "origin_name", media.get("filename", "")
                        )
                elements.append(
                    LabeledElement(
                        md5=hit.get("md5", ""),
                        label="good",
                        origin=origin,
                        origin_name=origin_name,
                        filename=hit.get("filename", ""),
                        category=hit.get("category", ""),
                    )
                )
        return cls(elements)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict.

        Returns:
            ``{"labels": [<element dict>, ...]}``.  The format is a
            superset of the legacy label-export format (which only had
            ``md5`` and ``label`` keys), so existing consumers remain
            compatible.
        """
        return {"labels": [e.to_dict() for e in self.elements]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelSet:
        """Reconstruct a ``LabelSet`` from a dict produced by :meth:`to_dict`.

        Also accepts the legacy label format (entries with only ``md5`` and
        ``label`` keys) for backward compatibility.
        """
        elements: list[LabeledElement] = []
        for entry in d.get("labels", []):
            if not isinstance(entry, dict):
                continue
            elements.append(LabeledElement.from_dict(entry))
        return cls(elements)


def _clip_to_element(media: dict[str, Any], label: str) -> LabeledElement:
    """Convert a media dict into a :class:`LabeledElement`."""
    return LabeledElement(
        md5=media["md5"],
        label=label,
        origin=media.get("origin"),
        origin_name=media.get("origin_name", media.get("filename", "")),
        filename=media.get("filename", ""),
        category=media.get("category", ""),
    )
