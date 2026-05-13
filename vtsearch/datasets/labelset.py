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
        metadata: Arbitrary per-element metadata that round-trips through
            serialisation.  Importers and external systems (e.g. Holder)
            can attach extra key-value data here (such as ``contentID``,
            ``mediaID``, ``media_url``).  ``None`` when no metadata is
            present.
        region_box: Normalised ``(x0, y0, x1, y1)`` box on the source
            image when the user drew a region as part of a yes-vote;
            ``None`` for image-level votes (the perpetual default for
            no-votes, and the v1 default for every vote).  See the
            patch-embedder v2 design for region-vote semantics.
    """

    md5: str
    label: str
    origin: dict[str, Any] | None = None
    origin_name: str = ""
    filename: str = ""
    category: str = ""
    metadata: dict[str, Any] | None = None
    region_box: tuple[float, float, float, float] | None = None

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
        if self.metadata is not None:
            d["metadata"] = self.metadata
        if self.region_box is not None:
            d["region_box"] = list(self.region_box)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabeledElement:
        """Reconstruct a :class:`LabeledElement` from a dict.

        ``region_box`` is accepted as either a list (its JSON form) or a
        tuple, and is always normalised to a 4-tuple of floats so the
        dataclass invariant holds regardless of where the dict came from.
        """
        rb = d.get("region_box")
        region_box: tuple[float, float, float, float] | None = None
        if rb is not None:
            region_box = (float(rb[0]), float(rb[1]), float(rb[2]), float(rb[3]))
        return cls(
            md5=d.get("md5", ""),
            label=d.get("label", ""),
            origin=d.get("origin"),
            origin_name=d.get("origin_name", ""),
            filename=d.get("filename", ""),
            category=d.get("category", ""),
            metadata=d.get("metadata"),
            region_box=region_box,
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
        *,
        expand_dupes: bool = True,
        vote_region_boxes: dict[int, tuple[float, float, float, float]] | None = None,
    ) -> LabelSet:
        """Build a ``LabelSet`` from the current media and vote state.

        Args:
            medias: The global medias dict.
            good_votes: Dict of media IDs voted "good".
            bad_votes: Dict of media IDs voted "bad".
            expand_dupes: When ``True`` (default), dupe-set representatives
                are expanded into one element per member so that an exported
                labelset records full provenance.  Set to ``False`` when the
                labelset will be re-imported into the same system (e.g.
                detector persistence) to avoid inflating the label
                count.
            vote_region_boxes: Optional ``media_id -> (x0, y0, x1, y1)``
                map populated by yes-votes that designated a region.  Each
                box is attached to the corresponding good vote's
                :class:`LabeledElement`.  Ignored for bad votes (no-votes
                are always image-level — see the patch-embedder v2 design).

        Returns:
            A new ``LabelSet`` containing one :class:`LabeledElement` per
            voted media, in vote-insertion order (good votes first, then bad).
        """
        region_boxes = vote_region_boxes or {}
        elements: list[LabeledElement] = []
        for cid in good_votes:
            media = medias.get(cid)
            if media:
                elements.extend(
                    _clip_to_elements(
                        media,
                        "good",
                        expand_dupes=expand_dupes,
                        region_box=region_boxes.get(cid),
                    )
                )
        for cid in bad_votes:
            media = medias.get(cid)
            if media:
                elements.extend(_clip_to_elements(media, "bad", expand_dupes=expand_dupes))
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
                :func:`~vtsearch.cli._build_multi_results_dict`.
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
                        origin_name = origin_name or media.get("origin_name", media.get("filename", ""))
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

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge(self, *others: LabelSet, conflict_policy: str = "drop") -> LabelSet:
        """Merge this labelset with one or more others.

        Elements are keyed by ``Origin`` (importer + params) when present,
        falling back to ``md5`` for legacy entries with no origin.  Elements
        with neither an origin nor an md5 are dropped (no way to dedupe).

        Behaviour:

        * Same key + same label across inputs → kept once.  Metadata dicts
          from later sources are shallow-merged onto earlier ones (later
          wins on key collision).
        * Same key + different labels → conflict; resolved per
          ``conflict_policy``.

        Args:
            *others: Additional :class:`LabelSet` instances to merge in.
            conflict_policy: Currently only ``"drop"`` is supported, which
                removes every entry that has a label disagreement across
                the input labelsets.

        Returns:
            A new :class:`LabelSet`.  Insertion order follows ``self``
            first, then each ``other`` in turn; on dedup the *first* entry
            wins position, with later metadata merged in.
        """
        if conflict_policy != "drop":
            raise ValueError(f"Unsupported conflict_policy: {conflict_policy!r}")

        # First pass: collect every label-claim per key so we can detect
        # disagreement before deciding what to keep.
        labels_by_key: dict[Any, set[str]] = {}
        for ls in (self, *others):
            for el in ls.elements:
                key = element_key(el)
                if key is None:
                    continue
                labels_by_key.setdefault(key, set()).add(el.label)

        conflicting_keys = {k for k, labels in labels_by_key.items() if len(labels) > 1}

        # Second pass: emit one element per non-conflicting key, in
        # first-seen order.  Shallow-merge metadata from later occurrences.
        merged: list[LabeledElement] = []
        seen: dict[Any, int] = {}
        for ls in (self, *others):
            for el in ls.elements:
                key = element_key(el)
                if key is None or key in conflicting_keys:
                    continue
                if key in seen:
                    existing = merged[seen[key]]
                    if el.metadata:
                        merged_meta = dict(existing.metadata or {})
                        merged_meta.update(el.metadata)
                        existing.metadata = merged_meta
                    continue
                seen[key] = len(merged)
                merged.append(
                    LabeledElement(
                        md5=el.md5,
                        label=el.label,
                        origin=dict(el.origin) if el.origin else None,
                        origin_name=el.origin_name,
                        filename=el.filename,
                        category=el.category,
                        metadata=dict(el.metadata) if el.metadata else None,
                        region_box=el.region_box,
                    )
                )
        return LabelSet(merged)


def element_key(el: LabeledElement) -> Any:
    """Return a hashable identity key for an element, or ``None`` if it has none.

    Prefers the element's ``Origin`` (importer + params) so that the same
    source media can be deduped across labelsets even when re-embedded with
    different embedders.  Falls back to ``md5`` for legacy entries that
    have no origin.
    """
    if el.origin:
        importer = el.origin.get("importer", "")
        params = el.origin.get("params") or {}
        try:
            params_key = tuple(sorted(params.items()))
        except TypeError:
            params_key = tuple(sorted((str(k), str(v)) for k, v in params.items()))
        return ("origin", importer, params_key, el.origin_name)
    if el.md5:
        return ("md5", el.md5)
    return None


def media_element_key(media: dict[str, Any]) -> Any:
    """Return the element-identity key for a media dict (origin-keyed when possible).

    Mirrors :func:`element_key` so that callers can match media items in a
    dataset against entries in a :class:`LabelSet` without first converting
    them to :class:`LabeledElement`.
    """
    origin = media.get("origin")
    origin_name = media.get("origin_name", media.get("filename", ""))
    md5 = media.get("md5", "")
    fake = LabeledElement(md5=md5, label="", origin=origin, origin_name=origin_name)
    return element_key(fake)


def _clip_to_elements(
    media: dict[str, Any],
    label: str,
    *,
    expand_dupes: bool = True,
    region_box: tuple[float, float, float, float] | None = None,
) -> list[LabeledElement]:
    """Convert a media dict into one or more :class:`LabeledElement` instances.

    When *expand_dupes* is ``True`` and the media is a dupe-set
    representative (origin importer is ``"dupe_set"``), one element is
    produced for each original member so that an exported labelset reflects
    the full duplicate set.  When ``False``, a single element is emitted
    using the representative's own MD5 and origin, which avoids inflating
    the label count for internal round-trip use cases (e.g. detector
    persistence).

    When *region_box* is provided (only ever populated for ``label == "good"``
    by the caller), it is attached to every emitted element so the region
    annotation rides along through label export / detector sync.  Dupe-set
    members share the representative's box — the user voted "this region is
    good in *this* image" and the representative is what the user actually
    saw, so cloning the box across structurally-identical members is the
    right default.
    """
    origin = media.get("origin")
    cm = media.get("custom_metadata") or None
    if expand_dupes and isinstance(origin, dict) and origin.get("importer") == "dupe_set":
        members = origin.get("members", [])
        if members:
            return [
                LabeledElement(
                    md5=m.get("md5", media["md5"]),
                    label=label,
                    origin=m.get("origin"),
                    origin_name=m.get("origin_name", m.get("filename", "")),
                    filename=m.get("filename", ""),
                    category=m.get("category", ""),
                    metadata=cm,
                    region_box=region_box,
                )
                for m in members
            ]

    # Non-dupe or missing members — single element
    return [
        LabeledElement(
            md5=media["md5"],
            label=label,
            origin=origin,
            origin_name=media.get("origin_name", media.get("filename", "")),
            filename=media.get("filename", ""),
            category=media.get("category", ""),
            metadata=cm,
            region_box=region_box,
        )
    ]
