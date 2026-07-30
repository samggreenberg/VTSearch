"""MediaCleaner ABC: optional 1→1 cleanup gates applied before embedding.

A *MediaCleaner* removes content-free regions from a media item so the
embedder spends its representational capacity on signal instead of
letterbox bars, leading silence, or PDF-extraction junk.  Like a
:class:`~vtscore.media.clipper.MediaClipper` it maps type X to type X, but
it differs in **cardinality** and **use**:

============  ==================  ==============================
              Clipper             Cleaner
============  ==================  ==============================
Cardinality   1 → N               1 → 1
UI            pick **one**        **all optional gates**, each
              per import          independently toggleable
============  ==================  ==============================

A clipper breaks large media into manageable sub-items; a cleaner tightens
each item in place.  Cleaners therefore run *after* the final
clipper/converter step, on the units that will actually be embedded, and
every enabled cleaner for the chain's final media type sees every unit.

Subclassing :class:`MediaClipper` reuses the whole descriptor stack
(``name`` / ``media_type`` / ``display_name`` / ``description`` /
``parameters`` / ``creation_questions`` / ``with_params`` / ``to_dict``) and
lets a cleaner ride the existing chain machinery as an ``n_out == 1`` step,
so cross-dataset label replay comes for free.  Cleaners live in their own
registry (``register_cleaner`` / ``get_cleaner`` / ``cleaners_for_type``) so
they never appear in a clipper chooser.

See ``docs/plans/media-cleaners.md`` for the design and the roster of
cleaners still to be written.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from vtscore.media.clipper import MediaClipper


class MediaCleaner(MediaClipper):
    """A 1→1 cleanup step run on each final unit before embedding.

    Subclasses implement :meth:`clean`; :meth:`clip` wraps it so the chain
    runner can treat a cleaner as an ordinary single-output chain step.
    """

    @property
    def default_enabled(self) -> bool:
        """Whether the import UI checks this cleaner's box by default.

        ``False`` for every cleaner that makes a judgment call about what
        counts as wasted content.  Override to ``True`` only for cleaners
        that fix an outright representation bug (e.g. an image the embedder
        would otherwise see sideways), where leaving it off means shipping
        known-wrong vectors.
        """
        return False

    def with_params(self, params: dict[str, Any]) -> "MediaCleaner":
        """Return a **new** cleaner of the same type with overridden parameters.

        Narrows :meth:`MediaClipper.with_params`' return type: a cleaner's
        parameter override is still a cleaner, so callers keep access to
        :meth:`clean`.  The default still returns ``self`` unchanged, which is
        correct for a cleaner with no parameters.
        """
        return self

    @abstractmethod
    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        """Return a media dict with wasted regions removed.

        Must return *media* unchanged (or an equal copy) when there is
        nothing to clean or the payload can't be decoded: like a clipper, a
        cleaner never aborts a load, and a degenerate input is a no-op
        rather than an error.  The chain runner detects "nothing changed"
        by comparing the payload before and after, and only then skips
        snapshotting the pre-clean version.
        """

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Adapt :meth:`clean` to the clipper interface (always one output)."""
        return [self.clean(media)]

    def to_dict(self) -> dict[str, Any]:
        """Return the clipper descriptor plus the cleaner-only default flag."""
        d = super().to_dict()
        d["default_enabled"] = self.default_enabled
        return d
