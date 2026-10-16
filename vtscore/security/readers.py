"""How a change to a registry entry's ``readers`` list ended.

Datasets and detectors share one access model: an entry is visible to its
creator, to every user in its ``readers`` list, and to everyone when that list
holds ``"*"``, and only the creator may change the list.  The two registries
(:func:`vtscore.datasets.registry.update_readers`,
:func:`vtscore.detectors.registry.update_detector_readers`) both answer such a
change with a :class:`ReadersUpdate`, so a caller that has to tell "not yours"
from "no such entry" (the readers routes pick 403 or 404 from it) branches on
the member, never on the wording of a message.

The enum lives here rather than in either registry so that importing the
light detector registry does not pull in the whole :mod:`vtscore.datasets`
package.
"""

from __future__ import annotations

import enum


class ReadersUpdate(enum.Enum):
    """The outcome of replacing an entry's ``readers`` list."""

    #: The list was replaced.
    UPDATED = "updated"
    #: No entry has that id.
    NOT_FOUND = "not_found"
    #: The entry exists, but the requesting user did not create it.
    NOT_CREATOR = "not_creator"

    def message(self, kind: str) -> str:
        """Say for a person why a change to a *kind* entry failed.

        *kind* names the entry (``"dataset"``, ``"detector"``).  Empty for
        :attr:`UPDATED`.  This is the ``err`` that the ``(ok, err)`` forms,
        :func:`vtscore.datasets.registry.set_readers` and
        :func:`vtscore.detectors.registry.set_detector_readers`, return.  It is
        prose: compare the member, not this text.
        """
        if self is ReadersUpdate.NOT_CREATOR:
            return f"Only the {kind} creator can modify readers"
        if self is ReadersUpdate.NOT_FOUND:
            return f"{kind.capitalize()} not found"
        return ""
