"""How a dataset import ended, as handed to code outside the load pipeline.

The load pipeline (:func:`vtscore.datasets.load_pipeline._run_origin_load_in_background`)
reports the end of an import through its ``on_finished`` callback with one
:class:`DatasetImported`.  The app passes that event on to the functions a
server admin names with ``--on-dataset-imported`` (see
:mod:`vtsearch.import_hooks`), which is how a deployment sends an email, posts
to a chat channel, or does anything else when a user's import completes.

The event is a frozen value rather than the live ``DatasetContext`` on purpose.
A hook reads what it needs to tell someone about the import; it never reaches
into the dataset's in-memory state, so that state stays free to change, and a
new field can be added here without breaking a hook that ignores it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

#: How an import ended: :data:`SUCCEEDED` or :data:`FAILED`.
ImportOutcome = Literal["succeeded", "failed"]

#: The import saved a dataset.
SUCCEEDED: Final = "succeeded"

#: The import stopped on an error and saved nothing.  A cancelled import is
#: not reported at all: the user who cancelled it already knows.
FAILED: Final = "failed"


@dataclass(frozen=True)
class DatasetImported:
    """One finished dataset import.

    :param outcome: :data:`SUCCEEDED` or :data:`FAILED`.  Compare against
        those two values exactly rather than treating "not succeeded" as
        failed, so a hook keeps doing the right thing if a new outcome is ever
        reported.
    :param dataset_id: The saved dataset's id; empty when the import failed.
    :param name: The dataset's display name: the saved name on success (the
        registry may have made it unique), the requested one on failure.
    :param user: The VTSearch username that started the import.  VTSearch
        stores no email addresses, so mapping a user to an address is the
        hook's job.
    :param media_type: The dataset's media type (``"image"``, ``"audio"``, …);
        empty when an import failed before it could tell.
    :param n_media: How many items the saved dataset holds; 0 on failure.
    :param origin: The importer and parameters the dataset was built from (a
        copy, so a hook cannot change the dataset's recorded origin).
    :param error: The message the user was shown; empty on success.
    """

    outcome: ImportOutcome
    dataset_id: str
    name: str
    user: str
    media_type: str
    n_media: int
    origin: dict[str, Any] = field(default_factory=dict)
    error: str = ""
