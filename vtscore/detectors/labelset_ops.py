"""Single entry point for the detector-labelset operations surface.

VTSearch has two unrelated families that both say "labels":

* **Labels, the detector concept** - a detector's saved *labelset*: the
  ``LabeledElement``s persisted alongside it, the votes synced into it, the
  training/scoring that runs against it, and the bookkeeping that keeps it
  consistent when datasets or detector files move.  That family is spread
  across five sibling modules in this package
  (:mod:`~vtscore.detectors.label_sync`,
  :mod:`~vtscore.detectors.label_restoration`,
  :mod:`~vtscore.detectors.labelset_elements`,
  :mod:`~vtscore.detectors.labelset_training`,
  :mod:`~vtscore.detectors.labelset_rename`) plus the
  :mod:`vtscore.datasets.labelset` data model.
* **Labels, the import/export plugin family** - the ``labels`` plugin
  package (:mod:`vtscore.labels`, served by ``vtsearch/routes/labels/``)
  that reads and writes label files in external formats.  That is a
  *plugin family* like importers/exporters, not the detector concept.

The split names blur which "labels" a reader means.  This module is the
facade for the first family: callers that need detector-labelset operations
import :mod:`vtscore.detectors.labelset_ops` once and reach the whole
surface through it, instead of remembering which of the five sibling
modules a given function lives in.  The sibling modules remain the
implementation homes; this is the discoverable seam.

That includes the concurrency contract: any caller doing its own
read-modify-write of a detector JSON file must hold
:data:`~vtscore.detectors.label_sync.label_sync_write_lock` across the whole
pass, and will usually want
:func:`~vtscore.detectors.label_sync.merge_labelsets_across_datasets` to
reconcile the result.  Both are re-exported here so a writer never has to
reach into a sibling module for them.
"""

from __future__ import annotations

from vtscore.detectors.label_restoration import restore_labels_from_detector
from vtscore.detectors.label_sync import (
    label_sync_write_lock,
    merge_labelsets_across_datasets,
    sync_labels_to_loaded_detector,
)
from vtscore.detectors.labelset_elements import (
    apply_element_vote_in_data,
    build_element_view,
    build_labels_detail,
    find_element_by_id,
    resolve_current_dataset_cid,
    resolve_element_to_path,
    stable_element_id,
)
from vtscore.detectors.labelset_rename import (
    detect_pending_labelset_move,
    move_labelset_file,
)
from vtscore.detectors.labelset_training import (
    build_xy_from_labelset,
    labelset_resolution_report,
    labelset_train_and_score,
    populate_label_embeddings,
    train_from_labelset,
)

__all__ = [
    # label_restoration
    "restore_labels_from_detector",
    # label_sync
    "label_sync_write_lock",
    "merge_labelsets_across_datasets",
    "sync_labels_to_loaded_detector",
    # labelset_elements
    "apply_element_vote_in_data",
    "build_element_view",
    "build_labels_detail",
    "find_element_by_id",
    "resolve_current_dataset_cid",
    "resolve_element_to_path",
    "stable_element_id",
    # labelset_rename
    "detect_pending_labelset_move",
    "move_labelset_file",
    # labelset_training
    "build_xy_from_labelset",
    "labelset_resolution_report",
    "labelset_train_and_score",
    "populate_label_embeddings",
    "train_from_labelset",
]
