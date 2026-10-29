"""Command-line interface utilities for VTSearch.

The only CLI workflow is autodetect: load a dataset (from pickle or via an
importer), score it against the detectors flagged for AutoFind in the settings
file, and export the results.  With ``save_dataset`` the source is first
imported through the GUI's own load pipeline and registered, so it shows up on
the dashboard, and the run then scores that saved dataset.

The package holds one stage per module, each reading only from those above it:

* :mod:`~vtscore.cli.progress` - the text / NDJSON event stream every stage
  reports on (``--progress-format``).  Depends on nothing.
* :mod:`~vtscore.cli.resume` - the ``--resume`` checkpoint format: run
  identity, chunk fingerprints, the exclusive-run lock.
* :mod:`~vtscore.cli.sources` - where a run's medias come from: a dataset
  pickle or an importer, whole or chunked, one dataset or several.
* :mod:`~vtscore.cli.detectors` - reading, re-clipping and training the run's
  detectors against its first chunk.
* :mod:`~vtscore.cli.scoring` - scoring a chunk against the trained heads and
  merging chunks' results.
* :mod:`~vtscore.cli.export` - resolving and running the exporter.
* :mod:`~vtscore.cli.labels` - importing labels into a detector.
* :mod:`~vtscore.cli.datasets` - saving a run's datasets to the dashboard, and
  multi-dataset runs.
* :mod:`~vtscore.cli.dry_run` - ``--dry-run``'s validation and plan.
* :mod:`~vtscore.cli.streaming` - ``--stream-results``.
* :mod:`~vtscore.cli.resumable` - ``--resume``, the checkpointed stream.
* :mod:`~vtscore.cli.run` - one source end to end: save, dry run, or detect
  and export.
* :mod:`~vtscore.cli.autodetect` - the four public ``autodetect_*_main``
  entry points.
* :mod:`~vtscore.cli.pipeline` - the YAML pipeline file (``--pipeline``),
  dispatched onto :mod:`~vtscore.cli.run`.

The public entry points are re-exported here, so ``vtscore.cli.X`` resolves as
it did when this was one module.  ``vtscore.cli_progress``,
``vtscore.cli_resume`` and ``vtscore.cli_pipeline`` are deprecated aliases of
:mod:`~vtscore.cli.progress`, :mod:`~vtscore.cli.resume` and
:mod:`~vtscore.cli.pipeline` - the same module objects, so their state is
shared.

**Patch targets** follow the caller, not this package: a stub has to be
installed where the name is looked up.  A caller that imports a public name
from this package lazily (as ``vtsearch.cli_main`` and
:mod:`~vtscore.cli.pipeline` do with :func:`import_labels_into_detector` and
the ``autodetect_*_main`` entry points) re-reads it here on every call, so
patching ``vtscore.cli`` reaches it.  A caller inside a stage module resolves
its own module global: stubbing ``_load_and_train_detectors`` for
:func:`~vtscore.cli.detectors._train_detectors_for_first_chunk` means patching
``vtscore.cli.detectors``, and stubbing ``_run_source`` for the pipeline
runner means patching ``vtscore.cli.run`` - not this package, where the
rebind is silently ignored.
"""

from __future__ import annotations

from vtscore.cli.autodetect import (
    autodetect_importer_main,
    autodetect_importer_main_chunked,
    autodetect_main,
    autodetect_main_chunked,
)
from vtscore.cli.labels import (
    DetectorNotFoundError,
    import_labels_into_detector,
    import_labels_into_detector_from_file,
)

__all__ = [
    "DetectorNotFoundError",
    "autodetect_importer_main",
    "autodetect_importer_main_chunked",
    "autodetect_main",
    "autodetect_main_chunked",
    "import_labels_into_detector",
    "import_labels_into_detector_from_file",
]
