"""Fixtures and helpers shared by the ``tests/`` and ``tests_lib/`` conftests.

The two suites need the same session bootstrap (fake embedders, startup
contexts, heap freeze), the same per-test reset of ``vtscore``'s process-global
state, and the same end-of-run summary printer.  They used to carry
near-verbatim copies of all of it, and the copies drifted: the library tier's
fake audio embedder fell back to a ``PYTHONHASHSEED``-salted ``hash()`` (giving
every in-memory media the same vector, non-deterministically across xdist
workers) and its reset fixture never dropped
``vtscore.detectors.dataset_sync``'s TTL mtime cache.  Both were silent
(issue #3424).

This package is the single source for those pieces.  What stays in each
conftest is what is genuinely tier-specific: the app tier's Flask ``client``,
settings isolation and autorun-processor reset; the library tier's Flask
blocker, native-thread caps and library-only ``CoreConfig`` builder.

**Contract:** nothing here may import ``flask``, ``werkzeug``,
``flask_smorest``, or any app-tier ``vtsearch`` module — ``tests_lib/`` imports
this package under the Flask blocker installed by
``scripts/check-vtscore-clean.py``.  Mutable app-tier objects (the ``medias``
map) are passed in by the caller rather than imported here.

Nothing under this package is collected as a test; it holds no ``test_*``
modules and is not on ``testpaths``.
"""
