"""Library-only test package for VTSearch.

Tests under :mod:`tests_lib` import only the ``vtscore`` candidate
subpackages of ``vtsearch`` (``datasets/``, ``detectors/``,
``embedding/``, …) and never reach into ``vtsearch.app``,
``vtsearch.routes``, ``vtsearch.settings``, or any other app-tier
module.  The split is the test-suite half of Phase 7 in
``docs/plans/extract-library.md``.
"""
