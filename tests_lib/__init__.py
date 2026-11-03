"""Library-only test package for VTSearch.

Tests under :mod:`tests_lib` exercise ``vtscore`` and never import
``vtsearch`` — not :mod:`vtsearch.app`, :mod:`vtsearch.routes` or
:mod:`vtsearch.settings`, and not :mod:`vtsearch.state` either, whose
``medias`` is an app-tier *proxy* rather than the library object.  The split
is seam 7 ("Test suite") in ``../vtscore/docs/architecture.md``.

That promise sat here unchecked until issue #3421, when ``conftest.py`` and
``fixtures/medias.py`` were both found importing ``vtsearch.state``.  Two
gates now hold it, and they see different things:

* ``./run-tests.sh vtscore-clean`` blocks ``flask``/``werkzeug``/
  ``flask_smorest`` at import time.  It cannot block ``vtsearch``: the
  library ships inside the same distribution, so the package must stay
  importable for the library's own optional, guarded references to it.
* ``tests_lib/meta/test_library_layering.py`` statically scans this tree for
  ``vtsearch`` imports *and* for ``mock.patch("vtsearch…")`` targets — an
  import in every sense that matters, and the one the AST scan alone misses.

The folders mirror ``vtscore``'s packages (#5403): a test of
``vtscore/eval/`` lives in ``eval/``, a test of a top-level module such as
``vtscore/host_seams.py`` at this package's root.  A folder is therefore not
a ``./run-tests.sh`` group by name, and
``tests_shared.pytest_plumbing.LIB_GROUPS`` maps each one to the group it runs
under (the table is in ``docs/TESTING.md``).

``tests_lib/meta/`` is a deliberate lodger rather than a library package: its
subject is the repository's own tooling, which satisfies the tier's contract
trivially.  See its ``__init__`` for why it is not a third top-level tree.
"""
