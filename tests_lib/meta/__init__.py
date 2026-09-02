"""Repo/tooling meta-tests: the subject is this repository, not the product.

Everything here tests something that is **not shipped as ``vtsearch`` or
``vtscore`` code** - the packaging and requirements files, the Dockerfiles, the
docs, the frontend's SCSS text, the helper scripts under ``scripts/`` and the
hooks under ``.claude/hooks/``, the ``run-tests.sh`` gates themselves, and the
test harness in ``tests_shared/``.  A test belongs here when a reader asking
"what does VTSearch do?" would never open it, and a reader asking "how is this
repo built and checked?" would.

They live under ``tests_lib/`` rather than a third top-level tree because they
satisfy the library tier's contract trivially - they import no ``vtsearch``
module, so ``./run-tests.sh vtscore-clean`` keeps proving it - and because the
group system already keys off the folder name, which is what actually makes
them findable and skippable (``./run-tests.sh meta``).

What they are *not* is ``core``.  They silted into that group because it was
the nearest folder, and ``./run-tests.sh core`` - the fast inner loop for
"basic app functionality" - ended up running a Dockerfile text parser and four
gate self-tests (issue #3421).
"""
