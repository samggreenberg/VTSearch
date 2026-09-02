"""Process-local memoization for the eval-harness sweeps.

``simulate_voting_iterations`` is by far the most expensive thing the Python
suite does: a single call trains a head and fits the safe-threshold mixtures
once per simulated voting step, so one sweep costs seconds.  The threshold /
variant-frame tests are written as one assertion per behaviour, and most of
them assert on *different columns of the same frame* — so the same sweep, with
byte-identical arguments, was being recomputed once per test.  Across
``tests_lib/detectors`` that duplication was roughly a quarter of the whole
pytest runtime.

:func:`memoize_sweep` wraps such a helper so identical argument sets are
computed once per worker process and replayed thereafter.  Two properties make
this sound:

* **The sweeps are deterministic.** Every call site passes an explicit
  ``seed``, and the planted fixtures are built from a seeded ``default_rng``,
  so a repeat call cannot produce different rows.
* **The call still happens inside the test body**, not in a module- or
  session-scoped fixture. That matters: higher-scoped fixtures are set up
  *before* the function-scoped autouse ``reset_contexts``, which would let a
  previous test's leaked ``config.TRAIN_EPOCHS`` (issue #3101) decide the
  training budget for a cached sweep. Memoizing the plain helper keeps the
  first, cache-filling call in exactly the state it runs in today.

**Cached results are shared, so treat them as read-only.** A test that mutates
returned rows would corrupt every later test that gets the same cached object.
Tests that genuinely need a fresh computation — determinism checks that compare
two independent runs, or anything passing a mutable output sink — must call the
underlying helper directly rather than the memoized wrapper.

For the cache to actually hit, the tests sharing it have to land on the same
xdist worker: a module that relies on this carries a
``pytestmark = pytest.mark.xdist_group(...)`` so ``--dist loadgroup`` keeps it
together. Without that, a 4-worker run scatters the calls and recomputes the
sweep on each worker that sees one.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def memoize_sweep(fn: _F) -> _F:
    """Memoize *fn* on its arguments for the life of the worker process.

    Unlike :func:`functools.lru_cache` this accepts keyword arguments whose
    values are unhashable (the sweeps take lists of rules / inclusion ks), by
    keying on a ``repr`` of the bound arguments instead of the values
    themselves. The keys are small, fully-determined literals at every call
    site, so ``repr`` is an exact identity here rather than an approximation.
    """
    cache: dict[str, Any] = {}

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = repr((args, sorted(kwargs.items())))
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
