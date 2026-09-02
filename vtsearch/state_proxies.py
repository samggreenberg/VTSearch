"""App-side proxy view over the library's ``DatasetContext`` / ``DetectorContext``.

Many app-layer call sites (Flask routes, app-tier tests) like to import
``medias``, ``good_votes``, etc. as module-level dict/list names and treat
them as the "current" container.  The library does not depend on those
names - its functions resolve the active context explicitly - but the app
keeps them as a convenience facade so existing code reads naturally.

These proxies live on the app side of the ``vtscore`` / ``vtsearch`` split
(see Phase 3 of ``../vtscore/docs/architecture.md``).  ``vtsearch.state``
re-exports them under their original names so ``from vtsearch.state import
medias`` keeps working; the library tier ``vtscore.state`` never imports the
proxy classes.

This module is the canonical app-tier state API (the proxy layer), not a
thin adapter, which is why it lives at ``vtsearch/state_proxies.py`` rather
than under ``vtsearch/shim/`` (which holds the genuine Flask glue).

.. warning::

   **The proxies' own built-in storage is permanently empty**, so anything
   that reads a proxy through the C-level ``dict`` / ``list`` slots instead
   of through an overridden Python method sees *nothing* and returns a
   confidently wrong answer rather than raising.  Two distinct cases:

   * **Unforwarded methods.**  Fixed by forwarding, and kept fixed by
     ``tests/core/test_state_proxies.py``, which fails when any public
     ``dict`` / ``list`` method is neither forwarded nor listed in that
     test's explicit blacklist.  Add a forward (or a blacklist entry with a
     reason) when a new Python version grows a container method.

   * **C fast paths, which cannot be forwarded at all.**  ``json.dumps()``
     and ``copy.copy()`` read a ``dict`` subclass's internal table directly
     via ``PyDict_Next``; no Python-level override can intercept them, so
     ``json.dumps(medias)`` returns ``"{}"``.  Unbound calls
     (``dict.keys(medias)``) bypass the overrides the same way.  **Never
     hand a proxy straight to a serializer** - materialise it first with
     ``dict(medias)`` / ``sorted(good_votes)`` / a comprehension, which all
     route through ``__iter__`` and are correct.  Every production call site
     does this today; ``test_state_proxies.py`` pins the known-bypassing
     operations so the boundary stays documented rather than surprising.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vtscore.state.core import get_active_context, get_active_detector_context


class _ProxyDict(dict):
    """Dict-like proxy that forwards all operations to a target dict.

    The *target_attr* names the attribute on the active context that holds
    the real dict.  The *context_fn* is called to find that context.
    Every method fetches the target lazily so switching contexts is instant.

    Inherits from ``dict`` so that ``isinstance(medias, dict)`` is ``True``
    and existing code that type-checks works.
    """

    def __init__(self, target_attr: str, context_fn: Callable[[], Any] | None = None) -> None:
        # Do NOT call super().__init__() with data - we are a proxy, not a
        # real container.  The empty super().__init__() satisfies the dict
        # constructor without storing anything in our own hash table.
        super().__init__()
        object.__setattr__(self, "_target_attr", target_attr)
        object.__setattr__(self, "_context_fn", context_fn or get_active_context)

    def _target(self) -> dict:
        ctx_fn = object.__getattribute__(self, "_context_fn")
        return getattr(ctx_fn(), object.__getattribute__(self, "_target_attr"))

    # -- Core dict methods, all forwarded -----------------------------------

    def __getitem__(self, key):
        return self._target().__getitem__(key)

    def __setitem__(self, key, value):
        self._target().__setitem__(key, value)

    def __delitem__(self, key):
        self._target().__delitem__(key)

    def __contains__(self, key):
        return self._target().__contains__(key)

    def __iter__(self):
        # Snapshot keys so `for k in proxy` is safe against concurrent
        # mutation of the underlying dict by another thread.
        return iter(list(self._target()))

    def __len__(self):
        return self._target().__len__()

    def __repr__(self):
        return f"_ProxyDict({object.__getattribute__(self, '_target_attr')!r}, {self._target()!r})"

    def __eq__(self, other):
        return self._target().__eq__(other)

    def __ne__(self, other):
        return self._target().__ne__(other)

    def __bool__(self):
        return bool(self._target())

    def get(self, key, default=None):
        return self._target().get(key, default)

    def keys(self):
        return self._target().keys()

    def values(self):
        return self._target().values()

    def items(self):
        return self._target().items()

    def pop(self, *args):
        return self._target().pop(*args)

    def setdefault(self, key, default=None):
        return self._target().setdefault(key, default)

    def update(self, *args, **kwargs):
        return self._target().update(*args, **kwargs)

    def clear(self):
        return self._target().clear()

    def copy(self):
        return self._target().copy()

    def __or__(self, other):
        return self._target().__or__(other)

    def __ior__(self, other):
        target = self._target()
        target.__ior__(other)
        return self

    def __ror__(self, other):
        return self._target().__ror__(other)

    def popitem(self):
        return self._target().popitem()

    def __reversed__(self):
        # Snapshot to avoid RuntimeError under concurrent mutation.
        return reversed(list(self._target()))


class _ProxyList(list):
    """List-like proxy forwarding to the active context's list attribute."""

    def __init__(self, target_attr: str, context_fn: Callable[[], Any] | None = None) -> None:
        super().__init__()
        object.__setattr__(self, "_target_attr", target_attr)
        object.__setattr__(self, "_context_fn", context_fn or get_active_context)

    def _target(self) -> list:
        ctx_fn = object.__getattribute__(self, "_context_fn")
        return getattr(ctx_fn(), object.__getattribute__(self, "_target_attr"))

    def __getitem__(self, index):
        return self._target().__getitem__(index)

    def __setitem__(self, index, value):
        self._target().__setitem__(index, value)

    def __delitem__(self, index):
        self._target().__delitem__(index)

    def __contains__(self, item):
        return self._target().__contains__(item)

    def __iter__(self):
        # Snapshot elements so iteration is safe against concurrent mutation.
        return iter(list(self._target()))

    def __len__(self):
        return self._target().__len__()

    def __repr__(self):
        return f"_ProxyList({object.__getattribute__(self, '_target_attr')!r}, {self._target()!r})"

    def __eq__(self, other):
        return self._target().__eq__(other)

    def __ne__(self, other):
        return self._target().__ne__(other)

    # Ordering comparisons.  ``list`` defines all four, so an unforwarded one
    # would compare the proxy's permanently-empty own storage and quietly
    # answer as if the context held nothing.
    def __lt__(self, other):
        return self._target().__lt__(other)

    def __le__(self, other):
        return self._target().__le__(other)

    def __gt__(self, other):
        return self._target().__gt__(other)

    def __ge__(self, other):
        return self._target().__ge__(other)

    def __bool__(self):
        return bool(self._target())

    def __reversed__(self):
        # Snapshot to avoid RuntimeError under concurrent mutation.
        return reversed(list(self._target()))

    def __add__(self, other):
        return self._target().__add__(other)

    def __radd__(self, other):
        return other.__add__(self._target())

    def __mul__(self, count):
        return self._target().__mul__(count)

    def __rmul__(self, count):
        return self._target().__rmul__(count)

    def __imul__(self, count):
        target = self._target()
        target.__imul__(count)
        return self

    def __iadd__(self, other):
        target = self._target()
        target.__iadd__(other)
        return self

    def append(self, item):
        return self._target().append(item)

    def extend(self, items):
        return self._target().extend(items)

    def insert(self, index, item):
        return self._target().insert(index, item)

    def remove(self, item):
        return self._target().remove(item)

    def pop(self, *args):
        return self._target().pop(*args)

    def clear(self):
        return self._target().clear()

    def copy(self):
        return self._target().copy()

    def index(self, value, *args):
        return self._target().index(value, *args)

    def count(self, value):
        return self._target().count(value)

    def sort(self, **kwargs):
        return self._target().sort(**kwargs)

    def reverse(self):
        return self._target().reverse()


# ---------------------------------------------------------------------------
# Module-level proxy instances - re-exported by ``vtsearch.state`` so app
# call sites can write ``from vtsearch.state import medias`` and treat
# ``medias`` like a normal dict that always points at the current request's
# active dataset.
#
# ``medias`` delegates to the active DatasetContext (dataset-intrinsic).
# All vote/label proxies delegate to the active DetectorContext.
# ---------------------------------------------------------------------------

# Per-name container types, declared (but not assigned) here so static checkers
# see the real dict/list shape of each proxy.  The instances themselves are
# generated from the ``_PROXY_SPECS`` table below, so the annotations here are
# the only place the shapes differ.
#
#   medias                Clips storage: id -> {id, type, duration, ...}
#   good_votes/bad_votes  Voting storage (insertion-ordered dict)
#   label_history         Combined label history: [(media_id, label, ts), ...]
#   vote_click_times      Click order (1-indexed) per media_id
#   vote_region_boxes     media_id -> (x0, y0, x1, y1) normalised box drawn on a
#                         yes-vote (patch-embedder v2); absent otherwise
#   last_learned_scores   media_id -> learned-sort score (float in [0, 1])
#   textsort_suggestions  Text queries that got a Good vote, most recent last
medias: dict[int, dict[str, Any]]
good_votes: dict[int, None]
bad_votes: dict[int, None]
label_history: list[tuple[int, str, float]]
vote_click_times: dict[int, int]
vote_region_boxes: dict[int, tuple[float, float, float, float]]
last_learned_scores: dict[int, float]
textsort_suggestions: list[str]

# (exported name, proxy class, context resolver).  Every proxy delegates to the
# active DetectorContext except ``medias``, which is dataset-intrinsic and
# delegates to the active DatasetContext.  Driving both the instances and
# ``__all__`` from this single table keeps the exported-name list from ever
# drifting out of sync with the instances.
_PROXY_SPECS: list[tuple[str, type, Callable[[], Any]]] = [
    ("medias", _ProxyDict, get_active_context),
    ("good_votes", _ProxyDict, get_active_detector_context),
    ("bad_votes", _ProxyDict, get_active_detector_context),
    ("label_history", _ProxyList, get_active_detector_context),
    ("vote_click_times", _ProxyDict, get_active_detector_context),
    ("vote_region_boxes", _ProxyDict, get_active_detector_context),
    ("last_learned_scores", _ProxyDict, get_active_detector_context),
    ("textsort_suggestions", _ProxyList, get_active_detector_context),
]

for _name, _proxy_cls, _context_fn in _PROXY_SPECS:
    globals()[_name] = _proxy_cls(_name, _context_fn)


# Generated from the table above (plus the two proxy classes) so the export
# list can't drift from the instances.  pyright can't evaluate this statically,
# hence the targeted ignore; the annotation-only declarations above already give
# it the per-name types.
__all__ = [  # pyright: ignore[reportUnsupportedDunderAll]
    "_ProxyDict",
    "_ProxyList",
    *sorted(name for name, _cls, _fn in _PROXY_SPECS),
]
