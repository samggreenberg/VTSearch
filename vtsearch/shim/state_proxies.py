"""App-side proxy view over the library's ``DatasetContext`` / ``DetectorContext``.

Many app-layer call sites (Flask routes, app-tier tests) like to import
``medias``, ``good_votes``, etc. as module-level dict/list names and treat
them as the "current" container.  The library does not depend on those
names — its functions resolve the active context explicitly — but the app
keeps them as a convenience facade so existing code reads naturally.

These proxies live on the app side of the future ``vtscore`` / ``vtsearch``
split (see Phase 3 of ``docs/plans/extract-library.md``).  ``vtsearch.state``
re-exports them under their original names so ``from vtsearch.state import
medias`` keeps working; the library subpackage ``vtsearch.state`` itself
never imports the proxy classes.
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
        # Do NOT call super().__init__() with data — we are a proxy, not a
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

    def __bool__(self):
        return bool(self._target())

    def __add__(self, other):
        return self._target().__add__(other)

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
# Module-level proxy instances — re-exported by ``vtsearch.state`` so app
# call sites can write ``from vtsearch.state import medias`` and treat
# ``medias`` like a normal dict that always points at the current request's
# active dataset.
#
# ``medias`` delegates to the active DatasetContext (dataset-intrinsic).
# All vote/label proxies delegate to the active DetectorContext.
# ---------------------------------------------------------------------------

# Clips storage: id -> {id, type, duration, file_size, embedding, media_bytes, ...}
medias: dict[int, dict[str, Any]] = _ProxyDict("medias")  # type: ignore[assignment]

# Voting storage (OrderedDict behavior via dict in Python 3.7+)
good_votes: dict[int, None] = _ProxyDict("good_votes", get_active_detector_context)  # type: ignore[assignment]
bad_votes: dict[int, None] = _ProxyDict("bad_votes", get_active_detector_context)  # type: ignore[assignment]

# Combined label history: [(media_id, label, timestamp), ...]
label_history: list[tuple[int, str, float]] = _ProxyList(  # type: ignore[assignment]
    "label_history",
    get_active_detector_context,
)

# Click-time tracking: media_id -> click order (1-indexed).
vote_click_times: dict[int, int] = _ProxyDict(  # type: ignore[assignment]
    "vote_click_times",
    get_active_detector_context,
)

# Per-good-vote region boxes: media_id -> (x0, y0, x1, y1) in normalised image
# coords.  Only populated when the user drew a region as part of a yes-vote;
# absent for image-level yes-votes and for every no-vote.  Patch-embedder v2.
vote_region_boxes: dict[int, tuple[float, float, float, float]] = _ProxyDict(  # type: ignore[assignment]
    "vote_region_boxes",
    get_active_detector_context,
)

# Last learned-sort scores: media_id -> score (float in [0, 1]).
last_learned_scores: dict[int, float] = _ProxyDict(  # type: ignore[assignment]
    "last_learned_scores",
    get_active_detector_context,
)

# Text-sort suggestions: text queries that received a Good vote, most recent last.
textsort_suggestions: list[str] = _ProxyList(  # type: ignore[assignment]
    "textsort_suggestions",
    get_active_detector_context,
)


__all__ = [
    "_ProxyDict",
    "_ProxyList",
    "bad_votes",
    "good_votes",
    "label_history",
    "last_learned_scores",
    "medias",
    "textsort_suggestions",
    "vote_click_times",
    "vote_region_boxes",
]
