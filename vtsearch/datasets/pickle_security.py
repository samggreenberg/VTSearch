"""Restricted pickle unpickler for safe dataset deserialization.

Provides :class:`RestrictedUnpickler` and :func:`safe_pickle_load` to prevent
arbitrary code execution when loading ``.pkl`` dataset files.
"""

from __future__ import annotations

import io
import pickle
from typing import Any


# Allowlist of (module, name) pairs that may be instantiated during unpickling.
# VTSearch pickles only contain plain Python containers and numpy arrays.
_PICKLE_SAFE_CLASSES: set[tuple[str, str]] = {
    # builtins (needed for dict/list/set/bytes subclasses & booleans)
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytes"),
    ("builtins", "bytearray"),
    ("builtins", "complex"),
    # collections
    ("collections", "OrderedDict"),
    # numpy – reconstruction helpers used by numpy's __reduce__
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "_core.multiarray._reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """An ``Unpickler`` that refuses to instantiate classes outside an allowlist.

    Plain Python primitives (int, float, str, None, bool, dict, list, tuple)
    are handled by the pickle protocol directly and never trigger
    ``find_class``.  This restricts only explicit class/callable references
    in the pickle stream.
    """

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _PICKLE_SAFE_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Forbidden pickle class: {module}.{name}. Only plain Python types and numpy arrays are allowed."
        )


def safe_pickle_load(f: io.BufferedIOBase, **kwargs: Any) -> Any:
    """Deserialise a pickle stream using the restricted unpickler.

    Drop-in replacement for ``pickle.load(f)`` that blocks arbitrary code
    execution.  Any extra keyword arguments (e.g. ``encoding``) are
    forwarded to the underlying ``Unpickler``.
    """
    return RestrictedUnpickler(f, **kwargs).load()
