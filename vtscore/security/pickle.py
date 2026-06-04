"""Restricted pickle unpickler for safe dataset deserialization.

Provides :class:`RestrictedUnpickler` and :func:`safe_pickle_load` to prevent
arbitrary code execution when loading ``.pkl`` dataset files.
"""

from __future__ import annotations

import io
import pickle
import struct
from typing import TYPE_CHECKING, Any


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


# numpy reconstruction callables. A real array reduces either via
# ``_frombuffer(buffer, dtype, shape, order)`` (protocol >=5) or via
# ``_reconstruct(...)`` + a BUILD/``__setstate__`` that carries the raw bytes
# (older protocols). Both consume the array's data buffer — which the peek
# unpickler stubs out to empty — so calling the real callables raises (e.g.
# "cannot reshape array of size 0"). The peek path swaps them for a stub.
_PICKLE_NUMPY_RECONSTRUCT: set[tuple[str, str]] = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "_core.multiarray._reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
}


class _PeekStubArray(list):
    """Placeholder a peek substitutes for a numpy array.

    Subclasses ``list`` so it reads as an empty sequence (``len() == 0``), and
    swallows ``__setstate__`` so the BUILD opcode that would feed an array its
    (stubbed-empty) raw bytes is a no-op instead of a buffer-size error.
    """

    def __setstate__(self, state: object) -> None:
        pass


def _peek_stub_numpy(*args: Any, **kwargs: Any) -> _PeekStubArray:
    """Stand-in for numpy's array/scalar reconstruction during a peek."""
    return _PeekStubArray()


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


class _PeekUnpickler(pickle._Unpickler):
    """Pure-Python unpickler that preserves dict/key structure but stubs out
    heavy leaf values.

    Used by :func:`peek_pickle_dataset_summary` to extract a dataset's media
    count and first-entry media type from a ``.pkl`` upload without paying
    the cost of materialising per-media embedding lists (millions of Python
    floats) or inline media bytes (audio/image/video blobs).

    Safety: ``find_class`` enforces the same allowlist as
    :class:`RestrictedUnpickler`, so an RCE payload is still rejected.
    """

    if TYPE_CHECKING:
        # pickle._Unpickler is the private pure-Python implementation; its
        # internal attributes are not exposed in typeshed. Declare the ones
        # this subclass touches so type-checkers can resolve them.
        stack: list[Any]

        def read(self, n: int) -> bytes: ...
        def readline(self) -> bytes: ...
        def append(self, value: Any) -> None: ...
        def pop_mark(self) -> list[Any]: ...

    # Typed as dict[int, Any] (not the typeshed-inferred Callable[[Unpickler], None])
    # because the dispatched methods take Self, which is contravariantly
    # incompatible with the public Unpickler signature.
    dispatch: dict[int, Any] = pickle._Unpickler.dispatch.copy()

    def find_class(self, module: str, name: str) -> Any:
        # Swap numpy's array/scalar reconstruction for a stub: the real
        # callables validate their data buffer, which this peek has emptied.
        if (module, name) in _PICKLE_NUMPY_RECONSTRUCT:
            return _peek_stub_numpy
        if (module, name) in _PICKLE_SAFE_CLASSES:
            return pickle._Unpickler.find_class(self, module, name)
        raise pickle.UnpicklingError(
            f"Forbidden pickle class: {module}.{name}. Only plain Python types and numpy arrays are allowed."
        )

    def load_binfloat(self) -> None:
        self.read(8)
        self.append(None)

    dispatch[pickle.BINFLOAT[0]] = load_binfloat

    def load_float(self) -> None:
        # Protocol 0 ASCII float: decimal text terminated by newline.
        self.readline()
        self.append(None)

    dispatch[pickle.FLOAT[0]] = load_float

    def load_binbytes(self) -> None:
        (size,) = struct.unpack("<I", self.read(4))
        self.read(size)
        self.append(b"")

    dispatch[pickle.BINBYTES[0]] = load_binbytes

    def load_binbytes8(self) -> None:
        (size,) = struct.unpack("<Q", self.read(8))
        self.read(size)
        self.append(b"")

    dispatch[pickle.BINBYTES8[0]] = load_binbytes8

    def load_short_binbytes(self) -> None:
        size = self.read(1)[0]
        self.read(size)
        self.append(b"")

    dispatch[pickle.SHORT_BINBYTES[0]] = load_short_binbytes

    def load_bytearray8(self) -> None:
        # Protocol 5 has a dedicated bytearray opcode that bypasses BINBYTES;
        # without this override, inline bytearray media bytes would be fully
        # materialised even at the highest protocol.
        (size,) = struct.unpack("<Q", self.read(8))
        self.read(size)
        self.append(bytearray())

    dispatch[pickle.BYTEARRAY8[0]] = load_bytearray8

    def load_append(self) -> None:
        # Discard the value instead of appending to the list below it.
        self.stack.pop()

    dispatch[pickle.APPEND[0]] = load_append

    def load_appends(self) -> None:
        # Drop everything between the mark and the top; the list below the
        # mark stays empty.
        self.pop_mark()

    dispatch[pickle.APPENDS[0]] = load_appends


def peek_pickle_dataset_summary(f: io.BufferedIOBase) -> Any:
    """Cheaply summarise a dataset pickle without instantiating embeddings.

    Returns the same top-level structure ``pickle.load`` would, but with
    embedding lists left empty and inline media-byte blobs replaced by
    ``b""``. Sufficient for reading the media count and the first entry's
    ``"media_type"`` field; not suitable for any operation that needs the
    underlying vectors or bytes.

    Rejects non-allowlisted classes the same way :func:`safe_pickle_load`
    does, so an RCE payload still raises ``UnpicklingError``.
    """
    return _PeekUnpickler(f).load()
