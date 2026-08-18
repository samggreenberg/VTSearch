"""Validation of externally-supplied ("pre-computed") embedding vectors.

Most vectors in VTSearch are produced by an embedder we control, so their
width, dtype and finiteness are guaranteed by construction.  A second class of
vector arrives from *outside*: an ``.npz`` manifest of pre-computed embeddings,
an importer's ``content_vectors`` / ``custom_metadata_map`` entry, or a re-ingest
source that re-supplies a stored vector.  Nothing about those is guaranteed -
they can be the wrong width (produced by a different model), the wrong dtype
(``float64`` from a research script, ``float16`` from a half-precision embed),
non-finite (a failed forward pass serialized verbatim), or not even rectangular
(a ragged per-key archive that numpy loads as ``dtype=object``).

Adopted unchecked, none of those fail where they are introduced.  They fail much
later and much worse: the matrix builder allocates ``(N, D)`` from the *first*
media's width and then raises a bare

    ValueError: could not broadcast input array from shape (768,) into shape (1152,)

on some unrelated request, naming neither the media, nor the embedder, nor the
manifest that supplied it.  A non-finite vector is worse still - it never raises
at all, it silently poisons every score, threshold compare and sort that touches
it.

This module is the one place that turns such a vector into either a clean
``float32`` array or a :class:`MismatchedVectorError` that says exactly what is
wrong and what to do about it.  Two tiers, because they have very different cost
profiles:

* :func:`normalize_vector` / :func:`normalize_vector_block` - **full**
  validation (shape, dtype, finiteness, width).  Run once per vector at
  *ingestion*, where an O(D) scan is free next to the file I/O that produced it.
* :func:`require_dim` / :func:`stack_vectors` - **shape-only** checks for the
  hot scoring path, where a per-request finiteness scan over the whole dataset
  would be a real cost.  These exist to convert a numpy broadcast failure into a
  locatable message, not to re-derive what ingestion already guaranteed.

Imports only :mod:`numpy`, so it stays a leaf alongside
:mod:`vtscore.embedding.normalize` with no risk of an import cycle through the
embedding package façade.  In particular it does **not** reach into the embedder
registry: callers that know an embedder's declared width pass it in as
*expected_dim* (see
:func:`vtscore.datasets.importers._npz_vectors.expected_dim_for_embedder`).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


class MismatchedVectorError(ValueError):
    """An embedding vector cannot be used as-is.

    Subclasses :class:`ValueError` deliberately: every existing ``except
    ValueError`` around the import and scoring paths (including the route layer's
    400-mapping) keeps working unchanged, while code that wants to single out a
    vector-shape problem can catch this specific type.
    """


#: Appended to every dimension-mismatch message.  Split out so the wording of
#: the fix stays identical whether the mismatch is caught at import (where the
#: manifest is still nameable) or at score time (where it isn't).
_DIM_ADVICE = (
    "Vectors of different widths cannot be compared, so they cannot share a dataset. "
    "This normally means the pre-computed vectors were produced by a different embedder "
    "than the one this dataset uses: re-export them from the matching embedder, or build "
    "the dataset with the embedder that produced them."
)


def _dim_mismatch_message(label: str, actual: int, expected: int, expected_source: str = "") -> str:
    """Compose the shared "wrong width" message.

    *expected_source* names where *expected* came from (an embedder name, "the
    dataset", a sibling column) so the reader can tell which of the two numbers
    is the one they control.
    """
    source = f" ({expected_source})" if expected_source else ""
    return f"{label}: vector is {actual}-dimensional but {expected} was expected{source}. {_DIM_ADVICE}"


def _as_numeric_array(vec: Any, label: str) -> np.ndarray:
    """Return *vec* as a numeric ndarray, or raise :class:`MismatchedVectorError`.

    Catches the two ways a non-rectangular or non-numeric payload reaches us: a
    ragged nested sequence (numpy 2.x refuses it outright; older shapes land as
    ``dtype=object``) and a string / structured array from a mis-keyed archive.
    """
    if vec is None:
        raise MismatchedVectorError(f"{label}: no vector supplied (None).")
    try:
        arr = np.asarray(vec)
    except Exception as exc:  # ragged nested sequences raise on numpy >= 1.24
        raise MismatchedVectorError(
            f"{label}: not convertible to a numeric array ({exc}). "
            "Each vector must be a flat, equal-length sequence of numbers."
        ) from exc
    if arr.dtype == object:
        raise MismatchedVectorError(
            f"{label}: loaded as a ragged / non-numeric array (numpy dtype 'object'). "
            "Each vector must be a flat, equal-length sequence of numbers."
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise MismatchedVectorError(
            f"{label}: has non-numeric dtype {arr.dtype!r}; expected a float (or integer) array."
        )
    return arr


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Narrow *arr* to contiguous ``float32``, letting overflow become ``inf`` quietly.

    A ``float64`` value too large for single precision casts to ``inf``, which
    numpy reports as a ``RuntimeWarning``.  That warning is noise here: the very
    next step is :func:`_reject_non_finite`, which turns the ``inf`` into a
    message that actually explains what went wrong and what to do about it.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        return np.ascontiguousarray(arr, dtype=np.float32)


def _reject_non_finite(arr: np.ndarray, label: str, *, row_axis: bool = False) -> None:
    """Raise unless every entry of the already-``float32`` *arr* is finite.

    Checked *after* the cast to ``float32`` so it also catches a ``float64``
    value whose magnitude overflows to ``inf`` on narrowing - a real hazard for
    vectors written by a script that never intended them to be stored at single
    precision.  With *row_axis* set the message names the first offending row,
    which is what makes a 100k-row manifest debuggable.
    """
    finite = np.isfinite(arr)
    if bool(finite.all()):
        return
    detail = ""
    if row_axis and arr.ndim == 2:
        bad_rows = np.flatnonzero(~finite.all(axis=1))
        detail = f" First offending row: index {int(bad_rows[0])} (of {len(bad_rows)} affected)."
    raise MismatchedVectorError(
        f"{label}: holds non-finite values - NaN, infinity, or a magnitude that overflows "
        f"float32.{detail} Non-finite entries silently poison every score, threshold "
        "comparison and sort they reach, so they are rejected at the door rather than stored."
    )


def normalize_vector(
    vec: Any,
    *,
    label: str,
    expected_dim: int | None = None,
    expected_source: str = "",
) -> np.ndarray:
    """Return *vec* as a validated, contiguous 1-D ``float32`` array.

    Rejects (with :class:`MismatchedVectorError`) a vector that is ``None``,
    ragged, non-numeric, not 1-D, empty, non-finite, or - when *expected_dim* is
    given - the wrong width.  A ``(1, D)`` or ``(D, 1)`` array is accepted and
    flattened, since a research script that saved one row per file very commonly
    keeps the leading axis.

    *label* identifies the vector in the error message; make it something the
    user can act on ("manifest row 41 ('cat.jpg')"), not an internal id.
    *expected_source* names the origin of *expected_dim* for the same reason.
    """
    arr = _as_numeric_array(vec, label)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise MismatchedVectorError(
            f"{label}: expected a 1-D vector, got shape {arr.shape}. "
            "A pre-computed embedding is a single flat row per media."
        )
    if arr.size == 0:
        raise MismatchedVectorError(f"{label}: vector is empty (zero-length), so it carries no embedding.")
    out = _to_float32(arr)
    _reject_non_finite(out, label)
    if expected_dim is not None and int(out.shape[0]) != expected_dim:
        raise MismatchedVectorError(_dim_mismatch_message(label, int(out.shape[0]), expected_dim, expected_source))
    return out


def normalize_vector_block(
    block: Any,
    *,
    label: str,
    expected_dim: int | None = None,
    expected_source: str = "",
) -> np.ndarray:
    """Return an ``(N, D)`` block of vectors as a validated ``float32`` array.

    The bulk counterpart of :func:`normalize_vector`, for the ``vectors`` array
    of an ``.npz`` manifest: one vectorised pass validates the whole archive, so
    a 100k-row manifest costs a single numpy scan rather than 100k Python calls.
    Rejects a block that is ragged, non-numeric, not 2-D, zero-width, non-finite,
    or the wrong width; a ``(D,)`` array is **not** silently promoted to one row,
    because a manifest that lost its row axis is far more likely to be a bug than
    a deliberate one-row archive.
    """
    arr = _as_numeric_array(block, label)
    if arr.ndim != 2:
        raise MismatchedVectorError(
            f"{label}: expected a 2-D (N, D) array of vectors, got shape {arr.shape}. "
            "Each row is one media's embedding."
        )
    if arr.shape[1] == 0:
        raise MismatchedVectorError(f"{label}: vectors are zero-width (shape {arr.shape}), so they carry no embedding.")
    out = _to_float32(arr)
    _reject_non_finite(out, label, row_axis=True)
    if expected_dim is not None and int(out.shape[1]) != expected_dim:
        raise MismatchedVectorError(_dim_mismatch_message(label, int(out.shape[1]), expected_dim, expected_source))
    return out


def vector_dim(vec: Any) -> int | None:
    """Return the trailing dimension of *vec*, or ``None`` if it has no shape.

    Deliberately cheap: reads ``.shape`` off an ndarray without copying or
    re-validating, so the hot scoring path can compare widths per media without
    paying for :func:`normalize_vector`.
    """
    shape = getattr(vec, "shape", None)
    if shape is None:
        try:
            shape = np.shape(vec)
        except Exception:
            return None
    return int(shape[-1]) if shape else None


def require_dim(vec: Any, expected_dim: int, *, label: str, expected_source: str = "") -> None:
    """Raise :class:`MismatchedVectorError` unless *vec* is a 1-D row of *expected_dim*.

    The scoring-path backstop.  Ingestion is supposed to have made this
    impossible, but a dataset can also acquire mixed widths by other routes (a
    pickle written before a validation rule existed, a plugin importer that
    writes ``media["embeddings"]`` directly), and when it does, the alternative
    is numpy's shape-only broadcast error from inside the matrix builder.  This
    costs one attribute read per media and yields a message that names the media
    and both widths.
    """
    shape = getattr(vec, "shape", None)
    if shape is None:
        shape = np.shape(vec)
    if len(shape) == 1 and int(shape[0]) == expected_dim:
        return
    if len(shape) != 1:
        raise MismatchedVectorError(
            f"{label}: expected a 1-D {expected_dim}-dimensional vector, got shape {tuple(shape)}."
        )
    raise MismatchedVectorError(_dim_mismatch_message(label, int(shape[0]), expected_dim, expected_source))


def stack_vectors(
    vecs: Iterable[Any],
    *,
    label: str,
    row_labels: Sequence[str] | None = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    """``np.stack`` over per-item vectors, naming the first row whose width differs.

    ``np.stack`` reports only "all input arrays must have the same shape", which
    on a training set of hundreds of votes says nothing about *which* vote is the
    odd one out.  This walks the rows once, compares each width against the
    first, and raises a message naming the offending row - using *row_labels*
    (e.g. a filename or content id per row) when the caller can supply them.

    Raises :class:`MismatchedVectorError` for an empty *vecs*: every caller here
    is building a training or scoring matrix that is meaningless with no rows.
    """
    rows = list(vecs)
    if not rows:
        raise MismatchedVectorError(f"{label}: no vectors to stack.")

    def _row_label(i: int) -> str:
        if row_labels is not None and i < len(row_labels):
            return f"{label} row {i} ({row_labels[i]})"
        return f"{label} row {i}"

    dim = vector_dim(rows[0])
    if dim is None:
        raise MismatchedVectorError(f"{_row_label(0)}: not an array-like vector.")
    for i, vec in enumerate(rows):
        require_dim(vec, dim, label=_row_label(i), expected_source=f"matching {_row_label(0)}")
    return np.stack([np.asarray(v, dtype=dtype) for v in rows])
