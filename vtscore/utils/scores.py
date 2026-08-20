"""Score-sanitization helpers shared by every score-emitting code path.

A trained MLP can produce ``NaN`` logits when training destabilises (bad
optimisation, corrupted input embeddings, AMP overflow on CUDA, extreme
class-weight bias from a large ``inclusion_value`` shift, etc.).
``torch.sigmoid(NaN)`` is ``NaN``, and ``json.dumps`` happily emits the
literal token ``NaN`` (and ``Infinity``/``-Infinity``) by default - invalid
JSON per RFC 7159, rejected by every browser ``JSON.parse``. A single
poisoned response breaks the Angular client until the user clears votes.

These helpers exist so every score-emitting path goes through one
sentinel:

* :data:`NON_FINITE_SCORE_SENTINEL` (``-1.0``) sits *outside* the ``[0, 1]``
  sigmoid range, so ``score >= threshold`` is always ``False`` for a
  sanitised score (broken items deterministically fall to the bottom of
  any sort).
* The frontend already treats missing scores as ``-1`` (see
  ``label-list.component.ts``'s ``learnedScores[id] ?? -1``), so a
  sanitised score renders identically to "no score yet" - no UI change.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

NON_FINITE_SCORE_SENTINEL: float = -1.0


def sigmoid_to_finite_scores(
    logits: torch.Tensor,
    *,
    default: float = NON_FINITE_SCORE_SENTINEL,
) -> list[float]:
    """Apply sigmoid to *logits* and return a list of JSON-safe floats.

    Squeezes the trailing dim, applies sigmoid, replaces every non-finite
    value with *default*, and copies the result to a Python list. Use this
    in place of ``torch.sigmoid(model(X)).squeeze(1).cpu().tolist()`` at
    every site whose output reaches a JSON response.

    Args:
        logits: Raw MLP output, shape ``(N, 1)`` or ``(N,)``.
        default: Sentinel substituted for ``NaN`` / ``+Inf`` / ``-Inf``.
            Defaults to :data:`NON_FINITE_SCORE_SENTINEL`.

    Returns:
        Python list of floats, all guaranteed ``math.isfinite``.
    """
    import torch  # noqa: PLC0415

    scores = torch.sigmoid(logits).squeeze(-1)
    scores = torch.nan_to_num(scores, nan=default, posinf=default, neginf=default)
    return scores.cpu().tolist()


def sigmoid_to_finite_array(
    logits: torch.Tensor,
    *,
    default: float = NON_FINITE_SCORE_SENTINEL,
) -> np.ndarray:
    """Apply sigmoid to *logits* and return a JSON-safe ``np.ndarray``.

    Array-native twin of :func:`sigmoid_to_finite_scores`: same squeeze /
    sigmoid / ``nan_to_num`` sanitisation, but ends in ``.cpu().numpy()``
    instead of ``.tolist()``. Use this at sites whose output feeds straight
    into numpy math (e.g. segmented max-pool) so the scores never make a
    round-trip through a Python list, which is a pure-Python, GIL-holding
    ``O(N)`` pass that matters in the background training thread.

    Args:
        logits: Raw MLP output, shape ``(N, 1)`` or ``(N,)``.
        default: Sentinel substituted for ``NaN`` / ``+Inf`` / ``-Inf``.
            Defaults to :data:`NON_FINITE_SCORE_SENTINEL`.

    Returns:
        A freshly-owned ``np.ndarray`` (the returned array does not alias any
        live tensor), every element guaranteed ``math.isfinite``. Dtype
        follows the input tensor (typically ``float32``); upcast at the call
        site if the downstream math wants ``float64``.
    """
    import torch  # noqa: PLC0415

    scores = torch.sigmoid(logits).squeeze(-1)
    scores = torch.nan_to_num(scores, nan=default, posinf=default, neginf=default)
    return scores.cpu().numpy()


def finite_or(value: float, default: float = NON_FINITE_SCORE_SENTINEL) -> float:
    """Return *value* if finite, otherwise *default*.

    Defensive guard for code that consumes already-stored float scores
    (e.g. ``DetectorContext.last_learned_scores``) so a regression at any
    write site cannot leak ``NaN``/``Infinity`` into a JSON response.
    """
    return value if math.isfinite(value) else default


def scored_mask(scores: "np.ndarray | list[float]") -> np.ndarray:
    """Boolean mask of the entries of *scores* that are real sigmoid outputs.

    ``True`` where a score lies in the ``[0, 1]`` sigmoid range and is finite;
    ``False`` for :data:`NON_FINITE_SCORE_SENTINEL` and for anything else out
    of range.  The sentinel means "this item could not be scored", which is a
    *different* statement from "this item scored low" - so any estimator
    fitted on a score distribution has to drop it rather than treat ``-1.0``
    as an observation.  Returned as a mask (rather than a filtered array) so
    callers holding parallel arrays - a score list and its labels - can drop
    the same positions from both.
    """
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(scores, dtype=np.float64)
    return np.isfinite(arr) & (arr >= 0.0) & (arr <= 1.0)


def scored_only(scores: "np.ndarray | list[float]") -> np.ndarray:
    """*scores* with every unscorable entry dropped; see :func:`scored_mask`.

    The population a threshold estimator may be fitted on.  Feeding the
    sentinel into a fit is not a rounding error but a sign flip: ``-1.0`` sits
    a full unit below the sigmoid range, so a handful of unscorable media pull
    a fitted cut *below zero*, at which point every real score clears it and
    the detector reports the whole dataset as a hit (issue #3180).
    """
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(scores, dtype=np.float64)
    return arr[scored_mask(arr)]
