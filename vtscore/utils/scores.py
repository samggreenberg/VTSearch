"""Score-sanitization helpers shared by every score-emitting code path.

A trained MLP can produce ``NaN`` logits when training destabilises (bad
optimisation, corrupted input embeddings, AMP overflow on CUDA, extreme
class-weight bias from a large ``inclusion_value`` shift, etc.).
``torch.sigmoid(NaN)`` is ``NaN``, and ``json.dumps`` happily emits the
literal token ``NaN`` (and ``Infinity``/``-Infinity``) by default — invalid
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
  sanitised score renders identically to "no score yet" — no UI change.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def finite_or(value: float, default: float = NON_FINITE_SCORE_SENTINEL) -> float:
    """Return *value* if finite, otherwise *default*.

    Defensive guard for code that consumes already-stored float scores
    (e.g. ``DetectorContext.last_learned_scores``) so a regression at any
    write site cannot leak ``NaN``/``Infinity`` into a JSON response.
    """
    return value if math.isfinite(value) else default
