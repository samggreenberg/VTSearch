"""Deprecated alias for :mod:`vtscore.training.thresholds.knobs`.

The non-finite score sentinel and the helpers that sanitise to it and drop it
were folded into :mod:`vtscore.training.thresholds.knobs`, beside the other
threshold sentinels, whose estimators are most of their callers (#5399).
``import vtscore.utils.scores`` still works and hands back that very module,
not a copy, but it warns.  Import from :mod:`vtscore.training.thresholds.knobs`
instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.training.thresholds import knobs as _knobs

if TYPE_CHECKING:
    from vtscore.training.thresholds.knobs import (  # noqa: F401
        NON_FINITE_SCORE_SENTINEL,
        finite_or,
        scored_mask,
        scored_only,
        sigmoid_to_finite_array,
        sigmoid_to_finite_scores,
    )

warnings.warn(
    "vtscore.utils.scores has moved to vtscore.training.thresholds.knobs; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _knobs
