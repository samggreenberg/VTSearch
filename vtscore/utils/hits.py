"""Deprecated alias for :mod:`vtscore.datasets.labelset`.

``build_media_hit`` and ``hit_custom_metadata`` were folded into
:mod:`vtscore.datasets.labelset`, beside the labelset elements that are the
other shape a scored media is exported as (#5399).  ``import
vtscore.utils.hits`` still works and hands back that very module, not a copy,
but it warns.  Import from :mod:`vtscore.datasets.labelset` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.datasets import labelset as _labelset

if TYPE_CHECKING:
    from vtscore.datasets.labelset import build_media_hit, hit_custom_metadata  # noqa: F401

warnings.warn(
    "vtscore.utils.hits has moved to vtscore.datasets.labelset; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _labelset
