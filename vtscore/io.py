"""Deprecated alias for :mod:`vtscore.utils.io`.

The loose top-level modules of ``vtscore`` moved into packages (#5398), and
this one moved to :mod:`vtscore.utils.io`.  ``import vtscore.io`` still works
and hands back that very module, not a copy - so its state is shared - but it
warns.  Import from :mod:`vtscore.utils.io` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.utils import io as _io

if TYPE_CHECKING:
    from vtscore.utils.io import *  # noqa: F403

warnings.warn(
    "vtscore.io has moved to vtscore.utils.io; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _io
