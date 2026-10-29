"""Deprecated alias for :mod:`vtscore.host.seams`.

The loose top-level modules of ``vtscore`` moved into packages (#5398), and
this one moved to :mod:`vtscore.host.seams`.  ``import vtscore.host_seams``
still works and hands back that very module, not a copy - so its state is
shared - but it warns.  Import from :mod:`vtscore.host.seams` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.host import seams as _seams

if TYPE_CHECKING:
    from vtscore.host.seams import *  # noqa: F403

warnings.warn(
    "vtscore.host_seams has moved to vtscore.host.seams; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _seams
