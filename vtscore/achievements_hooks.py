"""Deprecated alias for :mod:`vtscore.host.achievements`.

The loose top-level modules of ``vtscore`` moved into packages (#5398), and
this one moved to :mod:`vtscore.host.achievements`.
``import vtscore.achievements_hooks`` still works and hands back that very
module, not a copy - so its state is shared - but it warns.  Import from
:mod:`vtscore.host.achievements` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.host import achievements as _achievements

if TYPE_CHECKING:
    from vtscore.host.achievements import *  # noqa: F403

warnings.warn(
    "vtscore.achievements_hooks has moved to vtscore.host.achievements; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _achievements
