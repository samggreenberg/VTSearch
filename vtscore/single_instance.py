"""Deprecated alias for :mod:`vtscore.concurrency.single_instance`.

The loose top-level modules of ``vtscore`` moved into packages (#5398), and
this one moved to :mod:`vtscore.concurrency.single_instance`.
``import vtscore.single_instance`` still works and hands back that very module,
not a copy - so its state is shared - but it warns.  Import from
:mod:`vtscore.concurrency.single_instance` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.concurrency import single_instance as _single_instance

if TYPE_CHECKING:
    from vtscore.concurrency.single_instance import *  # noqa: F403

warnings.warn(
    "vtscore.single_instance has moved to vtscore.concurrency.single_instance; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _single_instance
