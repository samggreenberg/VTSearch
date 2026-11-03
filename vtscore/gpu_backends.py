"""Deprecated alias for :mod:`vtscore.config.gpu_backends`.

The loose top-level modules of ``vtscore`` moved into packages (#5398), and
this one moved to :mod:`vtscore.config.gpu_backends`.
``import vtscore.gpu_backends`` still works and hands back that very module,
not a copy - so its state is shared - but it warns.  Import from
:mod:`vtscore.config.gpu_backends` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.config import gpu_backends as _gpu_backends

if TYPE_CHECKING:
    from vtscore.config.gpu_backends import *  # noqa: F403

warnings.warn(
    "vtscore.gpu_backends has moved to vtscore.config.gpu_backends; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _gpu_backends
