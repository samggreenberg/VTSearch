"""Deprecated alias for :mod:`vtscore.cli.resume`.

The CLI modules became the :mod:`vtscore.cli` package (#5397), and this one
moved to :mod:`vtscore.cli.resume`.  ``import vtscore.cli_resume`` still works
and hands back that very module, not a copy - but it warns.  Import from
:mod:`vtscore.cli.resume` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.cli import resume as _resume

if TYPE_CHECKING:
    from vtscore.cli.resume import *  # noqa: F403

warnings.warn(
    "vtscore.cli_resume has moved to vtscore.cli.resume; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _resume
