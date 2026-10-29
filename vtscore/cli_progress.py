"""Deprecated alias for :mod:`vtscore.cli.progress`.

The CLI modules became the :mod:`vtscore.cli` package (#5397), and this one
moved to :mod:`vtscore.cli.progress`.  ``import vtscore.cli_progress`` still
works and hands back that very module, not a copy, so the output format set
through either name is the one every stage reads - but it warns.  Import from
:mod:`vtscore.cli.progress` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.cli import progress as _progress

if TYPE_CHECKING:
    from vtscore.cli.progress import *  # noqa: F403

warnings.warn(
    "vtscore.cli_progress has moved to vtscore.cli.progress; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _progress
