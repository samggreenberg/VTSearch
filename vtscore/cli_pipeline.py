"""Deprecated alias for :mod:`vtscore.cli.pipeline`.

The CLI modules became the :mod:`vtscore.cli` package (#5397), and this one
moved to :mod:`vtscore.cli.pipeline`.  ``import vtscore.cli_pipeline`` still
works and hands back that very module, not a copy - but it warns.  Import from
:mod:`vtscore.cli.pipeline` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.cli import pipeline as _pipeline

if TYPE_CHECKING:
    from vtscore.cli.pipeline import *  # noqa: F403

warnings.warn(
    "vtscore.cli_pipeline has moved to vtscore.cli.pipeline; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _pipeline
