"""Deprecated alias for :mod:`vtscore.state.votes`.

Click-time tracking was three functions on the active detector's
``click_counter`` / ``vote_click_times``, and the vote operations were their
only caller, so they were folded into :mod:`vtscore.state.votes` (#5399).
``import vtscore.state.clicks`` still works and hands back that very module,
not a copy - so a stub patched through either name is the one the votes read -
but it warns.  Import ``assign_click_time``, ``remove_click_time`` and
``get_vote_click_times`` from :mod:`vtscore.state.votes` instead (or keep using
the unchanged ``vtscore.state`` / ``vtsearch.state`` re-exports).
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.state import votes as _votes

if TYPE_CHECKING:
    from vtscore.state.votes import (  # noqa: F401
        assign_click_time,
        get_vote_click_times,
        remove_click_time,
    )

warnings.warn(
    "vtscore.state.clicks has moved to vtscore.state.votes; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _votes
