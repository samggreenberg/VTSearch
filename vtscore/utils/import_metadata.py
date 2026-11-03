"""Deprecated alias for :mod:`vtscore.media.torch_utils.torch_setup`.

The stat-free ``packages_distributions`` seed was folded into
:mod:`vtscore.media.torch_utils.torch_setup`, beside the embedder stack's other runtime
setup; the embedding loader calls both (#5399).  ``import
vtscore.utils.import_metadata`` still works and hands back that very module,
not a copy - so ``original_packages_distributions`` is the one captured
there - but it warns.  Import from :mod:`vtscore.media.torch_utils.torch_setup` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from vtscore.media.torch_utils import torch_setup as _torch_setup

if TYPE_CHECKING:
    from vtscore.media.torch_utils.torch_setup import (  # noqa: F401
        fast_packages_distributions,
        original_packages_distributions,
        seed_packages_distributions,
    )

warnings.warn(
    "vtscore.utils.import_metadata has moved to vtscore.media.torch_utils.torch_setup; import from there instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _torch_setup
