"""Runtime torch configuration for embedder code.

Lives in ``vtsearch/media/`` because every embedder (the bottom layer of the
sort/training stack) must run :func:`ensure_torch_configured` before invoking
torch.  Higher layers (``vtsearch.models``) call it via the same hook.
"""

import sys

from vtsearch.config import TORCH_THREADS


_torch_configured = False


def ensure_torch_configured() -> None:
    """Set ``torch.set_num_threads(TORCH_THREADS)`` the first time torch is used.

    Thread count comes from :data:`vtsearch.config.TORCH_THREADS`, which in
    turn reads ``VTSEARCH_TORCH_THREADS`` (default ``1``).  Safe to call
    multiple times — the configuration is applied only once.  Call this from
    any code path that imports torch before doing work (e.g. ``load_models``,
    ``train_model``).
    """
    global _torch_configured
    if _torch_configured:
        return
    if "torch" not in sys.modules:
        return
    import torch  # noqa: PLC0415

    torch.set_num_threads(TORCH_THREADS)
    _torch_configured = True
