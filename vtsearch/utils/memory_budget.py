"""Helpers for capping concurrency by a memory budget.

The auto-detect and auto-process endpoints fan out across all medias inside a
``ThreadPoolExecutor``.  Each worker materialises an ``N x D`` fp32 tensor of
embeddings (plus activations of the same order), which at ``N=100k, D=1152``
is roughly 450 MB per worker.  Eight unconstrained workers can therefore push
multi-GB transient allocations through the process; this module lets the
callers cap the worker count so peak memory stays inside a configured budget.
"""

from __future__ import annotations

import os

# Default share of available system memory we are willing to hand to the
# scoring fan-out.  The rest must stay free for embeddings, the loaded
# datasets, the model weights, and OS / framework overhead.
_DEFAULT_BUDGET_FRACTION = 0.25

# Floor for the per-process budget when we cannot read system memory: 1 GiB.
_FALLBACK_BUDGET_BYTES = 1 * 1024 * 1024 * 1024


def _available_memory_bytes() -> int:
    """Best-effort estimate of currently-available physical memory in bytes."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (OSError, ValueError):
        pass
    return _FALLBACK_BUDGET_BYTES


def cap_workers_by_memory(
    n_items: int,
    embed_dim: int,
    *,
    max_workers: int,
    bytes_per_element: int = 4,
    budget_fraction: float = _DEFAULT_BUDGET_FRACTION,
) -> int:
    """Cap *max_workers* so peak per-worker memory fits in the budget.

    Each worker is assumed to allocate ``n_items * embed_dim *
    bytes_per_element`` bytes for its score-all-medias pass.  We allow it to
    consume up to ``budget_fraction`` of currently-available memory, divided
    across the concurrent workers.  Always returns at least 1.
    """
    if max_workers <= 1 or n_items <= 0 or embed_dim <= 0:
        return max(1, max_workers)
    per_worker = n_items * embed_dim * bytes_per_element
    if per_worker <= 0:
        return max(1, max_workers)
    budget = int(_available_memory_bytes() * budget_fraction)
    return max(1, min(max_workers, budget // per_worker))
