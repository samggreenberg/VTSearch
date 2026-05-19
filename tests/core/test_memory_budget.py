"""Tests for ``vtsearch.concurrency.memory_budget.cap_workers_by_memory``.

The function caps a requested worker count so that ``n_items * embed_dim *
bytes_per_element`` per worker fits inside ``budget_fraction`` of currently-
available memory.  Always returns at least 1.
"""

from __future__ import annotations

import pytest

from vtsearch.concurrency import memory_budget
from vtsearch.concurrency.memory_budget import cap_workers_by_memory


@pytest.fixture
def fixed_memory(monkeypatch):
    """Pin ``_available_memory_bytes`` to a known value for deterministic math."""

    def _set(bytes_available: int) -> None:
        monkeypatch.setattr(memory_budget, "_available_memory_bytes", lambda: bytes_available)

    return _set


class TestEarlyReturns:
    def test_single_worker_short_circuits(self, fixed_memory):
        # Even with a tiny budget, max_workers=1 must return 1 without
        # consulting the budget.
        fixed_memory(0)
        assert cap_workers_by_memory(1_000_000, 1024, max_workers=1) == 1

    def test_zero_workers_returns_one(self, fixed_memory):
        # Floor of 1 — never return 0.
        fixed_memory(10 * 1024 * 1024 * 1024)
        assert cap_workers_by_memory(100, 128, max_workers=0) == 1

    def test_negative_workers_returns_one(self, fixed_memory):
        fixed_memory(10 * 1024 * 1024 * 1024)
        assert cap_workers_by_memory(100, 128, max_workers=-5) == 1

    def test_zero_items_uses_requested_workers(self, fixed_memory):
        # With no items, per-worker memory is 0 — return the request.
        fixed_memory(0)
        assert cap_workers_by_memory(0, 128, max_workers=8) == 8

    def test_zero_embed_dim_uses_requested_workers(self, fixed_memory):
        fixed_memory(0)
        assert cap_workers_by_memory(1000, 0, max_workers=8) == 8


class TestBudgetCapping:
    def test_budget_allows_all_workers(self, fixed_memory):
        # 8 GiB available, 25% budget = 2 GiB.  100 items * 128 dim * 4 B =
        # ~50 KB per worker — plenty of room for 8 workers.
        fixed_memory(8 * 1024 * 1024 * 1024)
        assert cap_workers_by_memory(100, 128, max_workers=8) == 8

    def test_budget_caps_below_requested(self, fixed_memory):
        # 1 GiB available, 25% budget = 256 MiB.
        # 100k items * 1152 dim * 4 B = ~439 MiB per worker → only 0 fit
        # mathematically (256/439 = 0).  Floor enforces 1.
        fixed_memory(1 * 1024 * 1024 * 1024)
        assert cap_workers_by_memory(100_000, 1152, max_workers=8) == 1

    def test_budget_allows_partial_workers(self, fixed_memory):
        # Construct a case that divides exactly.  Budget = 4 * per_worker.
        per_worker = 1000 * 100 * 4  # 400 KB
        # budget = 4 * 400 KB = 1.6 MB, so available = 1.6 MB / 0.25 = 6.4 MB
        fixed_memory(int(per_worker * 4 / 0.25))
        assert cap_workers_by_memory(1000, 100, max_workers=16) == 4

    def test_custom_budget_fraction(self, fixed_memory):
        # 1 GiB available, 50% budget = 512 MiB.  Per-worker = 256 MiB →
        # exactly 2 workers fit.
        fixed_memory(1024 * 1024 * 1024)
        per_worker = 256 * 1024 * 1024 // 4  # n*d*4 = 256 MiB
        cap = cap_workers_by_memory(
            per_worker,
            1,
            max_workers=8,
            budget_fraction=0.5,
        )
        assert cap == 2

    def test_custom_bytes_per_element(self, fixed_memory):
        # 16 MiB available, 25% = 4 MiB.  Per element = 8 bytes (fp64).
        # n=1024, d=128 → per worker = 1024*128*8 = 1 MiB → 4 fit.
        fixed_memory(16 * 1024 * 1024)
        cap = cap_workers_by_memory(1024, 128, max_workers=8, bytes_per_element=8)
        assert cap == 4

    def test_request_lower_than_budget_wins(self, fixed_memory):
        # Budget would allow 10 workers but request is only 3 — return 3.
        fixed_memory(10 * 1024 * 1024 * 1024)
        assert cap_workers_by_memory(100, 128, max_workers=3) == 3


class TestRealMemoryProbe:
    """Sanity-check the real (un-monkeypatched) probe returns a sane value."""

    def test_returns_positive_int(self):
        avail = memory_budget._available_memory_bytes()
        assert isinstance(avail, int)
        assert avail > 0

    def test_real_call_returns_floor_or_more(self):
        # No monkeypatching: with a heavy per-worker estimate the cap should
        # still be >= 1.
        assert cap_workers_by_memory(10_000_000, 4096, max_workers=8) >= 1
