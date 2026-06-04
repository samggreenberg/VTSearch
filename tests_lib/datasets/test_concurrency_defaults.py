"""Hardware-derived default for ``max_concurrent_dataset_embeddings``.

Unit-tests the scaling logic in
:func:`vtscore.embedding.loader.default_concurrent_embeddings` in isolation by
faking the GPU count, CPU core count, and total RAM. The settings layer's
integration with this value (surfaced via ``get_max_concurrent_dataset_embeddings``
and never persisted) is covered in ``tests/core/test_settings.py``.

The guard these tests pin down: a memory- or core-starved box must resolve to a
single embed worker (the old fully-serial behaviour), so raising the default for
capable hardware never freezes a small one.
"""

from __future__ import annotations

import pytest

from vtscore.embedding import loader

GIB = 1024 * 1024 * 1024


@pytest.fixture
def fake_hw(monkeypatch):
    """Pin (gpus, cpus, total RAM) and return the resolved default."""

    def _resolve(*, gpus: int, cpus: int, ram_gib: float) -> int:
        monkeypatch.setattr(loader, "_detect_cuda_devices", lambda: gpus)
        monkeypatch.setattr(loader.os, "cpu_count", lambda: cpus)
        monkeypatch.setattr(loader, "_total_memory_bytes", lambda: int(ram_gib * GIB))
        return loader.default_concurrent_embeddings()

    return _resolve


class TestDefaultConcurrentEmbeddings:
    def test_small_cpu_box_stays_serial(self, fake_hw):
        # The 2-core / 3.7 GB laptop this guard exists for: both factors floor to 1.
        assert fake_hw(gpus=0, cpus=2, ram_gib=3.7) == 1

    def test_ram_starved_many_cores_stays_serial(self, fake_hw):
        # 32 cores but only 3 GB RAM -> RAM is the binding constraint.
        assert fake_hw(gpus=0, cpus=32, ram_gib=3.0) == 1

    def test_core_starved_huge_ram_stays_serial(self, fake_hw):
        # 256 GB RAM but 2 cores -> cores are the binding constraint.
        assert fake_hw(gpus=0, cpus=2, ram_gib=256) == 1

    def test_midrange_workstation_scales_to_two(self, fake_hw):
        # 8 cores / 16 GB -> min(8 // 4, 16 // 4) = min(2, 4) = 2.
        assert fake_hw(gpus=0, cpus=8, ram_gib=16) == 2

    def test_large_box_caps_at_four(self, fake_hw):
        # 64 cores / 256 GB would compute far higher; the cap holds it at 4.
        assert fake_hw(gpus=0, cpus=64, ram_gib=256) == 4

    def test_unreadable_ram_falls_back_to_serial(self, fake_hw):
        # _total_memory_bytes() == 0 (couldn't read) -> conservative 1 even on a
        # many-core box, rather than guessing generously.
        assert fake_hw(gpus=0, cpus=64, ram_gib=0) == 1

    def test_single_gpu_unaffected_by_cpu_ram(self, fake_hw):
        # GPU path returns early and ignores the CPU/RAM scaling.
        assert fake_hw(gpus=1, cpus=2, ram_gib=3.7) == 1

    def test_multi_gpu_caps_at_two(self, fake_hw):
        assert fake_hw(gpus=4, cpus=64, ram_gib=256) == 2


class TestTotalMemoryBytes:
    def test_reads_a_positive_value_on_this_host(self):
        # On any real CI/dev box /proc/meminfo or sysconf yields > 0; the
        # zero-fallback path is exercised via the mocked test above.
        assert loader._total_memory_bytes() > 0
