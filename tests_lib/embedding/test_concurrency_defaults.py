"""Hardware-derived (and env-overridable) defaults for the dataset-ingest
concurrency knobs.

Unit-tests the scaling logic in
:func:`vtscore.embedding.loader.default_concurrent_embeddings` and the shared
:func:`vtscore.embedding.loader._env_concurrency_override` in isolation by
faking the CPU core count, total RAM, and environment. The settings layer's
integration with these values (surfaced via
``get_max_concurrent_dataset_embeddings`` / ``...downloads`` and never persisted)
is covered in ``tests/core/test_settings.py``.

The guard these tests pin down: a memory- or core-starved box must resolve to a
single embed worker (the old fully-serial behaviour), so raising the default for
capable hardware never freezes a small one. The companion guard: GPU presence no
longer caps the embed default - embedders run on CPU today, so a single-GPU
SLURM node scales by its cores/RAM instead of being throttled to one device.
"""

from __future__ import annotations

import pytest

from vtscore.embedding import loader

GIB = 1024 * 1024 * 1024


@pytest.fixture
def fake_hw(monkeypatch):
    """Pin (cpus, total RAM) and return the resolved embed default.

    Clears the env override so the hardware probe is exercised; the override
    path has its own tests below.
    """

    def _resolve(*, cpus: int, ram_gib: float) -> int:
        monkeypatch.delenv("VTSEARCH_MAX_CONCURRENT_EMBEDDINGS", raising=False)
        monkeypatch.setattr(loader.os, "cpu_count", lambda: cpus)
        monkeypatch.setattr(loader, "_total_memory_bytes", lambda: int(ram_gib * GIB))
        return loader.default_concurrent_embeddings()

    return _resolve


class TestDefaultConcurrentEmbeddings:
    def test_small_cpu_box_stays_serial(self, fake_hw):
        # The 2-core / 3.7 GB laptop this guard exists for: both factors floor to 1.
        assert fake_hw(cpus=2, ram_gib=3.7) == 1

    def test_ram_starved_many_cores_stays_serial(self, fake_hw):
        # 32 cores but only 3 GB RAM -> RAM is the binding constraint.
        assert fake_hw(cpus=32, ram_gib=3.0) == 1

    def test_core_starved_huge_ram_stays_serial(self, fake_hw):
        # 256 GB RAM but 2 cores -> cores are the binding constraint.
        assert fake_hw(cpus=2, ram_gib=256) == 1

    def test_midrange_workstation_scales_to_two(self, fake_hw):
        # 8 cores / 16 GB -> min(8 // 4, 16 // 4) = min(2, 4) = 2.
        assert fake_hw(cpus=8, ram_gib=16) == 2

    def test_single_gpu_grid_node_scales_by_cores(self, fake_hw):
        # The HLTCOE Grid lockout case: a single-GPU SLURM allocation with 8
        # cores / 48 GB. The old GPU branch returned 1 (one job per visible
        # device) and serialised every embed; now it scales by cores -> 2.
        assert fake_hw(cpus=8, ram_gib=48) == 2

    def test_large_box_caps_at_four(self, fake_hw):
        # 64 cores / 256 GB would compute far higher; the cap holds it at 4.
        assert fake_hw(cpus=64, ram_gib=256) == 4

    def test_unreadable_ram_falls_back_to_serial(self, fake_hw):
        # _total_memory_bytes() == 0 (couldn't read) -> conservative 1 even on a
        # many-core box, rather than guessing generously.
        assert fake_hw(cpus=64, ram_gib=0) == 1


class TestEnvConcurrencyOverride:
    @pytest.fixture(autouse=True)
    def _fat_hardware(self, monkeypatch):
        # Pin generous hardware so any test asserting the override value can't
        # be confused with a hardware-derived result (which caps at 4 anyway).
        monkeypatch.setattr(loader.os, "cpu_count", lambda: 64)
        monkeypatch.setattr(loader, "_total_memory_bytes", lambda: 256 * GIB)

    def test_embeddings_override_wins_over_hardware(self, monkeypatch):
        # The launcher lever: push past the auto cap of 4 on a fat node.
        monkeypatch.setenv("VTSEARCH_MAX_CONCURRENT_EMBEDDINGS", "8")
        assert loader.default_concurrent_embeddings() == 8

    def test_downloads_override_wins_over_hardware(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_MAX_CONCURRENT_DOWNLOADS", "10")
        assert loader.default_concurrent_downloads() == 10

    def test_override_clamps_above_max(self, monkeypatch):
        # Above the settings-layer clamp -> pinned to 16, not handed through.
        monkeypatch.setenv("VTSEARCH_MAX_CONCURRENT_EMBEDDINGS", "999")
        assert loader.default_concurrent_embeddings() == 16

    def test_override_clamps_below_min(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_MAX_CONCURRENT_EMBEDDINGS", "0")
        assert loader.default_concurrent_embeddings() == 1

    def test_blank_override_ignored(self, monkeypatch):
        # Blank -> autodetect runs; fat hardware caps at 4.
        monkeypatch.setenv("VTSEARCH_MAX_CONCURRENT_EMBEDDINGS", "   ")
        assert loader.default_concurrent_embeddings() == 4

    def test_non_integer_override_ignored(self, monkeypatch):
        # A launcher typo must not block startup: fall back to autodetect.
        monkeypatch.setenv("VTSEARCH_MAX_CONCURRENT_EMBEDDINGS", "lots")
        assert loader.default_concurrent_embeddings() == 4

    def test_unset_override_returns_none(self, monkeypatch):
        monkeypatch.delenv("VTSEARCH_MAX_CONCURRENT_EMBEDDINGS", raising=False)
        assert loader._env_concurrency_override("VTSEARCH_MAX_CONCURRENT_EMBEDDINGS") is None


class TestTotalMemoryBytes:
    def test_reads_a_positive_value_on_this_host(self):
        # On any real CI/dev box /proc/meminfo or sysconf yields > 0; the
        # zero-fallback path is exercised via the mocked test above.
        assert loader._total_memory_bytes() > 0
