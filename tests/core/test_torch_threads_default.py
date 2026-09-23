"""The server's thread default, which decides whether a vote scores on 1 core or 8.

The library default is 1 (``vtscore.config.TORCH_THREADS``) and stays 1: batch
work and the test suite want the low-memory setting, and a fat node's core count
has OOM'd jobs here before. The *server* is the other case, so it resolves its
own.
"""

import pytest

from vtsearch import torch_threads


class TestAllocatedCpus:
    def test_reads_the_affinity_mask_not_the_machine(self, monkeypatch):
        """``os.cpu_count()`` is the node; the cgroup is the allocation."""
        monkeypatch.setattr(torch_threads.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}, raising=False)
        monkeypatch.setattr(torch_threads.os, "cpu_count", lambda: 40)
        assert torch_threads.allocated_cpus() == 4

    def test_falls_back_to_cpu_count_without_affinity(self, monkeypatch):
        monkeypatch.delattr(torch_threads.os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(torch_threads.os, "cpu_count", lambda: 6)
        assert torch_threads.allocated_cpus() == 6

    def test_never_returns_zero(self, monkeypatch):
        monkeypatch.delattr(torch_threads.os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(torch_threads.os, "cpu_count", lambda: None)
        assert torch_threads.allocated_cpus() == 1


class TestResolve:
    def test_an_explicit_setting_wins(self, monkeypatch):
        monkeypatch.setattr(torch_threads, "allocated_cpus", lambda: 8)
        assert torch_threads.resolve({"VTSEARCH_TORCH_THREADS": "2"}) == 2

    def test_the_allocation_is_the_default(self, monkeypatch):
        """The bug this closes: unset used to mean 1, whatever the job held."""
        monkeypatch.setattr(torch_threads, "allocated_cpus", lambda: 8)
        assert torch_threads.resolve({}) == 8

    @pytest.mark.parametrize("raw", ["", "   ", "eight", "1.5"])
    def test_an_unusable_value_falls_back_rather_than_raising(self, monkeypatch, raw):
        """A stray ``export VTSEARCH_TORCH_THREADS=`` must not stop the server booting."""
        monkeypatch.setattr(torch_threads, "allocated_cpus", lambda: 8)
        assert torch_threads.resolve({"VTSEARCH_TORCH_THREADS": raw}) == 8

    def test_zero_and_negative_clamp_to_one(self, monkeypatch):
        monkeypatch.setattr(torch_threads, "allocated_cpus", lambda: 8)
        assert torch_threads.resolve({"VTSEARCH_TORCH_THREADS": "0"}) == 1
        assert torch_threads.resolve({"VTSEARCH_TORCH_THREADS": "-4"}) == 1


class TestAppPrelude:
    """app.py must publish what it resolved, or torch and OMP disagree.

    ``vtscore.config.TORCH_THREADS`` reads the same env var and drives
    ``torch.set_num_threads``. If app.py sets only OMP/MKL, torch silently wins
    with the library default of 1 and the OMP setting is decoration.
    """

    def test_the_prelude_exports_the_resolved_value(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parents[2] / "app.py").read_text(encoding="utf-8")
        head = src.split("# Configure structured logging", 1)[0]
        assert "OMP_NUM_THREADS" in head and "MKL_NUM_THREADS" in head
        assert "_THREADS_ENV" in head, "app.py must re-export the resolved count for vtscore.config"


class TestSuiteRunsAtTheLibraryDefault:
    """The *test* process must not inherit the server's allocation-sized default.

    ``tests/conftest.py`` imports ``app``, so the prelude above runs in every
    xdist worker. Left to size itself from the allocation it gave each worker one
    math thread per core -- workers x cores threads on the same cores -- and the
    full suite ran ~3x slower (~3 min -> ~10 min on 4 vCPUs) with nothing failing.
    """

    _PINNED = ("VTSEARCH_TORCH_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

    def test_conftest_pins_threads_before_its_first_heavy_import(self):
        """Source order, because the runtime check below cannot see it under xdist.

        The xdist master imports this conftest (and so ``app``) before spawning
        workers, which inherit whatever env it ended with -- so inside a worker
        a regression looks self-consistent. The one place it is visible is here.
        """
        import ast
        from pathlib import Path

        tree = ast.parse((Path(__file__).resolve().parents[1] / "conftest.py").read_text(encoding="utf-8"))
        pinned: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if names != ["os"]:
                    break
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "setdefault"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                ):
                    pinned.add(call.args[0].value)
        missing = set(self._PINNED) - pinned
        assert not missing, f"tests/conftest.py must setdefault {sorted(missing)} before importing anything but os"

    def test_no_native_threadpool_is_wider_than_the_library_setting(self):
        import torch
        from threadpoolctl import threadpool_info

        import vtscore.config as config

        expected = config.TORCH_THREADS
        assert torch.get_num_threads() == expected
        wide = [(p["internal_api"], p["num_threads"]) for p in threadpool_info() if p["num_threads"] > expected]
        assert not wide, f"native threadpools wider than TORCH_THREADS={expected}: {wide}"
