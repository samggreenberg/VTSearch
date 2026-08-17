"""Tests for ``scripts/slurm/pick_gpu.py``.

The picker replaces a hardcoded ``v100`` in the GRID launchers, which is how
every pile cell built before 2026-08-17 got embedded on the cluster's slowest
GPU while L40S/A100 nodes idled (issue #3144). Both halves are text-only: the
``scontrol show node --oneliner`` parsing, and the pure choice on top of it.
No scheduler needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location("pick_gpu", REPO_ROOT / "scripts" / "slurm" / "pick_gpu.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which blows up if the module isn't there yet.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pick_gpu = _load_module()


def _node(name: str, *, gres: str, used: str, state: str = "MIXED", partitions: str = "gpu") -> str:
    """One `scontrol show node --oneliner` record, in field order and all."""
    return (
        f"NodeName={name} Arch=x86_64 CoresPerSocket=16 CPUAlloc=8 CPUTot=32 "
        f"Gres={gres} NodeAddr={name} NodeHostName={name} "
        f"RealMemory=256000 State={state} Partitions={partitions} "
        f"BootTime=2026-08-01T00:00:00 GresUsed={used}"
    )


class TestParseNodeAvailability:
    def test_sums_free_and_total_across_nodes(self):
        out = "\n".join(
            [
                _node("rack4n01", gres="gpu:l40s:8", used="gpu:l40s:3"),
                _node("rack4n02", gres="gpu:l40s:4", used="gpu:l40s:4"),
                _node("rack1n01", gres="gpu:v100:8", used="gpu:v100:1"),
            ]
        )
        avail = pick_gpu.parse_node_availability(out)
        assert avail["l40s"] == pick_gpu.Availability("l40s", free=5, total=12)
        assert avail["v100"] == pick_gpu.Availability("v100", free=7, total=8)

    def test_strips_the_index_and_socket_decorations(self):
        """scontrol writes `gpu:l40s:8(S:0-1)` and `gpu:l40s:3(IDX:0-2)`."""
        out = _node("rack4n01", gres="gpu:l40s:8(S:0-1)", used="gpu:l40s:3(IDX:0-2)")
        assert pick_gpu.parse_node_availability(out)["l40s"].free == 5

    def test_ignores_nodes_in_another_partition(self):
        out = "\n".join(
            [
                _node("gpu01", gres="gpu:a100:4", used="gpu:a100:0"),
                _node("other01", gres="gpu:a100:8", used="gpu:a100:0", partitions="scavenge"),
            ]
        )
        assert pick_gpu.parse_node_availability(out)["a100"].total == 4

    def test_matches_a_partition_in_a_comma_list_not_a_substring(self):
        out = "\n".join(
            [
                _node("n1", gres="gpu:a100:4", used="gpu:a100:0", partitions="cpu,gpu,all"),
                # `gpu-preempt` must not match a request for `gpu`.
                _node("n2", gres="gpu:a100:8", used="gpu:a100:0", partitions="gpu-preempt"),
            ]
        )
        assert pick_gpu.parse_node_availability(out)["a100"].total == 4

    def test_drained_and_down_nodes_contribute_nothing(self):
        """Not even to `total` -- else a dead node advertises the shortest queue."""
        out = "\n".join(
            [
                _node("up", gres="gpu:a100:2", used="gpu:a100:0"),
                _node("drained", gres="gpu:a100:8", used="gpu:a100:0", state="IDLE+DRAIN"),
                _node("dead", gres="gpu:a100:8", used="gpu:a100:0", state="DOWN+NOT_RESPONDING"),
            ]
        )
        assert pick_gpu.parse_node_availability(out)["a100"] == pick_gpu.Availability("a100", free=2, total=2)

    def test_untyped_gres_is_not_a_candidate(self):
        """This cluster rejects `--gres=gpu:1`, so an untyped pool is unusable."""
        out = _node("n1", gres="gpu:8", used="gpu:0")
        assert pick_gpu.parse_node_availability(out) == {}

    def test_non_gpu_nodes_and_null_gres_are_skipped(self):
        out = "\n".join([_node("cpu01", gres="(null)", used="(null)"), _node("n1", gres="gpu:v100:4", used="gpu:v100:2")])
        avail = pick_gpu.parse_node_availability(out)
        assert set(avail) == {"v100"}

    def test_used_exceeding_total_clamps_to_zero_free(self):
        """Gres/GresUsed can disagree mid-update; never report negative free."""
        out = _node("n1", gres="gpu:v100:4", used="gpu:v100:6")
        assert pick_gpu.parse_node_availability(out)["v100"].free == 0

    def test_one_node_holding_two_types(self):
        out = _node("mixed", gres="gpu:a100:2,gpu:v100:2", used="gpu:a100:2,gpu:v100:0")
        avail = pick_gpu.parse_node_availability(out)
        assert avail["a100"].free == 0
        assert avail["v100"].free == 2

    def test_a_reason_field_with_spaces_does_not_shift_later_fields(self):
        """`Reason=` holds free text; field lookup must not split on whitespace."""
        line = _node("n1", gres="gpu:l40s:4", used="gpu:l40s:1") + " Reason=Not responding [root@2026-08-17T00:00:00]"
        assert pick_gpu.parse_node_availability(line)["l40s"].free == 3

    def test_garbage_input_yields_no_availability(self):
        assert pick_gpu.parse_node_availability("scontrol: error: Invalid node name\n") == {}


class TestSelectGpuType:
    def _avail(self, **kwargs: tuple[int, int]) -> dict[str, pick_gpu.Availability]:
        return {t: pick_gpu.Availability(t, free, total) for t, (free, total) in kwargs.items()}

    def test_prefers_the_fastest_type_that_is_free_not_the_most_free(self):
        """The motivating case: idle L40S beats a bigger pile of idle V100s."""
        gpu_type, _ = pick_gpu.select_gpu_type(self._avail(a100=(0, 4), l40s=(2, 8), v100=(8, 8)))
        assert gpu_type == "l40s"

    def test_never_returns_v100_when_a_faster_type_is_free(self):
        for avail in (self._avail(a100=(1, 4), v100=(8, 8)), self._avail(l40s=(1, 8), v100=(8, 8))):
            assert pick_gpu.select_gpu_type(avail)[0] != "v100"

    def test_need_skips_a_type_that_cannot_start_every_job(self):
        avail = self._avail(a100=(2, 4), l40s=(4, 8), v100=(8, 8))
        assert pick_gpu.select_gpu_type(avail, need=1)[0] == "a100"
        assert pick_gpu.select_gpu_type(avail, need=3)[0] == "l40s"
        assert pick_gpu.select_gpu_type(avail, need=5)[0] == "v100"

    def test_falls_back_to_the_most_free_when_nothing_meets_need(self):
        """Something startable beats a fast queue slot."""
        gpu_type, reason = pick_gpu.select_gpu_type(self._avail(a100=(1, 4), l40s=(3, 8), v100=(2, 8)), need=6)
        assert gpu_type == "l40s"
        assert "most" in reason

    def test_when_all_are_busy_picks_the_largest_pool(self):
        gpu_type, reason = pick_gpu.select_gpu_type(self._avail(a100=(0, 4), l40s=(0, 8), v100=(0, 24)))
        assert gpu_type == "v100"
        assert "largest pool" in reason

    def test_ignores_types_outside_the_candidate_list(self):
        """h100 is idle but the QOS caps it at 0, so requesting it pends forever."""
        gpu_type, _ = pick_gpu.select_gpu_type(self._avail(h100=(8, 8), v100=(1, 8)))
        assert gpu_type == "v100"

    def test_candidate_order_is_the_only_speed_signal(self):
        avail = self._avail(a100=(1, 4), v100=(1, 8))
        assert pick_gpu.select_gpu_type(avail, types=("v100", "a100"))[0] == "v100"

    def test_empty_availability_uses_the_fallback(self):
        gpu_type, reason = pick_gpu.select_gpu_type({}, fallback="a100")
        assert gpu_type == "a100"
        assert "fallback" in reason

    def test_default_fallback_is_not_the_slow_type(self):
        assert pick_gpu.DEFAULT_FALLBACK != "v100"
        assert pick_gpu.DEFAULT_TYPES[-1] == "v100"

    def test_premium_types_stay_out_of_the_defaults(self):
        """`4gpu_tier` forbids h100/h200 outright (GRID-PLAYBOOK section 2)."""
        assert "h100" not in pick_gpu.DEFAULT_TYPES
        assert "h200" not in pick_gpu.DEFAULT_TYPES


class TestMain:
    def test_explicit_vts_gpu_short_circuits_the_query(self, monkeypatch, capsys):
        def _boom():  # pragma: no cover - only reached if the short-circuit breaks
            raise AssertionError("scontrol must not be queried when VTS_GPU is set")

        monkeypatch.setenv("VTS_GPU", "h200")
        monkeypatch.setattr(pick_gpu, "_query_scontrol", _boom)

        assert pick_gpu.main([]) == 0
        assert capsys.readouterr().out.strip() == "h200"

    def test_prints_type_to_stdout_and_reason_to_stderr(self, monkeypatch, capsys):
        monkeypatch.delenv("VTS_GPU", raising=False)
        monkeypatch.setattr(
            pick_gpu,
            "_query_scontrol",
            lambda: "\n".join([_node("n1", gres="gpu:l40s:8", used="gpu:l40s:0"), _node("n2", gres="gpu:v100:8", used="gpu:v100:0")]),
        )
        assert pick_gpu.main([]) == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == "l40s"
        assert "pick_gpu: l40s" in captured.err

    def test_unreachable_scheduler_still_prints_a_usable_type(self, monkeypatch, capsys):
        monkeypatch.delenv("VTS_GPU", raising=False)
        monkeypatch.setattr(pick_gpu, "_query_scontrol", lambda: None)
        assert pick_gpu.main([]) == 0
        assert capsys.readouterr().out.strip() == pick_gpu.DEFAULT_FALLBACK

    def test_vts_gpu_types_env_overrides_the_candidate_list(self, monkeypatch, capsys):
        monkeypatch.delenv("VTS_GPU", raising=False)
        monkeypatch.setenv("VTS_GPU_TYPES", "v100, l40s")
        monkeypatch.setattr(
            pick_gpu,
            "_query_scontrol",
            lambda: "\n".join([_node("n1", gres="gpu:l40s:8", used="gpu:l40s:0"), _node("n2", gres="gpu:v100:8", used="gpu:v100:0")]),
        )
        assert pick_gpu.main([]) == 0
        assert capsys.readouterr().out.strip() == "v100"

    def test_explain_lists_every_type_including_non_candidates(self, monkeypatch, capsys):
        monkeypatch.delenv("VTS_GPU", raising=False)
        monkeypatch.setattr(
            pick_gpu,
            "_query_scontrol",
            lambda: "\n".join([_node("n1", gres="gpu:l40s:8", used="gpu:l40s:2"), _node("n2", gres="gpu:h100:4", used="gpu:h100:0")]),
        )
        assert pick_gpu.main(["--explain"]) == 0
        err = capsys.readouterr().err
        assert "6 free /   8 total" in err
        assert "h100" in err and "not a candidate" in err
