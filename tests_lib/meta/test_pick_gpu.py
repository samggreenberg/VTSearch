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
        out = "\n".join(
            [_node("cpu01", gres="(null)", used="(null)"), _node("n1", gres="gpu:v100:4", used="gpu:v100:2")]
        )
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


class TestUsageWithoutGresUsed:
    """The cluster does not write ``GresUsed``; usage has to come from ``AllocTRES``.

    Every fixture above supplies a ``GresUsed`` field, which is why nothing
    caught #3299: the HLTCOE cluster (Slurm 23.11.6) emits that field on **no**
    node, so the reader saw an empty string, computed ``free == total``
    everywhere and always returned the first candidate type. It advertised
    "a100 23/23 free" while all 23 A100s were allocated and 109 V100s idled.
    The records below are copied from that cluster rather than composed, so a
    future reader cannot describe a machine the code never meets.
    """

    #: Verbatim `scontrol show node --oneliner` for rack5n06, 2026-08-28: four
    #: A100s, all four allocated, and no GresUsed field anywhere on the line.
    BUSY_A100 = (
        "NodeName=rack5n06 Arch=x86_64 CoresPerSocket=24  CPUAlloc=32 CPUEfctv=96 CPUTot=96 CPULoad=5.06 "
        "AvailableFeatures=a100,intel,40gb ActiveFeatures=a100,intel,40gb Gres=gpu:a100:4 NodeAddr=rack5n06 "
        "NodeHostName=rack5n06 Version=23.11.6 OS=Linux 5.14.0-687.13.1.el9_8.x86_64 #1 SMP PREEMPT_DYNAMIC "
        "Tue Jun 2 11:33:56 EDT 2026  RealMemory=750000 AllocMem=393216 FreeMem=8524 Sockets=2 Boards=1 "
        "State=MIXED ThreadsPerCore=2 TmpDisk=6900000 Weight=1 Owner=N/A MCS_label=N/A Partitions=gpu  "
        "BootTime=2026-06-25T09:05:36 SlurmdStartTime=2026-07-08T10:05:26 LastBusyTime=2026-08-24T02:01:17 "
        "ResumeAfterTime=None CfgTRES=cpu=96,mem=750000M,billing=1950,gres/gpu=4,gres/gpu:a100=4 "
        "AllocTRES=cpu=32,mem=384G,gres/gpu=4,gres/gpu:a100=4 CapWatts=n/a CurrentWatts=0 AveWatts=0 "
        "ExtSensorsJoules=n/a ExtSensorsWatts=0 ExtSensorsTemp=n/a"
    )

    #: Same cluster, rack10n01: eight idle V100s. `AllocTRES=` is present and
    #: **empty**, which is a real measurement (zero) and not a missing field.
    IDLE_V100 = (
        "NodeName=rack10n01 Arch=x86_64 CoresPerSocket=20 CPUAlloc=0 CPUEfctv=80 CPUTot=80 CPULoad=0.01 "
        "AvailableFeatures=v100,intel ActiveFeatures=v100,intel Gres=gpu:v100:8 NodeAddr=rack10n01 "
        "NodeHostName=rack10n01 Version=23.11.6 RealMemory=1000000 AllocMem=0 FreeMem=900000 Sockets=2 "
        "Boards=1 State=IDLE ThreadsPerCore=2 TmpDisk=7100000 Weight=1 Owner=N/A MCS_label=N/A "
        "Partitions=gpu  BootTime=2026-06-13T19:13:24 SlurmdStartTime=2026-07-08T10:05:26 "
        "CfgTRES=cpu=80,mem=1000000M,billing=3208,gres/gpu=8,gres/gpu:v100=8 AllocTRES= "
        "CapWatts=n/a CurrentWatts=0 AveWatts=0"
    )

    def test_a_fully_allocated_node_reports_zero_free(self):
        avail = pick_gpu.parse_node_availability(self.BUSY_A100)
        assert avail["a100"] == pick_gpu.Availability("a100", free=0, total=4)

    def test_an_empty_alloctres_means_zero_used_not_unknown(self):
        avail = pick_gpu.parse_node_availability(self.IDLE_V100)
        assert avail["v100"] == pick_gpu.Availability("v100", free=8, total=8)

    def test_the_slow_type_wins_when_the_fast_one_is_full(self):
        """The whole point: this pair used to select a100 and pend for a day."""
        avail = pick_gpu.parse_node_availability(self.BUSY_A100 + "\n" + self.IDLE_V100)
        gpu_type, _ = pick_gpu.select_gpu_type(avail, need=3)
        assert gpu_type == "v100"

    def test_the_untyped_gres_gpu_total_is_not_double_counted(self):
        """`AllocTRES` carries both `gres/gpu=4` and `gres/gpu:a100=4`."""
        assert pick_gpu._node_gpus_used(self.BUSY_A100) == {"a100": 4}

    def test_gres_used_still_wins_where_a_cluster_writes_it(self):
        """Other Slurm builds do emit it; do not regress them onto AllocTRES."""
        line = _node("n1", gres="gpu:l40s:8", used="gpu:l40s:3") + " AllocTRES=cpu=8,gres/gpu:l40s=7"
        assert pick_gpu.parse_node_availability(line)["l40s"].free == 5

    def test_a_node_with_no_usage_field_at_all_is_not_counted_as_free(self):
        """Unmeasurable is not empty: it must not advertise a free GPU."""
        line = (
            "NodeName=n1 Arch=x86_64 Gres=gpu:a100:8 RealMemory=256000 State=MIXED Partitions=gpu "
            "BootTime=2026-08-01T00:00:00"
        )
        assert pick_gpu.parse_node_availability(line) == {}


class TestSelectGpuType:
    # No return annotation: pick_gpu is loaded by path, so its Availability is a
    # runtime attribute pyright cannot use in a type expression.
    def _avail(self, **kwargs: tuple[int, int]):
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
            lambda: "\n".join(
                [_node("n1", gres="gpu:l40s:8", used="gpu:l40s:0"), _node("n2", gres="gpu:v100:8", used="gpu:v100:0")]
            ),
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
            lambda: "\n".join(
                [_node("n1", gres="gpu:l40s:8", used="gpu:l40s:0"), _node("n2", gres="gpu:v100:8", used="gpu:v100:0")]
            ),
        )
        assert pick_gpu.main([]) == 0
        assert capsys.readouterr().out.strip() == "v100"

    def test_explain_lists_every_type_including_non_candidates(self, monkeypatch, capsys):
        monkeypatch.delenv("VTS_GPU", raising=False)
        monkeypatch.setattr(
            pick_gpu,
            "_query_scontrol",
            lambda: "\n".join(
                [_node("n1", gres="gpu:l40s:8", used="gpu:l40s:2"), _node("n2", gres="gpu:h100:4", used="gpu:h100:0")]
            ),
        )
        assert pick_gpu.main(["--explain"]) == 0
        err = capsys.readouterr().err
        assert "6 free /   8 total" in err
        assert "h100" in err and "not a candidate" in err
