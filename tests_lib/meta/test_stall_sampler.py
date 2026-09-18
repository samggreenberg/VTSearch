"""Tests for the host sampler in the #3853 kit.

The sampler runs on a GRID node beside a labeling session and there is no NFS
mount in CI, so its value here is entirely in the pure parsing functions: a
`mountstats` reader that mis-assigns a column would produce plausible,
wrong tables of exactly the kind this issue has already had to retract once.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SAMPLER = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "stall_3853" / "sample_host.py"


@pytest.fixture(scope="module")
def sampler():
    spec = importlib.util.spec_from_file_location("_stall_sample_host", SAMPLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A real mountstats shape: a non-NFS device first (must be skipped), then two
# NFS mounts, with the preamble sections that precede `per-op statistics`.
MOUNTSTATS = """device rootfs mounted on / with fstype rootfs
device /dev/sda1 mounted on /boot with fstype ext4
device storage:/export/sgreenberg mounted on /exp/sgreenberg with fstype nfs statvers=1.1
\topts:\trw,vers=3,rsize=1048576,wsize=1048576
\tage:\t7084800
\tbytes:\t1 2 3 4 5 6 7 8
\tevents:\t1 2 3 4 5 6 7 8 9 10
\txprt:\ttcp 892 0 1 0 2 3 4
\tper-op statistics
\t        NULL: 1 1 0 44 24 0 0 0 0
\t     GETATTR: 10626 10626 0 1000 2000 100 3000 3600 0
\t       WRITE: 2083 2083 0 5000 6000 1000 1500 1833 0
device flash:/expscratch mounted on /expscratch/sgreenberg with fstype nfs4 statvers=1.1
\tper-op statistics
\t       WRITE: 500 500 0 10 20 5 6 7 0
"""


class TestParseMountstats:
    def test_finds_nfs_mounts_and_skips_the_rest(self, sampler):
        """A non-NFS mount has no per-op section, so including it would make
        'mount missing' ambiguous between not-mounted and not-NFS."""
        parsed = sampler.parse_mountstats(MOUNTSTATS)
        assert set(parsed) == {"/exp/sgreenberg", "/expscratch/sgreenberg"}

    def test_columns_land_in_the_right_fields(self, sampler):
        """The load-bearing assertion: `queue`, `rtt` and `exec` are the 6th,
        7th and 8th numbers, and swapping them would silently rewrite every
        conclusion drawn from this sampler."""
        write = sampler.parse_mountstats(MOUNTSTATS)["/exp/sgreenberg"]["WRITE"]
        assert write == {
            "ops": 2083,
            "trans": 2083,
            "timeouts": 0,
            "bytes_sent": 5000,
            "bytes_recv": 6000,
            "queue_ms": 1000,
            "rtt_ms": 1500,
            "exec_ms": 1833,
            "errors": 0,
        }

    def test_empty_or_garbage_input_is_not_an_error(self, sampler):
        """It reads /proc on a node it does not control; a surprise there must
        cost a sample, not the session."""
        assert sampler.parse_mountstats("") == {}
        assert sampler.parse_mountstats("device broken mounted on\nper-op statistics\n") == {}


class TestDiffOps:
    def test_reports_per_operation_times_not_totals(self, sampler):
        """The correction this sampler exists to prevent: a cumulative counter
        read raw gives the average since the mount was made (82 days in the
        #3853 case, dominated by past bulk writes), which is how a '75 ms
        WRITE average' entered the record when steady state was ~1 ms."""
        before = sampler.parse_mountstats(MOUNTSTATS)["/exp/sgreenberg"]
        after = {
            "GETATTR": {**before["GETATTR"], "ops": 10726, "exec_ms": 3640, "queue_ms": 110, "rtt_ms": 3030},
            "WRITE": {**before["WRITE"], "ops": 2093, "exec_ms": 3833, "queue_ms": 1100, "rtt_ms": 1600},
        }
        diffed = sampler.diff_ops(before, after, ("GETATTR", "WRITE", "READ"))

        assert diffed["GETATTR"]["ops"] == 100
        assert diffed["GETATTR"]["exec"] == 0.4
        # 2000ms of exec over 10 writes: a slow interval, and the number a
        # reader would otherwise have to compute by hand.
        assert diffed["WRITE"] == {"ops": 10, "exec": 200.0, "queue": 10.0, "rtt": 10.0, "bytes": 0}
        assert "READ" not in diffed, "an op with no traffic must not produce a row"

    def test_no_traffic_in_an_interval_produces_nothing(self, sampler):
        """Dividing by a zero op count is the obvious crash; a quiet mount is
        the common case overnight."""
        before = sampler.parse_mountstats(MOUNTSTATS)["/exp/sgreenberg"]
        assert sampler.diff_ops(before, before, ("WRITE", "GETATTR")) == {}


class TestParseProcStat:
    def test_reads_faults_and_rss_past_a_comm_with_spaces(self, sampler):
        """`comm` is parenthesised and can hold spaces, so a naive field split
        shifts every column after it."""
        # Everything after the closing paren, so index 0 is `state` and
        # /proc field N lands at index N-3: minflt(10)->7, majflt(12)->9,
        # rss(24)->21.
        tail = ["0"] * 45
        tail[0] = "S"
        tail[7] = "1234"  # minflt
        tail[9] = "245"  # majflt
        tail[21] = "256"  # rss, in pages
        parsed = sampler.parse_proc_stat("4242 (python app.py) " + " ".join(tail))
        assert parsed is not None
        assert parsed["minflt"] == 1234
        assert parsed["majflt"] == 245
        assert parsed["rss_kb"] > 0

    def test_truncated_line_reads_as_unknown(self, sampler):
        assert sampler.parse_proc_stat("1 (sh) S 0 0") is None
