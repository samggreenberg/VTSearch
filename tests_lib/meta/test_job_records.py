"""The job-record ceiling: ``scripts/slurm/job_records.py`` and its two callers (#4701).

Slurm's ``MaxJobCount`` caps the job records ``slurmctld`` holds for the whole
cluster, and every array task counts against it, pending ones included. #4668
queued sixteen 720-task arrays at ``%6``: the first twelve held 8,500 of the
cluster's ~9,200 records, and the last four were refused while
``launch_cells.sh`` printed an empty job id and carried on.

What is pinned here:

* the decision itself, on the numbers #4668 met, at the ceilings' boundaries;
* the reading of ``scontrol show config`` and ``squeue``, through stubs on
  ``PATH`` (no scheduler needed);
* preflight check 18 (``--array-tasks``), and that it is opt-in;
* ``calibration/launch_cells.sh`` refusing such an array before ``sbatch`` sees
  it, and failing loudly when ``sbatch`` hands back no job id.

Meta-group: the subject is repo tooling under ``scripts/``, which has no other
test or type coverage.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "slurm" / "job_records.py"
_PREFLIGHT = _ROOT / "scripts" / "experiments" / "preflight.sh"
_LAUNCH_CELLS = _ROOT / "scripts" / "experiments" / "calibration" / "launch_cells.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("job_records", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


job_records = _load_module()

_CONFIG = """Configuration data as of 2026-10-08T14:55:00
MaxArraySize            = 10100
MaxJobCount             = 10000
MinJobAge               = 300 sec
PrivateData             = {private}
"""


def _stub_bin(tmp_path: Path, *, mine: int = 0, others: int = 0, private: str = "none") -> Path:
    """A `bin/` holding a stub `scontrol` and `squeue`, and an `sbatch` that logs its calls.

    `squeue` prints one user name per record (`-o %u`), and only `me`'s when
    SQUEUE_USERS reaches it, the way a real default filter would.  `sbatch`
    appends its argv to `sbatch.log` and prints `$STUB_JOBID`, so an unset
    STUB_JOBID is a refused submission; STUB_REFUSE_DEPENDENT refuses only a
    job that names a `--dependency`.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (tmp_path / "config.txt").write_text(_CONFIG.format(private=private))
    (tmp_path / "users.txt").write_text("me\n" * mine + "someone\n" * others)
    stubs = {
        "scontrol": f'cat "{tmp_path}/config.txt"\n',
        "squeue": (
            f'if [ -n "${{SQUEUE_USERS:-}}" ]; then grep -x me "{tmp_path}/users.txt"; exit 0; fi\n'
            f'cat "{tmp_path}/users.txt"\n'
        ),
        "sbatch": (
            f'echo "$*" >> "{tmp_path}/sbatch.log"\n'
            'case "$*" in *--dependency*) [ -n "${STUB_REFUSE_DEPENDENT:-}" ] && exit 1 ;; esac\n'
            'printf "%s" "${STUB_JOBID:-}"\n'
        ),
    }
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    return bin_dir


def _no_scheduler_bin(tmp_path: Path) -> Path:
    """A `bin/` whose `scontrol` and `squeue` fail as a missing command does (exit 127).

    A machine with Slurm installed (the GRID: /usr/bin/scontrol) would otherwise
    answer the "no scheduler" tests from the real cluster.
    """
    bin_dir = tmp_path / "noslurm"
    bin_dir.mkdir(exist_ok=True)
    for name in ("scontrol", "squeue"):
        path = bin_dir / name
        path.write_text(f"#!/bin/sh\necho '{name}: command not found' >&2\nexit 127\n")
        path.chmod(0o755)
    return bin_dir


def _env(tmp_path: Path, bin_dir: Path | None, **extra: str) -> dict[str, str]:
    path = f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/local/bin"
    if bin_dir is not None:
        path = f"{bin_dir}:{path}"
    # The GRID's venv python is linked against a libpython it finds only through
    # LD_LIBRARY_PATH; stripped of it, every subprocess exits 127 there (as
    # test_preflight_knobs found).
    keep = {"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")}
    return {"PATH": path, "HOME": str(tmp_path), "USER": "me", **keep, **extra}


def _cli(tmp_path: Path, bin_dir: Path | None, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603  # fixed argv, repo-local script path, no shell
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(tmp_path, bin_dir, **env),
    )


class TestJudge:
    """The decision, on numbers alone."""

    @staticmethod
    def _snap(cluster: int, mine: int, max_jobs: int = 10_000):
        return job_records.Snapshot(max_jobs=max_jobs, cluster=cluster, mine=mine)

    def test_4668s_loop_stops_at_its_fourth_array_not_its_thirteenth(self):
        """Sixteen 720-task arrays beside ~700 other records, re-checked before each one.

        Your quarter share (2,500 records) binds first: three arrays are 2,160,
        and a fourth would be 2,880.  The run that went through sbatch got twelve
        arrays in and the cluster to ~92% of MaxJobCount before anything said no.
        """
        others, mine, submitted = 700, 0, 0
        for _ in range(16):
            verdict = job_records.judge(720, self._snap(others + mine, mine), 50, 25)
            if verdict.tag != "FITS":
                break
            mine += 720
            submitted += 1
        assert submitted == 3
        assert verdict.tag == "OVER"
        assert verdict.fits == 2_500 - 2_160
        assert "you past 25%" in verdict.summary and "cluster past" not in verdict.summary

    def test_the_cluster_ceiling_binds_on_a_busy_cluster(self):
        """Other users' records count too: 4,800 of theirs leave you 200."""
        verdict = job_records.judge(720, self._snap(cluster=4_800, mine=0), 50, 25)
        assert verdict.tag == "OVER" and verdict.fits == 200
        assert "the cluster past 50%" in verdict.summary and "you past" not in verdict.summary

    def test_the_ceiling_itself_fits_and_one_more_does_not(self):
        snap = self._snap(cluster=1_000, mine=500)
        assert job_records.judge(2_000, snap, 50, 25).tag == "FITS"  # mine -> 2,500 exactly
        over = job_records.judge(2_001, snap, 50, 25)
        assert over.tag == "OVER" and over.fits == 2_000

    def test_the_advice_names_the_fix_and_the_throttle_trap(self):
        advice = "\n".join(job_records.judge(5_000, self._snap(0, 0), 50, 25).advice)
        assert "2,500 tasks fit right now" in advice
        assert "throttle limits running tasks, not records" in advice
        assert "chunk the array" in advice and "pack several cells per task" in advice

    def test_a_full_table_fits_nothing_rather_than_a_negative_count(self):
        assert job_records.judge(1, self._snap(cluster=9_000, mine=3_000), 50, 25).fits == 0

    def test_private_data_is_said_out_loud(self):
        snap = job_records.Snapshot(max_jobs=10_000, cluster=10, mine=10, private=True)
        verdict = job_records.judge(10, snap, 50, 25)
        assert verdict.tag == "FITS"
        assert any("lower bound" in a for a in verdict.advice)


class TestParsing:
    @pytest.mark.parametrize(
        ("private", "hidden"),
        (("none", False), ("jobs", True), ("accounts,jobs,usage", True), ("usage", False)),
    )
    def test_config(self, private, hidden):
        assert job_records.parse_config(_CONFIG.format(private=private)) == (10_000, hidden)

    def test_a_config_without_max_job_count_reads_as_none(self):
        assert job_records.parse_config("MaxArraySize = 10100\n") == (None, False)

    def test_records_count_every_line_and_yours_by_name(self):
        assert job_records.count_records("me\nsomeone\nme\n\nmeow\n", "me") == (4, 2)


class TestCli:
    def test_fits(self, tmp_path):
        proc = _cli(tmp_path, _stub_bin(tmp_path, mine=100, others=700), "--tasks", "720")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert proc.stdout.startswith("FITS: 720 array tasks fit: cluster 800 + 720 = 1,520 records")

    def test_over(self, tmp_path):
        proc = _cli(tmp_path, _stub_bin(tmp_path, mine=2_160, others=700), "--tasks", "720")
        assert proc.returncode == 1
        first, *rest = proc.stdout.splitlines()
        assert first.startswith("OVER: 720 array tasks would take you past 25% of MaxJobCount")
        assert rest and all(line.startswith("-> ") for line in rest)

    def test_a_default_squeue_filter_does_not_shrink_the_cluster(self, tmp_path):
        """With SQUEUE_USERS honoured, 4,800 other records would vanish and this would pass."""
        bin_dir = _stub_bin(tmp_path, mine=0, others=4_800)
        proc = _cli(tmp_path, bin_dir, "--tasks", "720", SQUEUE_USERS="me")
        assert proc.returncode == 1 and "the cluster past 50%" in proc.stdout

    def test_the_ceilings_come_from_the_environment(self, tmp_path):
        bin_dir = _stub_bin(tmp_path, mine=2_160, others=700)
        proc = _cli(tmp_path, bin_dir, "--tasks", "720", VTS_RECORDS_USER_PCT="40")
        assert proc.returncode == 0 and "(ceiling 4,000)" in proc.stdout

    def test_no_scheduler_is_unknown_not_fits(self, tmp_path):
        proc = _cli(tmp_path, _no_scheduler_bin(tmp_path), "--tasks", "720")
        assert proc.returncode == 3
        assert proc.stdout.startswith("UNKNOWN: could not run `scontrol show config`")

    def test_a_bad_percentage_is_a_usage_error(self, tmp_path):
        proc = _cli(tmp_path, _stub_bin(tmp_path), "--tasks", "1", "--cluster-pct", "150")
        assert proc.returncode == 2 and "(0, 100]" in proc.stderr


class TestPreflightCheck18:
    @staticmethod
    def _preflight(tmp_path: Path, bin_dir: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
        """Run the preflight with no VTS_REPO, so the repo checks stay out of the way."""
        return subprocess.run(  # noqa: S603  # fixed argv, repo-local script path, no shell
            ["bash", str(_PREFLIGHT), "--exp", str(tmp_path / "exp"), *args],  # noqa: S607 - bash from PATH
            capture_output=True,
            text=True,
            env=_env(tmp_path, bin_dir),
        )

    def test_over_fails_with_the_advice(self, tmp_path):
        proc = self._preflight(tmp_path, _stub_bin(tmp_path, mine=2_160, others=700), "--array-tasks", "720")
        assert "  FAIL  job records: 720 array tasks would take you past 25%" in proc.stdout
        assert "        -> 340 tasks fit right now" in proc.stdout
        assert proc.returncode == 1

    def test_fits_is_ok(self, tmp_path):
        proc = self._preflight(tmp_path, _stub_bin(tmp_path, mine=0, others=700), "--array-tasks", "720")
        assert "  ok    job records: 720 array tasks fit" in proc.stdout

    def test_an_unread_ceiling_fails(self, tmp_path):
        proc = self._preflight(tmp_path, _no_scheduler_bin(tmp_path), "--array-tasks", "720")
        assert "  FAIL  job records: could not run `scontrol show config`" in proc.stdout
        assert "an unread ceiling is not a passing one" in proc.stdout

    def test_the_check_is_opt_in(self, tmp_path):
        assert "job records" not in self._preflight(tmp_path, None).stdout

    def test_a_task_count_that_is_not_a_number_is_a_usage_error(self, tmp_path):
        proc = self._preflight(tmp_path, None, "--array-tasks", "lots")
        assert proc.returncode == 2 and "--array-tasks wants a whole number" in proc.stderr


class TestLaunchCells:
    """``calibration/launch_cells.sh`` against a stand-in worktree that counts 720 cells."""

    @staticmethod
    def _launch(tmp_path: Path, bin_dir: Path, **env: str) -> subprocess.CompletedProcess[str]:
        wt = tmp_path / "wt"
        calib = wt / "scripts" / "experiments" / "calibration"
        calib.mkdir(parents=True, exist_ok=True)
        (wt / "gridenv.sh").write_text("")
        (calib / "run_cells.py").write_text("print(720)\n")
        # launch_cells.sh counts its cells with a bare `python`, which a venv provides.
        python = bin_dir / "python"
        if not python.exists():
            python.symlink_to(sys.executable)
        return subprocess.run(  # noqa: S603  # fixed argv, repo-local script path, no shell
            ["bash", str(_LAUNCH_CELLS)],  # noqa: S607 - bash from PATH
            capture_output=True,
            text=True,
            env=_env(
                tmp_path,
                bin_dir,
                VTS_REPO=str(wt),
                CALIB_EXP=str(tmp_path / "exp"),
                CALIB_JOB_NAME="bu4668-b5_app_b1",
                **env,
            ),
        )

    @staticmethod
    def _sbatch_calls(tmp_path: Path) -> list[str]:
        log = tmp_path / "sbatch.log"
        return log.read_text().splitlines() if log.exists() else []

    def test_an_array_that_fits_is_submitted(self, tmp_path):
        proc = self._launch(tmp_path, _stub_bin(tmp_path, mine=0, others=700), STUB_JOBID="4242")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "FITS: 720 array tasks fit" in proc.stdout
        assert "cells array: 4242" in proc.stdout and "analyze: 4242" in proc.stdout
        assert "--array=0-719%8" in self._sbatch_calls(tmp_path)[0]

    def test_an_array_over_the_ceiling_never_reaches_sbatch(self, tmp_path):
        proc = self._launch(tmp_path, _stub_bin(tmp_path, mine=2_160, others=700), STUB_JOBID="4242")
        assert proc.returncode == 1
        assert "OVER: 720 array tasks would take you past 25%" in proc.stdout
        assert "not submitting bu4668-b5_app_b1's 720-task array" in proc.stderr
        assert self._sbatch_calls(tmp_path) == []

    def test_the_check_can_be_skipped_on_purpose(self, tmp_path):
        bin_dir = _stub_bin(tmp_path, mine=2_160, others=700)
        proc = self._launch(tmp_path, bin_dir, STUB_JOBID="4242", CALIB_SKIP_RECORD_CHECK="1")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "OVER" not in proc.stdout and len(self._sbatch_calls(tmp_path)) == 2

    def test_a_refused_array_fails_loudly_and_submits_no_analyze(self, tmp_path):
        """#4668's last four: sbatch printed no id, and the launcher used to echo an empty one."""
        proc = self._launch(tmp_path, _stub_bin(tmp_path, mine=0, others=700))
        assert proc.returncode == 1
        assert "bu4668-b5_app_b1's cells array was REFUSED by sbatch" in proc.stderr
        assert "MaxJobCount" in proc.stderr
        assert "cells array: \n" not in proc.stdout
        assert len(self._sbatch_calls(tmp_path)) == 1  # the analyze step never went in
        assert not (tmp_path / "exp" / "logs" / ".cells_jobid").exists()

    def test_a_refused_analyze_step_fails_but_keeps_the_arrays_id(self, tmp_path):
        bin_dir = _stub_bin(tmp_path, mine=0, others=700)
        proc = self._launch(tmp_path, bin_dir, STUB_JOBID="4242", STUB_REFUSE_DEPENDENT="1")
        assert proc.returncode == 1
        assert "bu4668-b5_app_b1's analyze step was REFUSED by sbatch" in proc.stderr
        assert (tmp_path / "exp" / "logs" / ".cells_jobid").read_text() == "4242\n"
