"""The per-user memory gate: `preflight.sh` check 8 and the QOS cap it reads (#4839).

`sacctmgr` prints a QOS's memory cap in the largest unit that divides it
exactly.  The GRID's `cpu_limit` cap was 1,100,000 MB, printed `mem=1100000M`
when the check was written (#3129); by 2026-10-13 it was 1,126,400 MB, printed
`mem=1100G`.  The check read only the digits, took that for 1,100 MB, and failed
every array: 60 tasks x 4G came out at 22342% of the allowance.

What is pinned here, through a stub `sacctmgr` on ``PATH`` (no scheduler needed):

* the cap is read in every unit `sacctmgr` writes, and the verdict is the same
  whichever one the number happens to land in;
* the tightest of several QOS is chosen by size, not by its digits;
* a cap in a unit the check cannot read fails it, rather than being guessed.

Meta-group: the subject is repo tooling under ``scripts/``, which has no other
test or type coverage.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PREFLIGHT = _ROOT / "scripts" / "experiments" / "preflight.sh"


def _stub_bin(tmp_path: Path, caps: dict[str, str], *, partition_qos: str = "cpu_limit") -> Path:
    """A `bin/` whose Slurm tools report *caps* ({qos: MaxTRESPU}) and nothing else.

    `scontrol` names *partition_qos* as the cpu partition's QOS, `sacctmgr show
    assoc` names every QOS in *caps*, and `squeue` has no jobs, so the check's
    candidates are exactly the keys of *caps*.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    qos_cases = "".join(f'  *"show qos {q} "*) echo "  {tres} " ;;\n' for q, tres in caps.items())
    stubs = {
        "scontrol": f'echo "PartitionName=cpu AllowQos=ALL QoS={partition_qos} State=UP"\n',
        "squeue": "exit 0\n",
        "sacctmgr": (
            f'case "$* " in\n  *"show assoc"*) echo " {",".join(caps)} " ;;\n{qos_cases}  *) exit 1 ;;\nesac\n'
        ),
    }
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    return bin_dir


def _preflight(tmp_path: Path, bin_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the preflight with no VTS_REPO, so the repo checks stay out of the way."""
    # The GRID's venv python is linked against a libpython it finds only through
    # LD_LIBRARY_PATH (test_preflight_knobs); the preflight's own probes need it.
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
        "USER": "me",
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    }
    return subprocess.run(  # noqa: S603  # fixed argv, repo-local script path, no shell
        ["bash", str(_PREFLIGHT), "--exp", str(tmp_path / "exp"), *args],  # noqa: S607 - bash from PATH
        capture_output=True,
        text=True,
        env=env,
    )


def _memory_line(proc: subprocess.CompletedProcess[str]) -> str:
    lines = [ln for ln in proc.stdout.splitlines() if "allowance" in ln or "per-user memory" in ln]
    assert len(lines) == 1, proc.stdout
    return lines[0]


class TestCapUnits:
    @pytest.mark.parametrize(
        ("tres", "cap_g"),
        [
            ("cpu=360,mem=1100G", 1100),  # the 2026-10-13 cap, #4839
            ("cpu=240,mem=1100000M", 1074),  # the #3129-era cap
            ("cpu=360,mem=1T", 1024),
            ("cpu=360,mem=1.1T", 1126),  # a fractional product, still a whole number of MB
            ("cpu=360,mem=1126400", 1100),  # no suffix is MB
        ],
    )
    def test_each_unit_reads_as_its_size(self, tmp_path, tres, cap_g):
        proc = _preflight(tmp_path, _stub_bin(tmp_path, {"cpu_limit": tres}), "--mem", "4G", "--conc", "60")
        line = _memory_line(proc)
        assert line.startswith("  ok    per-user memory: 240G of "), line
        assert f" of {cap_g}G (" in line and "under QOS cpu_limit" in line

    def test_the_issue_array_passes_at_its_real_share(self, tmp_path):
        """#4835's Up and Up runs: 60 x 4G against the 1100G cap is 22%, not 22342%."""
        proc = _preflight(
            tmp_path, _stub_bin(tmp_path, {"cpu_limit": "cpu=360,mem=1100G"}), "--mem", "4G", "--conc", "60"
        )
        assert _memory_line(proc) == "  ok    per-user memory: 240G of 1100G (22%) under QOS cpu_limit"

    def test_an_array_over_a_g_cap_still_fails(self, tmp_path):
        proc = _preflight(
            tmp_path, _stub_bin(tmp_path, {"cpu_limit": "cpu=360,mem=1100G"}), "--mem", "16G", "--conc", "70"
        )
        assert (
            _memory_line(proc) == "  FAIL  this array claims 1120G of your 1100G allowance under QOS cpu_limit (102%)"
        )
        assert proc.returncode == 1


class TestTightestCap:
    def test_the_tighter_cap_wins_across_units(self, tmp_path):
        """1000000M is tighter than 1100G, though 1100 < 1000000 by digits alone."""
        caps = {"cpu_limit": "cpu=360,mem=1100G", "4gpu_tier": "gres/gpu=4,mem=1000000M"}
        proc = _preflight(tmp_path, _stub_bin(tmp_path, caps), "--mem", "4G", "--conc", "60")
        assert _memory_line(proc) == "  ok    per-user memory: 240G of 977G (25%) under QOS 4gpu_tier"


class TestUnreadableCap:
    def test_an_unknown_unit_fails_rather_than_guessing(self, tmp_path):
        proc = _preflight(tmp_path, _stub_bin(tmp_path, {"cpu_limit": "cpu=360,mem=1P"}), "--mem", "4G", "--conc", "60")
        line = _memory_line(proc)
        assert line.startswith("  FAIL  per-user memory allowance: a QOS cap in a unit this check cannot read"), line
        assert "cpu_limit:mem=1P" in line and "NOT CHECKED" in line
        assert proc.returncode == 1

    def test_an_unknown_unit_is_not_hidden_by_a_readable_one(self, tmp_path):
        """The unreadable cap might be the tighter one, so a readable neighbour does not pass the check."""
        caps = {"cpu_limit": "cpu=360,mem=1100G", "odd_tier": "cpu=8,mem=2P"}
        proc = _preflight(tmp_path, _stub_bin(tmp_path, caps), "--mem", "4G", "--conc", "60")
        assert (
            "  FAIL  per-user memory allowance: a QOS cap in a unit this check cannot read (odd_tier:mem=2P)"
            in proc.stdout
        )

    def test_no_cap_at_all_is_still_its_own_failure(self, tmp_path):
        proc = _preflight(tmp_path, _stub_bin(tmp_path, {"cpu_limit": "cpu=360"}), "--mem", "4G", "--conc", "60")
        assert _memory_line(proc).startswith("  FAIL  per-user memory allowance: could not read the QOS cap")
