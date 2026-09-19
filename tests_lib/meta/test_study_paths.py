"""A reader whose study output is gone must say so, and say where it went (#4001).

The finished study output dirs under ``/expscratch/sgreenberg`` were deleted on
2026-09-18, and the analysis scripts still default ``--exp`` to them.  Without a
check the first ``open()`` raises on some file three directories down, which
names neither the study nor the reason.

The guard is for **readers only**.  ``launch_*.sh`` and ``run_*.sbatch`` create
their study dir (they all ``mkdir -p``), so a must-exist check there would break
the re-run that is the whole point of allowing the delete.  That asymmetry is
what ``test_launchers_are_not_guarded_because_they_create_the_dir`` pins.

The last two tests are #4006's, and they cover the two places where a body guard
is the wrong instrument because the failure happens before the body runs: an
``#SBATCH --output=`` slurmd opens first, and a stale default inside a ``--wrap``
that only reaches a compute node after a queue wait.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"
EXPERIMENTS = CALIB.parent


def _load_study_paths():
    """Import ``study_paths`` by path, as the calibration scripts import it."""
    spec = importlib.util.spec_from_file_location("_calib_study_paths", CALIB / "study_paths.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(CALIB))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CALIB))
    return module


def test_an_existing_dir_passes_through_unchanged(tmp_path: Path) -> None:
    study_paths = _load_study_paths()
    assert study_paths.require_study_dir(tmp_path) == tmp_path
    assert study_paths.require_study_dir(str(tmp_path)) == tmp_path


def test_a_missing_dir_exits_naming_the_path_and_the_record(tmp_path: Path, capsys) -> None:
    study_paths = _load_study_paths()
    gone = tmp_path / "scale-3156-final"
    with pytest.raises(SystemExit) as exc:
        study_paths.require_study_dir(gone, "--exp")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert str(gone) in err, "the message must name the directory that is missing"
    assert "#4001" in err and "deleted-20260918" in err, "it must point at where the deletion is recorded"
    assert "--exp" in err, "it must name the flag to point somewhere else"


def test_a_file_is_not_a_study_dir(tmp_path: Path) -> None:
    study_paths = _load_study_paths()
    f = tmp_path / "results.json"
    f.write_text("{}")
    with pytest.raises(SystemExit):
        study_paths.require_study_dir(f)


@pytest.mark.parametrize(
    "rel",
    [
        "calibration/analyze_overview.py",
        "calibration/analyze_phases.py",
        "calibration/analyze_scale.py",
        "calibration/analyze_tail_overlap.py",
        "calibration/figures_overview.py",
        "calibration/figures_scale.py",
        "calibration/figures_trajectory.py",
        "calibration/per_env_acq_2877.py",
    ],
)
def test_every_patched_reader_calls_the_guard(rel: str) -> None:
    text = (EXPERIMENTS / rel).read_text()
    assert "require_study_dir" in text, f"{rel} reads a deleted study dir and must check for it"


@pytest.mark.parametrize(
    "rel",
    [
        "calibration/analyse_good_mining.sh",
        "calibration/status_acq_2877.sh",
        "calibration/probe_acq_divergence.sh",
        "drive_cold/analyze_drive_3521.sbatch",
        "timing_r2/analyze_timing_3345.sbatch",
    ],
)
def test_every_patched_shell_reader_carries_the_guard_inline(rel: str) -> None:
    """Shell cannot import the helper -- ``sbatch`` copies the script, so a
    relative ``source`` need not resolve. These carry the check inline."""
    text = (EXPERIMENTS / rel).read_text()
    assert "deleted-20260918" in text and "#4001" in text
    assert "exit 2" in text


@pytest.mark.parametrize(
    "rel",
    [
        "calibration/launch_scale.sh",
        "calibration/launch_acq_2877.sh",
        "calibration/launch_good_mining.sh",
        "calibration/launch_incl_3196.sh",
        "calibration/launch_calseed_3796.sh",
        "enrich/launch_enrich.sh",
        "fastproc/launch_fastproc.sh",
        "precision/launch_precision.sh",
    ],
)
def test_launchers_are_not_guarded_because_they_create_the_dir(rel: str) -> None:
    """#4001: guarding a launcher would break re-running the deleted study."""
    text = (EXPERIMENTS / rel).read_text()
    assert "deleted-20260918" not in text, f"{rel} creates its study dir; it must not require one"


# --------------------------------------------------------------------------
# #4006: a guard in the body cannot save a job whose LOG is the missing file.
# --------------------------------------------------------------------------

_OUTPUT = re.compile(r"^#SBATCH\s+--output=(\S+)", re.MULTILINE)


@pytest.mark.parametrize("sbatch", sorted(EXPERIMENTS.rglob("*.sbatch")), ids=lambda p: p.name)
def test_no_sbatch_logs_into_a_purgeable_study_dir(sbatch: Path) -> None:
    """``--output=`` on ``/expscratch`` must sit at the user's scratch root.

    Slurm opens the batch output file before the first line of the script runs,
    so a path *inside* a study dir that is gone kills the job with nowhere to
    say why -- the log IS the missing file.  #4004's "launchers create their own
    dir, so leave them alone" reasoning does not reach this header: ``mkdir -p``
    in the body is already too late, which is why the launchers are in here
    beside the analyzers.

    Scoped to ``/expscratch`` because that is the purgeable mount #4001 emptied;
    a log under a durable ``/exp`` tree is fine and ``umap_params/`` keeps one.
    """
    for path in _OUTPUT.findall(sbatch.read_text()):
        if not path.startswith("/expscratch/"):
            continue
        tail = path.split("/", 3)[3:]  # everything below /expscratch/<user>/
        assert tail and "/" not in tail[0], (
            f"{sbatch.name} logs to {path}, which is nested inside a purgeable study dir; "
            "slurmd opens this before the script runs, so the job dies with no log at all"
        )


def test_the_gmm_launcher_checks_the_prepares_its_wrap_reads() -> None:
    """A stale default inside a ``--wrap`` fails on the compute node (#4006).

    That is the expensive shape: accepted, queued, and only then dead on a
    directory the submitter could have been told about at once.  This launcher
    creates its own study dir and is deliberately unguarded on it (#4001); what
    it checks are the two *other* studies' outputs it reads as inputs.
    """
    text = (EXPERIMENTS / "gmm_init" / "launch_gmm_3585.sh").read_text()
    assert 'SCALE_PREPARE="${SCALE_PREPARE:-' in text, "the wrap's default must be a named variable, not inline"
    assert "--results $SCALE_PREPARE " in text, "the wrap must read that variable"
    assert 'for prep in "$REUSE_PREPARE" "$SCALE_PREPARE"' in text, "both prepares are read, so both are checked"
    assert "deleted-20260918" in text and "#4001" in text and "exit 2" in text
    assert "CALIB_EXP" not in text.split("for prep in")[1].split("done")[0], (
        "the launcher creates CALIB_EXP; requiring it would break the re-run (#4001)"
    )
