"""Preflight check 12: the knobs a run pins, against what the app ships (#4549).

Check 12 is the gate that makes a study *declare* every knob it pins off
production (`--diverges <knob>`).  Until #4549 it read the detector's knobs and
none of the session's, so a launcher could run an opening, a preference or a
spot check nobody ships and preflight said ok: `CALIB_STARTUP_SCHEDULE`,
`CALIB_SPOT_CHECK`, `CALIB_MIN_PRECISION`, `CALIB_BETA`, `CALIB_WALK_SHAPE` and
eleven siblings were never compared.

Pinned here, against the probe `scripts/experiments/calibration/preflight_knobs.py`:

* a clean environment MATCHES, and a pin equal to the shipped value passes;
* every session knob diverges when set off the app's value;
* a knob another knob makes inert is not flagged (a shape needs a balance),
  so no study declares a fictional divergence;
* a value the harness itself refuses is REFUSED, which no declaration excuses;
* an unset knob whose harness default has gone stale is caught too.

And, through `preflight.sh` itself, that the bash side reads all four tags.

Nothing here tests shipped ``vtsearch``/``vtscore`` behaviour, which is why it
lives in the ``meta`` group.  The calibration modules are loose scripts rather
than package members, so they are loaded by path.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CALIB = _ROOT / "scripts" / "experiments" / "calibration"
_PREFLIGHT = _ROOT / "scripts" / "experiments" / "preflight.sh"

#: App env vars check 12 reads beside the ``CALIB_*`` ones.
_APP_KNOBS = ("VTSEARCH_SVM_HEAD_C", "VTSEARCH_TRAIN_EPOCHS", "VTSEARCH_TRAIN_PATIENCE")


def _load(name: str, path: Path):
    """Import one calibration script by path, its directory on ``sys.path`` only while it loads."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


@pytest.fixture(scope="module")
def knobs():
    return _load("_calib_preflight_knobs", _CALIB / "preflight_knobs.py")


@pytest.fixture
def check(monkeypatch, knobs):
    """Run the probe under exactly *env*: every knob it could read starts unset."""

    def run(**env: str) -> list[str]:
        for name in list(os.environ):
            if name.startswith("CALIB_") or name in _APP_KNOBS:
                monkeypatch.delenv(name)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        cfg = _load("_calib_experiment_config", _CALIB / "experiment_config.py")
        return knobs.probe(cfg)

    return run


def _tagged(lines: list[str], tag: str) -> dict[str, list[str]]:
    rows = [line.split("\t") for line in lines if line.startswith(tag + "\t")]
    return {row[1]: row[2:] for row in rows}


def test_a_clean_environment_matches(check):
    out = check()
    assert "MATCHES" in out
    assert not _tagged(out, "DIVERGES") and not _tagged(out, "REFUSED")


def test_the_apps_own_dry_runs_match(check):
    """#4731: spelling out the app's 16 is the app, not an arm."""
    out = check(CALIB_GOOD_DRY_RUN="16", CALIB_QUOTA_DRY_BADS="16")
    assert not _tagged(out, "DIVERGES") and not _tagged(out, "REFUSED")


# One off-production value per session knob, and the knob name a launcher
# declares for it.  The names the launchers already used (`startup_schedule`,
# `spot_check`, `opening_diversity`) are kept exactly.
_DIVERGENT = [
    ({"CALIB_STARTUP_SCHEDULE": "g3@top,b4@mid,g20+dry1/16@top"}, "startup_schedule"),
    ({"CALIB_OPENING_DIVERSITY": "0.85/1"}, "opening_diversity"),
    ({"CALIB_MORE_WALK": "detector"}, "more_walk"),
    ({"CALIB_NEW_WALK": "hard"}, "new_walk"),
    ({"CALIB_BAND_SHARE": "8"}, "band_share"),
    ({"CALIB_SIGMA_FLOOR": "absolute"}, "sigma_floor"),
    ({"CALIB_SEED_EXAMPLES": "1"}, "seed_examples"),
    ({"CALIB_STRATIFY_TARGET": "1"}, "stratify_target"),
    ({"CALIB_GOOD_DRY_RUN": "off"}, "good_dry_run"),
    ({"CALIB_GOOD_DRY_RUN": "8"}, "good_dry_run"),
    ({"CALIB_QUOTA_DRY_BADS": "off"}, "quota_dry_bads"),
    ({"CALIB_CENTROID_LINE": "midpoint"}, "centroid_line"),
    ({"CALIB_BETA": "off"}, "beta"),
    ({"CALIB_BETA": "4"}, "beta"),
    ({"CALIB_WALK_SHAPE": "walk"}, "walk_shape"),
    ({"CALIB_WALK_PICKS": "3"}, "walk_picks"),
    ({"CALIB_WALK_TOL": "0.01"}, "walk_tol"),
    ({"CALIB_WALK_FINE": "1"}, "walk_fine"),
    ({"CALIB_WALK_GUARD": "0.5"}, "walk_guard"),
    ({"CALIB_SPOT_CHECK": "end"}, "spot_check"),
    ({"CALIB_SPOT_CHECK": "off"}, "spot_check"),
    ({"CALIB_WEAK_D": "0.75"}, "weak_d"),
    ({"CALIB_WEAK_MIN_T": "3"}, "weak_min_t"),
    ({"CALIB_WEAK_REPEAT": "0"}, "weak_repeat"),
    ({"CALIB_WEAK_PHASE": "any"}, "weak_phase"),
    ({"CALIB_ACQ_INCLUSION_OFFSET": "0", "CALIB_ACQ_RANK_PERCENTILE": "0.02"}, "acq_rank_percentile"),
    ({"CALIB_ACQ_INCLUSION_OFFSET": "0", "CALIB_ACQ_P_CROSSING": "0.5"}, "acq_p_crossing"),
]


@pytest.mark.parametrize(("env", "knob"), _DIVERGENT, ids=[f"{k}={'/'.join(e.values())}" for e, k in _DIVERGENT])
def test_a_session_knob_off_production_diverges(check, env, knob):
    diverges = _tagged(check(**env), "DIVERGES")
    assert knob in diverges, diverges
    got, shipped = diverges[knob]
    assert got and shipped


def test_pins_at_the_shipped_values_pass(check):
    """Naming today's production explicitly is not a divergence; the check compares values, not presence."""
    from vtscore.eval.startup_schedule import PRODUCTION_STARTUP
    from vtscore.training import thresholds as T

    out = check(
        CALIB_STARTUP_SCHEDULE=PRODUCTION_STARTUP,
        CALIB_BETA="%g" % T.DEFAULT_BETA,
        CALIB_SPOT_CHECK="weak",
        CALIB_WEAK_D="%g" % T.WEAK_SEPARATION_D,
        CALIB_WEAK_MIN_T=str(T.WEAK_CHECK_MIN_VOTES),
        CALIB_WEAK_REPEAT=str(T.WEAK_CHECK_COOLDOWN),
        CALIB_WEAK_PHASE="learned",
        CALIB_WALK_PICKS="0",
        CALIB_WALK_TOL="0",
        CALIB_WALK_FINE="0",
    )
    assert "MATCHES" in out, out


def test_acq_p_crossing_off_is_the_app_while_the_shipped_factor_is_none(check):
    """`off` forces the offset cut; the app takes it too exactly when ACQUISITION_ARGMAX_FACTOR is None."""
    from vtscore.training import thresholds as T

    flagged = "acq_p_crossing" in _tagged(check(CALIB_ACQ_P_CROSSING="off"), "DIVERGES")
    assert flagged == (T.ACQUISITION_ARGMAX_FACTOR is not None)


def test_a_swept_beta_is_not_also_a_shape_or_acquisition_divergence(check):
    """The app's shape and acquisition factor are read at the run's OWN beta, not the default one."""
    diverges = _tagged(check(CALIB_BETA="4"), "DIVERGES")
    assert set(diverges) == {"beta"}, diverges


def test_a_shape_without_a_balance_is_inert(check):
    """The Inclusion arm has no walk; the harness resolves any shape to None there."""
    diverges = _tagged(check(CALIB_BETA="off", CALIB_WALK_SHAPE="walk"), "DIVERGES")
    assert set(diverges) == {"beta"}, diverges


@pytest.mark.parametrize(
    ("env", "knob"),
    [
        ({"CALIB_SPOT_CHECK": "of"}, "spot_check"),
        ({"CALIB_WEAK_PHASE": "learnt"}, "weak_phase"),
        ({"CALIB_STARTUP_SCHEDULE": "g3@topp"}, "startup_schedule"),
        ({"CALIB_WALK_SHAPE": "wlak"}, "walk_shape"),
        ({"CALIB_BETA": "100"}, "beta"),
        ({"CALIB_MORE_WALK": "learned"}, "more_walk"),
        ({"CALIB_NEW_WALK": "boundary"}, "new_walk"),
        ({"CALIB_BAND_SHARE": "0"}, "band_share"),
        ({"CALIB_BAND_SHARE": "one"}, "band_share"),
        ({"CALIB_SIGMA_FLOOR": "abs"}, "sigma_floor"),
        ({"CALIB_GOOD_DRY_RUN": "0"}, "good_dry_run"),
        ({"CALIB_QUOTA_DRY_BADS": "sixteen"}, "quota_dry_bads"),
        ({"CALIB_GOOD_DRY_RUN": "on"}, "good_dry_run"),
        ({"CALIB_CENTROID_LINE": "tail"}, "centroid_line"),
        ({"CALIB_CENTROID_LINE_VARIANTS": "count"}, "centroid_line_variants"),
        ({"CALIB_CENTROID_LINE_VARIANTS": "count@0"}, "centroid_line_variants"),
    ],
)
def test_a_value_the_harness_refuses_is_refused(check, env, knob):
    """A typo'd value would kill every cell after the array is queued; preflight is the place to stop it."""
    out = check(**env)
    assert knob in _tagged(out, "REFUSED"), out
    assert "MATCHES" not in out


def test_an_unset_knob_on_a_stale_harness_default_is_caught(check, monkeypatch):
    """Both sides of the knob: not pinning is only the app if the harness resolves there (#3400)."""
    from vtscore.eval import voting_iterations as VI

    real = VI.simulate_voting_iterations
    sig = inspect.signature(real)
    stale = sig.replace(
        parameters=[p.replace(default="end") if p.name == "spot_check" else p for p in sig.parameters.values()]
    )

    def drifted(*args, **kwargs):  # pragma: no cover - only its signature is read
        return real(*args, **kwargs)

    drifted.__signature__ = stale  # type: ignore[attr-defined]
    monkeypatch.setattr(VI, "simulate_voting_iterations", drifted)
    got, _ = _tagged(check(), "DIVERGES")["spot_check"]
    assert "harness default; CALIB_SPOT_CHECK is unset" in got


class TestPreflightWiring:
    """The bash half: `preflight.sh` reads every tag the probe prints.

    Run against a stub worktree whose probe prints fixed lines, so this pins the
    parsing without needing a real run's environment.  The stub has no
    ``vtscore/``, so check 5 does not run and leaves the probe enabled.
    """

    _LINES = [
        "SKIPPED\tanchored grid (stub)",
        "REFUSED\tspot_check\tof\tmust be one of end, off, weak",
        "DIVERGES\tbeta\t4\t<unset> = the balance at beta 1",
        "DIVERGES\tstartup_schedule\tg3@top\tg3@top,b4@mid",
    ]

    def _preflight(self, tmp_path: Path, *args: str, probe: bool = True) -> str:
        calib = tmp_path / "repo" / "scripts" / "experiments" / "calibration"
        calib.mkdir(parents=True)
        (calib / "experiment_config.py").write_text("")
        if probe:
            (calib / "preflight_knobs.py").write_text("print(%r)\n" % "\n".join(self._LINES))
        proc = subprocess.run(  # noqa: S603  # fixed argv, repo-local script path, no shell
            ["bash", str(_PREFLIGHT), "--exp", str(tmp_path / "exp"), *args],  # noqa: S607 - bash from PATH
            capture_output=True,
            text=True,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(tmp_path),
                "VTS_REPO": str(tmp_path / "repo"),
                # The GRID's venv python is linked against a libpython it finds
                # only through LD_LIBRARY_PATH; stripped of it, the stub probe
                # never runs and the knob check reads as "could not compare".
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            },
        )
        return proc.stdout + proc.stderr

    def test_each_tag_reaches_the_gate(self, tmp_path):
        out = self._preflight(tmp_path, "--diverges", "beta,spot_check")
        assert "ok    knob check skipped: anchored grid (stub)" in out
        assert "ok    declared divergence on 'beta'" in out
        assert "FAIL  UNDECLARED divergence from production: startup_schedule = g3@top" in out
        # Declared, and still a failure: a declaration cannot make a value valid.
        assert "FAIL  spot_check = of is a value the harness refuses" in out
        assert "could not compare" not in out

    def test_a_worktree_without_the_probe_fails_rather_than_skipping(self, tmp_path):
        out = self._preflight(tmp_path, probe=False)
        assert "predates calibration/preflight_knobs.py" in out
