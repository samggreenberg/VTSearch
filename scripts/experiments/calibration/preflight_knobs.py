"""Preflight check 12's probe: the knobs this run pins, against what the app ships.

``preflight.sh`` runs this from the worktree the jobs will import (``VTS_REPO``),
so it reads the same ``experiment_config`` they do, and parses its stdout: one
tab-separated line per finding.

* ``MATCHES`` - nothing diverges;
* ``DIVERGES <knob> <got> <shipped>`` - passes only if the study declared
  ``<knob>`` with ``--diverges``;
* ``REFUSED <knob> <got> <why>`` - a value the harness itself refuses, which no
  declaration excuses: every cell would fail on it after the array is queued;
* ``SKIPPED <what>`` - a family this run does not enable, reported rather than
  passed, because a skipped check is not a passed one.

Moved out of a heredoc in ``preflight.sh`` so it can be tested (#4549); the
design notes below are the check's own.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path
from typing import Any

#: The values the harness accepts for the two closed-set knobs of the weak check
#: (#4496).  ``simulate_voting_iterations`` validates them inline, not through a
#: function this can call, so they are spelled out; a set that drifts from the
#: harness's fails closed here (a REFUSED line), never open.
SPOT_CHECKS = ("end", "off", "weak")
WEAK_PHASES = ("any", "learned")


def env(name: str) -> str | None:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else None


def probe(C: Any) -> list[str]:
    """Check 12's findings for the run *C* (``experiment_config``, imported under its env)."""
    from vtscore.eval.live_threshold_rules import SIGMA_FLOORS
    from vtscore.eval.startup_schedule import PRODUCTION_STARTUP, parse_startup_schedule
    from vtscore.eval.voting_iterations import (
        MORE_WALKS,
        PRODUCTION_HEAD,
        PRODUCTION_PATCH_STYLE,
        resolve_acquisition_factor,
        resolve_walk_shape,
        simulate_voting_iterations,
    )
    from vtscore.training import thresholds as T

    rows: list[tuple[str, str, str]] = []
    refused: list[tuple[str, str, str]] = []
    skipped: list[str] = []

    def pinned(knob, var, shipped):
        """A scalar knob: unset means the harness resolves it to the shipped value."""
        v = env(var)
        if v is not None and v != str(shipped):
            rows.append((knob, v, str(shipped)))

    def must_contain(knob, var, shipped, effective):
        """A set-valued knob: the shipped value has to be IN what the run resolves
        it to, or the run has no arm to compare its challengers against.

        *effective* is the resolved list off ``experiment_config``, so this catches a
        stale harness default exactly as it catches a stale launcher pin - and says
        which of the two it is, because the remedy differs (drop the pin vs. fix the
        default).
        """
        got = [str(x).strip() for x in effective]
        if str(shipped) in got:
            return
        source = "pinned in %s" % var if env(var) else "harness default; %s is unset" % var
        rows.append((knob, "%s (%s)" % (",".join(got), source), "a set containing " + str(shipped)))

    def resolved(knob, var, effective, shipped, want=None):
        """A knob the run resolves to *effective* - its pin, else a harness default.

        Compared as the run will USE it, so an unset knob whose harness default has
        gone stale is caught as surely as a stale pin, and named as such.
        """
        if effective == shipped:
            return
        got = env(var) or "%s (harness default; %s is unset)" % (effective, var)
        rows.append((knob, got, want or str(shipped)))

    def refuse(knob, var, exc):
        refused.append((knob, env(var) or "<unset>", str(exc)))

    def harness_default(param):
        """What the harness runs when ``run_cells.py`` leaves *param* out (it does when the knob is unset)."""
        return inspect.signature(simulate_voting_iterations).parameters[param].default

    pinned("head", "CALIB_HEAD", PRODUCTION_HEAD)
    # The pipeline, the vote order and the standalone cut (#3959): unset is the app's
    # own on all three, so any value is a run-level arm the study must declare.
    pinned("trainer", "CALIB_TRAINER", "app")
    pinned("strategy", "CALIB_STRATEGY", "autopilot")
    pinned("standalone_cut", "CALIB_STANDALONE_CUT", "raw")

    # The heads' own fit knobs are app env vars, not CALIB_* ones (#3197), so a
    # launcher that exports them changes the detector without touching any knob
    # above.  Their shipped values are the literal defaults in `config/runtime.py`,
    # read from its source because `vtscore.config` has already resolved them from
    # THIS run's environment - comparing the env var against the imported constant
    # would compare the pin against itself.
    from vtscore.config import runtime as _RT

    _RT_SRC = inspect.getsource(_RT)

    def pinned_app_env(knob, var):
        v = env(var)
        if v is None:
            return
        m = re.search(r'os\.environ\.get\(\s*"%s"\s*,\s*"([^"]*)"' % re.escape(var), _RT_SRC)
        if m is None:
            rows.append((knob, v, "<shipped default not found in config/runtime.py>"))
            return
        try:
            same = float(v) == float(m.group(1))
        except ValueError:
            same = v == m.group(1)
        if not same:
            rows.append((knob, v, m.group(1)))

    pinned_app_env("svm_head_c", "VTSEARCH_SVM_HEAD_C")
    pinned_app_env("train_epochs", "VTSEARCH_TRAIN_EPOCHS")
    pinned_app_env("train_patience", "VTSEARCH_TRAIN_PATIENCE")
    pinned("acq_offset", "CALIB_ACQ_INCLUSION_OFFSET", T.ACQUISITION_INCLUSION_OFFSET)
    pinned("calibrate_count", "CALIB_CALIBRATE_COUNT", 2)
    # The LIVE cut rule (#3557) - unset resolves to FOLD_ANCHOR_CUT_RULE inside the
    # harness.  Distinct from `cut_rule` below, which is the set of RE-CUTS riding
    # the trajectory: this one moves the trajectory itself (acquisition re-cuts the
    # same estimator), so a pinned value is a run-level arm and must be declared.
    pinned("live_cut_rule", "CALIB_LIVE_CUT_RULE", T.FOLD_ANCHOR_CUT_RULE)
    # A RETIRED live threshold rule (#4184) - unset is the shipped fold-anchored
    # cut.  Any value replaces the cut acquisition reads, so it is always a
    # run-level divergence the study must declare.
    v = env("CALIB_LIVE_THRESHOLD")
    if v is not None:
        rows.append(("live_threshold", v, "<unset> = the shipped fold-anchored cut"))
    # The Train/Calibrate split of each calibration fold (#3287/#3290).  The
    # shipped default is no longer one scalar: unset resolves per embedder through
    # `production_split_for` (PRODUCTION_SPLIT_BY_SPACE), exactly as the app does,
    # so an unset env var IS the production arm.  A pinned scalar can match at
    # most one space on a run that mixes them, so - like CALIB_BLEND_SCHEDULE - an
    # explicit pin is always a divergence the study must declare.
    v = env("CALIB_CALIBRATION_FRACTION")
    if v is not None:
        per_space = ", ".join("%s=%g" % (k, f) for k, f in sorted(T.PRODUCTION_SPLIT_BY_SPACE.items()))
        rows.append(("calibration_fraction", v, "<unset> = the app's per-space default (%s)" % per_space))

    # The app has no safe-thresholds switch any more (#2799): fusion is always on.
    # Read off the resolved config rather than the env var, because until #3400 the
    # harness default was 0: an unset var passed this check while the run measured
    # the unfused control - the one arm the app can no longer produce.
    if not C.SAFE_THRESHOLDS:
        rows.append(("safe_thresholds", env("CALIB_SAFE_THRESHOLDS") or "<unset> = 0", "1 (the app has no switch)"))

    # The #3796 calibration-split draw.  Production pins the split to
    # CALIBRATION_SPLIT_SEED and #2934 pinned it on purpose, so an unset env var IS
    # the production arm and ANY list is a divergence - including a one-element list
    # holding today's constant, which freezes the arm against a pin that can move.
    # The sweep is legitimate and is the only thing that can measure the pin's cost;
    # what it may not be is silent, because a grid whose cells calibrate off
    # nineteen splits nobody ships looks exactly like a grid that does not.
    v = env("CALIB_CALIBRATION_SEEDS")
    if v is not None:
        rows.append(("calibration_seed", v, "<unset> = the app's pinned split (%d)" % T.CALIBRATION_SPLIT_SEED))

    # An explicit schedule overrides the app's per-mode default (#2841).
    v = env("CALIB_BLEND_SCHEDULE")
    if v is not None:
        rows.append(("blend_schedule", v, "<unset> = the app's per-mode default"))

    # The #3314 adaptive fold count.  The app has no such thing: `calibrate_count`
    # is a constant there, so ANY schedule is a divergence and has to be declared -
    # including one whose early phase happens to equal today's constant, since the
    # knob's whole effect is that the count stops being one.  Checked separately
    # from `calibrate_count` above because the two can be set together and mean
    # different arms (the schedule's tail IS `calibrate_count`).
    v = env("CALIB_FOLD_COUNT_SCHEDULE")
    if v is not None:
        rows.append(("fold_count_schedule", v, "<unset> = a constant calibrate_count, as the app has"))

    # The #3308 voted-media exclusion floor, which #3312 sweeps as an arm axis.
    # Unset resolves through the app's own `resolve_exclusion_floor`, so an unset
    # env var IS the production arm.  Every other value is a divergence - INCLUDING
    # a numeric pin that happens to equal today's shipped floor, because pinning it
    # freezes the arm against a constant that can move underneath the study.
    v = env("CALIB_EXCLUDE_VOTED")
    if v is not None and v.strip().lower() not in ("", "default", "app"):
        rows.append(
            (
                "exclusion_floor",
                v,
                "<unset> = the app's own floor (currently %g)" % T.resolve_exclusion_floor(None),
            )
        )

    # The anchored/fold-anchored grid (#2852) is emitted only under CALIB_ANCHORED=1
    # and is off by default.  Checking its knobs unconditionally makes every study
    # that does not use the family declare a divergence it does not have - and a
    # declared-but-fictional divergence is worse than no check, because the next
    # reader cannot tell the real ones from the noise.  Check them when the family is
    # actually on; say plainly that they were skipped when it is not.
    if os.environ.get("CALIB_ANCHORED") == "1":
        must_contain("cut_rule", "CALIB_ANCHORED_RULES", T.FOLD_ANCHOR_CUT_RULE, C.ANCHORED_RULES)
        must_contain("fold_combine", "CALIB_ANCHORED_FOLD_COMBINES", T.FOLD_ANCHOR_COMBINE, C.ANCHORED_FOLD_COMBINES)
        must_contain(
            "anchor_weight",
            "CALIB_ANCHORED_WEIGHTS",
            "%g" % T.FOLD_ANCHOR_WEIGHT,
            ["%g" % w for w in C.ANCHORED_WEIGHTS],
        )
    else:
        skipped.append("anchored grid (CALIB_ANCHORED is not 1, so no anchored row is emitted)")
    must_contain("patch_style", "CALIB_PATCH_STYLES", PRODUCTION_PATCH_STYLE, C.PATCH_STYLES)

    # --- The session the simulated user runs (#4549) --------------------------
    # Everything below changes the trajectory or the line, not just what is
    # recorded, and went unpoliced until #4549: a launcher could run an opening,
    # a preference or a check nobody ships, and preflight said ok.  Where the
    # harness resolves a knob through a function (the preference, the walk's
    # shape, the acquisition factor), that function is the comparison, so a knob
    # another knob makes inert is not flagged and the study is not made to
    # declare a divergence it does not have.

    # The opening (#3267), against the app's own in its grammar - parsed, so a
    # respelling of today's opening matches and a malformed one is refused here
    # rather than by every cell.  Unset is the app's opening by construction.
    if C.STARTUP_SCHEDULE is not None:
        try:
            if parse_startup_schedule(C.STARTUP_SCHEDULE) != parse_startup_schedule(PRODUCTION_STARTUP):
                rows.append(("startup_schedule", C.STARTUP_SCHEDULE, PRODUCTION_STARTUP))
        except ValueError as exc:
            refuse("startup_schedule", "CALIB_STARTUP_SCHEDULE", exc)
    # #4197's diversity pass is an experiment knob with no app counterpart.
    if C.OPENING_DIVERSITY is not None:
        rows.append(("opening_diversity", C.OPENING_DIVERSITY, "<unset> = the app's opening, no diversity pass"))
    # #4637's More walk on the detector's top: an arm the app does not take.
    if C.MORE_WALK not in MORE_WALKS:
        refuse("more_walk", "CALIB_MORE_WALK", "must be one of %s" % ", ".join(MORE_WALKS))
    elif C.MORE_WALK != harness_default("more_walk"):
        rows.append(("more_walk", C.MORE_WALK, "<unset> = seed, the app's walk down the text sort"))
    # #4482's band picks: an arm the app does not take.
    if C.BAND_SHARE is not None:
        if isinstance(C.BAND_SHARE, int) and C.BAND_SHARE >= 1:
            rows.append(("band_share", str(C.BAND_SHARE), "<unset> = the app's own picks, no band picks"))
        else:
            refuse("band_share", "CALIB_BAND_SHARE", "must be a positive integer, one pick in N")
    # #4699's example opening and stratified split: the app's example sort, but
    # not the opening a dataset with a typed query takes, and an eval protocol
    # rather than an app setting. Declared, so a study cannot take either silently.
    if C.SEED_EXAMPLES is not None:
        rows.append(("seed_examples", str(C.SEED_EXAMPLES), "<unset> = the text or known-good opening"))
    if C.STRATIFY_TARGET:
        rows.append(("stratify_target", "1", "<unset> = the plain random split"))
    # #4731's Good-walk dry run and the quota's second tier: unset is the app's, ``off`` the
    # pre-#4731 arm, another count an arm.
    from vtscore.detectors.label_quota import DRY_BAD_QUOTA  # noqa: PLC0415
    from vtscore.eval.autopilot_flow import MORE_DRY_RUN  # noqa: PLC0415

    for knob, var, val, app, app_text in (
        (
            "good_dry_run",
            "CALIB_GOOD_DRY_RUN",
            C.GOOD_DRY_RUN,
            MORE_DRY_RUN,
            f"<unset> = the app's: the Good phase also ends after {MORE_DRY_RUN} misses with a Good in hand",
        ),
        (
            "quota_dry_bads",
            "CALIB_QUOTA_DRY_BADS",
            C.QUOTA_DRY_BADS,
            DRY_BAD_QUOTA,
            f"<unset> = the app's: a Good and {DRY_BAD_QUOTA} Bads get the trained head",
        ),
    ):
        if val is None or val == app:
            continue
        if val == "off" or (isinstance(val, int) and val >= 1):
            rows.append((knob, str(val), app_text))
        else:
            refuse(knob, var, "must be a positive integer or 'off'")
    # #4668's spread floor before #4492: a retired piece of the labels line.
    if C.SIGMA_FLOOR not in SIGMA_FLOORS:
        refuse("sigma_floor", "CALIB_SIGMA_FLOOR", "must be one of %s" % ", ".join(SIGMA_FLOORS))
    elif C.SIGMA_FLOOR != "relative":
        rows.append(("sigma_floor", C.SIGMA_FLOOR, "<unset> = relative, the app's floor (#4492)"))

    # The balance the line is drawn at (#4413).  Unset is the app's default;
    # the precision floor went in #4421 (experiment_config refuses its knob).
    def preference(beta):
        return "the balance at beta %g" % beta if beta is not None else "the Inclusion arm"

    shipped_beta = T.resolve_line_knobs(None)
    shipped_pref = "<unset> = the app's default preference, %s" % preference(shipped_beta)
    run_beta = shipped_beta
    if C.BETA is not None:
        try:
            run_beta = T.resolve_line_knobs(C.BETA)
        except ValueError as exc:
            refuse("beta", "CALIB_BETA", exc)
        else:
            if run_beta != shipped_beta:
                rows.append(("beta", env("CALIB_BETA") or str(C.BETA), shipped_pref))

    # The check's shape (#4427): the app's is `check_shape` at the run's own beta,
    # so a study sweeping beta is not also made to declare a shape it left alone.
    # No balance, no walk: the harness resolves any shape to None there.
    try:
        if resolve_walk_shape(C.WALK_SHAPE, run_beta) != resolve_walk_shape(None, run_beta):
            rows.append(("walk_shape", C.WALK_SHAPE, "<unset> = the app's check_shape at the run's beta"))
    except ValueError as exc:
        refuse("walk_shape", "CALIB_WALK_SHAPE", exc)
    # The walk's other arms (#4427): the app calls `SpotCheck.start_balance` with
    # none of them, so its defaults are the shipped values.  `run_cells.py` always
    # passes all four, so what is compared is the resolved config.
    walk_app = inspect.signature(T.SpotCheck.start_balance).parameters
    resolved(
        "walk_picks", "CALIB_WALK_PICKS", C.WALK_PICKS, walk_app["picks"].default, "<unset> = the schedule's picks"
    )
    resolved("walk_tol", "CALIB_WALK_TOL", C.WALK_TOL, walk_app["tol"].default)
    resolved("walk_fine", "CALIB_WALK_FINE", C.WALK_FINE, walk_app["fine"].default)
    resolved("walk_guard", "CALIB_WALK_GUARD", C.WALK_GUARD, walk_app["guard"].default, "<unset> = no guard")

    # When the simulated user checks (#4496).  The app's Autopilot runs the weak
    # check, so `weak` is shipped, at the app's constants; `run_cells.py` leaves an
    # unset knob out, so the run gets the harness default - compared too, so a
    # stale default is caught on a launcher that pins nothing.
    spot = C.SPOT_CHECK if C.SPOT_CHECK is not None else harness_default("spot_check")
    if spot not in SPOT_CHECKS:
        refuse("spot_check", "CALIB_SPOT_CHECK", "must be one of %s" % ", ".join(SPOT_CHECKS))
    else:
        resolved("spot_check", "CALIB_SPOT_CHECK", spot, "weak")
    for knob, var, value, param, shipped in (
        ("weak_d", "CALIB_WEAK_D", C.WEAK_D, "weak_separation", T.WEAK_SEPARATION_D),
        ("weak_min_t", "CALIB_WEAK_MIN_T", C.WEAK_MIN_T, "weak_min_t", T.WEAK_CHECK_MIN_VOTES),
        ("weak_repeat", "CALIB_WEAK_REPEAT", C.WEAK_REPEAT, "weak_repeat", T.WEAK_CHECK_COOLDOWN),
    ):
        resolved(knob, var, value if value is not None else harness_default(param), shipped)
    # The app has no detector to read separation from during its opening, so it
    # can only prompt once the flow has left it: `learned`.
    phase = C.WEAK_PHASE if C.WEAK_PHASE is not None else harness_default("weak_phase")
    if phase not in WEAK_PHASES:
        refuse("weak_phase", "CALIB_WEAK_PHASE", "must be one of %s" % ", ".join(WEAK_PHASES))
    else:
        resolved("weak_phase", "CALIB_WEAK_PHASE", phase, "learned")

    # Where acquisition samples.  The rank pin has no app counterpart; the P-aware
    # factor resolves at the run's beta to the app's, so `off` matches while the
    # shipped ACQUISITION_ARGMAX_FACTOR is None and a number never does.
    if C.ACQ_RANK_PERCENTILE is not None:
        rows.append(("acq_rank_percentile", env("CALIB_ACQ_RANK_PERCENTILE") or "", "<unset> = the app's offset cut"))
    # #3546's arms: neither has an app counterpart yet.
    if C.SMART_GATE != "app":
        rows.append(("smart_gate", C.SMART_GATE, "app (the Smart light gates Hard -> New / Done)"))
    if C.LABEL_QUOTA is not None:
        rows.append(
            (
                "label_quota",
                "off",
                "app (the Goods' centroid under 3 Goods and 4 Bads, or a Good and 16 Bads; #4643, #4731)",
            )
        )
    if C.ACQ_ORIGIN != "line":
        rows.append(("acq_origin", C.ACQ_ORIGIN, "line (the app counts the offset from the line)"))
    from vtscore.eval.voting_iterations import resolve_acquisition_target  # noqa: PLC0415

    if resolve_acquisition_target(C.ACQ_TARGET_P, run_beta) != resolve_acquisition_target(None, run_beta):
        rows.append(("acq_target_p", env("CALIB_ACQ_TARGET_P") or "", "<unset> = the app's target pick precision"))
    if resolve_acquisition_factor(C.ACQ_P_CROSSING, run_beta) != resolve_acquisition_factor(None, run_beta):
        rows.append(
            (
                "acq_p_crossing",
                env("CALIB_ACQ_P_CROSSING") or "",
                "<unset> = the app's factor (%s)" % T.ACQUISITION_ARGMAX_FACTOR,
            )
        )

    out = ["SKIPPED\t%s" % s for s in skipped]
    out += ["REFUSED\t%s\t%s\t%s" % r for r in refused]
    out += ["DIVERGES\t%s\t%s\t%s" % r for r in rows]
    if not rows and not refused:
        out.append("MATCHES")
    return out


def main(argv: list[str]) -> int:
    calib = Path(argv[1]) / "scripts" / "experiments" / "calibration"
    sys.path.insert(0, str(calib))
    import common

    common.setup_env()
    # Imported with the RUN'S OWN ENVIRONMENT, so every knob reads the value this
    # run will actually use - the env var if it set one, else the harness
    # default.  Re-deriving the defaults here as literals is what let #3400's three
    # stale ones sit unnoticed: the check compared a launcher's pin against the app
    # and never noticed that *not pinning* resolved to a study-era value.  An unset
    # knob is only "the shipped arm" if the harness resolves it there, so that is
    # what gets compared.
    import experiment_config as C

    print("\n".join(probe(C)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
