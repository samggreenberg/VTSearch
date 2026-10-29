#!/usr/bin/env python3
"""Drift gate: the eval framework's default arm vs. the app it is meant to model.

`vtscore.eval` exists to measure *deviations* from the shipped algorithm.  That
only means something if the framework's **default arm** is the shipped
algorithm.  When it isn't, every experiment run against it is measuring a
detector nobody uses, and the damage is silent and retroactive - the numbers
still look fine.

Most of the harness is safe by construction because it *delegates*: the
`max_patch` style calls `pool_box_from_media` / `bad_negative_vecs` /
`media_score_rows` rather than re-deriving them, so it cannot drift.  This gate
covers the parts that can't delegate:

* **ported** - app logic re-implemented in the harness, because the original is
  unreachable (it lives in TypeScript) or unusable (it is wrapped in
  interactive, lock-guarded, single-detector caches).  A copy goes stale the
  moment the original moves.
* **default** - places where the harness resolves "no explicit arm" to whatever
  the app currently defaults to.  When the app's default changes, the harness
  keeps handing out the old one under the name "default".

Each mirror pins a digest of **both** sides.  A copy stays faithful only while
neither half moves without the other, so either half moving is the same event
and both have to be watched:

* `app-changed` - the original moved.  Reconcile the harness copy to it.
* `harness-changed` - the copy moved on its own, with the original standing
  still.  Confirm the copy still says what the app says; a harness edit that
  quietly re-points the default arm is drift that the app side can never
  reveal, and it is the direction the Smart-indicator plumbing actually drifted
  in (#2923).

Either way, once reconciled (or once you have confirmed nothing is owed), re-pin:

    python scripts/check-eval-app-sync.py --update

Digests ignore comments, docstrings and formatting, so only real logic changes
trip the gate.

A few harness sides are too coarse to digest - one function serving several
mirrors and much else besides - and say so in `no_harness_pin=`.  Those keep the
app-side pin alone; the fix, when the blind spot starts to matter, is to extract
the reproduction into its own helper (as #3403 did for the two `*_default`
mirrors) rather than to pin a thousand lines.

Adding a mirror: append a `Mirror(...)` to `MIRRORS` and run `--update`.  The
harness side is pinned by default, so opting out is a decision you have to write
down.  If the harness *intentionally* differs from the app, say so in
`divergence=` - the text is printed whenever the mirror trips, so the next
person reconciling it knows which differences are deliberate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import keyword
import re
import sys
import textwrap
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PINS_PATH = REPO_ROOT / "scripts" / "eval-app-sync.pins.json"

AUTOPILOT_TS = "frontend/src/app/services/autopilot-state.service.ts"
LABEL_VIEW_TS = "frontend/src/app/components/label-view/label-view.component.ts"
AUTO_SELECT_TS = "frontend/src/app/utils/auto-select-next.ts"


@dataclass(frozen=True)
class Mirror:
    """One app-side surface the eval harness reproduces rather than calls.

    Attributes:
        id: Stable key into the pins file.  Renaming one re-pins from scratch.
        app: The app-side source. ``py:<dotted.path.to.symbol>`` for Python,
            or ``py:<dotted.module>.<symbol>,<symbol>`` when the original is
            spread over more than one top-level name in one module;
            ``ts:<repo-relative file>::<anchor>`` for a TypeScript block (the
            anchor is matched literally, then brace-matched from the first
            ``{`` after it).
        harness: Where the reproduction lives, as ``<repo-relative file>::<symbol>``,
            or ``::<symbol>,<symbol>`` when the reproduction is spread over
            more than one name.  A name is top-level, or ``Class.method`` for
            a reproduction that lives on a method.  Each is resolved by
            parsing, so a name surviving only inside a comment does not count
            as present, and deleting or renaming the harness side trips the
            gate.
        kind: ``ported`` or ``default`` - see the module docstring.
        note: What is reproduced, and what to re-check when either side moves.
        divergence: A *declared, intentional* difference from the app.  Not an
            exemption - the digest is still pinned - but it tells whoever
            reconciles this mirror which differences are on purpose.
        no_harness_pin: Why this mirror's harness side carries no digest of its
            own.  Absent (the default) means it does: a harness symbol that is
            this mirror's dedicated counterpart should be pinned, so opting out
            is a decision that has to be written down rather than defaulted
            into.  The reason is printed with the mirror whenever it trips.
    """

    id: str
    app: str
    harness: str
    kind: str
    note: str
    divergence: str | None = None
    no_harness_pin: str | None = None


MIRRORS: list[Mirror] = [
    # ------------------------------------------------------------------ ported
    Mirror(
        id="autopilot.phase_machine",
        app=f"ts:{AUTOPILOT_TS}::checkPhaseTransition(goodCount",
        harness="vtscore/eval/autopilot_flow.py::next_phase,stop_lights,STOP_CAP",
        kind="ported",
        note=(
            "The phase ordering and every transition trigger of the simulated Autopilot user. "
            "Lives in TypeScript, so there is nothing to import - it is a hand copy. If you "
            "add, remove or reorder a phase, or change what gates one, port the same change. "
            "The 'more' walk (#4282) keeps history the counts cannot: its run of misses, read "
            "off which vote count rose. The app keeps it in checkPhaseTransition; the harness "
            "in AutopilotFlow._note_more_vote, which next_phase reads as more_done. A document "
            "dataset (stop_rule 'dry_run', #4488) takes its own branch after the opening - the "
            "walk with no Good target, its dry run as done - ported as next_phase(dry_run_stop=, "
            "ran_dry=) and AutopilotFlow(dry_run_stop=). The Good walk's dry run (#4731: moreDryRun "
            "misses with a Good in hand end the Good phase and spend the walk) is the same kind of "
            "history: the app's goodMisses/goodRanDry, the harness's AutopilotFlow good_misses/"
            "good_dry, read by next_phase as good_ran_dry. The Spot Check step is history too: the app's "
            "spotCheck (a check running: the phase is 'spot') and doneChecked (the run's check before Done "
            "has closed, so all green reads 'done' rather than 'spot'), moved by beginSpotCheck / "
            "endSpotCheck / skipDoneCheck; the harness's AutopilotFlow spot_check / done_checked, moved by "
            "begin_spot_check / end_spot_check / skip_done_check, read by next_phase as checking / "
            "done_checked. Whether the check before Done runs is done_check_due, which both call. "
            "Done's branch also opens on the labels line's level (#4878): the app's checkPhaseTransition reads "
            "levelMet (the labeling status's level.met, from the library's stop_level_reading) beside the three "
            "lights, and every light green or the level met is that branch; the harness's AutopilotFlow, under its "
            "default stop_rule 'lights|level', hands next_phase every light green when stop_lights says the rule "
            "stops, reading stop_level_met on the step's labels line. And on the click cap (#4903): the app's "
            "checkPhaseTransition opens that branch once the votes reach STOP_CAP_VOTES (atCap); the harness's "
            "AutopilotFlow passes stop_lights at_cap once they reach its stop_cap (STOP_CAP by default), and "
            "stop_lights says stop under every rule. The two numbers are pinned literally by "
            "TestPortedConstants and the service spec."
        ),
        divergence=(
            "checkPhaseTransition also latches doneReached (#4621): once a run reaches done, the "
            "panel keeps its Done step checked while the phase falls back to hard or new. It is "
            "computed from nextPhase and never feeds it, so the phase sequence - all next_phase "
            "reproduces - is unchanged and the harness, which draws no panel, has nothing to port. "
            "The app decides the check before Done once the sort after the vote settles; the harness "
            "decides it the step the flow reaches 'spot' (_settle_done_check), so a skipped check never "
            "stamps a row. The app's check, of either kind, deals its first pick at the advance after the "
            "step begins (#5136): the item already on screen, the previous step's pick, is voted first, as "
            "a phase change waits for the next advance; the harness deals the check's round at the step "
            "after the one that made it due, one vote sooner. The harness also skips a check whose opening bands do not fit the voting "
            "budget left, as it starts no prompted check that cannot: the app has no budget. And the "
            "harness's end-of-budget check (spot_check 'end' and 'weak', rows with phase 'check' past "
            "max_steps) is its measurement of the final line, which no app user can start since the "
            "Check button went: it is never a click. The app also latches doneChecked when a run's first "
            "labeling-status reading already says Done, every light green or the level met (the Train view "
            "re-entered on a finished detector), and when its first labelset reading already holds the cap "
            "(noteInitialLabelset); a harness run starts with no votes, where none can hold, so "
            "it has nothing to port. A run that reaches the cap still in its opening (cappedInOpening, #4903) "
            "is told Autopilot has stopped and offered the re-sort prompt, but its phase stays the opening's: "
            "display only, like doneReached, so next_phase has nothing to port; the harness's "
            "capped_in_opening reads the same condition for a reader asking where such a session stopped. "
            "The harness's early check (begin_spot_check('early'), early_check=N, #4664) "
            "is an experiment arm the app does not run: a check at a fixed vote count, which latches nothing."
        ),
    ),
    Mirror(
        id="autopilot.vote_targets",
        app=f"ts:{AUTOPILOT_TS}::const INITIAL_STATE",
        harness="vtscore/eval/autopilot_flow.py::GOOD_TARGET,BAD_TARGET,MORE_TARGET,MORE_DRY_RUN",
        kind="ported",
        note=(
            "goodToStart / badToStart / moreToStart / moreDryRun are copied as GOOD_TARGET / "
            "BAD_TARGET / MORE_TARGET / MORE_DRY_RUN, which decide how many votes the simulation "
            "spends before its first learned sort. Pinned literally "
            "by tests_lib/detectors/test_autopilot_flow.py::TestPortedConstants. Both harness "
            "constants are named here, because the mirror is the pair: watching only GOOD_TARGET "
            "would leave a change to BAD_TARGET alone as silent as the app-side half used to be."
        ),
    ),
    Mirror(
        id="autopilot.phase_sort_select",
        app=f"ts:{LABEL_VIEW_TS}::onAutopilotStop(",
        harness="vtscore/eval/al_strategies.py::_select_phase_faithful",
        kind="ported",
        note=(
            "Which Sort and which Select each Autopilot phase drives - good=text+top, "
            "bad=text+hard, more=text+top, hard=learned+hard, new=learned+new, and spot (stopped inside the "
            "Spot Check step) learned+hard, the harness's fallback branch. The simulated user picks the "
            "next item from exactly this pairing, so re-pointing a phase at a different sort "
            "or select here silently changes what every study's vote order means. The load- "
            "bearing row is `bad`: it is still on the TEXT sort, so the harness must not "
            "consult detector scores there even though a model exists for measurement."
        ),
        divergence=(
            "onAutopilotStop applies the mapping when the user stops Autopilot; the live "
            "mapping is the phase subscription in the same component, which additionally "
            "kicks off the sort request. This anchor is the one place the whole table is "
            "written out in one block, so it is what the digest watches. On a document dataset "
            "(stop_rule 'dry_run', #4488) 'more' and 'done' draw off learned+top; the harness "
            "does not model that row, because vtscore.eval simulates photo datasets. Documents are "
            "simulated by scripts/experiments/fullmarks/sota_documents.py, which clicks the top of "
            "the retrained structural ranking at every step. The harness's more_walk='detector' (#4637) "
            "is an experiment arm that walks a photo dataset's 'more' on the detector's top; the "
            "default, 'seed', is the app's row. Likewise new_walk='hard' (#4671) is an arm that takes "
            "the Hard pick in 'new' instead of the atlas's; the default, 'atlas', is the app's row."
        ),
    ),
    Mirror(
        id="autopilot.auto_select_next",
        app=f"ts:{AUTO_SELECT_TS}::export function autoSelectNext(",
        harness="vtscore/eval/al_strategies.py::_hard_pick_by_index",
        kind="ported",
        note=(
            "The app's auto-advance rule - which item each Select mode shows next. `top` takes "
            "the highest-ranked unvoted row; `hard` takes the unvoted row nearest the "
            "acquisition cut BY RANK INDEX, not by score; `new` defers to the coverage atlas. "
            "_hard_pick_by_index is the `hard` branch verbatim, and it is the branch every "
            "simulated vote after the seed phase goes through, so a change to the rule that "
            "does not reach the harness silently re-points every study's vote order. Rank space "
            "is the load-bearing detail: a score-space argmin biases toward whichever side of "
            "the line is denser, which is the whole reason the app measures in indices. Note "
            "the cutoff index is computed over the FULL window, voted rows included, so it does "
            "not slide as votes accumulate - the harness's `ordered` list must stay the full "
            "ranking rather than the pool. Also re-check the tie rule: both sides scan "
            "ascending by index with a strict `<`, so an exact tie takes the higher-ranked row. "
            "Extracted out of the label-view component (#3428) so this digest covers the pick "
            "rule alone rather than the component's sort plumbing; "
            "frontend/src/app/utils/auto-select-next.spec.ts states the same cases executably."
        ),
        divergence=(
            "The harness mirrors only the `hard` branch here. `top` is trivial and is "
            "reproduced inline by the phase-faithful strategy; `new` is not ported at all - "
            "_atlas_next DELEGATES to CoverageAtlas.next_sample, the same call the app's New "
            "pick makes through /api/coverage-atlas/next, so it cannot drift and needs no pin. "
            "The app's guard that a `new` pick still requires a loaded ranking is a UI "
            "precondition (the window steers the probe's scores) with no harness counterpart, "
            "as is `excludeId`: it covers the instant between casting a vote and that vote "
            "landing in goodVotes/badVotes, which a simulation never observes because it "
            "records the vote before asking for the next pick."
        ),
    ),
    Mirror(
        id="check.schedule",
        app="py:vtscore.training.thresholds.spot_check.balance_schedule",
        harness="scripts/experiments/calibration/analyze_line_estimate_4383.py::floor_schedule,kept_count,band_edges,BASE_FINE,M_PICKS",
        kind="ported",
        note=(
            "Where the band walk starts and what a band costs (#4388, the owner's ruling on "
            "#4383; the balance's cap since #4413): the starting candidate is the precision "
            "floor's schedule count for the preset the balance leans toward - K(P) = "
            "32 * 2**max(0, floor(log2(0.5 / P))) (the #4267 ruling) at P = 50% for beta <= 1 "
            "(32) and at P = 10% above it (128) - over the bands 8 / 16 / 32 / 64 / ... "
            "(band_edges at BASE_FINE), with 5 picks a band (M_PICKS). The owner priced the "
            "walk from these counts (grow-fine on the #4220 precision frames, "
            "docs/experiments/2026-09-30-line-estimate-4383/REPORT.md; the F-beta stop in "
            "docs/experiments/2026-10-01-fbeta-line-4411/REPORT.md, which reads the cap through "
            "kept_count); tests_lib/sorting/test_spot_check.py pins (32, 3, 5) and (128, 5, 5) "
            "literally. The eval harness itself delegates to the library (voting_iterations "
            "calls balance_schedule through SpotCheck and balance_count), so the pair pinned "
            "here is library vs. the priced reference."
        ),
        divergence=(
            "INTENTIONAL: the reference is keyed by the floor P it was priced at (floor_schedule, "
            "which also censuses each band at P >= 1); the library keys the same two counts by "
            "beta and validates beta in [0.25, 4]."
        ),
    ),
    Mirror(
        id="check.likely_range",
        app="py:vtscore.training.thresholds.spot_check.likely_range",
        harness="scripts/experiments/state_of_app/check_range_4483.py::hg_bounds",
        kind="ported",
        note=(
            "The bound one part of a checked set carries (#4483, the owner's ruling of 2026-10-15; "
            "before it #4272's Clopper-Pearson with the tail split over the bands): the exact "
            "hypergeometric bound for a part of known size from the picks inside it, each tail at "
            "alpha (range_tail no longer splits), exact (s / K) once the picks cover the part, 0-100% "
            "with none. The parts are summed size-weighted over the set the line kept when the check "
            "started (SpotCheck.range_over). #4483 priced it on 3,568 real closing checks and simulated "
            "checks at clicks 25-100: it held the truth in at least 98.9% of checks in every regime, "
            "at 0.55-0.76 of the Clopper-Pearson width. If the bound, the tail or the census rule "
            "moves on either side, move the other."
        ),
        divergence=(
            "INTENTIONAL: the reference returns a band's (lo, hi) share for the pricing's own sums and "
            "takes a zero-pick band as 0-100% before scipy; the library returns a LikelyRange record "
            "through hypergeometric_lower / hypergeometric_upper, which search only the counts the "
            "picks leave possible (the same bound)."
        ),
    ),
    Mirror(
        id="check.bands",
        app="py:vtscore.training.thresholds.spot_check.SpotCheck",
        harness="scripts/experiments/calibration/analyze_line_estimate_4383.py::draw_audits,Audits",
        kind="ported",
        note=(
            "What the band walk audits (#4388, the owner's ruling on #4383): 5 picks drawn "
            "uniformly from each band, never twice from one band; the set under test is the top "
            "b bands and its share right is band-stratified (each band's share weighted by its "
            "size, Audits.union_estimate). SpotCheck is the app's live state machine over one "
            "fixed unvoted ranking; draw_audits / Audits are what the rule was priced with. If "
            "either changes the bands, the picks a band or the estimate, the other has to follow "
            "or the pricing no longer describes the shipped check. Where the walk stops is the "
            "check.walk mirror's."
        ),
        divergence=(
            "INTENTIONAL: the app's check can end 'cancelled' (the user closed the step), "
            "carries a fingerprint for the stale flag, reports its state to a client, and draws "
            "one band at a time as the walk reaches it; the reference draws every band's picks "
            "up front from planted labels (the same picks the walk would touch) and asks no user. "
            "Since #4483 the app's check also records the count the line kept when it started "
            "(kept) and bounds that set from its picks (parts_over / range_over), which the "
            "reference, an estimate of the walk's own set, does not model."
        ),
    ),
    Mirror(
        id="check.walk",
        app="py:vtscore.training.thresholds.spot_check.SpotCheck",
        harness="scripts/experiments/calibration/analyze_fbeta_line_4411.py::rule_fb_walk,_edge_tp",
        kind="ported",
        note=(
            "Where the band walk stops (#4413, priced in #4411): the F-beta estimate of the top "
            "b bands - their audited positives (_edge_tp) over the mixture's count of the "
            "ranking's positives - and the walk deeper while it rises, shallower from a start "
            "whose first deeper step does not, ties to the smaller set; a walk that starts on "
            "the ranking's last band steps shallower at once (#4424). If either side changes "
            "the stop, the other has to follow or the pricing no longer describes the shipped "
            "check."
        ),
        divergence=(
            "INTENTIONAL: the app's walk carries #4427's arms (picks, tol, fine, guard and the "
            "shallower-only walk), all off by default save the shallower-only walk that "
            "check_shape turns on above beta 1, and its shape decides whether the walk's end "
            "moves the line (balance_count). Where the mixture gives no count of at least one "
            "positive, the app's walk reads recall against the balance's cap lowered to the "
            "unvoted count (walk_positives, #4419), where rule_fb_walk falls back to the "
            "audits' own tp for no fit (which leaves its F-beta a function of precision alone) "
            "and keeps a count below one as it is."
        ),
    ),
    Mirror(
        id="thresholds.balance_default",
        app="py:vtscore.state.__init__.line_knobs",
        harness="vtscore/training/thresholds/spot_check.py::resolve_line_knobs",
        kind="default",
        note=(
            "What balance a detector's line is drawn at when nobody pinned one (#4413): the app's "
            "line_knobs hands the trainer the detector's beta, seeded from the user's setting "
            "(DEFAULT_BETA when unset); the harness's resolve_line_knobs resolves beta=None to the "
            "same constant. The values cannot drift - both sides read the same constants, and "
            "tests/sorting/test_balance_routes.py pins UserSettings' defaults against the wire - so "
            "this digest watches the *resolution*: if the app's preference starts depending on "
            "something else (the dataset, the embedder, a per-detector default), that has to reach "
            "the harness too. The per-detector balance of #4665 is not such a default: it is the "
            "user's own pick, asked for when the detector is created, and the form starts on the "
            "user's setting - DEFAULT_BETA when unset - so a detector nobody chose a balance for "
            "still resolves to the harness's default. Every harness entry point that draws a line "
            "reaches the default through it - voting_iterations, al_benchmark, and since #4971 "
            "runner.eval_learned_sort (python -m vtscore.eval --mode learned) - so a caller that "
            "skips it is caught by that caller's tests, not by this digest."
        ),
        divergence=(
            "INTENTIONAL: the harness accepts 'off' (the Inclusion arm) and a pinned beta where the "
            "app has per-user settings; the DEFAULT arm passes None and resolves here. The app's own "
            "'no balance' is a library caller's CoreConfig(beta=None), which line_knobs hands on as "
            "beta None - the harness's 'off'; the app itself always sets a balance."
        ),
    ),
    Mirror(
        id="autopilot.startup_default",
        app=f"ts:{AUTOPILOT_TS}::const INITIAL_STATE",
        harness="vtscore/eval/startup_schedule.py::PRODUCTION_STARTUP",
        kind="default",
        note=(
            "issue #3267 made the Autopilot opening a parameter, so the harness now has a "
            "spelling of the app's own opening - 'g3@top,b4@mid' again since #4740 (from #4282 "
            "to #4740 it was 'g3@top,b4@mid,g20+dry1/16@top', the More walk) - that a study's control arm "
            "runs. If goodToStart/badToStart move, or the opening stops being 'top of the "
            "seed sort then its cutoff', this constant has to move with them or every #3267 "
            "study measures its deviations from an opening nobody ships. "
            "tests_lib/detectors/test_startup_schedule.py pins it two ways: literally against "
            "GOOD_TARGET/BAD_TARGET, and behaviourally by requiring it to reproduce a "
            "default-arm run click for click."
        ),
        divergence=(
            "Since #4731 the app's Good phase also ends when its walk runs dry - moreDryRun picks in "
            "a row without a Good, with a Good in hand - and the More walk is then spent. The "
            "schedule grammar cannot say that: a round's '+dry<m>/<w>' stop fires with no Good in "
            "hand, and a later round never learns an earlier one ran dry. So PRODUCTION_STARTUP is "
            "the app's opening only for a session whose Good walk finds its third Good before 16 "
            "misses in a row; the default arm (no schedule) is the app in every session. A #3267 "
            "study comparing schedules against PRODUCTION_STARTUP compares openings that both lack "
            "the dry exit."
        ),
    ),
    Mirror(
        id="thresholds.text_sort_acquisition",
        app="py:vtscore.training.thresholds.gmm.text_sort_cuts",
        harness="vtscore/eval/al_strategies.py::_sort_threshold",
        kind="default",
        note=(
            "Which of a typed-query sort's two lines the opening samples against (#4136). The app's "
            "text route returns text_sort_cuts: the display line as `threshold` (the guarded rule "
            "by default, #3826) and the mixture midpoint as `acq_threshold`, and the Hard select "
            "reads `acq_threshold ?? threshold`. The harness's _sort_threshold reads "
            "text_sort_acquisition_threshold for a typed query, and startup_schedule's `@mid` cut "
            "does the same, so the Bad phase and the `b4@mid` round sample at the midpoint whatever "
            "VTSEARCH_TEXT_SORT_CUT paints green. That split is the whole point: with the guarded "
            "line as the sampling position the #3826 A/B put the Bad phase at the 2.5th percentile "
            "of the ranking and the first detectors were worse at every beta. If text_sort_cuts "
            "starts putting anything but the midpoint in acq_threshold, or _sort_threshold starts "
            "reading the display line, the other side has to move with it. "
            "tests_lib/sorting/test_text_sort_threshold.py pins the identity bit for bit and runs "
            "the harness's opening under both rules to show the picks do not move. Since #4603 the "
            "display line also takes the balance (by count at beta <= 1); the acquisition cut does "
            "not, so the harness's opening is unaffected, and text_baseline.py scores each preset's "
            "display line for the review's click 0."
        ),
        divergence=(
            "The app side is the library function the route calls, not the route: "
            "vtsearch/routes/sorting.py only forwards the pair into windowed_sort_response, and "
            "the frontend's `acq_threshold ?? threshold` fallback is pinned by the "
            "autopilot.auto_select_next mirror's TypeScript side."
        ),
    ),
    Mirror(
        id="progress.smart_status",
        app="py:vtscore.detectors.labeling_progress._compute_smart_status",
        harness="vtscore/eval/autopilot_flow.py::smart_status,smart_detail",
        kind="ported",
        note=(
            "The Smart indicator - has F-beta stopped rising? (issue #5134; FPR + FNR going flat "
            "before) - one of the three gates the phase machine reads. Both sides are one-line "
            "wrappers over `vtscore.detectors.cost_trend.rising_status_from_scores` (issue #3832's "
            "test with its sign turned over), so the rule itself - the per-class minimum, the "
            "flatness threshold, and the slope's significance against the window's own scatter - "
            "cannot drift; what this pin watches is the plumbing around it. If either wrapper grows "
            "a step of its own, move that step into the shared module instead. `smart_detail` is "
            "the harness's whole-dict entry point (issue #3560) and is pinned beside "
            "`smart_status`, which is a one-liner over it."
        ),
        divergence=(
            "The harness takes the window of F-beta points as an argument instead of reading a "
            "`_ProgressCache`'s `steps` model cache, which is built for one interactive detector "
            "advancing a vote at a time. The *rule* is shared rather than copied; only the input "
            "plumbing differs - and only in where the models come from, not in how they are "
            "scored: the caller (`step_trainers._labelset_window_readings`) re-scores the whole "
            "window against the *current* labelset every step, as `_eval_cached_models` does. "
            "Handing in a history of frozen per-step points instead would silently change the "
            "statistic the slope measures (issue #2923), which is not a declared divergence. "
            "The F-beta arithmetic underneath both is `cost_trend.best_fbeta`, shared, at "
            "`cost_trend.smart_beta` of the session's balance. The harness also keeps the A/B's "
            "other readings as arms (`AutopilotFlow.smart_signal`, #5134): `cost` (the app before "
            "#5134), `ap`, `fbeta`, `heldout_*`, `line`; the default arm is `fbeta_best`, the app's. "
            "Two consequences of #3757 are declared here rather than pinned. (1) The app windows "
            "its last SMART_WINDOW *models*, because most of its label steps have no model - only "
            "the ones a learned sort ran against do; the harness trains one per step, so its "
            "last-SMART_WINDOW-steps slice is the identical set and the `voting_iterations` caller "
            "says so at the `del recent_steps[:-SMART_WINDOW]`. (2) Both sides score a model in "
            "its own serving geometry - the app through `scoring_rows_for_snap`, the harness "
            "through `score_sim_set_with_model` on the arm's style - rather than on whole-image "
            "vectors a patch head was never fitted on. Those live in the scorers, not in the rule "
            "pinned here, so a change to either scorer trips the `progress.eval_geometry` mirror "
            "below instead."
        ),
    ),
    Mirror(
        id="progress.eval_geometry",
        app="py:vtscore.detectors.labeling_progress._score_step",
        harness="vtscore/eval/step_trainers.py::_labelset_window_readings",
        kind="ported",
        note=(
            "How a cached detector is *scored* when the Smart indicator measures it. Both sides "
            "must score a head the way that head is served - max-pooled over the rows it was "
            "fitted against - or the indicator measures a geometry nobody ships, which is half "
            "of issue #3757. If you change which rows either side scores, or the pooling over "
            "them, change both. Each point is the model's best F-beta over every cut "
            "(`cost_trend.best_fbeta`, #5134) on both sides, so neither reads a cut."
        ),
        divergence=(
            "The app reaches the rows through `scoring_rows_for_snap`, which is bound to the live "
            "dataset context and its cached region matrix; the harness has arms rather than one "
            "shipped geometry, so it routes through the arm's style object "
            "(`score_sim_set_with_model`) and keeps the whole-image `predict` for the whole_image "
            "/ SVM arms, whose heads are fitted in that space. Both reduce to the same rule - "
            "score a head where it is served - and both delegate the arithmetic on top to "
            "`best_fbeta`. The harness reads the A/B's other units of the same scores beside it "
            "(`ap`, `fbeta` at the model's own Smart cut), which the app does not."
        ),
    ),
    Mirror(
        id="progress.eval_cached_models",
        app=(
            "py:vtscore.detectors.labeling_progress."
            "_eval_cached_models,_model_step_indices,_build_eval_rows,_smart_status_uncached"
        ),
        harness="vtscore/eval/step_trainers.py::_labelset_window_readings",
        kind="ported",
        note=(
            "Which models the Smart indicator re-scores, against which eval set, and at which "
            "balance: the input plumbing progress.smart_status declares and progress.eval_geometry "
            "sits inside (#5005). The app windows its last SMART_WINDOW model-bearing steps "
            "(`_model_step_indices`, sliced in `_smart_status_uncached`), builds one eval set from "
            "the current votes' scoring rows (`_build_eval_rows`), and reads every model's best "
            "F-beta at the session's balance (`_eval_cached_models`, `cost_trend.smart_beta`, "
            "#5134). `_labelset_window_readings` re-scores the window its caller hands it against "
            "the current votes at the beta it is passed. This is the plumbing that drifted once "
            "(#2923): re-scoring the whole window against the *current* labelset is the statistic, "
            "and a point read at another balance than the session's moves every point of the "
            "slope. If either side changes which models are in the window, what they are scored "
            "against, or the balance they are read at, change the other. How a model is scored is "
            "progress.eval_geometry's."
        ),
        divergence=(
            "The harness's window is sliced, and its balance chosen, by its caller: "
            "`voting_iterations` keeps `del recent_steps[:-SMART_WINDOW]` and passes the run's "
            "beta, inline in a function too large to digest, so a change to either is caught by "
            "tests_lib/detectors/test_smart_signal_5134.py, not here. It trains a model every step, "
            "so its last SMART_WINDOW steps are the app's last SMART_WINDOW models "
            "(progress.smart_status, consequence (1)). The app reads the beta from the request "
            "(`line_knobs()`, snapshotted for its workers) and keys its Smart memo on it; a harness "
            "run has one balance. Three smaller differences: the app rounds each point to four "
            "places (`_score_step`, which also feeds the chart) and regresses on the rounded value, "
            "where the harness keeps the float, which can move the light only on a window within a "
            "rounding step of one of the rule's thresholds; the app returns its own yellow, with "
            "its own reason, while fewer than SMART_MIN_POINTS steps carry a model, where the "
            "harness leaves that test to `rising_status_from_scores`, whose answer is the same "
            "yellow; and the harness returns empty lists for a labelset missing either class, "
            "which the app never scores because `_smart_status_uncached` stops at MIN_PER_CLASS "
            "first. The `cost` arm's window (`_labelset_error_costs`, the app before #5134) is no "
            "longer a mirror: nothing in the app prices FPR + FNR."
        ),
    ),
    Mirror(
        id="progress.stable_status",
        app="py:vtscore.detectors.labeling_progress._compute_stable_status",
        harness="vtscore/eval/autopilot_flow.py::stable_status,stable_detail",
        kind="ported",
        note=(
            "The Stable indicator - confident prediction-flip rate over the whole pool, plus the "
            "flip-rate plateau test (issue #3831). Both sides are one-line wrappers over "
            "`vtscore.detectors.stability.stable_status_from_entries`, so the rule itself cannot "
            "drift; what this pin watches is the plumbing around it. If either wrapper grows a "
            "step of its own, move that step into the shared module instead. `stable_detail` is "
            "the harness's whole-dict entry point (issue #3560), pinned beside `stable_status` "
            "for the reason given on progress.smart_status: the app returns the whole "
            "sub-object, and the harness keeping the three flip rates is convergence, not "
            "divergence."
        ),
        divergence=(
            "Same input plumbing divergence as progress.smart_status: flip counts are passed "
            "in rather than read from a `_ProgressCache`'s `steps`. That plumbing is "
            "progress.step_stability's."
        ),
    ),
    Mirror(
        id="progress.step_stability",
        app="py:vtscore.detectors.labeling_progress._compute_step_stability",
        harness="vtscore/eval/autopilot_flow.py::AutopilotFlow.record_step",
        kind="ported",
        note=(
            "The Stable indicator's inputs: one `stability_entry` per retraining (#5005). The app "
            "scores the pool in serving geometry, drops the ids labeled so far, cuts at the served "
            "line, and counts flips against the previous model's snapshot with the whole pool as "
            "the denominator; `record_step` takes the still-unlabeled pool's scores, the cut and "
            "`num_pool` from its caller and counts the same way. The counting itself is "
            "`vtscore.detectors.stability.stability_entry`, shared, so what this pins is what each "
            "side hands it: which items, at which cut, against which baseline, over which "
            "denominator. If either side changes one of those, change the other. `record_step` also "
            "replaces the Smart window with the costs it is handed; that half is the harness end of "
            "progress.eval_cached_models, pinned here because it is one method."
        ),
        divergence=(
            "The app rebases its baseline when the balance changes (#4745): a snapshot drawn at "
            "another beta becomes the new baseline instead of being compared against. "
            "`record_step` has no balance argument, because a harness run draws every step's line "
            "at the run's one balance, so the rebase could never fire. Its scores are handed in: "
            "`voting_iterations` scores the remaining pool in the arm's geometry "
            "(`step_trainers._score_pool`) and passes `num_pool=len(sim_ids)`, the haystack with "
            "labeled items included, as the app divides by its whole scored pool. The app records "
            "an entry only for a label set a sort trained a model on, and keeps its baseline across "
            "the steps between (`_resolve_step_model`); the harness trains every step, so it has no "
            "such gap. At the edges: the app drops its baseline when a model arrives with no pool "
            "to score (a sort racing the caller's decision to skip the pool), where "
            "`record_step(scores=None)` keeps it, and no caller passes None; and once the pool is "
            "exhausted the app records a zero-flip entry even with no baseline, where the harness "
            "records one only once it has a baseline. "
            "#5134's arm: `record_step` also folds `windows` and `history`, the Smart candidates in "
            "the objective's units (AutopilotFlow.smart_signal), which the app does not read. The "
            "default arm passes neither (smart_readings off, smart_signal 'cost'), and with them the "
            "Stable entry and the cost window are untouched."
        ),
    ),
    Mirror(
        id="progress.span_status",
        app="py:vtscore.detectors.labeling_progress._compute_span_status",
        harness="vtscore/eval/autopilot_flow.py::span_status,span_target",
        kind="ported",
        note=(
            "The Span indicator - coverage-atlas breadth - which drives the new -> done "
            "transition. Re-check the green target and the yellow cutoff. `span_target` is "
            "pinned beside it because the harness's green bar was extracted into it (issue "
            "#3560) - it is the app's own `green_at = min(SPAN_GREEN, tree_total)`, split out "
            "so that a study *reporting* how far short a run fell reads the same number the "
            "light was decided by. The degenerate-tree branch still short-circuits ahead of "
            "it on both sides."
        ),
        divergence=(
            "The app reads its green target from the `autopilot_goal_diversity` setting "
            "(`get_setting`); the "
            "harness takes it per-run so a sweep can vary it, defaulting to the same value. "
            "Both sides *report* the resolved bar: the harness as `span_target` on every "
            "emitted row, the app as the indicator's `target` key, which the autopilot panel "
            "paces its Diversity light against (issue #4319). Reporting only, not a difference "
            "in the rule."
        ),
    ),
    # ----------------------------------------------------------------- default
    Mirror(
        id="training.train_and_threshold",
        app="py:vtscore.detectors.training.train_and_threshold",
        harness="vtscore/eval/step_trainers.py::_style_train_and_calibrate",
        kind="default",
        note=(
            "The app's canonical train + calibrate pipeline, which the harness reproduces step "
            "for step (fold calibration, full-data fit, fold-anchored threshold). A new stage "
            "here - or a changed fold rule or ordering - has to reach the harness or its "
            "default arm trains a detector the app no longer ships. Note "
            "_app_train_and_calibrate is the single-vector path and reproduces the same shape. "
            "The head is the one knob mirrored by name: this function's `hidden_dim` "
            "(LINEAR_SVM_PRIOR_HEAD since #4785) must equal "
            "`resolve_hidden_dim(step_model.PRODUCTION_HEAD, ...)`, which "
            "tests_lib/detectors/test_harness_linear_head.py pins by training this pipeline for "
            "real - so a head change fails the suite as well as tripping this digest. Since #4269 "
            "the app reads no stored Inclusion: its conformal cut is at "
            "PRECISION_FLOOR_FALLBACK_INCLUSION (0). The harness keeps `inclusion` as an arm knob "
            "whose default, 0, is that cut. Since #4413 the app draws the line at the detector's "
            "balance and hands its beta to _fused_threshold - since #4665 *det_ctx*'s own "
            "(vtscore.state.detector_beta: the balance the detector keeps, else the user's), "
            "`line_knobs` only with no context; the harness's default arm is the "
            "balance's at DEFAULT_BETA (resolve_line_knobs, the thresholds.balance_default mirror). "
            "The harness trainer takes no beta, so where the app reads it is not a stage it copies."
        ),
    ),
    Mirror(
        id="acquisition.balance_cut",
        app="py:vtscore.state.core.detector_acquisition_threshold",
        harness="vtscore/eval/voting_iterations.py::resolve_acquisition_factor",
        kind="default",
        note=(
            "Where Autopilot's Hard / New picks sample under a balance (#4409, shipped from the pricing in "
            "docs/experiments/2026-10-01-acquisition-fbeta-4409): the app's detector_acquisition_threshold "
            "takes the beta and returns the library's acquisition_threshold - the score at "
            "ACQUISITION_ARGMAX_FACTOR of the F-beta argmax's depth over the unvoted ranking - falling "
            "through to the line - 4 re-cut with no mixture estimate. The harness's resolve_acquisition_factor "
            "turns acq_p_crossing=None into that same constant under a balance (and None, the offset cut, "
            "otherwise), and its step reads the same library function. The factor cannot drift - both read "
            "one constant - so this digest watches the resolution: a second input to the app's choice (the "
            "corpus size, the fit's confidence) has to reach the harness too."
        ),
        divergence=(
            "INTENTIONAL: the harness accepts 'off' (the offset cut under a balance, the pricing's control) "
            "and a pinned factor where the app has the constant; the DEFAULT arm passes None and resolves here."
        ),
    ),
    Mirror(
        id="acquisition.target_precision",
        app="py:vtscore.state.core.detector_acquisition_threshold",
        harness="vtscore/eval/voting_iterations.py::resolve_acquisition_target",
        kind="default",
        note=(
            "Where Autopilot's Hard / New picks sample under a balance since #3546 "
            "(docs/experiments/2026-10-07-acquisition-cut-3546): the app's detector_acquisition_threshold "
            "reads the library's target_precision_threshold off ctx.labels_line at "
            "ACQUISITION_TARGET_PRECISION - the score where the line's corpus posterior falls below 0.5 - "
            "and falls through to the line - 4 re-cut with no labels line. The harness's "
            "resolve_acquisition_target turns acq_target_p=None into that same constant under a balance "
            "(and None, the offset cut, otherwise), and its step reads the same library function on the "
            "step's labels line (details['find_line']), with the offset cut as the same fallback. The "
            "target cannot drift - both read one constant - so this digest watches the resolution."
        ),
        divergence=(
            "INTENTIONAL: the harness accepts 'off' (the offset cut under a balance, the #3546 control and every "
            "offset arm) and a pinned target where the app has the constant; the DEFAULT arm passes None."
        ),
    ),
    Mirror(
        id="training.fused_threshold",
        app="py:vtscore.detectors.training_calibration._fused_threshold",
        harness="vtscore/eval/voting_iterations.py::_safe_threshold_for_step",
        kind="default",
        note=(
            "How the cross-calibration cut and the population estimate are fused into the "
            "shipped threshold. The harness's reported operating point is only comparable to "
            "the app's if this rule matches. Since #4272 this is where the line is drawn under "
            "a preference: the set the balance keeps, through the shared balance_line over a "
            "LineRanking built from the final model's scores with the voted items marked - "
            "delegated, so which line a balance draws cannot drift; what this digest watches is the "
            "ranking's inputs (every scored item, unscorable ones dropped, the trainer's voted set), "
            "which the harness has to build the same way, the mixture's proposal for the unchecked "
            "line (#4389, #4413: fbeta_count on that ranking, anchored on the labels the step "
            "trained on, handed to balance_line and balance_state as `proposal`, which the harness "
            "has to hand the same way), and the order of the fallbacks under it "
            "(the fold-anchored cut at PRECISION_FLOOR_FALLBACK_INCLUSION with no balance, the "
            "schedule blend with no fitted cut). Neither side builds the #4220 precision estimate "
            "any more (#4362): it stopped drawing the line in #4272 and lost its last reader, "
            "the Find Stats curve, in #4360. Since #4269 the app hands reporting_line "
            "PRECISION_FLOOR_FALLBACK_INCLUSION (0) rather than a stored Inclusion; the harness "
            "hands it the arm's `inclusion`, whose default is that 0. Since #4452 the line under a balance "
            "is the labels' line (_labels_line over fit_labels_line, cut at `beta`) when the retrain fitted "
            "a class model, and the fallbacks otherwise; the harness's default arm passes DEFAULT_BETA and "
            "draws the same line in _preference_line_for_step. The count line (balance_line over "
            "fbeta_count's proposal) survives only in the harness, as the forced-check-shape arm. The "
            "harness's labels_scale (#4910) carries each fold's held-out scores onto the final model's scale "
            "through that fold's haystack before the class model is fitted, an arm the app does not take; "
            "its default, None, pools them as scored, as _labels_line does."
        ),
        no_harness_pin=(
            "The harness side is _safe_threshold_for_step, the whole production-threshold path (150 lines, named "
            "by three mirrors, and carrying the arm knobs and per-fold timing that no mirror is about). Its "
            "digest would trip on edits with nothing to do with this mirror, and a pin people re-run --update on "
            "without reading is worse than no pin. The app-side digest still covers the direction that matters "
            "here. When this blind spot starts to matter, the fix is to extract the reproduction into its own "
            "helper - what #3403 did for the two *_default mirrors, giving them a harness side small enough to "
            "pin - not to digest the whole function."
        ),
    ),
    Mirror(
        id="training.calibration_score_rows",
        app="py:vtscore.detectors.training_calibration._calibration_score_rows",
        harness="vtscore/eval/voting_iterations.py::_safe_threshold_for_step",
        kind="default",
        note=(
            "Calibrating in *inference* geometry: each voted bag collapses over the rows the "
            "scorer will max-pool, not the rows the fold model trained on. Changing which rows "
            "the app calibrates over silently moves every threshold the harness reports."
        ),
        no_harness_pin=(
            "The harness side is _safe_threshold_for_step, the whole production-threshold path (150 lines, named "
            "by three mirrors, and carrying the arm knobs and per-fold timing that no mirror is about). Its "
            "digest would trip on edits with nothing to do with this mirror, and a pin people re-run --update on "
            "without reading is worse than no pin. The app-side digest still covers the direction that matters "
            "here. When this blind spot starts to matter, the fix is to extract the reproduction into its own "
            "helper - what #3403 did for the two *_default mirrors, giving them a harness side small enough to "
            "pin - not to digest the whole function."
        ),
    ),
    Mirror(
        id="training.blend_schedule_default",
        app="py:vtscore.detectors.training_calibration._blend_schedule_for_snap",
        harness="vtscore/eval/voting_iterations.py::_resolve_production_defaults",
        kind="default",
        note=(
            "How the app picks a safe-threshold blend schedule when none is named (per voting "
            "mode). _resolve_production_defaults resolves blend_schedule=None through "
            "production_schedule_for to match; if the app's choice becomes conditional on "
            "something else, that condition has to reach the harness too."
        ),
    ),
    Mirror(
        id="training.split_fraction_default",
        app="py:vtscore.detectors.training_calibration.resolve_calibration_fraction",
        harness="vtscore/eval/voting_iterations.py::_resolve_production_defaults",
        kind="default",
        note=(
            "How the app resolves calibration_fraction when the user has no explicit setting "
            "(#3287/#3290): the per-SPACE production split - 0.3 when the detector learns in a "
            "single-vector space, 0.5 on a patch grid, 0.5 when unknown - keyed on the "
            "embedder's supports_patch_regions capability, NOT on the voting mode. "
            "_resolve_production_defaults resolves calibration_fraction=None through the same "
            "production_split_for table, keyed on whether any media carries a patch_grid (the "
            "harness's spelling of 'built by a patch embedder'). The values themselves cannot "
            "drift - both sides read PRODUCTION_SPLIT_BY_SPACE - so what this digest watches is "
            "the *predicate*: if the app's choice starts depending on something patch_grid "
            "presence can't see (the voting mode, the dataset, a per-user knob), that condition "
            "has to reach the harness too."
        ),
    ),
    Mirror(
        id="thresholds.vote_exclusion_floor",
        app="py:vtscore.training.thresholds.anchored.resolve_exclusion_floor",
        harness="vtscore/eval/voting_iterations.py::_safe_threshold_for_step",
        kind="default",
        note=(
            "How the #3308 voted-media exclusion resolves its remainder floor when no override "
            "is given (#3312). The exclusion DECISION itself is not mirrored and must never "
            "become mirrored: both sides call apply_vote_exclusion / drop_voted, so the "
            "all-or-nothing contract and the filtering are shared code rather than two copies. "
            "What this digest watches is the *default*: the harness's exclusion_min_remainder "
            "arm knob defaults to None, and None has to keep meaning 'whatever a live detector "
            "does'. If the app's floor stops being one constant - keyed on the dataset size, "
            "the voting mode, a per-user setting - that resolution has to reach the harness "
            "too, or every arm in the #3312 grid is measured against a baseline nobody runs."
        ),
        divergence=(
            "INTENTIONAL: the harness accepts an explicit floor (0 = exclude unconditionally, "
            "math.inf = the pre-#3308 behaviour) where the app has no such setting. That is the "
            "#3312 arm axis and is exactly the kind of deliberate deviation the harness exists "
            "to measure; the DEFAULT arm still passes None and so resolves here."
        ),
        no_harness_pin=(
            "The harness side is _safe_threshold_for_step, the whole production-threshold path (150 lines, named "
            "by three mirrors, and carrying the arm knobs and per-fold timing that no mirror is about). Its "
            "digest would trip on edits with nothing to do with this mirror, and a pin people re-run --update on "
            "without reading is worse than no pin. The app-side digest still covers the direction that matters "
            "here. When this blind spot starts to matter, the fix is to extract the reproduction into its own "
            "helper - what #3403 did for the two *_default mirrors, giving them a harness side small enough to "
            "pin - not to digest the whole function."
        ),
    ),
    Mirror(
        id="thresholds.fold_anchored_fit_then_cut",
        app="py:vtscore.training.thresholds.anchored.fold_anchored_gmm_threshold",
        harness="vtscore/eval/arms_inclusion.py::_cut_inclusion_arms",
        kind="default",
        note=(
            "The app composes a fold-anchored threshold as fit_fold_anchored_cut(...) then "
            "cut.threshold_at(inclusion). The #2865 sweep re-cuts ONE fit per anchor weight "
            "across the whole (rule, combine, k) grid, so it calls those two halves itself "
            "instead of calling this function per grid point. The two are identical only for as "
            "long as this function has nothing BETWEEN the fit and the cut. If a stage is added "
            "there - a clamp, a smoothing pass, a provenance-dependent substitution - the sweep "
            "silently stops measuring the shipped estimator, and its rows would still look "
            "perfectly reasonable. Re-check that the composition is still fit-then-cut. "
            "Checked at #3166: the empty-interval snap that fixed the saturated-distribution "
            "threshold went INSIDE threshold_at, not between the fit and the cut, precisely so "
            "the sweep picks it up for free and this composition stayed untouched."
        ),
        divergence=(
            "INTENTIONAL: the harness does NOT reproduce this function's terminal fallbacks (the "
            "final model's unanchored midpoint, then its median) when no fold yields a fit. Both "
            "are inclusion-blind by construction, so they would enter the #2865 frame as arms "
            "that trivially lose the knob-liveness comparison while saying nothing about the cut "
            "rule under test; the sweep skips that anchor weight instead. Fitting once and "
            "re-cutting is also what production itself does for the acquisition cut and Smart's "
            "pricing (recut_detector_threshold), so the sweep measures the object the app "
            "re-cuts rather than a chain of independent retrains."
        ),
    ),
    Mirror(
        id="thresholds.rate_cut_no_root",
        app="py:vtscore.training.thresholds.gmm._rate_cut",
        harness="vtscore/eval/cut_rules.py::gaussian_cuts",
        kind="default",
        note=(
            "What the 'rate' rule does on a fit whose density crossing has no root between the "
            "component means. This has moved twice already - midpoint (pre-2026-08-06), bare "
            "edge, then continued past the edge at the rule's first-order slope (#2896) - and "
            "each move silently changed what the harness's *_rate arms mean relative to the "
            "app. Pinned so the next move prompts a decision instead of going unnoticed for "
            "days, which is exactly how #2900 happened."
        ),
        divergence=(
            "INTENTIONAL, decided in #2900: gaussian_cuts reports NaN and _safe_gmm_variant_rows "
            "substitutes that fit's MIDPOINT, where production continues past the component "
            "mean. The decomposition family compares cut rules against each other on one fit, so "
            "it wants a neutral, rule-independent stand-in - production's continuation exists "
            "only for 'rate' and would make it incomparable to its cross/priorfree siblings (at "
            "inclusion 0 it would break the rate == priorfree identity every report in "
            "docs/experiments/2026-08-04-gmm-cut/ relies on). The divergence is recorded per row in "
            "cut_fallback_kind ('midpoint' vs 'continued'/'degenerate_midpoint'), so an analysis "
            "that needs the shipped path filters on it. The fold-anchored family calls "
            "gmm_cut_from_fit directly and is the faithful stand-in for the app. When re-pinning "
            "this mirror, re-check that the divergence is still the one you want AND that "
            "cut_fallback still fires on the same fits in both families - the flag being "
            "comparable is what keeps fallback_rate aggregates joinable across them."
        ),
    ),
    Mirror(
        id="training.typed_query_prior",
        app="py:vtscore.detectors.query_head.typed_query_prior_vector",
        harness="vtscore/eval/voting_iterations.py::_fits_from_typed_query",
        kind="default",
        note=(
            "A typed-query detector's head starts from its typed query as well as from the Goods' "
            "centroid (#4384, #4859): every fit of a training pass runs inside "
            "svm_prior.typed_query_prior, whose query feature sits at QUERY_PRIOR_SCALE (10). The app "
            "enters it in labelset_train_and_score and train_from_labelset (Find, AutoFind, a load, "
            "an export, the CLI, and the Train view's learned sort, which passes its detector's "
            "registry query, learned_sort.resolve_active_text_query), with the query embedded by "
            "typed_query_prior_vector where typed_query_prior_applies. The harness's default arm "
            "runs the whole session inside it when the caller hands it the run's query_vector "
            "(run_cells.cell_query_vector; run_voting_iterations_eval's query_vectors). If the app "
            "enters the prior elsewhere, reads another space or another scale, the harness's "
            "wrapper has to move with it."
        ),
        divergence=(
            "INTENTIONAL: the app embeds the detector's text_query itself, with the user's enrich "
            "setting, on a known single-vector space with a text tower (typed_query_prior_applies); "
            "the harness is handed the cell's query vector (embedded at SEED_ENRICH, the app's "
            "default) and applies it on a typed-query run (seed_scores, no example opening) whose "
            "media carry no patch grid - the same split on every production path. CALIB_QUERY_PRIOR "
            "(query_prior_scale) is an arm: unset is the app's scale, 0 the head before #4859."
        ),
    ),
    Mirror(
        id="detectors.label_quota_default",
        app="py:vtscore.detectors.labelset_training.train_from_labelset",
        harness="vtscore/eval/voting_iterations.py::simulate_voting_iterations",
        kind="default",
        note=(
            "The label quota (#4643): train_from_labelset - Test, AutoFind, a load and the CLI - "
            "gives the Goods' centroid from the first Good until the resolved labels hold "
            "GOOD_QUOTA Goods and BAD_QUOTA Bads, or a Good and DRY_BAD_QUOTA Bads (#4731) "
            "(label_quota.quota_from_groups), the trained head from there, and nothing with no Good. "
            "A detector made from a typed query (its text_query) gives that query's own set instead "
            "of the centroid and of nothing, from its first vote until the trained head (#4758, "
            "install_query_head). The harness's default arm (label_quota=None on "
            "the app trainer) scores the withheld half the same way: a row from the first Good, "
            "the centroid's metrics under the quota (label_quota.label_quota, delegated), the "
            "head's from it; a run that opens on a typed query (seed_scores, no seed_examples) "
            "writes a row from the first vote, the typed query's set under the quota "
            "(_query_test, detector_tier 'query'; query_tier=False is the arm before). "
            "Past the quota a typed-query detector on a single-vector space keeps its query's set "
            "until QUERY_HOLD_LABELS labels (#4736, label_quota.query_holds, read by _query_tier and "
            "cached_head_is_current); the harness calls the same query_holds on its vote counts "
            "(query_hold, default QUERY_HOLD_LABELS; 0 is the arm before), writes the query's row "
            "untagged with app_trained 0 and keeps the head's rows tagged hold:head. "
            "Under the quota on a patch space, once the typed query has missed - a Good, and a Bad "
            "before the Goods quota (QUERY_MISS_BADS) - the detector gives the Goods' centroid in place of its "
            "query (#4800, label_quota.query_missed, read by _query_tier and cached_head_is_current); "
            "the harness calls the same query_missed on its vote counts (query_miss, default on; "
            "False is the arm before) and writes the centroid's row untagged there. "
            "The trained head it then fits starts from the typed query too (#4859): mirrored by "
            "training.typed_query_prior. "
            "If the app's tiers move - a new tier, a quota read from somewhere "
            "other than the counts, a tier that needs a Bad - the harness's rows have to move "
            "with them, and so do the notes that describe them (curves.py, viewer.py and its "
            "reading note, the state-of-the-app and grid-experiments skills)."
        ),
        divergence=(
            "INTENTIONAL: the harness counts votes, not resolved labels - every simulated vote "
            "resolves - and keeps the Train side (acquisition, lights, spot check) on the trained "
            "head wherever there is a Good and a Bad, as the Train view's learned sort does "
            "(labelset_train_and_score with label_quota=False). The hold's space is the run's "
            "region_aware (its media carry a patch grid) where the app reads the embedder's "
            "capability (_patch_space_for): the same split on every production path, and an "
            "unknown embedder, which the app does not hold, never reaches the harness. The "
            "miss's space is read the same way."
        ),
        no_harness_pin=(
            "simulate_voting_iterations is the whole voting loop; the quota's own rows are "
            "_centroid_test and step_trainers._centroid_step, pinned by "
            "detectors.label_quota_centroid."
        ),
    ),
    Mirror(
        id="detectors.label_quota_centroid",
        app="py:vtscore.detectors.labelset_heads.install_centroid_head",
        harness="vtscore/eval/step_trainers.py::_centroid_step",
        kind="ported",
        note=(
            "The Goods' centroid under the label quota (#4643). Both sides call "
            "centroid_head.fit_centroid_head, so the head and its cut are delegated; what is "
            "ported is what they are handed. The app hands it the resolved Good rows of "
            "build_xy_from_labelset (a boxed Good's patch, else its image-level vector) and the "
            "haystack it scores (scoring_rows_for_snap's rows: the active dataset in Test, the "
            "searched one in a cold Find); the harness hands it each Good vote's training "
            "vector (the style's good_vec, else good_training_vec - the vectors its trained "
            "head fits on) and the withheld half's scorer (_centroid_test). Both draw the line by "
            "the media type's rule at a balance (#4732, #5325: centroid_line_rule, face crops' "
            "own, else CENTROID_LINE_RULE): the app reads the detector's media_type and draws "
            "at its own balance (_centroid_beta), the harness resolves its centroid_line default "
            "by production_centroid_line and draws at the run's beta. Both count unvoted media only "
            "(#4779): the app masks the labelset's labelled media out of the line's count "
            "(fit_centroid's line_mask), and the harness's withheld half holds no voted media. "
            "The harness's priced lines (#4732's variant rows) score the same probe once a click "
            "and draw each line off it (_centroid_cut_step, #4800), the set _centroid_test gives. "
            "If the app changes which vectors average, which corpus the line is cut on, or "
            "which balance draws it, change the harness to match."
        ),
    ),
    Mirror(
        id="training.hard_view_goods",
        app="py:vtscore.detectors.labelset_matrix.hard_view_good_vectors",
        harness="vtscore/eval/voting_iterations.py::_fits_weigh_hard_views",
        kind="ported",
        note=(
            "A hard-view face's Good weighs svm_prior.HARD_VIEW_GOOD_WEIGHT in the head's fit (#5413). "
            "The rule is delegated: both sides name rows to svm_prior.hard_view_rows and "
            "centroid_prior_direction weighs them (hard_view_good_weights). What is ported is which "
            "rows are named. The app names its labelset's Goods whose resolved media is a hard view "
            "(imagekit_bridge.media_hard_view), by their cached training vectors; the harness names "
            "every hard-view face of its pool by media_embedding, the vector good_training_vec fits on "
            "a single-vector dataset. The two agree because the rule weighs Goods only. If the app "
            "changes which rows it names, or reads the verdict from elsewhere, change the harness."
        ),
    ),
    Mirror(
        id="training.example_centroid",
        app="py:vtscore.training.query_sort.example_sort_cuts_from_paths",
        harness="vtscore/eval/example_opening.py::example_sort",
        kind="ported",
        note=(
            "The example opening's sort (#4699): cosine to the centroid of the starting photos' "
            "L2-normalised vectors, each weighing the same. Face studies patch the harness side "
            "(#5376 swaps a hard-view face out, #5389 weighs it less), so a change to how the app "
            "averages its examples changes what those studies measured against. If the app changes "
            "which examples average or how they are weighted, change the harness to match."
        ),
        divergence=(
            "The app re-ranks a structural (SIFT/VLAD) dataset against every example (Stage 2); "
            "the harness ranks by the centroid alone and refuses a patch dataset."
        ),
    ),
    Mirror(
        id="detectors.label_quota_query",
        app="py:vtscore.detectors.labelset_heads.install_query_head",
        harness="vtscore/eval/step_trainers.py::_query_step",
        kind="ported",
        note=(
            "A typed-query detector's head under the label quota (#4758). The app embeds the "
            "detector's text_query in the searched collection's text slot, scores every media of "
            "it there and fits query_head.fit_query_head: the query's unit vector as a "
            "Linear(D, 1), cut by centroid_cut(rule='text') - text_sort_cuts' display line - at the "
            "detector's balance, over every media scored, centred in its gap (the centroid's "
            "_gap_centre). The harness reads the same cosines off the opening's text sort "
            "(seed_scores, the run's typed query embedded in its text space) and builds the same "
            "logit, CENTROID_LOGIT_SCALE * (cosine - cut), with the cut drawn by the same two "
            "functions on the withheld half. If the app changes the rule, the balance it draws "
            "at, the corpus the line counts or the head's logit, change the harness to match."
        ),
        divergence=(
            "INTENTIONAL: the harness never re-embeds the query - seed_scores already holds the "
            "cosine the app's head computes, and on a paired arm the text space is not the one the "
            "run's vectors are in, so the step scores by media id (StepModel.id_scores). Its line "
            "is drawn on the withheld half, the corpus a Test there searches, as the centroid's "
            "is; on the labelled collection the app's line also counts the voted media, as the "
            "text sort on screen does."
        ),
    ),
    Mirror(
        id="line_test.budgets_default",
        app="py:vtscore.training.thresholds.line_test.LineBudgets",
        harness="vtscore/eval/line_test_arm.py::line_test_row,simulate_line_test",
        kind="default",
        note=(
            "Test mode's targets and budgets (#4523, #4527): the Test arm runs the app's phase "
            "machine (LineTest.draw / record / line_phase, delegated) at LineBudgets' defaults "
            "when a run names none, so a changed default - the width a phase stops at, a pick "
            "budget, the dry-run share, the model's weight - silently changes what the arm's "
            "default rows measure. When LineBudgets moves, re-read the arm's inputs "
            "(line_test_inputs: Find's labels line on the withheld half, the posteriors "
            "aligned with the ranking) and the verdict rule (read_verdict: LEAN_GAIN, "
            "RETRAIN_BAR), which the app does not implement yet (#4524 takes it from the "
            "study) and the study's report prices."
        ),
    ),
    Mirror(
        id="find.training_domain",
        app="py:vtsearch.training_domain.domain_shift_against",
        harness="vtscore/eval/find_corpus.py::training_domain_flags,FindCorpus.matrix,FindCorpus.pool_matrix",
        kind="ported",
        note=(
            "The Training-domain overlap chip (#4961) read against #4960's Find corpus at the snapshot "
            "clicks. The report itself is delegated - both sides call vtscore.coverage.domain_shift."
            "domain_shift_report - so what is ported is how the app's route calls it: at the report's "
            "default alpha, with the reference dataset's media_embedding vectors as the reference and "
            "media_embedding of every media of the dataset searched (FindCorpus.matrix) as the matrix, and "
            "refused (NaN here) with no vectors on either side or a reference too small to calibrate "
            "(ReferenceTooSmall). Since #3348 it needs no coverage atlas and runs on patch embedders. If the "
            "route changes its alpha, the vectors it stacks or a refusal, follow it; the harness reads "
            "shifted / frac_atypical / z_score off the report, so a renamed key surfaces as a KeyError in "
            "the tests."
        ),
        divergence=(
            "INTENTIONAL: the harness's reference is the sim pool (FindCorpus.pool_matrix over every pool "
            "id), which the app's reference dataset models. The route's embedder-mismatch refusal is the "
            "harness's same-embedder rule, enforced when the corpus is built (build_find_corpus refuses "
            "another vector width or patch grids on one side) and by the runner reading the Find corpus out "
            "of the cell's own embedder's pickle. The route's same-dataset refusal has no harness case: a "
            "Find corpus never holds the sim pool, and a share-0 corpus (the withheld half) is read on "
            "purpose, as the no-shift baseline."
        ),
    ),
    Mirror(
        id="find.evidence_coverage",
        app="py:vtsearch.routes.detectors.scoring.find_evidence_coverage",
        harness=(
            "vtscore/eval/find_corpus.py::evidence_flags,pool_baseline,vote_vectors,FindCorpus.matrix,"
            "FindCorpus.pool_matrix"
        ),
        kind="ported",
        note=(
            "The Evidence coverage chip read against #4960's Find corpus at the snapshot clicks. The report "
            "is delegated - both sides call vtscore.detectors.evidence_coverage.evidence_coverage_report at "
            "its defaults - so what is ported is its inputs: the labelset's rows split Good / Bad (the app's "
            "build_xy_from_labelset: a Good's one vector, a Bad's flooded image-level + patch negatives on a "
            "patch dataset; vote_vectors assembles the trainers' same rows), the whole-image vector of every "
            "scored media (FindCorpus.matrix), and each predicted positive where its Find score clears the "
            "Find threshold (the caller, _find_corpus_columns, passes score >= the line's cut on that "
            "corpus). The verdict's training baseline (#5089) is reference_cutoffs on both sides: the app "
            "draws it (evidence_baseline.draw_baseline) on the training dataset's scored media less its labelled ones, "
            "each called at that Find's threshold; the harness (pool_baseline) on the sim pool's unvoted "
            "media (FindCorpus.pool_matrix), found as a Find there draws it. The route reports nothing with "
            "no labelled rows or no scored media; so does evidence_flags. If the route changes which rows it "
            "reads, the threshold, the baseline's media or a default, follow it."
        ),
        divergence=(
            "INTENTIONAL: the app's labelset vectors are the label embeddings re-derived at detector load; "
            "the harness reads the same media's vectors straight off the run's medias, and a Good's is the "
            "row its trainers fit (the ground-truth box pooled under region voting), which is what the app's "
            "region vote stores. The app's baseline off the training dataset is the one recorded on the "
            "detector at its last test there, or drawn from its last learned sort there once the votes "
            "stop (#5120: the same draw_baseline, at the sort's head and line, which a Find there reuses "
            "above the label quota and nowhere else, so the sort leaves none below it); a vote restamps it "
            "(no verdict until the next). The harness draws it fresh at every snapshot click, so it models "
            "a user whose detector carries a baseline at the click's votes, the case the verdict is priced on."
        ),
    ),
]


class MirrorError(Exception):
    """A mirror could not be resolved at all - the app or harness side moved."""


# --------------------------------------------------------------------- digests


def _rel(path: Path) -> str:
    """*path* as the repo-relative string an error message should show."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:  # A path from outside the tree - only reachable from tests.
        return str(path)


def _source_slice(lines: list[str], start: tuple[int, int], end: tuple[int, int]) -> str:
    """The literal source text between two `(row, col)` token positions."""
    (srow, scol), (erow, ecol) = start, end
    if srow == erow:
        return lines[srow - 1][scol:ecol]
    parts = [lines[srow - 1][scol:]]
    parts.extend(lines[row - 1] for row in range(srow + 1, erow))
    parts.append(lines[erow - 1][:ecol])
    return "".join(parts)


def _collapse_fstrings(tokens: list[tokenize.TokenInfo], lines: list[str]) -> list[tokenize.TokenInfo]:
    """Re-join PEP 701's split f-string tokens into the one token <=3.11 emits.

    Python 3.12 stopped tokenizing an f-string as a single STRING and started
    emitting `FSTRING_START` / `FSTRING_MIDDLE` / `FSTRING_END` around the real
    tokens of each replacement field.  That is a pure tokenizer change - the
    code means the same thing - but it changes the token *text*, so a digest
    taken on 3.12+ disagreed with one taken on 3.11 for any mirrored
    function containing an f-string, and `--update` just moved the failure to
    the other half of the supported range instead of converging (issue #3117).

    Splicing the original source back out by token position restores the older
    single-token form on every interpreter, so a pin travels between them.
    """
    start_type = getattr(tokenize, "FSTRING_START", None)
    if start_type is None:  # <=3.11 already emits one STRING token.
        return tokens
    end_type = tokenize.FSTRING_END
    out: list[tokenize.TokenInfo] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type != start_type:
            out.append(tok)
            i += 1
            continue
        # Nested f-strings are legal on 3.12+, so match by depth, not first END.
        depth = 0
        j = i
        while j < len(tokens):
            if tokens[j].type == start_type:
                depth += 1
            elif tokens[j].type == end_type:
                depth -= 1
                if depth == 0:
                    break
            j += 1
        else:  # pragma: no cover - tokenize raises on an unterminated string first.
            raise MirrorError("unterminated f-string while normalizing source")
        end = tokens[j]
        text = _source_slice(lines, tok.start, end.end)
        out.append(tokenize.TokenInfo(tokenize.STRING, text, tok.start, end.end, tok.line))
        i = j + 1
    return out


def _normalize_python(source: str) -> str:
    """Source text stripped of comments, docstrings and formatting.

    Token-based rather than AST-based on purpose: `ast.unparse` output is not
    guaranteed stable across the Python versions this repo supports (>=3.11),
    which would make the pins fail for whoever is not on the pinning machine's
    interpreter.  Token text is *nearly* stable - see `_collapse_fstrings` for
    the one place it isn't, and how that is normalized away.

    Magic trailing commas are dropped as well, so that `ruff format` wrapping a
    call across lines - which it will do for a change as innocent as a longer
    variable name - doesn't read as a logic change.  The exception is a comma
    that is the only one in a parenthesized expression or a subscript: there it
    is what makes `(x,)` a tuple rather than `x`, and `x[0,]` a tuple index
    rather than `x[0]`, so it is kept.
    """
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER)
    out: list[str] = []
    prev_meaningful: int | None = None
    prev_tok: tokenize.TokenInfo | None = None
    # One entry per open bracket: (does a lone trailing comma matter, commas so far).
    groups: list[tuple[bool, int]] = []
    text = textwrap.dedent(source) + "\n"
    tokens = _collapse_fstrings(
        list(tokenize.generate_tokens(io.StringIO(text).readline)),
        text.splitlines(keepends=True),
    )
    for i, tok in enumerate(tokens):
        if tok.type in skip:
            continue
        nxt = next((t for t in tokens[i + 1 :] if t.type not in skip), None)
        prev, prev_tok = prev_tok, tok
        if tok.type == tokenize.STRING and prev_meaningful in (None, tokenize.NEWLINE, tokenize.INDENT):
            # A string that is an entire statement: a docstring.
            if nxt is not None and nxt.type == tokenize.NEWLINE:
                continue
        # After an expression, `(` is a call (or a `def` parameter list) and `[`
        # a subscript; anywhere else they open a parenthesized expression and a list.
        if tok.string == "(":
            groups.append((not _ends_expression(prev), 0))
        elif tok.string == "[":
            groups.append((_ends_expression(prev), 0))
        elif tok.string == "{":
            groups.append((False, 0))
        elif tok.string in (")", "]", "}") and groups:
            groups.pop()
        elif tok.string == "," and groups:
            sensitive, commas = groups[-1]
            if nxt is not None and nxt.string in (")", "]", "}") and not (sensitive and commas == 0):
                continue
            groups[-1] = (sensitive, commas + 1)
        out.append(tok.string.strip() if tok.type == tokenize.NEWLINE else tok.string)
        prev_meaningful = tok.type
    return "\n".join(s for s in out if s)


def _ends_expression(tok: tokenize.TokenInfo | None) -> bool:
    """Whether *tok* can end an expression, so a `(` after it calls and a `[` subscripts.

    Keywords are excluded so `return (x,)` and `in [a,]` read as a tuple and a
    list.  Soft keywords (`match`, `case`) are not: `case (x,):` digests as if
    it were a call, which only loses sensitivity there, never adds any.
    """
    if tok is None:
        return False
    if tok.type == tokenize.NAME:
        return not keyword.iskeyword(tok.string)
    return tok.type in (tokenize.NUMBER, tokenize.STRING) or tok.string in (")", "]", "}")


_TS_LINE_COMMENT = re.compile(r"//[^\n]*")
_TS_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _normalize_typescript(source: str) -> str:
    """TypeScript block stripped of comments and collapsed whitespace."""
    stripped = _TS_BLOCK_COMMENT.sub(" ", source)
    stripped = _TS_LINE_COMMENT.sub(" ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def _ts_block(path: Path, anchor: str) -> str:
    """The brace-delimited block introduced by *anchor* in *path*.

    Comments are removed before brace-matching so a brace inside a comment
    cannot unbalance the scan.
    """
    text = _TS_BLOCK_COMMENT.sub(" ", path.read_text(encoding="utf-8"))
    text = _TS_LINE_COMMENT.sub(" ", text)
    start = text.find(anchor)
    if start < 0:
        raise MirrorError(f"anchor {anchor!r} not found in {_rel(path)}")
    if text.find(anchor, start + len(anchor)) >= 0:
        raise MirrorError(f"anchor {anchor!r} is ambiguous in {_rel(path)} (matches more than once)")
    open_idx = text.find("{", start)
    if open_idx < 0:
        raise MirrorError(f"no block follows anchor {anchor!r} in {_rel(path)}")
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise MirrorError(f"unbalanced block for anchor {anchor!r} in {_rel(path)}")


def _py_symbol_source(path: Path, symbol: str) -> str:
    """The source of *symbol* in *path*, found by parsing not importing.

    *symbol* is a top-level name, or ``Class.method`` for a member of a
    top-level class, so a reproduction that lives on a method can be pinned
    as it stands rather than lifted out of its class first (#5005).

    Parsing keeps the gate fast and dependency-free: it runs before the test
    suite has installed torch, and reading a module for its text should never
    execute it.
    """
    text = path.read_text(encoding="utf-8")
    body = ast.parse(text).body
    lines = text.splitlines()
    *classes, name = symbol.split(".")
    for cls in classes:
        node = next((n for n in body if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if node is None:
            raise MirrorError(f"{_rel(path)} has no class {cls!r} - did it move or get renamed?")
        body = node.body
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
            return "\n".join(lines[start : node.end_lineno])
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    where = "member" if classes else "top-level"
    raise MirrorError(f"{_rel(path)} has no {where} {symbol!r} - did it move or get renamed?")


def _app_source(mirror: Mirror) -> str:
    """The normalized app-side source *mirror* is pinned against."""
    kind, _, ref = mirror.app.partition(":")
    if kind == "py":
        module_path, _, symbols = ref.rpartition(".")
        path = REPO_ROOT / (module_path.replace(".", "/") + ".py")
        if not path.exists():
            raise MirrorError(f"module {module_path} ({_rel(path)}) does not exist")
        return "\n".join(_normalize_python(_py_symbol_source(path, s.strip())) for s in symbols.split(","))
    if kind == "ts":
        rel, _, anchor = ref.partition("::")
        path = REPO_ROOT / rel
        if not path.exists():
            raise MirrorError(f"{rel} does not exist")
        return _normalize_typescript(_ts_block(path, anchor))
    raise MirrorError(f"unknown app source kind {kind!r} in {mirror.app!r}")


def _harness_source(mirror: Mirror) -> str:
    """The normalized source of the harness symbol(s) *mirror* names.

    Resolved by parsing rather than by searching the file text.  The existence
    check used to be ``symbol in path.read_text()``, which a name surviving only
    inside a comment - or inside the docstring explaining that the reproduction
    was removed - satisfied just as well as the reproduction itself.
    """
    rel, _, symbols = mirror.harness.partition("::")
    path = REPO_ROOT / rel
    if not path.exists():
        raise MirrorError(f"harness file {rel} does not exist")
    names = [s.strip() for s in symbols.split(",") if s.strip()]
    if not names:
        raise MirrorError(f"harness ref {mirror.harness!r} names no symbol")
    parts: list[str] = []
    for name in names:
        try:
            parts.append(_normalize_python(_py_symbol_source(path, name)))
        except MirrorError as exc:
            raise MirrorError(f"harness symbol {name!r} not found in {rel} - was it renamed or removed?") from exc
    return "\n".join(parts)


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _digests(mirror: Mirror) -> dict[str, str]:
    """Every side of *mirror* that carries a pin, keyed as the pins file is.

    Resolving the harness side is not conditional on pinning it: a mirror whose
    reproduction has been deleted is unresolvable whether or not its digest is
    recorded, and that is the failure the manifest most wants to hear about.
    """
    harness = _harness_source(mirror)
    pins = {"app": _digest(_app_source(mirror))}
    if mirror.no_harness_pin is None:
        pins["harness"] = _digest(harness)
    return pins


# ----------------------------------------------------------------------- pins


def _load_pins() -> dict[str, dict[str, str]]:
    """The recorded digests, as ``{mirror id: {side: digest}}``."""
    if not PINS_PATH.exists():
        return {}
    raw = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise SystemExit(
                f"{_rel(PINS_PATH)} records {key!r} as a bare digest, which is the "
                "one-sided format from before the harness side was pinned too. "
                "Run: python scripts/check-eval-app-sync.py --update"
            )
    return raw


def _write_pins(pins: dict[str, dict[str, str]]) -> None:
    PINS_PATH.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class Drift:
    mirror: Mirror
    reason: str
    detail: str = ""
    extras: list[str] = field(default_factory=list)


def check() -> list[Drift]:
    """Every mirror where either side moved, or whose two sides no longer resolve.

    One Drift per mirror, not per side: both halves of a copy moving together is
    the ordinary shape of a faithful port, and reporting it twice would print
    the mirror's whole note twice for one reconciliation.
    """
    pins = _load_pins()
    drifts: list[Drift] = []
    seen: set[str] = set()
    for mirror in MIRRORS:
        if mirror.id in seen:
            raise SystemExit(f"duplicate mirror id {mirror.id!r} in MIRRORS")
        seen.add(mirror.id)
        try:
            digests = _digests(mirror)
        except MirrorError as exc:
            drifts.append(Drift(mirror, "unresolvable", str(exc)))
            continue
        pinned = pins.get(mirror.id, {})
        reasons: list[str] = []
        details: list[str] = []
        for side, digest in digests.items():
            recorded = pinned.get(side)
            if recorded is None:
                reasons.append(f"{side}-unpinned")
                details.append(f"{side}: no digest recorded yet")
            elif recorded != digest:
                reasons.append(f"{side}-changed")
                details.append(f"{side}: pinned {recorded[:12]}, now {digest[:12]}")
        orphans = sorted(set(pinned) - set(digests))
        if orphans:
            reasons.append("side-not-pinned-anymore")
            details.append(
                "digests recorded for sides this mirror no longer pins: "
                + ", ".join(orphans)
                + " (--update drops them)"
            )
        if reasons:
            drifts.append(Drift(mirror, ", ".join(reasons), "; ".join(details)))
    stale = sorted(set(pins) - {m.id for m in MIRRORS})
    if stale:
        drifts.append(
            Drift(
                MIRRORS[0],
                "stale-pins",
                "pins recorded for mirrors that no longer exist: " + ", ".join(stale),
                extras=stale,
            )
        )
    return drifts


def update() -> int:
    pins: dict[str, dict[str, str]] = {mirror.id: _digests(mirror) for mirror in MIRRORS}
    _write_pins(pins)
    unpinned = [m.id for m in MIRRORS if m.no_harness_pin is not None]
    print(f"Pinned {len(pins)} eval/app mirrors to {_rel(PINS_PATH)}")
    if unpinned:
        print(f"  ({len(unpinned)} app-side only, by declaration: {', '.join(unpinned)})")
    return 0


def _report(drifts: list[Drift]) -> None:
    print("The eval framework and the app code it mirrors no longer agree.")
    print("")
    print("The eval default arm has to BE the app's algorithm - that is the only")
    print("thing that makes a deviation arm meaningful. Reconcile each mirror below,")
    print("then re-pin with:  python scripts/check-eval-app-sync.py --update")
    for drift in drifts:
        mirror = drift.mirror
        print("")
        if drift.reason == "stale-pins":
            print(f"  * {drift.detail}")
            print("    Run --update to drop them.")
            continue
        print(f"  * {mirror.id}  [{mirror.kind}, {drift.reason}]")
        print(f"      app:     {mirror.app.partition(':')[2]}")
        print(f"      harness: {mirror.harness}")
        print(f"      {mirror.note}")
        if mirror.divergence:
            print(f"      DECLARED DIVERGENCE: {mirror.divergence}")
        if mirror.no_harness_pin:
            print(f"      HARNESS SIDE NOT PINNED: {mirror.no_harness_pin}")
        if drift.detail:
            print(f"      {drift.detail}")
    print("")
    print("app-changed:     the original moved. Reconcile the harness copy to it, then --update.")
    print("harness-changed: the copy moved while the original stood still. Re-read the two")
    print("                 against each other: this is the direction a harness edit can")
    print("                 silently re-point the default arm, which is what #2923 was.")
    print("")
    print("If the harness already tracks the change (or is unaffected), --update alone")
    print("is the right answer. If the harness now *intentionally* differs, record why")
    print("in the mirror's divergence= field in scripts/check-eval-app-sync.py.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--update",
        action="store_true",
        help="re-pin every mirror to the current source on both sides, after reconciling them",
    )
    args = parser.parse_args(argv)
    if args.update:
        return update()
    drifts = check()
    if drifts:
        _report(drifts)
        return 1
    print(f"Eval/app sync: {len(MIRRORS)} mirrors up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
