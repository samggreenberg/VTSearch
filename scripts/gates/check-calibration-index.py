#!/usr/bin/env python3
"""Gate: ``scripts/experiments/calibration/``'s README index is total, and the
directory takes no new study files.

The directory is deliberately flat -- every concluded study's files in one
namespace -- because these are archival cluster scripts whose paths are quoted
by the reports they produced, and whose sibling imports (``import common``) work
only because the launcher ``cd``s here.  Issue #3409 proposed splitting them into
per-study subdirectories; that trade was declined (ungated 80-file churn, ~110
stale doc references) in favour of the thing the subdirectories were actually
wanted for: **knowing which files belong to which study.**

That answer is the README's index, and an index nobody checks is an index that
decays back into a heap of unclassified files -- which is how the directory got
here.  So this makes the mapping total and un-rottable in both directions:

* every ``.py`` / ``.sh`` in the directory appears in the index exactly once, so
  a new study's files cannot land unclassified;
* every file the index names exists, so a deletion cannot leave a phantom row.

An index stops a heap from being *unclassified*; it does not stop it growing,
and the directory doubled (124 to 231 files) in the two months after #3409.
#5406 asked again for the move, and the move had only got dearer: ~150 files in
other study directories now import the shared layer from here too.  What it
settled for instead is closing the directory to new studies, which is what
every study since ``fhibe/``, ``gp_head/`` and ``state_of_app/`` already does:
it lives in its own ``scripts/experiments/<study>/`` and reaches the pipeline
by putting this directory on ``sys.path``.  So, third:

* a file filed under **The studies** must be one of the
  :data:`GRANDFATHERED` names, the study files already here when #5406 closed
  the directory.  A new file here is either shared -- and is filed under **The
  shared layer**, where its row says what it is to every study -- or it belongs
  in its own study's directory.  The list must stay exact: a grandfathered file
  that is deleted, or promoted to the shared layer, comes off it, so a deleted
  name cannot quietly come back.

Pure stdlib and ~20ms, so it runs in ``run-tests.sh``'s cheap serial stage.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / "scripts" / "experiments" / "calibration"
README = CALIB / "README.md"

BEGIN = "<!-- BEGIN INDEX -->"
END = "<!-- END INDEX -->"

#: The heading that splits the index region: the shared layer above it, the
#: per-study rows below.
STUDIES_HEADING = "### The studies"

#: Backticked ``name.py`` / ``name.sh`` tokens, bare (no slash): the index names
#: files relative to the directory it documents.
TOKEN_RE = re.compile(r"`([A-Za-z0-9_]+\.(?:py|sh))`")

#: A **parenthesised** mention is a cross-reference, not a classification -- the
#: #2808 row noting that its analysis reuses ``analyze_spikes.py`` must not make
#: that file look like it is filed under two studies.  Stripping parentheses
#: before tokenizing fails safe: a filename accidentally left inside them reads
#: as unclassified (a loud failure), never as classified twice.
PARENS_RE = re.compile(r"\([^()]*\)")

#: The study files the directory held when #5406 closed it to new studies.  It
#: only ever shrinks.
GRANDFATHERED: frozenset[str] = frozenset(
    {
        "analyse_all.sh",
        "analyse_good_mining.sh",
        "analyze_ab.py",
        "analyze_acq.py",
        "analyze_acqcut_3546.py",
        "analyze_acqtarget_4746.py",
        "analyze_anchored.py",
        "analyze_autopilot_4671.py",
        "analyze_bench.py",
        "analyze_bench_interaction.py",
        "analyze_blend_3551.py",
        "analyze_blend_ab_3551.py",
        "analyze_buildup_4668.py",
        "analyze_calfrac.py",
        "analyze_calseed_3796.py",
        "analyze_calsplit_4583.py",
        "analyze_centroid_count_4759.py",
        "analyze_centroid_line_4732.py",
        "analyze_cross_corpus_4739.py",
        "analyze_cut.py",
        "analyze_cutincl.py",
        "analyze_drystop_4222.py",
        "analyze_evidence_5089.py",
        "analyze_exclusion.py",
        "analyze_fbeta_line_4411.py",
        "analyze_find_typed_query_4758.py",
        "analyze_fitq_3329.py",
        "analyze_floor_candidate_4267.py",
        "analyze_folds.py",
        "analyze_folds_2897.py",
        "analyze_folds_3314.py",
        "analyze_gmm_cap_3827.py",
        "analyze_hinge_3557.py",
        "analyze_horizon.py",
        "analyze_incl_3196.py",
        "analyze_line_estimate_4383.py",
        "analyze_line_test_4523.py",
        "analyze_mixin.py",
        "analyze_overview.py",
        "analyze_pframes_4220.py",
        "analyze_phases.py",
        "analyze_progression_4184.py",
        "analyze_provenance_4256.py",
        "analyze_random_verification.py",
        "analyze_rate.py",
        "analyze_safe.py",
        "analyze_scale.py",
        "analyze_siblings_4197.py",
        "analyze_spikes.py",
        "analyze_split_rounding_4864.py",
        "analyze_startup.py",
        "analyze_tail_overlap.py",
        "analyze_textgood_4222.py",
        "analyze_transfer.py",
        "analyze_upandup_4835.py",
        "build_coco_pickle.py",
        "bump_acq_2877_throttle.sh",
        "ceiling_line_4490.py",
        "chain_folds_2897_ab.sh",
        "check_prefix_3547.py",
        "compare_floor_opening_4287.py",
        "compare_prevalence_4201.py",
        "drive_chunks.sh",
        "error_report.py",
        "export_rank_frames.py",
        "figure_drystop_4222.py",
        "figure_fbeta_line_4411.py",
        "figure_gmm_cap_3827.py",
        "figure_line_estimate_4383.py",
        "figure_line_test_4523.py",
        "figure_pframes_4220.py",
        "figure_prevalence_4201.py",
        "figure_provenance_4256.py",
        "figure_textgood_4222.py",
        "figures_3546.py",
        "figures_3547.py",
        "figures_4582.py",
        "figures_4583.py",
        "figures_4864.py",
        "figures_centroid_count_4759.py",
        "figures_centroid_line_4732.py",
        "figures_find_typed_query_4758.py",
        "figures_fitq_3329.py",
        "figures_hinge_3557.py",
        "figures_incl_3196.py",
        "figures_overview.py",
        "figures_scale.py",
        "figures_trajectory.py",
        "folds_combine_3115.py",
        "frontier_3319.py",
        "frontier_3547.py",
        "goods_4671.py",
        "harvest_3547.py",
        "label_noise.py",
        "launch_ab_cells.sh",
        "launch_acq_2877.sh",
        "launch_acq_3319.sh",
        "launch_acq_3547.sh",
        "launch_acq_incl.sh",
        "launch_acq_incl_vg.sh",
        "launch_acq_region_2905.sh",
        "launch_acqcut_3546.sh",
        "launch_acqtarget_4746.sh",
        "launch_anchored.sh",
        "launch_autopilot_4671.sh",
        "launch_bands.sh",
        "launch_bench.sh",
        "launch_blend_3551.sh",
        "launch_buildup_4668.sh",
        "launch_calfrac_3287.sh",
        "launch_calseed_3796.sh",
        "launch_calsplit_4583.sh",
        "launch_cross_corpus_4739.sh",
        "launch_cut.sh",
        "launch_errdump.sh",
        "launch_exclusion_3308.sh",
        "launch_fitq_3329.sh",
        "launch_folds_2861.sh",
        "launch_folds_2897.sh",
        "launch_folds_2897_ab.sh",
        "launch_folds_3115.sh",
        "launch_folds_3314.sh",
        "launch_good_mining.sh",
        "launch_hinge_3557.sh",
        "launch_horizon.sh",
        "launch_incl_2865.sh",
        "launch_incl_3196.sh",
        "launch_line_test_4523.sh",
        "launch_linhead_2808.sh",
        "launch_logreg_4114.sh",
        "launch_mixin.sh",
        "launch_progression_4184.sh",
        "launch_rate_2861.sh",
        "launch_safe.sh",
        "launch_safe_ab.sh",
        "launch_scale.sh",
        "launch_smartgate_4359.sh",
        "launch_spike_2847.sh",
        "launch_split_rounding_4864.sh",
        "launch_svmlog_3197.sh",
        "launch_tail_2881.sh",
        "launch_textgood_4222.sh",
        "launch_transfer_2883.sh",
        "make_bench_figs.py",
        "make_cut_2846_fig.py",
        "make_cutincl_figs.py",
        "make_error_sheets.py",
        "make_folds_3115_figs.py",
        "make_linhead_figs.py",
        "make_rate_figs.py",
        "make_startup_sheets.py",
        "measure_bands_3679.py",
        "per_env_acq_2877.py",
        "pick_sheets.py",
        "price_pooled_taper_4560.py",
        "probe_acq_divergence.sh",
        "probe_startup_cuts.py",
        "rescore_fbeta_4582.py",
        "run_3292.sh",
        "sam3_score_5213.py",
        "scratch_folds_3310.py",
        "screens_examples_4739.py",
        "selftest_analyze_ab.py",
        "selftest_analyze_acq.py",
        "selftest_analyze_anchored.py",
        "selftest_analyze_blend_3551.py",
        "selftest_analyze_calfrac.py",
        "selftest_analyze_calseed_3796.py",
        "selftest_analyze_calsplit_4583.py",
        "selftest_analyze_cut.py",
        "selftest_analyze_cutincl.py",
        "selftest_analyze_drystop_4222.py",
        "selftest_analyze_exclusion.py",
        "selftest_analyze_fitq_3329.py",
        "selftest_analyze_floor_candidate_4267.py",
        "selftest_analyze_folds_2897.py",
        "selftest_analyze_folds_3314.py",
        "selftest_analyze_hinge_3557.py",
        "selftest_analyze_incl_3196.py",
        "selftest_analyze_line_estimate_4383.py",
        "selftest_analyze_line_test_4523.py",
        "selftest_analyze_mixin.py",
        "selftest_analyze_pframes_4220.py",
        "selftest_analyze_progression_4184.py",
        "selftest_analyze_provenance_4256.py",
        "selftest_analyze_random_verification.py",
        "selftest_analyze_rate.py",
        "selftest_analyze_spikes.py",
        "selftest_analyze_startup.py",
        "selftest_analyze_textgood_4222.py",
        "selftest_analyze_transfer.py",
        "selftest_ceiling_line_4490.py",
        "selftest_split_rounding_4864.py",
        "sigma_objective_4584.py",
        "spike_examples_3547.py",
        "spike_timing_3547.py",
        "split_cadence_4864.py",
        "status_acq_2877.sh",
        "status_folds_2897.sh",
        "status_folds_3314.sh",
        "text_baseline.py",
        "theory_bench.py",
        "theory_kappa_bench.py",
        "worked_cell_3329.py",
    }
)


def _names(region: str) -> list[str]:
    # Markdown links are ``[text](target)``; the targets are paths, not entries.
    return TOKEN_RE.findall(PARENS_RE.sub(" ", region))


def indexed_names(text: str) -> tuple[list[str], list[str], str | None]:
    """The filenames the index files under the shared layer and under the studies.

    Each list is in index order.  On a malformed README both are empty and the
    third element says what is wrong with it.
    """
    if BEGIN not in text or END not in text:
        return [], [], f"README.md is missing its {BEGIN} / {END} sentinels"
    region = text.split(BEGIN, 1)[1].split(END, 1)[0]
    if STUDIES_HEADING not in region:
        return [], [], f"README.md's index is missing its '{STUDIES_HEADING}' heading"
    shared, studies = region.split(STUDIES_HEADING, 1)
    return _names(shared), _names(studies), None


def check(text: str, on_disk: set[str]) -> list[str]:
    """Every problem with the README ``text`` against the scripts ``on_disk``."""
    shared, studies, err = indexed_names(text)
    if err:
        return [err]

    named = shared + studies
    listed = set(named)
    problems: list[str] = []

    dupes = sorted({n for n in named if named.count(n) > 1})
    if dupes:
        problems.append(
            "listed more than once (a file belongs to exactly one study, or to the shared layer): " + ", ".join(dupes)
        )

    missing = sorted(on_disk - listed)
    if missing:
        problems.append(
            "in the directory but not in the README index -- list them under the shared layer if every "
            "study may read them, or move them to their study's own scripts/experiments/<study>/ "
            "(this directory takes no new study files, #5406): " + ", ".join(missing)
        )

    phantom = sorted(listed - on_disk)
    if phantom:
        problems.append("named by the README index but not on disk -- drop the entry: " + ", ".join(phantom))

    new_study_files = sorted(set(studies) - GRANDFATHERED)
    if new_study_files:
        problems.append(
            "filed under a study, but this directory takes no new study files (#5406) -- a study's "
            "scripts live in its own scripts/experiments/<study>/ and put this directory on sys.path "
            "(see the README's index), and a script every study may read is filed under the shared "
            "layer instead: " + ", ".join(new_study_files)
        )

    retired = sorted(GRANDFATHERED - set(studies))
    if retired:
        problems.append(
            "no longer a study file here (deleted, or promoted to the shared layer) -- drop it from "
            "GRANDFATHERED in scripts/gates/check-calibration-index.py: " + ", ".join(retired)
        )

    return problems


def main() -> int:
    if not README.is_file():
        print(f"FAIL: {README} does not exist", file=sys.stderr)
        return 1

    on_disk = {p.name for p in CALIB.iterdir() if p.suffix in (".py", ".sh")}
    problems = check(README.read_text(encoding="utf-8"), on_disk)

    if problems:
        print("TESTS BLOCKED: scripts/experiments/calibration/README.md index is out of date", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"calibration index OK: {len(on_disk)} scripts, all classified, no new study files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
