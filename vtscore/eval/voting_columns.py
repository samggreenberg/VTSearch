"""Output-row schemas for the voting-iterations eval.

Every frame :mod:`vtscore.eval.voting_iterations` emits declares its column
order here, as a module-level tuple, so that the schema of a study's CSV is one
short file rather than four hundred lines wedged between the harness's helpers.
The tuples compose - most of them lead with :data:`IDENT_COLUMNS` - which is
what keeps a row's identifying prefix identical across the main frame and every
side frame, and therefore what lets an analyzer join them.

These are **public names**.  They were private (``_VOTING_COLUMNS`` and
siblings) while they lived in the harness module, but the study runners under
`scripts/experiments/` have always imported them to widen a frame with their own
per-cell columns, so the underscore only ever asserted something untrue; moving
them here drops it.

.. note::

   :data:`FIT_QUALITY_ROW_COLUMNS` is the *row* schema for the goodness-of-fit
   side frame.  It is not
   :data:`vtscore.eval.fit_quality.FIT_QUALITY_COLUMNS`, which is the narrower
   tuple of fit *metrics* that this one splats in after its identifying prefix.
"""

from __future__ import annotations

from vtscore.eval.fit_quality import FIT_QUALITY_COLUMNS


#: The three-term decomposition the skyline unlocks (issue #3322), NaN on every
#: row of a run that did not ask for :data:`SKYLINE_TRAIN_FULL`.
#:
#: ``cost = skyline_oracle_cost + training_regret + regret`` - the learnability
#: floor, what the interactive loop left on the table, and what the cut rule gave
#: away on the ranking it got.  ``oracle_cost`` alone conflates the first two:
#: a cell can be expensive because no linear head in this embedding separates the
#: class, or because 10-200 clicks did not find the head that does, and those two
#: diagnoses buy different things (a better embedder vs. a better acquisition
#: loop).  ``training_regret`` is defined on **rankings** -
#: ``oracle_cost(mortal) - oracle_cost(skyline)`` - which is what makes the
#: telescope exact.
#:
#: The honest columns re-base the same split off the cross-fitted reference, so
#: ``cost = skyline_oracle_cost_honest + training_regret_honest + regret_honest``
#: telescopes too.  **The terms share noise by construction** (they sum to a
#: pinned total), so the sum-pinned caution `row_metrics.operating_metrics` documents for
#: ``rule_inefficiency`` / ``calibration_shift`` transfers verbatim: do not read
#: one half moving as an effect when the knob also moves the yardstick.
#:
#: **Negative ``training_regret`` is legal and is information, not a bug.**
#: Unlike the threshold oracle, the skyline is not a per-run optimum over the
#: same object: region votes carry box information an image-labelled skyline
#: lacks, and small samples get lucky.  Nothing clamps it.
SKYLINE_COLUMNS: tuple[str, ...] = (
    "skyline_oracle_cost",
    "skyline_oracle_cost_honest",
    "training_regret",
    "training_regret_honest",
)


#: Identifying columns every emitted row (main or sweep) leads with.  ``phase``
#: and ``app_trained`` ride along so any downstream analysis - including the
#: calibration study's threshold rows - can filter to the steps at which the app
#: would actually have had a trained detector on screen.
IDENT_COLUMNS: tuple[str, ...] = (
    "seed",
    "dataset",
    "category",
    "strategy",
    "trainer",
    "head",
    "style",
    "prevalence_arm",
    "realized_prevalence",
    "t",
    "n_good",
    "n_bad",
    #: The simulation set this cell voted out of and fitted its threshold's
    #: population estimate on, and what is left of it after this step's votes
    #: (issue #3312).  ``n_remainder`` is exactly the count
    #: :func:`~vtscore.training.thresholds.apply_vote_exclusion` compares
    #: against the #3308 floor, so the pair reconstructs per step both whether
    #: the exclusion fired and how big it could possibly have been - the effect
    #: is bounded by ``1 - n_remainder / n_haystack``, the votes' share of the
    #: haystack, which is the axis that study bands on.
    "n_haystack",
    "n_remainder",
    "phase",
    "app_trained",
    #: The parameterised opening this run took (issue #3267), verbatim - so a
    #: pooled frame says which arm each row came from without depending on the
    #: directory it was read out of.  Empty on every run that took the app's
    #: own opening, which is every study before #3267.
    "startup_schedule",
    # --- Acquisition/reporting decoupling (docs/ML.md, threshold calibration).
    #: The threshold handed to the *selector* this step - cut
    #: ``acq_inclusion_offset`` inclusion steps below ``threshold``.  Equal to it
    #: on steps with no fold-anchored fit to re-cut, and at offset 0.
    "acq_threshold",
    #: Where each threshold sits in the **pool** score distribution the selector
    #: actually ranks - the two are emitted together on purpose.  Autopilot's
    #: ``hard`` pick works in rank space (:func:`~vtscore.eval.al_strategies.
    #: _hard_pick_by_index`), so "did the sampling position move, and how far"
    #: is a question about these two numbers, not about the thresholds.  Without
    #: them a sign error in the acquisition cut is invisible.
    "acq_pool_percentile",
    "report_pool_percentile",
)

#: Canonical column order for the voting-iterations result frame.  Kept in one
#: place so :func:`run_voting_iterations_eval` and downstream tooling agree.
#: Every column that is a **wall clock**, and therefore the set a determinism
#: check has to exclude.  Defined here, beside the code that emits them, because
#: it was defined in three test files instead: adding one timing column (#3314's
#: `final_score_seconds`) then broke four tests in two of the three, each with
#: its own private copy of this list, and the fourth would have gone on silently
#: comparing a column it no longer covered.  A new timing column now joins one
#: tuple and every consumer follows.
TIMING_COLUMNS: frozenset[str] = frozenset(
    {
        "elapsed_seconds",
        "train_seconds",
        "xcal_seconds",
        "final_score_seconds",
        "pool_score_seconds",
        "test_score_seconds",
        # Per-K calibration clocks (#3314), on the fold-count arms only.
        "fold_seconds",
        "fold_fit_seconds",
        "fold_score_seconds",
        "anchored_seconds",
        "cal_seconds",
    }
)


VOTING_COLUMNS: tuple[str, ...] = (
    *IDENT_COLUMNS,
    "cost",
    "fpr",
    "fnr",
    #: The operating point in the words a reader picks off a menu (#3281).
    #: ``recall`` is exactly ``1 - fnr`` and is emitted anyway: asking someone to
    #: invert an FNR in their head is where the reading errors come from.  One
    #: definition for all three, in ``calibration_metrics.detection_metrics``.
    "precision",
    "recall",
    "f1",
    #: The counts behind them, so a rate can be re-derived, weighted or pooled
    #: without going back to the cells.
    "n_test_pos",
    "n_test_neg",
    "n_flagged",
    "auroc",
    "average_precision",
    #: The fold count the STEP lived at (#3314).  Equal to the run's
    #: `calibrate_count` everywhere except under `fold_count_schedule`, where it
    #: is what the schedule resolved for this step's vote count.  On the plain
    #: frame as well as the calibration one, because it describes the step and
    #: not the study: a frame that cannot say how many folds a row was cut with
    #: cannot be pooled with one that can.
    "calibrate_count",
    "train_seconds",
    "xcal_seconds",
    #: The final model's own pass over the haystack, on the shipped
    #: safe-threshold path (#3314); NaN when safe thresholds are off, where
    #: there is no such pass.  A wall clock like its neighbours, so it varies
    #: run to run.
    "final_score_seconds",
    "pool_score_seconds",
    "test_score_seconds",
    "backend",
    "device",
    "elapsed_seconds",
)

#: Column order for the per-click **pick log** (issue #3267): one row for every
#: vote the simulated user casts, emitted only when the caller passes a
#: ``pick_sink``.
#:
#: The main frame cannot answer the questions this study asks.  It starts at the
#: first *trainable* step - before one Good and one Bad vote coexist there is no
#: model, no threshold and no metrics row - so the opening, which is the whole
#: subject here, is exactly the part it does not record.  This frame records
#: every click instead: what was picked, whether it turned out to be a positive,
#: and **where on the seed sort it came from**, which is what makes "why was this
#: arm better" answerable rather than merely visible in the totals.
PICK_COLUMNS: tuple[str, ...] = (
    "seed",
    "dataset",
    "category",
    "startup_schedule",
    "style",
    "t",
    "phase",
    #: Index of the schedule round this click was spent in, or -1 outside one.
    "startup_round",
    #: The round's cut on the seed sort, and where that lands in the sort's own
    #: score distribution - the sampling *position*, which is what the arms
    #: actually differ by.  NaN / -1 outside a round.
    "startup_cut",
    "startup_cut_percentile",
    "picked_id",
    #: Ground truth for the click: 1 if the item was a positive.
    "picked_label",
    #: Where the picked item sat in the seed sort - as a 0-based rank over the
    #: whole sort and as a percentile (0 = top).  Together with ``picked_label``
    #: this is the mining record: how deep the arm had to reach for each
    #: positive it found.
    "picked_seed_rank",
    "picked_seed_percentile",
    #: The seed-sort similarity of the picked item.
    "picked_seed_score",
    #: The detector score the *previous* step's model gave this item, and the
    #: acquisition cut it was picked against.  NaN before a model exists.
    "picked_detector_score",
    "acq_threshold",
    #: Whether this click was spent PAST the written schedule, held on its last
    #: round because one vote class was still empty.  An arm's opening is only
    #: as long as it was written where this is False, so it is what makes a
    #: length-matched control actually length-matched - and a cell whose whole
    #: horizon is held is total Good-starvation, the phenomenon #3267 is about.
    "startup_held",
    #: How many such clicks have been spent so far in this trajectory.
    "startup_extended_clicks",
    #: Running vote totals **after** this click.
    "n_good",
    "n_bad",
    #: Pool items still unlabelled after this click.
    "n_pool",
)

#: Column order for the calibration study's main per-step frame (issue #2781),
#: emitted only when ``emit_calibration_metrics``.  One row per ``pool_variant``;
#: under ``safe_thresholds`` additionally one row per safe-threshold GMM variant
#: (issue #2799), tagged in ``gmm_variant`` (``""`` on every other row).  The
#: fold-count arms (issues #2897, #3116, #3115) ride the same tag as
#: ``folds_k{K}_{xcal,blend,anchored,anchored_qmedian,tmean,tmedian,qmean,qmedian}``
#: and additionally fill ``fold_count`` / ``fold_seconds`` / ``fold_fit_seconds``
#: / ``fold_score_seconds`` / ``anchored_seconds`` / ``cal_seconds`` /
#: ``n_cal_scores`` / ``n_folds_used``.
CALIBRATION_COLUMNS: tuple[str, ...] = (
    *IDENT_COLUMNS,
    "pool_variant",
    "gmm_variant",
    "schedule",
    "threshold",
    "threshold_provenance",
    "degenerate",
    "threshold_percentile",
    "xcal_threshold",
    "gmm_cut",
    "blend_weight",
    "cut_fallback",
    "cut_fallback_kind",
    "cut_fail_reason",
    "raw_cut_cost",
    "raw_cut_fpr",
    "raw_cut_fnr",
    "cost",
    "fpr",
    "fnr",
    "precision",
    "recall",
    "f1",
    "n_test_pos",
    "n_test_neg",
    "n_flagged",
    "auroc",
    "average_precision",
    "oracle_threshold",
    "oracle_cost",
    "oracle_fpr",
    "oracle_fnr",
    "regret",
    "oracle_threshold_honest",
    "oracle_cost_honest",
    "regret_honest",
    "cal_oracle_threshold",
    "cal_oracle_cost",
    "rule_inefficiency",
    "calibration_shift",
    "calibration_shift_honest",
    *SKYLINE_COLUMNS,
    "n_pool_rows",
    "fold_count",
    "fold_seconds",
    "fold_fit_seconds",
    "fold_score_seconds",
    "anchored_seconds",
    "cal_seconds",
    "n_cal_scores",
    "n_folds_used",
    #: The fold count the STEP lived at (#3314).  Equal to the run's
    #: `calibrate_count` everywhere except under `fold_count_schedule`, where it
    #: is what the schedule resolved for this step's vote count.
    "calibrate_count",
    "train_seconds",
    #: The final model's pass over the haystack (#3314): app work, paid once
    #: per step whatever the fold count, so it belongs in the denominator of a
    #: cost ratio rather than in the calibration term.
    "final_score_seconds",
    "xcal_seconds",
    "pool_score_seconds",
    "test_score_seconds",
    "backend",
    "device",
    "elapsed_seconds",
)

#: Column order for the cut-decomposition side frame (issue #2836): one row per
#: (step, geometry), written to a separate CSV by the runner.  Carries the fitted
#: mixture parameters, the two families' fit quality, the label-supervised class
#: moments, and every cut in the decomposition chain, so the analyzer can test
#: the derivation offline without re-running the simulation.
#:
#: The chain telescopes ``tau_cross -> tau_priorfree -> tau_supervised ->
#: tau_sim_oracle -> tau_test_oracle``; each consecutive pair differs in exactly
#: one assumption (prior/loss, component identification, Gaussian shape, finite
#: sim set), so the terms sum to the total error of today's rule.
CUT_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    *IDENT_COLUMNS,
    "geometry",
    "sim_n",
    "sim_prevalence",
    "fallback_median",
    # Fitted Gaussian mixture.
    "gmm_ok",
    "w_lo",
    "mu_lo",
    "var_lo",
    "w_hi",
    "mu_hi",
    "var_hi",
    "gmm_loglik",
    "pred_offset_equal_var",
    "gmm_logit_loglik",
    # Fitted Gumbel + Normal mixture.  Its component parameters are in LOGIT
    # units (that is where the extreme-value limit lives and where it is fitted);
    # its log likelihood is converted back to score space so the two families are
    # directly comparable.  Reported per component, with ``evt_gumbel_is_low``
    # saying which mode the Gumbel landed on - #2836 assumed that was always the
    # low one and threw away every fit that said otherwise, which #2846 measured
    # at 14 % of production-like fits.
    "evt_ok",
    "evt_fit_fail",
    "evt_gumbel_is_low",
    "evt_w_gumbel",
    "evt_loc",
    "evt_scale",
    "evt_mu",
    "evt_var",
    "evt_loglik",
    "evt_loglik_gain",
    # Label-supervised class moments (diagnostic only).
    "s_mu_neg",
    "s_var_neg",
    "s_mu_pos",
    "s_var_pos",
    "s_prevalence",
    # The cut chain.
    "tau_mid",
    "tau_cross",
    "tau_priorfree",
    "tau_rate",
    "tau_gumbel_cross",
    "tau_gumbel_priorfree",
    "tau_gumbel_rate",
    "tau_gumbel_any_cross",
    "tau_gumbel_any_priorfree",
    "tau_gumbel_any_rate",
    # #2881's tail-quantile sweep, one column per swept alpha (in milli-alpha).
    "tau_tail_a040",
    "tau_tail_a080",
    "tau_tail_a110",
    "tau_tail_a158",
    "tau_tail_a220",
    "tau_tail_a300",
    "tau_tail_a400",
    "tau_bagfit_mid",
    "tau_bagfit_priorfree",
    "tau_supervised",
    "tau_sim_oracle",
    "tau_sim_oracle_f050",
    "tau_sim_oracle_f100",
    "tau_sim_oracle_f250",
    "tau_sim_oracle_f500",
    "tau_sim_oracle_bag",
    "tau_sim_oracle_smooth",
    "tau_test_oracle",
    # #2883: the reference point, honestly.  `tau_test_oracle` above is the
    # argmin of the empirical cost on the test sample itself, so it is a sample
    # minimum and the gap measured against it is biased high.  The cross-fitted
    # pair chooses the cut and pays for it on disjoint folds; the two costs
    # bracket the population optimum the chain actually wants.
    "tau_test_oracle_honest",
    "cost_test_oracle_naive",
    "cost_test_oracle_honest",
    # Sample sizes the last link's variance should scale with, recorded so the
    # scaling claim is read off the run rather than off the dataset's nominal
    # size (thinning, the prevalence arm and per-category positives all move it).
    "sim_n_pos",
    "test_n",
    "test_n_pos",
    # Where the true optimum sits in each fitted Bad component's upper tail.
    "oracle_lo_sf_gauss",
    "oracle_lo_sf_evt",
)

#: Column order for the **goodness-of-fit** side frame (issue #3329): one row per
#: (step, scope), where *scope* is either a calibration fold of the **shipped**
#: fold-anchored cut or the labelled sim set under one pooling geometry.
#:
#: This frame exists because every other diagnostic in this module is
#: *relative* - ``evt_loglik_gain`` prices one family against another, the
#: ``tau_*`` chain prices one cut against another - and a misfit both sides share
#: cancels in every one of them.  ``vtscore.eval.fit_quality`` scores a fit
#: against its own data instead, and this is where those statistics leave the
#: run.
#:
#: Two scopes, because the two questions live on different populations and
#: conflating them would silently compare a fold model's haystack with the final
#: model's sim set:
#:
#: * ``fold<i>`` - fold *i* of the shipped :class:`FoldAnchoredCut`, scored
#:   against **its own** haystack.  Label-free, so it carries the distance and
#:   anchoring columns only.  Nothing in the eval tier has ever read
#:   ``FoldAnchoredCut.fits``; these are the shipped mixture's first observations.
#: * ``sim:<geometry>`` - the unanchored mixture on the labelled sim scores.
#:   Carries the class-shape and identification columns, which need labels.
FIT_QUALITY_ROW_COLUMNS: tuple[str, ...] = (
    *IDENT_COLUMNS,
    "scope",
    "fold_index",
    "n_folds",
    "n_fit_sample",
    "fit_ok",
    # The fitted mixture itself, so a row is self-contained: an analyzer can
    # re-derive any statistic here without needing the run's other frames.
    "fq_w_lo",
    "fq_mu_lo",
    "fq_var_lo",
    "fq_w_hi",
    "fq_mu_hi",
    "fq_var_hi",
    "fq_cut",
    *FIT_QUALITY_COLUMNS,
)

#: Emit a goodness-of-fit row every this many steps (plus the first three, where
#: the fit moves fastest).  The mixture evolves slowly against the vote count, so
#: a row per step would multiply the fit cost by the horizon to resolve a curve
#: that a fifth of the points already resolves.
FIT_QUALITY_STRIDE_DEFAULT = 5

#: Column order for the inclusion-budget sweep side frame (long format, one row
#: per (step, inclusion k)); written to a separate CSV by the runner.
INCLUSION_SWEEP_COLUMNS: tuple[str, ...] = (
    *IDENT_COLUMNS,
    "inclusion_k",
    "alpha",
    "sweep_threshold",
    "sweep_fpr",
    "sweep_fnr",
    "excess_fnr",
)

#: Column order for the **cut-rule x inclusion** side frame (issue #2865): one
#: row per (step, fold-anchored arm, inclusion ``k``), written to its own CSV.
#:
#: Distinct from :data:`INCLUSION_SWEEP_COLUMNS`, which sweeps the *conformal*
#: rule's budget and asks whether its ``alpha(k)`` guarantee holds.  This frame
#: sweeps the **fold-anchored** estimator's cut *rules* and asks which one
#: should answer the knob at all - so every row is scored under the cost weights
#: **of its own k** (not the run's reporting inclusion), against the oracle at
#: that same k, which is what makes an arm's regret comparable across the knob.
#:
#: ``admitted_frac`` is the second decision number and the one with no analogue
#: anywhere else in the harness: a rule that moves the *threshold* without
#: moving the *admitted set* has not restored the knob.  Because the cut is
#: carried to the final model as a quantile, a whole band of the slider can
#: realize to one admitted set on a cleanly separated haystack - so the
#: analyzer's headline is how many distinct admitted sets survive across the
#: nominal range, per arm.
CUT_INCLUSION_COLUMNS: tuple[str, ...] = (
    *IDENT_COLUMNS,
    "arm",
    "cut_rule",
    "anchor_weight",
    "combine",
    "qtilt_step",
    "inclusion_k",
    "fold_quantile",
    "cut_threshold",
    "cut_cost",
    "cut_fpr",
    "cut_fnr",
    "k_oracle_threshold",
    "k_oracle_cost",
    "cut_regret",
    "admitted_frac",
    "n_admitted",
    "n_test",
)
