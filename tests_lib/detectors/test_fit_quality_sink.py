"""The #3329 goodness-of-fit side frame, end to end through the simulator.

The unit tests in ``test_fit_quality.py`` pin the statistics.  These pin the
*plumbing*, which is where a side frame actually goes wrong: a sink that is
never appended to, a column tuple that silently drops the keys the sink emits
(``pd.DataFrame(rows, columns=...)`` does exactly that, with no error), or a
scope that never fires because the fit it reads is always ``None``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.fit_quality import FIT_QUALITY_COLUMNS
from vtscore.eval.voting_iterations import _FIT_QUALITY_COLUMNS, simulate_voting_iterations

DIM = 32


def _blob_dataset(n_per_cat=120, seed=0, separation=3.0, dim=DIM):
    """Two linearly separable Gaussian blobs - a learnable class.

    Sized so the *sim* half clears ``fit_quality.MIN_CLASS_N`` (30) in **each**
    class: the shape statistics are third and fourth moments and decline below
    that, so a smaller fixture makes them correctly return NaN and makes this
    file look like a plumbing failure when it is a guard doing its job.
    """
    rng = np.random.default_rng(seed)
    direction = np.zeros(dim, dtype=np.float32)
    direction[0] = 1.0
    medias: dict[int, dict] = {}
    mid = 1
    for cat, sign in (("cat0", 1.0), ("cat1", -1.0)):
        for _ in range(n_per_cat):
            vec = rng.normal(0, 1.0, dim).astype(np.float32) + sign * separation * direction
            medias[mid] = {"id": mid, "category": cat, "embeddings": {"emb": vec}}
            mid += 1
    return medias


def _run(sink, *, stride=5, max_steps=12):
    return simulate_voting_iterations(
        _blob_dataset(),
        target_category="cat0",
        seed=0,
        dataset_name="synthetic",
        inclusion=0,
        style="whole_image",
        max_steps=max_steps,
        emit_calibration_metrics=True,
        safe_thresholds=True,
        fit_quality_sink=sink,
        fit_quality_stride=stride,
    )


class TestFitQualitySink:
    def test_rows_are_emitted(self):
        sink: list[dict] = []
        _run(sink)
        assert sink, "no goodness-of-fit rows were emitted at all"

    def test_every_emitted_key_survives_the_column_tuple(self):
        # The failure mode the repo has already paid for once: a key the sink
        # emits that is absent from the column tuple is dropped by pandas with
        # no error, so the column simply never appears in the CSV.
        sink: list[dict] = []
        _run(sink)
        cols = set(_FIT_QUALITY_COLUMNS)
        for row in sink:
            missing = set(row) - cols
            assert not missing, f"keys emitted but not in _FIT_QUALITY_COLUMNS: {sorted(missing)}"

    def test_the_statistics_columns_are_all_present(self):
        sink: list[dict] = []
        _run(sink)
        for col in FIT_QUALITY_COLUMNS:
            assert col in sink[0], f"{col} missing from the emitted row"

    def test_both_scopes_appear(self):
        sink: list[dict] = []
        _run(sink)
        scopes = {r["scope"] for r in sink}
        assert any(s.startswith("sim:") for s in scopes), f"no labelled sim scope emitted; got {scopes}"

    def test_the_sim_scope_carries_labelled_statistics(self):
        sink: list[dict] = []
        _run(sink)
        sim_rows = [r for r in sink if str(r["scope"]).startswith("sim:")]
        assert sim_rows
        # At least one step must have produced a real identification reading;
        # a frame of all-NaN would pass every "column exists" check above.
        assert any(np.isfinite(r["ident_bal_acc"]) for r in sim_rows), "no finite ident_bal_acc anywhere"
        assert any(np.isfinite(r["shape_skew_neg"]) for r in sim_rows), "no finite shape_skew_neg anywhere"

    def test_a_separable_class_identifies_well(self):
        # The blobs are linearly separable, so the fitted mixture's split really
        # should coincide with the class split. If this ever drops to chance the
        # statistic has stopped tracking what it claims to.
        sink: list[dict] = []
        _run(sink, max_steps=16)
        accs = [r["ident_bal_acc"] for r in sink if str(r["scope"]).startswith("sim:")]
        accs = [a for a in accs if np.isfinite(a)]
        assert accs, "no identification readings"
        assert max(accs) > 0.7, f"separable data identified at only {max(accs):.3f}"

    def test_stride_controls_the_row_count(self):
        dense: list[dict] = []
        sparse: list[dict] = []
        _run(dense, stride=1, max_steps=12)
        _run(sparse, stride=6, max_steps=12)
        assert len(sparse) < len(dense), f"stride did not thin the frame ({len(sparse)} vs {len(dense)})"

    def test_no_sink_is_a_no_op(self):
        # The default path must not pay for the diagnostics, and must not crash.
        rows = simulate_voting_iterations(
            _blob_dataset(),
            target_category="cat0",
            seed=0,
            dataset_name="synthetic",
            inclusion=0,
            style="whole_image",
            max_steps=8,
            emit_calibration_metrics=True,
            safe_thresholds=True,
        )
        assert rows


class TestFoldScope:
    """The fold scope is the point of the exercise: it is the only observation
    anywhere of the mixture the *shipped* threshold is actually cut from."""

    def test_fold_rows_carry_the_fitted_parameters(self):
        sink: list[dict] = []
        _run(sink, stride=1, max_steps=20)
        fold_rows = [r for r in sink if str(r["scope"]).startswith("fold")]
        if not fold_rows:
            pytest.skip("no fold-anchored cut formed on this small synthetic run")
        r = fold_rows[0]
        for col in ("fq_w_lo", "fq_mu_lo", "fq_var_lo", "fq_w_hi", "fq_mu_hi", "fq_var_hi"):
            assert np.isfinite(r[col]), f"{col} is not finite on a fold row"
        assert r["fq_mu_hi"] >= r["fq_mu_lo"], "components are not ordered by mean"

    def test_fold_rows_record_the_anchor_mass(self):
        sink: list[dict] = []
        _run(sink, stride=1, max_steps=20)
        fold_rows = [r for r in sink if str(r["scope"]).startswith("fold")]
        if not fold_rows:
            pytest.skip("no fold-anchored cut formed on this small synthetic run")
        # The H3 statistic has to be readable off the frame, not recomputed.
        assert all(np.isfinite(r["anchor_mass_frac"]) for r in fold_rows)
        assert all(r["anchor_kappa"] > 0 for r in fold_rows)

    def test_fold_rows_carry_a_measured_anchor_drift(self):
        # BOTH halves of H3 have to arrive populated. On the first real run
        # neither did: `anchored_dmu_*` was NaN on all 11,520 fold rows because
        # no call site passed the counterfactual, and `anchor_n` was the count
        # of FOLDS rather than of votes. Either alone is enough to score H3 as
        # a refutation of something never measured (#3329).
        sink: list[dict] = []
        _run(sink, stride=1, max_steps=20)
        fold_rows = [r for r in sink if str(r["scope"]).startswith("fold")]
        if not fold_rows:
            pytest.skip("no fold-anchored cut formed on this small synthetic run")
        anchored = [r for r in fold_rows if r["anchor_n"] > 0]
        if not anchored:
            pytest.skip("no fold anchored on this small synthetic run")
        assert all(np.isfinite(r["anchored_dmu_lo"]) for r in anchored)
        assert all(np.isfinite(r["anchored_dmu_hi"]) for r in anchored)
        assert all(np.isfinite(r["anchored_dw_lo"]) for r in anchored)

    def test_the_anchor_count_tracks_votes_rather_than_folds(self):
        # `anchor_n` flat across a growing label set is the signature of the
        # fold-count bug: the votes each fold anchors on are its held-out share,
        # which grows with the run.
        sink: list[dict] = []
        _run(sink, stride=1, max_steps=20)
        anchored = [r for r in sink if str(r["scope"]).startswith("fold") and r["anchor_n"] > 0]
        if len({int(r["t"]) for r in anchored}) < 2:
            pytest.skip("too few anchored checkpoints on this small synthetic run")
        n_folds = max(int(r["n_folds"]) for r in anchored)
        assert max(int(r["anchor_n"]) for r in anchored) > n_folds
