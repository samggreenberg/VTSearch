"""Tests for the MLP-vs-SVM experiment machinery.

Covers the additions made for the MLP-vs-SVM study: the wider
SVM grid (poly/sigmoid kernels, gamma multiplier, backend field), the shared
trainer registry + parameterised-name resolver, the trainer-pluggable voting
simulation (with the MLP path held numerically unchanged), prevalence control,
the text-sort seed-score glue, and the timing microbenchmark.

All tests use small synthetic embeddings so no model downloads are needed.
"""

import numpy as np
import pytest

from vtscore.eval.timing_benchmark import run_timing_benchmark
from vtscore.eval.sweep_trainers import _as_scores, _parse_trainer_spec, resolve_trainer
from vtscore.eval.voting_iterations import (
    thin_haystack,
    _downsample_to_prevalence,
    _prevalence,
    run_voting_iterations_eval,
    simulate_voting_iterations,
)
from vtscore.training.svm import train_svm

from tests_shared.rows import assert_same_rows, drop_timing


def _separable_clips(dim=16, n_per_cat=40, n_cats=2, seed=0):
    """``n_cats`` well-separated Gaussian clusters, one category each."""
    rng = np.random.RandomState(seed)
    medias = {}
    mid = 1
    centres = np.eye(n_cats, dim, dtype=np.float32) * 2.0
    for c in range(n_cats):
        for _ in range(n_per_cat):
            emb = (centres[c] + rng.normal(0, 0.25, dim)).astype(np.float32)
            medias[mid] = {"id": mid, "embeddings": {"emb": emb}, "category": f"cat{c}"}
            mid += 1
    return medias


# ---------------------------------------------------------------------------
# Wider SVM grid: poly / sigmoid kernels, gamma multiplier, backend field
# ---------------------------------------------------------------------------


class TestSVMWiderGrid:
    def _blobs(self, seed=0):
        rng = np.random.RandomState(seed)
        X = np.vstack([rng.normal(1.0, 0.3, (30, 8)), rng.normal(-1.0, 0.3, (30, 8))]).astype(np.float32)
        y = np.array([1] * 30 + [0] * 30, dtype=np.int32)
        return X, y

    @pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
    def test_all_kernels_fit_and_score(self, kernel):
        X, y = self._blobs()
        clf = train_svm(X, y, kernel=kernel, seed=1)  # type: ignore[arg-type]
        p = clf.predict_proba(X)
        assert p.shape == (60,)
        assert np.all((p >= 0.0) & (p <= 1.0))

    def test_poly_degree_is_honoured(self):
        X, y = self._blobs()
        # Different degrees should be constructible and produce valid scores.
        for degree in (2, 3):
            clf = train_svm(X, y, kernel="poly", degree=degree, seed=1)
            assert clf.kernel == "poly"
            assert np.all((clf.predict_proba(X) >= 0.0) & (clf.predict_proba(X) <= 1.0))

    def test_gamma_mult_changes_the_fit(self):
        X, y = self._blobs()
        narrow = train_svm(X, y, kernel="rbf", gamma_mult=0.25, seed=1).predict_proba(X)
        wide = train_svm(X, y, kernel="rbf", gamma_mult=4.0, seed=1).predict_proba(X)
        # A 16x change in bandwidth must move the scores somewhere.
        assert not np.allclose(narrow, wide)

    def test_gamma_mult_one_is_plain_scale(self):
        X, y = self._blobs()
        a = train_svm(X, y, kernel="rbf", gamma="scale", gamma_mult=1.0, seed=1).predict_proba(X)
        b = train_svm(X, y, kernel="rbf", gamma="scale", seed=1).predict_proba(X)
        assert np.allclose(a, b)

    def test_backend_field_records_cpu(self):
        X, y = self._blobs()
        clf = train_svm(X, y, kernel="rbf", backend="sklearn", seed=1)
        assert clf.backend == "sklearn-cpu"

    def test_explicit_cuml_without_gpu_raises(self):
        X, y = self._blobs()
        # No cuML on a CPU box -> an explicit cuML request must fail loudly.
        with pytest.raises(Exception):
            train_svm(X, y, kernel="linear", backend="cuml", seed=1)


# ---------------------------------------------------------------------------
# Trainer registry + parameterised-name resolver
# ---------------------------------------------------------------------------


class TestTrainerResolver:
    def test_base_names_resolve(self):
        for name in ("mlp", "svm_linear", "svm_rbf"):
            assert callable(resolve_trainer(name))

    def test_parse_parameterised_svm(self):
        assert _parse_trainer_spec("svm_rbf@C=3,gamma=scale") == ("rbf", {"C": 3.0, "gamma": "scale"})
        assert _parse_trainer_spec("svm_rbf@gamma=4x") == ("rbf", {"gamma_mult": 4.0})
        assert _parse_trainer_spec("svm_poly@degree=2,C=0.3") == ("poly", {"degree": 2, "C": 0.3})

    def test_resolved_parameterised_trainer_runs(self):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.normal(1, 0.3, (20, 8)), rng.normal(-1, 0.3, (20, 8))]).astype(np.float32)
        y = np.array([1] * 20 + [0] * 20, dtype=np.int32)
        predict = resolve_trainer("svm_rbf@C=3,gamma=0.5x")(X, y, 0)
        # A PredictFn may return scores or (scores, std); coerce to the score array.
        scores = _as_scores(predict(X))
        assert scores.shape == (40,)

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            resolve_trainer("random_forest")
        with pytest.raises(KeyError):
            resolve_trainer("svm_banana")


# ---------------------------------------------------------------------------
# Trainer-pluggable voting simulation
# ---------------------------------------------------------------------------


class TestTrainerPluggableVoting:
    def test_app_pipeline_is_the_default_and_deterministic(self):
        clips = _separable_clips(seed=1)
        rows_default = simulate_voting_iterations(clips, "cat0", seed=3, max_steps=25)
        rows_explicit = simulate_voting_iterations(clips, "cat0", seed=3, max_steps=25, trainer="app")
        rerun = simulate_voting_iterations(clips, "cat0", seed=3, max_steps=25)
        # Omitting `trainer` must equal trainer="app", and a rerun must reproduce,
        # on every non-timing column (wall-clock columns vary run to run).  This
        # guards against silent drift in the app-pipeline path.
        assert_same_rows(drop_timing(rows_default), drop_timing(rows_explicit))
        assert_same_rows(drop_timing(rows_default), drop_timing(rerun))
        assert rows_default[-1]["trainer"] == "app"

    def test_legacy_mlp_trainer_name_still_resolves_to_the_app_pipeline(self):
        """#3764: ``trainer="mlp"`` was the app pipeline, not an MLP.

        The spelling stays accepted so archived launch scripts keep running, but
        it is normalised on the way in, so result rows carry exactly one name.
        """
        clips = _separable_clips(seed=1)
        rows_legacy = simulate_voting_iterations(clips, "cat0", seed=3, max_steps=25, trainer="mlp")
        rows_new = simulate_voting_iterations(clips, "cat0", seed=3, max_steps=25, trainer="app")
        assert_same_rows(drop_timing(rows_legacy), drop_timing(rows_new))
        assert rows_legacy[-1]["trainer"] == "app"

    def test_head_is_rejected_on_a_standalone_estimator(self):
        """``head=`` names what the app pipeline fits; an SVM arm has no head."""
        clips = _separable_clips(seed=1)
        with pytest.raises(ValueError, match="only applies to trainer='app'"):
            simulate_voting_iterations(clips, "cat0", seed=0, max_steps=5, trainer="svm_linear", head="linear")

    def test_new_columns_present(self):
        clips = _separable_clips(seed=1)
        rows = simulate_voting_iterations(clips, "cat0", seed=0, max_steps=20, trainer="svm_linear")
        assert rows
        row = rows[-1]
        for key in (
            "trainer",
            "prevalence_arm",
            "realized_prevalence",
            "auroc",
            "average_precision",
            "train_seconds",
            "xcal_seconds",
            "pool_score_seconds",
            "test_score_seconds",
            "backend",
            "device",
        ):
            assert key in row
        assert row["trainer"] == "svm_linear"
        assert row["backend"] == "sklearn-cpu"
        # #2916: the SVM trainers fit no torch head, so naming one on their rows
        # would attribute them to an architecture they never trained.
        assert row["head"] == ""
        assert row["prevalence_arm"] == "natural"

    def test_svm_trajectory_learns(self):
        clips = _separable_clips(seed=2, n_per_cat=50)
        rows = simulate_voting_iterations(clips, "cat0", seed=0, max_steps=40, trainer="svm_rbf")
        assert rows
        # On well-separated data the SVM should reach a low cost by the end.
        assert rows[-1]["cost"] < 0.5

    def test_run_eval_sweeps_trainers(self):
        clips = {"toy": _separable_clips(seed=1)}
        df = run_voting_iterations_eval(clips, seeds=[0], max_steps=20, trainers=["app", "svm_linear", "svm_rbf"])
        assert set(df["trainer"].unique()) == {"app", "svm_linear", "svm_rbf"}


# ---------------------------------------------------------------------------
# Prevalence control
# ---------------------------------------------------------------------------


class TestPrevalenceControl:
    def test_downsample_hits_target(self):
        # 200 positives, 800 negatives -> natural 20%; downsample to 5%.
        clips = _separable_clips(n_per_cat=200, n_cats=5, seed=0)  # cat0 = 200/1000 = 20%
        rng = np.random.RandomState(0)
        out = _downsample_to_prevalence(clips, "cat0", 0.05, rng)
        assert out is not None
        prev = _prevalence(out, "cat0")
        assert prev <= 0.05 + 1e-6
        assert prev == pytest.approx(0.05, abs=0.01)

    def test_downsample_refuses_below_floor(self):
        clips = _separable_clips(n_per_cat=40, n_cats=2, seed=0)  # cat0 = 40 positives
        rng = np.random.RandomState(0)
        # Target 0.1% of ~40 negatives leaves << 15 positives -> refuse.
        assert _downsample_to_prevalence(clips, "cat0", 0.0001, rng) is None

    def test_rare_arm_recorded_or_skipped(self):
        clips = _separable_clips(n_per_cat=200, n_cats=5, seed=0)
        rows = simulate_voting_iterations(clips, "cat0", seed=0, max_steps=20, target_prevalence=0.05)
        assert rows
        assert rows[0]["prevalence_arm"] == "rare_0.05"
        assert rows[0]["realized_prevalence"] <= 0.05 + 1e-6

    def test_natural_arm_is_unchanged(self):
        clips = _separable_clips(seed=1)
        with_none = simulate_voting_iterations(clips, "cat0", seed=0, max_steps=15)
        with_natural = simulate_voting_iterations(clips, "cat0", seed=0, max_steps=15, target_prevalence=None)
        assert_same_rows(drop_timing(with_none), drop_timing(with_natural))


class TestPrecisionFrames:
    """#4220: the per-image evidence and truth a precision-floor estimator is priced on."""

    def _run(self, steps, sink):
        clips = _separable_clips(n_per_cat=80, n_cats=5, seed=0)
        return simulate_voting_iterations(
            clips,
            "cat0",
            seed=0,
            max_steps=20,
            style="whole_image",
            safe_thresholds=True,
            emit_calibration_metrics=True,
            precision_frame_sink=sink,
            precision_frame_steps=steps,
        )

    def test_one_frame_per_requested_step_with_consistent_shapes(self):
        sink: list = []
        self._run((12, 20), sink)
        assert [int(f["t"]) for f in sink] == [12, 20]
        for f in sink:
            assert f["test_scores"].shape == f["test_labels"].shape
            assert f["vote_scores"].shape == f["vote_labels"].shape == (int(f["t"]),)
            assert f["fold_cal_scores"].shape == f["fold_cal_labels"].shape == f["fold_cal_fold"].shape
            assert f["fold_hay_scores"].shape == f["fold_hay_fold"].shape
            assert set(np.unique(f["fold_cal_fold"])) == set(np.unique(f["fold_hay_fold"])), (
                "every fold has its haystack"
            )
            assert 0 < f["test_labels"].sum() < len(f["test_labels"])

    def test_each_calibration_vote_is_named_with_its_phase(self):
        """#4224: the frame says which vote (and which phase) each held-out score is.

        The mapping runs through the trainer's row order (Goods, then Bads), so
        a slip there would pair a score with the wrong vote; the labels catch it.
        """
        from vtscore.eval.labels import media_is_positive

        clips = _separable_clips(n_per_cat=80, n_cats=5, seed=0)
        sink: list = []
        simulate_voting_iterations(
            clips,
            "cat0",
            seed=0,
            max_steps=20,
            style="whole_image",
            safe_thresholds=True,
            emit_calibration_metrics=True,
            precision_frame_sink=sink,
            precision_frame_steps=(12, 20),
        )
        for f in sink:
            votes = f["fold_cal_vote"].tolist()
            assert len(votes) == len(f["fold_cal_scores"]) == len(f["fold_cal_phase"]) > 0
            truth = [1 if media_is_positive(clips[v], "cat0") else 0 for v in votes]
            assert truth == f["fold_cal_labels"].astype(int).tolist()
            assert set(f["fold_cal_phase"].tolist()) <= {"good", "bad", "more", "hard", "new", "done"}

    def test_recording_does_not_change_the_run(self):
        """The sink only reads: rows with and without it are the same."""
        plain = self._run(None, None)
        recorded = self._run((12, 20), [])
        assert_same_rows(drop_timing(plain), drop_timing(recorded))


class TestRankFrames:
    """#4357: where the positives sit in the test half and the session's unvoted pool."""

    def _run(self, steps, sink, pframes=None):
        clips = _separable_clips(n_per_cat=80, n_cats=5, seed=0)
        return simulate_voting_iterations(
            clips,
            "cat0",
            seed=0,
            max_steps=20,
            style="whole_image",
            safe_thresholds=True,
            emit_calibration_metrics=True,
            skyline_arms=["skyline_train_full"],
            rank_frame_sink=sink,
            rank_frame_steps=steps,
            precision_frame_sink=pframes,
            precision_frame_steps=steps if pframes is not None else None,
        )

    def test_step_last_and_skyline_frames(self):
        sink: list = []
        rows = self._run((10, 20), sink)
        assert [(f["kind"], f["t"]) for f in sink] == [
            ("step", 10),
            ("step", 20),
            ("last", 20),
            ("skyline_train_full", 0),
        ]
        for f in sink:
            ranks = [int(r) for r in f["test_pos_ranks"].split()]
            assert len(ranks) == f["n_test_pos"] > 0
            assert ranks == sorted(set(ranks)) and ranks[-1] < f["n_test"]
        sky = sink[-1]
        assert (sky["n_pool"], sky["n_pool_pos"], sky["pool_pos_ranks"]) == (-1, -1, ""), "a skyline has no pool"
        # The ``last`` frame is the ranking the spot check drew from: the last
        # ordinary step, with every check row after it.
        ordinary = [r["t"] for r in rows if r["phase"] not in ("check", "")]
        assert sink[2]["t"] == max(ordinary)
        assert min(r["t"] for r in rows if r["phase"] == "check") > sink[2]["t"]
        # The pool is the unvoted remainder: it shrinks by one per vote.
        n_sim = sink[0]["n_pool"] + 10
        assert sink[1]["n_pool"] == n_sim - 20

    def test_test_ranks_are_the_test_scores_sorted(self):
        """The ranks are the #4220 precision frame's own test half, sorted by score."""
        sink: list = []
        pframes: list = []
        self._run((10, 20), sink, pframes)
        for f, pf in zip([f for f in sink if f["kind"] == "step"], pframes, strict=True):
            order = np.argsort(-pf["test_scores"].astype(np.float64), kind="stable")
            expected = np.flatnonzero(pf["test_labels"][order] == 1).tolist()
            assert [int(r) for r in f["test_pos_ranks"].split()] == expected

    def test_step_frames_record_the_shipped_lines_count_on_the_test_half(self):
        """#4389: a cold Find over the test half plus the votes; the skyline drew no line."""
        from vtscore.training.thresholds import LineRanking, floor_count, mixture_count

        sink: list = []
        pframes: list = []
        self._run((10, 20), sink, pframes)
        for f, pf in zip([f for f in sink if f["kind"] == "step"], pframes, strict=True):
            test_s = pf["test_scores"].astype(np.float64)
            vote_s = pf["vote_scores"].astype(np.float64)
            n_test = len(test_s)
            votes = list(range(n_test, n_test + len(vote_s)))
            corpus = LineRanking.from_scores([*range(n_test), *votes], [*test_s, *vote_s], votes)
            labels = {v: bool(lab >= 0.5) for v, lab in zip(votes, pf["vote_labels"], strict=True)}
            for p in (0.1, 0.5, 0.9):
                want = min(floor_count(p, None, mixture_count(corpus, p, labels)), n_test)
                got = f[f"test_line_k_p{round(p * 100):d}"]
                assert got == want, (f["t"], p)
                assert 1 <= got <= min(floor_count(p, None), n_test), "the mixture only lowers the count"
        sky = sink[-1]
        assert [sky[f"test_line_k_p{q}"] for q in (10, 50, 90)] == [-1, -1, -1]

    def test_recording_does_not_change_the_run(self):
        plain = self._run(None, None)
        recorded = self._run((10, 20), [])
        assert_same_rows(drop_timing(plain), drop_timing(recorded))


class TestHaystackPrevalence:
    """The #4184/#4201 arm: thin the simulation half's negatives, touch nothing else."""

    def test_thinning_hits_target_and_keeps_every_positive(self):
        clips = _separable_clips(n_per_cat=200, n_cats=5, seed=0)  # cat0 = 20%
        ids = sorted(clips)
        pos = {c for c in ids if clips[c]["category"] == "cat0"}
        out = thin_haystack(clips, ids, "cat0", 0.5, seed=0)
        assert pos <= set(out)
        assert _prevalence({c: clips[c] for c in out}, "cat0") == pytest.approx(0.5, abs=0.01)
        assert out == thin_haystack(clips, ids, "cat0", 0.5, seed=0), "deterministic in the seed"

    def test_a_pool_already_richer_is_left_alone(self):
        clips = _separable_clips(n_per_cat=200, n_cats=5, seed=0)
        ids = sorted(clips)
        assert thin_haystack(clips, ids, "cat0", 0.05, seed=0) == ids

    def test_test_set_is_the_natural_runs(self):
        """Same held-out set, so a cell pairs with its natural twin; only the pool shrinks."""
        clips = _separable_clips(n_per_cat=100, n_cats=5, seed=0)
        natural = simulate_voting_iterations(clips, "cat0", seed=0, max_steps=12)
        thinned = simulate_voting_iterations(clips, "cat0", seed=0, max_steps=12, haystack_prevalence=0.5)
        assert thinned and thinned[0]["prevalence_arm"] == "haystack_0.5"
        assert thinned[0]["realized_prevalence"] == pytest.approx(0.5, abs=0.02)
        for a, b in zip(natural, thinned):
            assert (a["n_test_pos"], a["n_test_neg"]) == (b["n_test_pos"], b["n_test_neg"])
        assert thinned[-1]["n_haystack"] < natural[-1]["n_haystack"]

    def test_refuses_both_prevalence_arms(self):
        clips = _separable_clips(seed=1)
        with pytest.raises(ValueError, match="two different arms"):
            simulate_voting_iterations(
                clips, "cat0", seed=0, max_steps=5, target_prevalence=0.05, haystack_prevalence=0.05
            )


# ---------------------------------------------------------------------------
# Text-sort seed-score glue
# ---------------------------------------------------------------------------


class TestSeedScores:
    def test_build_seed_scores(self, monkeypatch):
        from dataclasses import dataclass

        import vtscore.eval.seed_scores as ss

        @dataclass
        class _Q:
            text: str
            target_category: str

        clips = _separable_clips(dim=8, n_per_cat=10, n_cats=2, seed=0)
        fake_eval = {"toy": {"queries": [_Q("a cat0 thing", "cat0"), _Q("a cat1 thing", "cat1")]}}
        monkeypatch.setattr("vtscore.eval.config.EVAL_DATASETS", fake_eval, raising=False)

        # Embed each query as the corresponding cluster centre so the ranking is meaningful.
        centres = {"cat0": np.eye(1, 8, 0, dtype=np.float32)[0], "cat1": np.eye(1, 8, 1, dtype=np.float32)[0]}

        def fake_embed(text, media_type, enrich=False, embedder_name=""):
            return centres["cat0"] if "cat0" in text else centres["cat1"]

        monkeypatch.setattr("vtscore.embedding.helpers.embed_text_query", fake_embed)

        out = ss.build_seed_scores({"toy": clips}, media_type="image", embedder_name="siglip")
        assert set(out["toy"].keys()) == {"cat0", "cat1"}
        # Every media id is scored.
        assert set(out["toy"]["cat0"].keys()) == set(clips.keys())
        # cat0 medias should on average outscore cat1 medias for the cat0 query.
        cat0_ids = [cid for cid, m in clips.items() if m["category"] == "cat0"]
        cat1_ids = [cid for cid, m in clips.items() if m["category"] == "cat1"]
        s = out["toy"]["cat0"]
        assert np.mean([s[i] for i in cat0_ids]) > np.mean([s[i] for i in cat1_ids])


# ---------------------------------------------------------------------------
# Timing microbenchmark
# ---------------------------------------------------------------------------


class TestTimingBenchmark:
    def test_cpu_smoke(self):
        df = run_timing_benchmark(
            trainers=["mlp", "svm_linear"],
            train_sizes=[16, 64],
            infer_sizes=[100, 500],
            dim=16,
            repeats=2,
            warmup=1,
        )
        assert set(df["phase"].unique()) == {"train", "infer"}
        assert (df["median_seconds"] >= 0).all()
        assert {"torch", "sklearn", "gpu"}.issubset(df.columns)
        # sklearn on a CPU box.
        assert (df[df.trainer == "svm_linear"]["backend"] == "sklearn-cpu").all()
