import io
import unittest.mock

import numpy as np
import torch

import app as app_module


class TestSortClips:
    def test_returns_all_clips(self, client):
        resp = client.post("/api/sort", json={"text": "high pitched beep"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == app_module.NUM_MEDIAS
        assert "threshold" in data

    def test_result_contains_id_and_similarity(self, client):
        resp = client.post("/api/sort", json={"text": "low tone"})
        data = resp.get_json()
        for entry in data["results"]:
            assert "id" in entry
            assert "similarity" in entry

    def test_sorted_by_descending_similarity(self, client):
        resp = client.post("/api/sort", json={"text": "a beeping sound"})
        data = resp.get_json()
        similarities = [e["similarity"] for e in data["results"]]
        assert similarities == sorted(similarities, reverse=True)

    def test_all_media_ids_present(self, client):
        resp = client.post("/api/sort", json={"text": "sine wave"})
        data = resp.get_json()
        ids = {e["id"] for e in data["results"]}
        assert ids == set(range(1, app_module.NUM_MEDIAS + 1))

    def test_similarity_values_in_range(self, client):
        resp = client.post("/api/sort", json={"text": "high pitch"})
        data = resp.get_json()
        for entry in data["results"]:
            assert -1.0 <= entry["similarity"] <= 1.0

    def test_empty_text_returns_400(self, client):
        resp = client.post("/api/sort", json={"text": ""})
        assert resp.status_code == 400

    def test_missing_text_returns_422(self, client):
        # Schema-level validation: the marshmallow ``SortRequestSchema``
        # rejects requests without a ``text`` key as 422 with the
        # standard ``errors`` envelope.
        resp = client.post("/api/sort", json={"other": "field"})
        assert resp.status_code == 422
        assert "text" in resp.get_json()["errors"]["json"]

    def test_whitespace_only_returns_400(self, client):
        resp = client.post("/api/sort", json={"text": "   "})
        assert resp.status_code == 400


class TestTrainAndScore:
    def test_returns_list_of_scored_clips(self):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        results, threshold, _model = app_module.train_and_score(
            app_module.medias, app_module.good_votes, app_module.bad_votes
        )
        assert len(results) == app_module.NUM_MEDIAS
        assert isinstance(threshold, float)
        for entry in results:
            assert "id" in entry
            assert "score" in entry

    def test_scores_between_zero_and_one(self):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        results, threshold, _model = app_module.train_and_score(
            app_module.medias, app_module.good_votes, app_module.bad_votes
        )
        for entry in results:
            assert 0.0 <= entry["score"] <= 1.0

    def test_results_sorted_descending(self):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        results, threshold, _model = app_module.train_and_score(
            app_module.medias, app_module.good_votes, app_module.bad_votes
        )
        scores = [e["score"] for e in results]
        assert scores == sorted(scores, reverse=True)

    def test_good_clips_scored_higher_than_bad(self):
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        results, threshold, _model = app_module.train_and_score(
            app_module.medias, app_module.good_votes, app_module.bad_votes
        )
        score_map = {e["id"]: e["score"] for e in results}
        avg_good = np.mean([score_map[i] for i in app_module.good_votes])
        avg_bad = np.mean([score_map[i] for i in app_module.bad_votes])
        assert avg_good > avg_bad

    def test_order_changes_after_new_vote(self):
        """After adding a vote and retraining, the sort order should change."""
        app_module.good_votes.update({k: None for k in [1, 2, 3, 4, 5]})
        app_module.bad_votes.update({k: None for k in [16, 17, 18, 19, 20]})
        results_before, _, _m = app_module.train_and_score(
            app_module.medias, app_module.good_votes, app_module.bad_votes
        )
        order_before = [e["id"] for e in results_before]

        # Add a new good vote on a media that was in the middle
        app_module.good_votes[10] = None
        results_after, _, _m = app_module.train_and_score(
            app_module.medias, app_module.good_votes, app_module.bad_votes
        )
        order_after = [e["id"] for e in results_after]

        assert order_before != order_after, "Sort order did not change after adding a new vote"


class TestBuildModel:
    """Tests for the build_model helper."""

    def test_build_model_returns_sequential(self):
        from vtscore.training.mlp import build_model

        model = build_model(64)
        assert isinstance(model, torch.nn.Sequential)

    def test_build_model_output_is_logits(self):
        """build_model should NOT include sigmoid — output can be outside [0,1]."""
        from vtscore.training.mlp import build_model

        # Use a seeded generator so the random weights are deterministic —
        # without this, the test is flaky because random initialisation can
        # occasionally produce weights that map extreme input into [0, 1].
        gen = torch.Generator().manual_seed(42)
        model = build_model(32, generator=gen)
        model.eval()
        # Use extreme input to push output well outside [0, 1]
        X = torch.ones(1, 32) * 100.0
        with torch.no_grad():
            logit = model(X).item()
        # Raw logit is unbounded — with extreme input it should land outside [0, 1]
        assert isinstance(logit, float)
        assert logit < 0.0 or logit > 1.0, f"Expected unbounded logit outside [0,1] with extreme input, got {logit}"

    def test_build_model_has_no_sigmoid_layer(self):
        from vtscore.training.mlp import build_model

        model = build_model(64)
        for layer in model:
            assert not isinstance(layer, torch.nn.Sigmoid)

    def test_build_model_state_dict_keys(self):
        from vtscore.training.mlp import build_model

        model = build_model(128)
        keys = set(model.state_dict().keys())
        # 4 layers: Linear(0), ReLU(1), Dropout(2), Linear(3)
        assert keys == {"0.weight", "0.bias", "3.weight", "3.bias"}


class TestTrainModelConfig:
    """Tests for training configuration: reproducibility, weight decay, loss function."""

    def test_deterministic_training(self):
        """Same inputs should produce the same model (manual seed)."""
        from vtscore.training.mlp import train_model

        rng = np.random.RandomState(0)
        X = torch.tensor(rng.randn(10, 32).astype(np.float32))
        y = torch.tensor([1.0] * 5 + [0.0] * 5).unsqueeze(1)

        model1 = train_model(X, y, 32)
        model2 = train_model(X, y, 32)

        # Both models should produce identical scores
        with torch.no_grad():
            scores1 = torch.sigmoid(model1(X)).squeeze(1).tolist()
            scores2 = torch.sigmoid(model2(X)).squeeze(1).tolist()
        assert scores1 == scores2

    def test_weight_decay_is_applied(self):
        """Weight decay should keep weights smaller than without it."""
        import vtscore.config as config
        from vtscore.training.mlp import _auto_hidden_dim, build_model

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 200
        try:
            rng = np.random.RandomState(7)
            X = torch.tensor(rng.randn(20, 16).astype(np.float32))
            y = torch.tensor([1.0] * 10 + [0.0] * 10).unsqueeze(1)

            # Train with weight decay (default: 1e-4)
            from vtscore.training.mlp import train_model

            model = train_model(X, y, 16)

            # Train without weight decay for comparison (use same local
            # generator approach as train_model for identical init weights)
            hidden_dim = _auto_hidden_dim(len(X))
            g = torch.Generator()
            g.manual_seed(42)
            model_no_wd = build_model(16, hidden_dim=hidden_dim, dropout=config.MLP_DROPOUT, generator=g)
            optimizer = torch.optim.Adam(model_no_wd.parameters(), lr=0.001, weight_decay=0.0)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            model_no_wd.train()
            for _ in range(200):
                optimizer.zero_grad()
                loss = loss_fn(model_no_wd(X), y)
                loss.backward()
                optimizer.step()
            model_no_wd.eval()

            # Weight magnitudes with decay should be <= without decay
            wd_norm = sum(p.norm().item() for p in model.parameters())
            no_wd_norm = sum(p.norm().item() for p in model_no_wd.parameters())
            assert wd_norm <= no_wd_norm
        finally:
            config.TRAIN_EPOCHS = saved

    def test_train_model_outputs_logits(self):
        """train_model should return a model that outputs raw logits."""
        from vtscore.training.mlp import train_model

        rng = np.random.RandomState(5)
        X = torch.tensor(rng.randn(6, 16).astype(np.float32))
        y = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).unsqueeze(1)

        model = train_model(X, y, 16)
        with torch.no_grad():
            raw = model(X).squeeze(1).tolist()
            sigmoided = torch.sigmoid(model(X)).squeeze(1).tolist()

        # Raw logits and sigmoided scores should differ
        assert raw != sigmoided
        # Sigmoided scores should be in [0, 1]
        for s in sigmoided:
            assert 0.0 <= s <= 1.0


class TestTrainModelEpochs:
    """Tests for env-tunable epochs and early-stopping on loss plateau."""

    @staticmethod
    def _count_optimizer_steps(fn):
        """Run ``fn`` and return the number of ``Adam.step`` invocations.

        The training loop calls ``optimizer.step()`` exactly once per epoch,
        so this counter equals the number of epochs actually executed.
        """
        calls = {"n": 0}
        real_step = torch.optim.Adam.step

        def counting_step(self, *args, **kwargs):
            calls["n"] += 1
            return real_step(self, *args, **kwargs)

        with unittest.mock.patch.object(torch.optim.Adam, "step", counting_step):
            fn()
        return calls["n"]

    def test_early_stop_fires_on_loss_plateau(self):
        """With a small patience, training stops well before TRAIN_EPOCHS."""
        import vtscore.config as config
        from vtscore.training import mlp

        saved_epochs = config.TRAIN_EPOCHS
        saved_patience = config.TRAIN_PATIENCE
        config.TRAIN_EPOCHS = 500
        try:
            rng = np.random.RandomState(11)
            # Trivially separable so the loss plateaus quickly.
            good = rng.randn(8, 16).astype(np.float32) + 5.0
            bad = rng.randn(8, 16).astype(np.float32) - 5.0
            X = torch.tensor(np.vstack([good, bad]))
            y = torch.tensor([1.0] * 8 + [0.0] * 8).unsqueeze(1)

            config.TRAIN_PATIENCE = 0
            full_epochs = self._count_optimizer_steps(lambda: mlp.train_model(X, y, 16))

            config.TRAIN_PATIENCE = 5
            stopped_epochs = self._count_optimizer_steps(lambda: mlp.train_model(X, y, 16))

            assert full_epochs == 500
            assert stopped_epochs < full_epochs
        finally:
            config.TRAIN_EPOCHS = saved_epochs
            config.TRAIN_PATIENCE = saved_patience

    def test_patience_zero_disables_early_stop(self):
        """``TRAIN_PATIENCE=0`` should always run the full ``TRAIN_EPOCHS``."""
        import vtscore.config as config
        from vtscore.training import mlp

        saved_epochs = config.TRAIN_EPOCHS
        saved_patience = config.TRAIN_PATIENCE
        config.TRAIN_EPOCHS = 42
        config.TRAIN_PATIENCE = 0
        try:
            rng = np.random.RandomState(2)
            X = torch.tensor(rng.randn(8, 16).astype(np.float32))
            y = torch.tensor([1.0] * 4 + [0.0] * 4).unsqueeze(1)
            n_epochs = self._count_optimizer_steps(lambda: mlp.train_model(X, y, 16))
            assert n_epochs == 42
        finally:
            config.TRAIN_EPOCHS = saved_epochs
            config.TRAIN_PATIENCE = saved_patience


class TestCalibrationSkippedForTinyLabels:
    """``train_and_score`` should skip cross-calibration when there are
    too few labels for the result to be useful — regardless of
    ``safe_thresholds``.  Calibration costs two 200-epoch trainings per
    call, and below the blend's ramp floor those trainings are either
    discarded (safe_thresholds=True) or trained on too little data to
    be reliable (safe_thresholds=False)."""

    def test_skips_calibration_when_safe_and_under_six_labels(self):
        """With safe_thresholds=True and n_labels<6, calculate_cross_calibration_threshold
        must not be invoked — its output is entirely discarded by the blender."""
        from vtscore.detectors import training as detector_training
        from vtscore.training import thresholds

        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4, 5]})  # 5 labels < 6

        with unittest.mock.patch.object(
            thresholds,
            "calculate_cross_calibration_threshold",
            side_effect=AssertionError("calibration should be skipped for tiny label sets"),
        ) as patched:
            _, threshold, _model = detector_training.train_and_score(
                app_module.medias,
                app_module.good_votes,
                app_module.bad_votes,
                safe_thresholds=True,
            )
        patched.assert_not_called()
        assert 0.0 <= threshold <= 1.0

    def test_skips_calibration_when_safe_off_and_under_six_labels(self):
        """With safe_thresholds=False and n_labels<6, calibration is still
        skipped — fold trainings are unreliable with so few labels, and the
        gate is purely a function of n_labels."""
        from vtscore.detectors import training as detector_training
        from vtscore.training import thresholds

        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4, 5]})

        with unittest.mock.patch.object(
            thresholds,
            "calculate_cross_calibration_threshold",
            side_effect=AssertionError("calibration should be skipped for tiny label sets"),
        ) as patched:
            detector_training.train_and_score(
                app_module.medias,
                app_module.good_votes,
                app_module.bad_votes,
                safe_thresholds=False,
            )
        patched.assert_not_called()

    def test_still_calibrates_when_enough_labels(self):
        """With safe_thresholds=True and n_labels>=6, calibration still runs."""
        from vtscore.detectors import training as detector_training
        from vtscore.training import thresholds

        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})  # 6 labels

        with unittest.mock.patch.object(
            thresholds,
            "calculate_cross_calibration_threshold",
            wraps=thresholds.calculate_cross_calibration_threshold,
        ) as patched:
            detector_training.train_and_score(
                app_module.medias,
                app_module.good_votes,
                app_module.bad_votes,
                safe_thresholds=True,
            )
        assert patched.call_count == 1


class TestCalibrateCountEnvDefault:
    """``VTSEARCH_CALIBRATE_COUNT`` should drive the settings default."""

    def test_default_calibrate_count_constant_exists(self):
        from vtscore import config


        assert isinstance(config.DEFAULT_CALIBRATE_COUNT, int)
        assert config.DEFAULT_CALIBRATE_COUNT >= 1


class TestCalibrationCache:
    """When the same labels are passed twice in a row with the same settings,
    the cross-calibration trainings should be skipped on the second call."""

    def _det_ctx(self):
        from vtscore.state.core import DetectorContext

        return DetectorContext("test-det")

    def _seed_six_labels(self):
        # Six labels puts us above the ``< 6`` skip floor so calibration
        # actually runs (and can therefore be cached).
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})

    def test_second_call_with_same_inputs_skips_calibration(self):
        from vtscore.detectors import training as detector_training
        from vtscore.training import thresholds

        self._seed_six_labels()
        det_ctx = self._det_ctx()

        detector_training.train_and_score(
            app_module.medias,
            app_module.good_votes,
            app_module.bad_votes,
            det_ctx=det_ctx,
        )
        assert det_ctx.calibration_cache is not None

        with unittest.mock.patch.object(
            thresholds,
            "calculate_cross_calibration_threshold",
            side_effect=AssertionError("calibration should be cached on repeat call"),
        ) as patched:
            detector_training.train_and_score(
                app_module.medias,
                app_module.good_votes,
                app_module.bad_votes,
                det_ctx=det_ctx,
            )
        patched.assert_not_called()

    def test_second_call_returns_same_threshold(self):
        from vtscore.detectors import training as detector_training

        self._seed_six_labels()
        det_ctx = self._det_ctx()

        _, t1, _ = detector_training.train_and_score(
            app_module.medias,
            app_module.good_votes,
            app_module.bad_votes,
            det_ctx=det_ctx,
        )
        _, t2, _ = detector_training.train_and_score(
            app_module.medias,
            app_module.good_votes,
            app_module.bad_votes,
            det_ctx=det_ctx,
        )
        assert t1 == t2

    def test_label_change_invalidates_cache(self):
        from vtscore.detectors import training as detector_training
        from vtscore.training import thresholds

        self._seed_six_labels()
        det_ctx = self._det_ctx()

        detector_training.train_and_score(
            app_module.medias,
            app_module.good_votes,
            app_module.bad_votes,
            det_ctx=det_ctx,
        )
        assert det_ctx.calibration_cache is not None
        first_key = det_ctx.calibration_cache[0]

        # Flip one media's label — calibration must recompute.
        app_module.good_votes.pop(3)
        app_module.bad_votes[3] = None

        with unittest.mock.patch.object(
            thresholds,
            "calculate_cross_calibration_threshold",
            wraps=thresholds.calculate_cross_calibration_threshold,
        ) as patched:
            detector_training.train_and_score(
                app_module.medias,
                app_module.good_votes,
                app_module.bad_votes,
                det_ctx=det_ctx,
            )
        assert patched.call_count == 1
        assert det_ctx.calibration_cache is not None
        assert det_ctx.calibration_cache[0] != first_key

    def test_inclusion_change_invalidates_cache(self):
        from vtscore.detectors import training as detector_training
        from vtscore.training import thresholds

        self._seed_six_labels()
        det_ctx = self._det_ctx()

        detector_training.train_and_score(
            app_module.medias,
            app_module.good_votes,
            app_module.bad_votes,
            inclusion_value=0,
            det_ctx=det_ctx,
        )

        with unittest.mock.patch.object(
            thresholds,
            "calculate_cross_calibration_threshold",
            wraps=thresholds.calculate_cross_calibration_threshold,
        ) as patched:
            detector_training.train_and_score(
                app_module.medias,
                app_module.good_votes,
                app_module.bad_votes,
                inclusion_value=2,
                det_ctx=det_ctx,
            )
        assert patched.call_count == 1

    def test_no_cache_when_det_ctx_missing(self):
        """Without a det_ctx, every call must recompute calibration."""
        from vtscore.detectors import training as detector_training
        from vtscore.training import thresholds

        self._seed_six_labels()

        with unittest.mock.patch.object(
            thresholds,
            "calculate_cross_calibration_threshold",
            wraps=thresholds.calculate_cross_calibration_threshold,
        ) as patched:
            detector_training.train_and_score(
                app_module.medias,
                app_module.good_votes,
                app_module.bad_votes,
            )
            detector_training.train_and_score(
                app_module.medias,
                app_module.good_votes,
                app_module.bad_votes,
            )
        assert patched.call_count == 2

    def test_settings_default_matches_config(self):
        from vtsearch import settings

        from vtscore import config

        assert settings._DEFAULTS["calibrate_count"] == config.DEFAULT_CALIBRATE_COUNT


class TestLearnedSort:
    def test_returns_all_clips(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == app_module.NUM_MEDIAS
        assert "threshold" in data

    def test_result_fields(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        data = resp.get_json()
        for entry in data["results"]:
            assert "id" in entry
            assert "score" in entry

    def test_sorted_descending(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        data = resp.get_json()
        scores = [e["score"] for e in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_all_media_ids_present(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        data = resp.get_json()
        ids = {e["id"] for e in data["results"]}
        assert ids == set(range(1, app_module.NUM_MEDIAS + 1))

    def test_only_good_votes_returns_400(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 400

    def test_only_bad_votes_returns_400(self, client):
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 400

    def test_scores_in_valid_range(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        data = resp.get_json()
        for entry in data["results"]:
            assert 0.0 <= entry["score"] <= 1.0


class TestLearnedSortAsync:
    """The endpoint now hands the work off to a background thread and the
    client polls ``/api/learned-sort/result?job_id=...`` until done."""

    def test_async_returns_job_id_then_polling_yields_result(self, client):
        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import learned_sort_jobs

        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})

        resp = client.post("/api/learned-sort", json={})
        assert resp.status_code == 200
        envelope = resp.get_json()
        assert envelope["status"] in ("running", "done")
        job_id = envelope["job_id"]

        # Wait for the background thread to finish before polling, since the
        # test client otherwise sees the still-running snapshot.
        _wait_for_job(learned_sort_jobs)

        result = client.get(f"/api/learned-sort/result?job_id={job_id}").get_json()
        assert result["status"] == "done"
        assert result["job_id"] == job_id
        assert "results" in result and len(result["results"]) > 0
        assert "threshold" in result

    def test_unchanged_votes_short_circuit_to_cached(self, client):
        """The signature cache lets re-sorts skip training entirely."""
        from vtscore.concurrency.async_jobs import learned_sort_jobs

        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})

        first = client.post("/api/learned-sort", json={"wait": True}).get_json()
        assert first["status"] == "done"
        first_job_id = first["job_id"]

        # Second call with the same signature should reuse the cached result —
        # job_id is the original job's id and we get back done immediately
        # without ``wait=true``.
        second = client.post("/api/learned-sort", json={}).get_json()
        assert second["status"] == "done"
        assert second["job_id"] == first_job_id

        # Cache invalidates when votes change.
        app_module.bad_votes.update({5: None})
        third = client.post("/api/learned-sort", json={"wait": True}).get_json()
        assert third["status"] == "done"
        assert third["job_id"] != first_job_id

        learned_sort_jobs.reset_for_tests()

    def test_polling_unknown_job_returns_404(self, client):
        # 404s are intercepted by the app-level ``NotFound`` errorhandler
        # in ``app.py``, which renders the legacy
        # ``{"error": "Not Found", "request_id": ...}`` envelope.
        # Frontends rely on the HTTP status code for the missing-job
        # branch rather than a body field.
        resp = client.get("/api/learned-sort/result?job_id=does-not-exist")
        assert resp.status_code == 404

    def test_polling_without_job_id_returns_422(self, client):
        # Schema-level validation: the marshmallow
        # ``LearnedSortResultQuerySchema`` rejects requests without a
        # ``job_id`` query parameter as 422 with the standard ``errors``
        # envelope.
        resp = client.get("/api/learned-sort/result")
        assert resp.status_code == 422
        assert "job_id" in resp.get_json()["errors"]["query"]


class TestEvalTrainAndScoreAsync:
    """The eval train-and-score endpoint mirrors the learned-sort pattern:
    return a job envelope, poll a result endpoint, short-circuit unchanged
    runs via the signature cache."""

    def _seed_history(self):
        from vtsearch.state import label_history

        # A handful of "good" votes are enough to exercise the smart metric.
        for cid, lbl in [(1, "good"), (2, "good"), (3, "bad"), (4, "bad")]:
            if lbl == "good":
                app_module.good_votes[cid] = None
            else:
                app_module.bad_votes[cid] = None
            label_history.append((cid, lbl, 0.0))

    def test_wait_returns_metric_inline(self, client):
        self._seed_history()
        resp = client.post("/api/eval/train-and-score", json={"metric": "smart", "wait": True})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "done"
        assert data["metric"] == "smart"
        assert "error_cost" in data

    def test_async_polls_to_done(self, client):
        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import eval_jobs

        self._seed_history()
        envelope = client.post("/api/eval/train-and-score", json={"metric": "stable"}).get_json()
        assert envelope["status"] in ("running", "done")
        job_id = envelope["job_id"]

        _wait_for_job(eval_jobs)

        result = client.get(f"/api/eval/train-and-score/result?job_id={job_id}").get_json()
        assert result["status"] == "done"
        assert result["metric"] == "stable"
        assert "stability" in result

    def test_invalid_metric_rejected(self, client):
        resp = client.post("/api/eval/train-and-score", json={"metric": "bogus", "wait": True})
        assert resp.status_code == 422


class TestExampleSort:
    def test_sort_with_audio_file(self, client):
        # Create a test WAV file in memory
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        result_data = resp.get_json()
        assert "results" in result_data
        assert "threshold" in result_data
        assert len(result_data["results"]) == app_module.NUM_MEDIAS

    def test_sort_results_sorted_descending(self, client):
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        result_data = resp.get_json()
        similarities = [e["similarity"] for e in result_data["results"]]
        assert similarities == sorted(similarities, reverse=True)

    def test_sort_similarity_in_valid_range(self, client):
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        result_data = resp.get_json()
        for entry in result_data["results"]:
            assert -1.0 <= entry["similarity"] <= 1.0

    def test_sort_no_file(self, client):
        resp = client.post("/api/example-sort", data={})
        assert resp.status_code == 400

    def test_sort_empty_filename(self, client):
        data = {"file": (io.BytesIO(b""), "")}
        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400


class TestTextsortSuggestions:
    def test_get_empty(self, client):
        resp = client.get("/api/textsort-suggestions")
        assert resp.status_code == 200
        assert resp.get_json() == {"suggestions": []}

    def test_add_and_get(self, client):
        resp = client.post("/api/textsort-suggestions", json={"text": "dog barking"})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}

        resp = client.get("/api/textsort-suggestions")
        assert resp.get_json()["suggestions"] == ["dog barking"]

    def test_multiple_suggestions_ordered(self, client):
        client.post("/api/textsort-suggestions", json={"text": "birds"})
        client.post("/api/textsort-suggestions", json={"text": "rain"})
        client.post("/api/textsort-suggestions", json={"text": "thunder"})

        resp = client.get("/api/textsort-suggestions")
        assert resp.get_json()["suggestions"] == ["birds", "rain", "thunder"]

    def test_duplicate_moves_to_end(self, client):
        client.post("/api/textsort-suggestions", json={"text": "birds"})
        client.post("/api/textsort-suggestions", json={"text": "rain"})
        client.post("/api/textsort-suggestions", json={"text": "birds"})

        resp = client.get("/api/textsort-suggestions")
        assert resp.get_json()["suggestions"] == ["rain", "birds"]

    def test_empty_text_returns_400(self, client):
        resp = client.post("/api/textsort-suggestions", json={"text": ""})
        assert resp.status_code == 400

    def test_missing_text_returns_422(self, client):
        # Schema-level validation: the marshmallow
        # ``TextsortSuggestionRequestSchema`` rejects requests without a
        # ``text`` key as 422 with the standard ``errors`` envelope.
        resp = client.post("/api/textsort-suggestions", json={"other": "x"})
        assert resp.status_code == 422
        assert "text" in resp.get_json()["errors"]["json"]

    def test_whitespace_only_returns_400(self, client):
        resp = client.post("/api/textsort-suggestions", json={"text": "   "})
        assert resp.status_code == 400

    def test_cleared_with_votes(self, client):
        """Suggestions are cleared when votes are cleared."""
        client.post("/api/textsort-suggestions", json={"text": "cat meowing"})
        resp = client.get("/api/textsort-suggestions")
        assert len(resp.get_json()["suggestions"]) == 1

        from vtsearch.state import clear_votes

        clear_votes()

        resp = client.get("/api/textsort-suggestions")
        assert resp.get_json()["suggestions"] == []


class TestLoadEmbedderConcurrentCallback:
    """Verify _load_embedder_with_progress does not trample _on_progress."""

    def test_lock_exists(self):
        """The module-level lock must exist to serialise concurrent callers."""
        import threading

        from vtsearch.routes.sorting import _embedder_load_lock

        assert isinstance(_embedder_load_lock, type(threading.Lock()))

    def test_concurrent_calls_restore_original_callback(self):
        """Two threads calling _load_embedder_with_progress must leave
        _on_progress set to the *original* callback, not a stale lambda."""
        import threading
        import time
        from unittest.mock import MagicMock

        from vtsearch.routes.sorting import _load_embedder_with_progress

        original_cb = MagicMock(name="original_cb")
        mock_mt = MagicMock()
        mock_mt._model = None  # force "needs loading"
        mock_mt._on_progress = original_cb

        def slow_load_models():
            """Simulate a slow load; first call loads, second sees it loaded."""
            time.sleep(0.05)
            mock_mt._model = True  # mark as loaded

        mock_mt.load_models = slow_load_models

        with unittest.mock.patch("vtscore.media.get", return_value=mock_mt):
            t1 = threading.Thread(target=_load_embedder_with_progress, args=("audio", 5))
            t2 = threading.Thread(target=_load_embedder_with_progress, args=("audio", 5))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        # The lock ensures thread 1 finishes (restores original_cb) before
        # thread 2 enters; thread 2 sees _model is loaded and returns early.
        assert mock_mt._on_progress is original_cb, (
            "_on_progress was not restored to the original callback after concurrent calls"
        )

    def test_callback_restored_after_load_error(self):
        """_on_progress must be restored even when load_models raises."""
        import pytest
        from unittest.mock import MagicMock

        from vtsearch.routes.sorting import _load_embedder_with_progress

        original_cb = MagicMock(name="original_cb")
        mock_emb = MagicMock()
        mock_emb._model = None
        mock_emb._on_progress = original_cb
        mock_emb.load_models.side_effect = RuntimeError("boom")

        with unittest.mock.patch("vtsearch.routes.sorting._get_embedder_for_loaded_data", return_value=mock_emb):
            with pytest.raises(RuntimeError):
                _load_embedder_with_progress("audio", 5)

        assert mock_emb._on_progress is original_cb
