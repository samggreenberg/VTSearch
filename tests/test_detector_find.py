"""Detector find-label tests.

Covers POST /api/find-label, demo-origin resolution, and stamp-demo-origin.
"""

import numpy as np

import app as app_module
from helpers import train_detector_from_votes


class TestFindLabel:
    """Tests for POST /api/find-label — score all medias and apply Good/Bad labels."""

    def _create_model_with_detector(self, client):
        """Helper: train a detector, register it as autorun, register in model registry."""
        from vtsearch.models.registry import register_model, reset_for_tests
        from vtsearch.utils import add_autorun_detector

        reset_for_tests()

        # Train a detector from votes
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        detector = train_detector_from_votes()

        # Register as autorun detector
        det_name = "find-label-det"
        add_autorun_detector(
            det_name,
            "audio",
            weights=detector["weights"],
            threshold=detector["threshold"],
        )

        # Register in model registry
        entry = register_model(
            name="Find Label Test Model",
            media_type="audio",
            trainable=False,
            detector_name=det_name,
        )

        # Clear the training votes
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        return entry["id"]

    def test_find_label_marks_all_items(self, client):
        """find-label should mark every loaded media as good or bad."""
        model_id = self._create_model_with_detector(client)
        resp = client.post("/api/find-label", json={"model_id": model_id})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        total = data["good_count"] + data["bad_count"]
        assert total == app_module.NUM_MEDIAS

    def test_find_label_activates_selected_dataset(self, client):
        """find-label with dataset_id should score against that dataset, not the active one.

        Reproduces the bug: user selects Dataset A + Detector 1 on the dashboard
        and clicks Find, but scoring runs against Dataset B because it was the
        most recently loaded into the labeling interface.
        """
        from vtsearch.utils import (
            DatasetContext,
            get_thread_dataset_context,
            register_context,
            set_thread_dataset_context,
            snapshot_medias,
            unregister_context,
        )

        model_id = self._create_model_with_detector(client)

        # Remember the default dataset (has test medias)
        original_ctx = get_thread_dataset_context()
        original_id = original_ctx.dataset_id if original_ctx else None
        original_count = len(snapshot_medias())

        # Create a second dataset with different (fewer) items
        rng = np.random.default_rng(99)
        ctx_b = DatasetContext("dataset_b")
        for i in range(1, 4):
            ctx_b.medias[i] = {
                "id": i,
                "md5": f"dataset_b_{i}",
                "embedding": rng.standard_normal(512).astype(np.float32),
            }
        register_context(ctx_b)

        try:
            # Switch active to dataset_b — simulates user browsing Dataset B
            set_thread_dataset_context(ctx_b)
            assert get_thread_dataset_context().dataset_id == "dataset_b"

            # Call find-label with dataset_id pointing to the ORIGINAL dataset
            resp = client.post(
                "/api/find-label",
                json={"model_id": model_id, "dataset_id": original_id},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            # Should have scored the original dataset's items, not dataset_b's 3 items
            total = data["good_count"] + data["bad_count"]
            assert total == original_count
        finally:
            set_thread_dataset_context(original_ctx)
            unregister_context("dataset_b")

    def test_find_label_overwrites_existing_votes(self, client):
        """find-label must mark ALL items even when votes already exist.

        Reproduces the bug: train on Dataset A, load Dataset B (which reuses
        integer IDs), run Find — previously, items whose IDs matched stale
        votes were skipped and never marked.
        """
        model_id = self._create_model_with_detector(client)

        # Simulate stale votes from a previous dataset (all IDs overlap)
        app_module.good_votes.update({k: None for k in range(1, 11)})
        app_module.bad_votes.update({k: None for k in range(11, 21)})

        resp = client.post("/api/find-label", json={"model_id": model_id})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        total = data["good_count"] + data["bad_count"]
        assert total == app_module.NUM_MEDIAS

    def test_find_label_trainable_model_with_labelset(self, client, tmp_path):
        """find-label should train on-the-fly from a trainable model's labelset.

        Reproduces the core bug: user creates a trainable model, labels items
        on Dataset A (labelset auto-saved), loads Dataset B, runs Find.
        Previously returned 400 because no pre-trained weights existed.
        """
        from vtsearch.models.registry import register_model, reset_for_tests
        from vtsearch.routes.trainable_models import _write_model
        from vtsearch.datasets.labelset import LabelSet
        from vtsearch.utils import bad_votes, good_votes, snapshot_medias

        reset_for_tests()

        # Simulate labeling: vote on some items
        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [18, 19, 20]})

        # Build a labelset from the current votes (as sync_labels_to_loaded_model does)
        snap = snapshot_medias()
        labelset = LabelSet.from_clips_and_votes(snap, good_votes, bad_votes, expand_dupes=False)

        # Write a trainable model file with the labelset but NO weights
        from vtsearch.settings import get_trainable_models_dir, set_trainable_models_dir

        original_dir = get_trainable_models_dir()
        set_trainable_models_dir(tmp_path)
        try:
            tm_name = "test-find-trainable"
            tm_path = tmp_path / f"{tm_name}.json"
            _write_model(
                tm_path,
                {
                    "name": tm_name,
                    "text_query": "",
                    "examples": [],
                    "labelset": labelset.to_dict(),
                },
            )

            # Register in model registry as a trainable model (no detector_name weights)
            entry = register_model(
                name="Trainable Find Test",
                media_type="audio",
                trainable=True,
                trainable_model_name=tm_name,
            )
            model_id = entry["id"]

            # Clear training votes (simulates loading a new dataset)
            good_votes.clear()
            bad_votes.clear()

            # Run find-label — should train on-the-fly from the labelset
            resp = client.post("/api/find-label", json={"model_id": model_id})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            total = data["good_count"] + data["bad_count"]
            assert total == app_module.NUM_MEDIAS
        finally:
            set_trainable_models_dir(original_dir)

    def test_find_label_cross_dataset_resolves_from_origin(self, client, tmp_path):
        """find-label should resolve labels from origin when MD5s don't match.

        Simulates the true cross-dataset scenario: train on Dataset A (labels
        saved with origins pointing to files on disk), load completely
        different Dataset B (no MD5 overlap), click Find.  The resolver must
        follow each label's origin trail to the original file, embed it, and
        train an MLP on-the-fly.
        """
        from unittest.mock import patch

        from vtsearch.models.registry import register_model, reset_for_tests
        from vtsearch.routes.trainable_models import _write_model
        from vtsearch.settings import get_trainable_models_dir, set_trainable_models_dir
        from vtsearch.utils import medias

        reset_for_tests()

        # --- Phase 1: build label folder on disk (simulates Dataset A files) ---
        label_folder = tmp_path / "dataset_a"
        label_folder.mkdir()
        for i in range(3):
            (label_folder / f"good_{i}.wav").write_bytes(f"good_audio_{i}".encode())
        for i in range(3):
            (label_folder / f"bad_{i}.wav").write_bytes(f"bad_audio_{i}".encode())

        label_origin = {
            "importer": "server_folder",
            "params": {"path": str(label_folder), "media_type": "audio"},
        }

        # Build labelset entries with origins pointing to the folder
        label_entries = []
        for i in range(3):
            label_entries.append(
                {
                    "md5": f"dataset_a_good_{i}",
                    "label": "good",
                    "origin": label_origin,
                    "origin_name": f"good_{i}.wav",
                    "filename": f"good_{i}.wav",
                }
            )
        for i in range(3):
            label_entries.append(
                {
                    "md5": f"dataset_a_bad_{i}",
                    "label": "bad",
                    "origin": label_origin,
                    "origin_name": f"bad_{i}.wav",
                    "filename": f"bad_{i}.wav",
                }
            )

        # --- Phase 2: write trainable model with these labels ---
        original_dir = get_trainable_models_dir()
        set_trainable_models_dir(tmp_path)
        try:
            tm_name = "test-cross-dataset"
            tm_path = tmp_path / f"{tm_name}.json"
            _write_model(
                tm_path,
                {
                    "name": tm_name,
                    "text_query": "",
                    "media_type": "audio",
                    "examples": [],
                    "labelset": {"labels": label_entries},
                },
            )

            entry = register_model(
                name="Cross Dataset Test",
                media_type="audio",
                trainable=True,
                trainable_model_name=tm_name,
            )
            model_id = entry["id"]

            # --- Phase 3: replace medias with completely different Dataset B ---
            # (no MD5 overlap, forcing origin resolution)
            saved = dict(medias)
            medias.clear()
            rng = np.random.default_rng(42)
            for i in range(1, 21):
                medias[i] = {
                    "id": i,
                    "type": "audio",
                    "embedding": rng.standard_normal(512).astype(np.float32),
                    "md5": f"dataset_b_md5_{i}",
                    "filename": f"dataset_b_{i}.wav",
                    "origin": {"importer": "server_folder", "params": {"path": "/other"}},
                    "origin_name": f"dataset_b_{i}.wav",
                }

            # Mock embed_file to return deterministic vectors
            good_emb = rng.standard_normal(512).astype(np.float32)
            bad_emb = rng.standard_normal(512).astype(np.float32)

            def fake_embed(path, media_type):
                from pathlib import Path

                name = Path(path).name
                if "good" in name:
                    return good_emb.copy()
                return bad_emb.copy()

            try:
                with patch("vtsearch.models.resolver.embed_file", side_effect=fake_embed):
                    resp = client.post("/api/find-label", json={"model_id": model_id})

                assert resp.status_code == 200, f"Expected 200 but got {resp.status_code}: {resp.get_json()}"
                data = resp.get_json()
                assert data["ok"] is True
                total = data["good_count"] + data["bad_count"]
                assert total == 20  # all Dataset B items scored
            finally:
                medias.clear()
                medias.update(saved)
        finally:
            set_trainable_models_dir(original_dir)

    def test_find_label_cross_dataset_error_includes_diagnostics(self, client, tmp_path):
        """When origin resolution fails, the error response should include diagnostics."""
        from vtsearch.models.registry import register_model, reset_for_tests
        from vtsearch.routes.trainable_models import _write_model
        from vtsearch.settings import get_trainable_models_dir, set_trainable_models_dir
        from vtsearch.utils import medias

        reset_for_tests()

        # Labels point to a nonexistent folder — resolution will fail
        bad_origin = {
            "importer": "server_folder",
            "params": {"path": "/nonexistent/dataset_a"},
        }
        label_entries = [
            {
                "md5": "no_match_g",
                "label": "good",
                "origin": bad_origin,
                "origin_name": "good.wav",
                "filename": "good.wav",
            },
            {
                "md5": "no_match_b",
                "label": "bad",
                "origin": bad_origin,
                "origin_name": "bad.wav",
                "filename": "bad.wav",
            },
        ]

        original_dir = get_trainable_models_dir()
        set_trainable_models_dir(tmp_path)
        try:
            tm_name = "test-diag"
            _write_model(
                tmp_path / f"{tm_name}.json",
                {
                    "name": tm_name,
                    "text_query": "",
                    "media_type": "audio",
                    "examples": [],
                    "labelset": {"labels": label_entries},
                },
            )

            entry = register_model(
                name="Diag Test",
                media_type="audio",
                trainable=True,
                trainable_model_name=tm_name,
            )

            # Ensure medias have no MD5 overlap
            saved = dict(medias)
            medias.clear()
            rng = np.random.default_rng(99)
            for i in range(1, 6):
                medias[i] = {
                    "id": i,
                    "type": "audio",
                    "embedding": rng.standard_normal(512).astype(np.float32),
                    "md5": f"other_{i}",
                    "filename": f"other_{i}.wav",
                    "origin": {"importer": "test", "params": {}},
                    "origin_name": f"other_{i}.wav",
                }

            try:
                resp = client.post("/api/find-label", json={"model_id": entry["id"]})
                assert resp.status_code == 400
                data = resp.get_json()

                # Error message should describe the resolution failure
                assert "could not be trained" in data["error"]
                assert "failed to resolve" in data["error"]

                # Should include structured diagnostics
                diag = data["resolution_diagnostic"]
                assert diag["total_labels"] == 2
                assert diag["md5_matched"] == 0
                assert diag["needed_resolution"] == 2
                assert diag["resolved_from_origin"] == 0
                assert diag["failed_resolution"] == 2
                assert "sample_failures" in diag
                assert len(diag["sample_failures"]) > 0
                assert diag["sample_failures"][0]["origin"]["importer"] == "server_folder"
            finally:
                medias.clear()
                medias.update(saved)
        finally:
            set_trainable_models_dir(original_dir)

    def test_find_label_does_not_overwrite_training_labels(self, client, tmp_path):
        """Running Find must NOT overwrite the model's saved training labels.

        Reproduces the bug: train on Dataset A (6 labels), load Dataset B,
        run Find → the model's labelset on disk should still have exactly 6
        training labels, not the N scoring labels from Dataset B.
        """
        from vtsearch.datasets.labelset import LabelSet
        from vtsearch.models.registry import add_loaded_model_id, register_model, reset_for_tests
        from vtsearch.models.label_sync import sync_labels_to_loaded_model
        from vtsearch.routes.trainable_models import _read_model, _write_model
        from vtsearch.settings import get_trainable_models_dir, set_trainable_models_dir
        from vtsearch.utils import bad_votes, good_votes, set_thread_detector_context, snapshot_medias
        from vtsearch.utils.state_core import DetectorContext, register_detector_context

        reset_for_tests()

        # --- Phase 1: simulate training on "Dataset A" (6 labels) ---
        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [18, 19, 20]})

        snap = snapshot_medias()
        labelset = LabelSet.from_clips_and_votes(snap, good_votes, bad_votes, expand_dupes=False)
        original_label_count = len(labelset)
        assert original_label_count == 6

        original_dir = get_trainable_models_dir()
        set_trainable_models_dir(tmp_path)
        try:
            tm_name = "persist-test"
            tm_path = tmp_path / f"{tm_name}.json"
            _write_model(
                tm_path,
                {
                    "name": tm_name,
                    "text_query": "",
                    "examples": [],
                    "labelset": labelset.to_dict(),
                },
            )

            entry = register_model(
                name="Persist Test",
                media_type="audio",
                trainable=True,
                trainable_model_name=tm_name,
            )
            model_id = entry["id"]
            add_loaded_model_id(model_id)
            det_ctx = DetectorContext(model_id)
            register_detector_context(det_ctx)
            set_thread_detector_context(det_ctx)

            # --- Phase 2: simulate loading Dataset B (clear votes) ---
            good_votes.clear()
            bad_votes.clear()

            # --- Phase 3: run Find → scores all N medias, applies labels ---
            resp = client.post("/api/find-label", json={"model_id": model_id})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            total_scored = data["good_count"] + data["bad_count"]
            assert total_scored == app_module.NUM_MEDIAS

            # --- Phase 4: trigger sync (as a subsequent vote would) ---
            sync_labels_to_loaded_model()

            # --- Assert: the saved labelset must still have 6 labels ---
            saved = _read_model(tm_path)
            saved_labels = saved["labelset"]["labels"]
            assert len(saved_labels) == original_label_count, (
                f"Expected {original_label_count} training labels but got "
                f"{len(saved_labels)} — find-label scoring overwrote training data"
            )
        finally:
            set_trainable_models_dir(original_dir)

    def test_find_mode_cleared_on_model_load(self, client):
        """Loading a new model should clear find mode so training syncs resume."""
        from vtsearch.models.registry import is_find_mode, reset_for_tests, set_find_mode

        reset_for_tests()
        set_find_mode(True)
        assert is_find_mode()

        # Loading null model via endpoint clears find mode
        client.post("/api/models/registry/load", json={"model_id": None})
        assert not is_find_mode()

    def test_find_label_missing_model_id(self, client):
        resp = client.post("/api/find-label", json={})
        assert resp.status_code == 400

    def test_find_label_unknown_model_id(self, client):
        resp = client.post("/api/find-label", json={"model_id": "nonexistent"})
        assert resp.status_code == 404

    def test_find_label_reports_progress(self, client):
        """find-label should update find_progress with discrete steps."""
        from vtsearch.utils.progress import find_progress

        model_id = self._create_model_with_detector(client)

        # Capture progress snapshots by monkey-patching update
        snapshots = []
        original_update = find_progress.update

        def capturing_update(*args, **kwargs):
            original_update(*args, **kwargs)
            snapshots.append(find_progress.get())

        find_progress.update = capturing_update
        try:
            resp = client.post("/api/find-label", json={"model_id": model_id})
        finally:
            find_progress.update = original_update

        assert resp.status_code == 200

        # Should have progress snapshots from each phase
        assert len(snapshots) >= 3

        # Step 1: resolving model
        step1 = [s for s in snapshots if s.get("step") == 1]
        assert len(step1) >= 1
        assert step1[0]["status"] == "running"
        assert step1[0]["total_steps"] == 4

        # Step 3: scoring (step 2 skipped when weights exist)
        step3 = [s for s in snapshots if s.get("step") == 3]
        assert len(step3) >= 1
        assert "Scoring" in step3[0]["message"]
        assert step3[0]["total"] == app_module.NUM_MEDIAS

        # Step 4: applying labels
        step4 = [s for s in snapshots if s.get("step") == 4]
        assert len(step4) >= 1
        assert "Applying labels" in step4[0]["message"]

        # Final state should be idle
        final = find_progress.get()
        assert final["status"] == "idle"

    def test_find_label_progress_resets_on_error(self, client):
        """find_progress should reset to idle when find-label returns an error."""
        from vtsearch.models.registry import reset_for_tests
        from vtsearch.utils.progress import find_progress

        reset_for_tests()
        resp = client.post("/api/find-label", json={})
        assert resp.status_code == 400
        data = find_progress.get()
        assert data["status"] == "idle"

    def test_find_label_progress_resets_on_not_found(self, client):
        """find_progress should reset to idle when model is not found."""
        from vtsearch.utils.progress import find_progress

        resp = client.post("/api/find-label", json={"model_id": "nonexistent"})
        assert resp.status_code == 404
        data = find_progress.get()
        assert data["status"] == "idle"

    def test_find_label_progress_visible_via_endpoint(self, client):
        """GET /api/find/progress should reflect find-label progress."""
        resp = client.get("/api/find/progress")
        assert resp.status_code == 200
        data = resp.get_json()
        # Should be idle when nothing is running
        assert data["status"] == "idle"


class TestFindLabelDemoOrigin:
    """Tests for find-label with demo dataset origins — cross-dataset resolution."""

    def test_find_label_demo_origin_resolves(self, client, tmp_path):
        """find-label should resolve labels from demo origins when MD5s don't match.

        Simulates: train on demo dataset A, load dataset B, click Find.
        The resolver follows the demo origin to find files on disk.
        """
        from unittest.mock import patch

        from vtsearch.models.registry import register_model, reset_for_tests
        from vtsearch.routes.trainable_models import _write_model
        from vtsearch.settings import get_trainable_models_dir, set_trainable_models_dir
        from vtsearch.utils import medias

        reset_for_tests()

        # --- Phase 1: set up demo-like files on disk ---
        img_dir = tmp_path / "caltech-101" / "101_ObjectCategories"
        (img_dir / "kangaroo").mkdir(parents=True)
        for i in range(3):
            (img_dir / "kangaroo" / f"image_{i:04d}.jpg").write_bytes(f"good_img_{i}".encode())
        (img_dir / "butterfly").mkdir(parents=True)
        for i in range(3):
            (img_dir / "butterfly" / f"image_{i:04d}.jpg").write_bytes(f"bad_img_{i}".encode())

        # Build labelset with demo origins
        demo_origin = {"importer": "demo", "params": {"name": "caltech101_s"}}
        label_entries = []
        for i in range(3):
            label_entries.append(
                {
                    "md5": f"demo_a_good_{i}",
                    "label": "good",
                    "origin": demo_origin,
                    "origin_name": f"kangaroo/image_{i:04d}.jpg",
                    "filename": f"kangaroo/image_{i:04d}.jpg",
                }
            )
        for i in range(3):
            label_entries.append(
                {
                    "md5": f"demo_a_bad_{i}",
                    "label": "bad",
                    "origin": demo_origin,
                    "origin_name": f"butterfly/image_{i:04d}.jpg",
                    "filename": f"butterfly/image_{i:04d}.jpg",
                }
            )

        # --- Phase 2: write trainable model ---
        original_dir = get_trainable_models_dir()
        set_trainable_models_dir(tmp_path)

        import vtsearch.datasets.importers.demo as demo_mod

        old_source_dirs = demo_mod._SOURCE_DIRS
        demo_mod._SOURCE_DIRS = {"caltech101": img_dir}

        try:
            tm_name = "test-demo-cross"
            _write_model(
                tmp_path / f"{tm_name}.json",
                {
                    "name": tm_name,
                    "text_query": "",
                    "media_type": "image",
                    "examples": [],
                    "labelset": {"labels": label_entries},
                },
            )

            entry = register_model(
                name="Demo Cross Dataset Test",
                media_type="image",
                trainable=True,
                trainable_model_name=tm_name,
            )
            model_id = entry["id"]

            # --- Phase 3: load completely different dataset B ---
            saved = dict(medias)
            medias.clear()
            rng = np.random.default_rng(42)
            for i in range(1, 11):
                medias[i] = {
                    "id": i,
                    "type": "image",
                    "embedding": rng.standard_normal(512).astype(np.float32),
                    "md5": f"dataset_b_md5_{i}",
                    "filename": f"dataset_b_{i}.jpg",
                    "origin": {"importer": "demo", "params": {"name": "food101_a"}},
                    "origin_name": f"dataset_b_{i}.jpg",
                }

            # Mock embed_file to return deterministic vectors
            good_emb = rng.standard_normal(512).astype(np.float32)
            bad_emb = rng.standard_normal(512).astype(np.float32)

            def fake_embed(path, media_type):
                if "kangaroo" in str(path):
                    return good_emb.copy()
                return bad_emb.copy()

            try:
                with patch("vtsearch.models.resolver.embed_file", side_effect=fake_embed):
                    resp = client.post("/api/find-label", json={"model_id": model_id})

                assert resp.status_code == 200, f"Expected 200 but got {resp.status_code}: {resp.get_json()}"
                data = resp.get_json()
                assert data["ok"] is True
                total = data["good_count"] + data["bad_count"]
                assert total == 10  # all Dataset B items scored
            finally:
                medias.clear()
                medias.update(saved)
        finally:
            demo_mod._SOURCE_DIRS = old_source_dirs
            set_trainable_models_dir(original_dir)

    def test_find_label_demo_origin_empty_params_fails_with_warning(self, client, tmp_path):
        """Demo labels with empty origin params should produce a user-friendly warning.

        Simulates old pickles that stored demo origins without the dataset name.
        """
        from vtsearch.models.registry import register_model, reset_for_tests
        from vtsearch.routes.trainable_models import _write_model
        from vtsearch.settings import get_trainable_models_dir, set_trainable_models_dir
        from vtsearch.utils import medias

        reset_for_tests()

        # Origin with empty params (simulates old pickle bug)
        bad_origin = {"importer": "demo", "params": {}}
        label_entries = [
            {
                "md5": "old_good",
                "label": "good",
                "origin": bad_origin,
                "origin_name": "cat/img_001.jpg",
                "filename": "cat/img_001.jpg",
            },
            {
                "md5": "old_bad",
                "label": "bad",
                "origin": bad_origin,
                "origin_name": "dog/img_002.jpg",
                "filename": "dog/img_002.jpg",
            },
        ]

        original_dir = get_trainable_models_dir()
        set_trainable_models_dir(tmp_path)
        try:
            tm_name = "test-empty-demo"
            _write_model(
                tmp_path / f"{tm_name}.json",
                {
                    "name": tm_name,
                    "text_query": "",
                    "media_type": "image",
                    "examples": [],
                    "labelset": {"labels": label_entries},
                },
            )

            entry = register_model(
                name="Empty Demo Test",
                media_type="image",
                trainable=True,
                trainable_model_name=tm_name,
            )

            saved = dict(medias)
            medias.clear()
            rng = np.random.default_rng(99)
            for i in range(1, 6):
                medias[i] = {
                    "id": i,
                    "type": "image",
                    "embedding": rng.standard_normal(512).astype(np.float32),
                    "md5": f"other_{i}",
                    "filename": f"other_{i}.jpg",
                    "origin": {"importer": "demo", "params": {"name": "eurosat_a"}},
                    "origin_name": f"other_{i}.jpg",
                }

            try:
                resp = client.post("/api/find-label", json={"model_id": entry["id"]})
                assert resp.status_code == 400
                data = resp.get_json()

                # Should have a user-friendly warning field
                assert "warning" in data
                assert "could not be resolved" in data["warning"]
                assert "2" in data["warning"]  # total labels

                # Error message should mention resolution failure
                assert "failed to resolve" in data["error"]
            finally:
                medias.clear()
                medias.update(saved)
        finally:
            set_trainable_models_dir(original_dir)


class TestStampDemoOrigin:
    """Tests for _stamp_demo_origin ensuring pickle-cached loads get correct origins."""

    def test_stamps_origin_with_dataset_name(self):
        from vtsearch.datasets.loader import _stamp_demo_origin

        medias = {
            1: {"id": 1, "origin": {"importer": "demo", "params": {}}},
            2: {"id": 2, "origin": None},
        }
        _stamp_demo_origin(medias, "caltech101_s")
        for media in medias.values():
            assert media["origin"]["importer"] == "demo"
            assert media["origin"]["params"]["name"] == "caltech101_s"

    def test_stamps_converter_when_provided(self):
        from vtsearch.datasets.loader import _stamp_demo_origin

        medias = {1: {"id": 1, "origin": None}}
        _stamp_demo_origin(medias, "caltech101_s", converter_name="image2text")
        assert medias[1]["origin"]["params"]["converter"] == "image2text"

    def test_each_media_gets_independent_dict(self):
        from vtsearch.datasets.loader import _stamp_demo_origin

        medias = {1: {"id": 1, "origin": None}, 2: {"id": 2, "origin": None}}
        _stamp_demo_origin(medias, "caltech101_s")
        # Mutating one should not affect the other
        medias[1]["origin"]["params"]["extra"] = "test"
        assert "extra" not in medias[2]["origin"]["params"]
