"""Tests for the labelset-driven right-pane API surface.

These cover three concerns introduced together so that a detector loaded
against a different dataset still shows its training labels on the right:

* ``GET /api/detectors/<name>/labels-detail`` — read.
* ``POST /api/detectors/<name>/labels/<id>/vote`` — toggle on a
  saved labelset element (origin-keyed, dataset-agnostic).
* ``sync_labels_to_loaded_detector`` is now non-destructive across datasets:
  a vote in dataset B no longer wipes labels saved from dataset A.
"""

from __future__ import annotations

import shutil

import pytest

from vtsearch.settings import get_detectors_dir


@pytest.fixture(autouse=True)
def clean_detectors_dir():
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_cross_dataset_model(tm_name: str = "cross-ds-model", *, mark_loaded: bool = True) -> str:
    """Write a detector whose labelset references a *different* dataset.

    The labels' origins use importer ``"ds_a"`` so they don't resolve against
    the test medias (which use importer ``"test"``).

    When *mark_loaded* is ``True`` (default), also registers a
    :class:`DetectorContext` and marks the model as loaded — convenient for
    tests that exercise the labels-detail / vote APIs without running the
    real load task.  When ``False``, only the on-disk model + registry
    entry are created, so a subsequent ``POST /api/detectors/registry/load``
    triggers the full load task.
    """
    from vtsearch.models.detector_registry import (
        add_loaded_detector_id,
        register_detector,
        reset_for_tests,
    )
    from vtsearch.models.detector_store import _detector_path, _write_detector
    from vtsearch.utils.state_core import (
        DetectorContext,
        register_detector_context,
        set_thread_detector_context,
    )

    reset_for_tests()

    labelset = {
        "labels": [
            {
                "md5": "a1" * 16,
                "label": "good",
                "origin": {"importer": "ds_a", "params": {"size": "100"}},
                "origin_name": "alpha.wav",
                "filename": "alpha.wav",
            },
            {
                "md5": "b2" * 16,
                "label": "good",
                "origin": {"importer": "ds_a", "params": {"size": "100"}},
                "origin_name": "beta.wav",
                "filename": "beta.wav",
            },
            {
                "md5": "c3" * 16,
                "label": "bad",
                "origin": {"importer": "ds_a", "params": {"size": "100"}},
                "origin_name": "gamma.wav",
                "filename": "gamma.wav",
            },
        ]
    }
    _write_detector(
        _detector_path(tm_name),
        {
            "name": tm_name,
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "labelset": labelset,
        },
    )

    entry = register_detector(
        name=tm_name,
        media_type="audio",
    )
    detector_id = entry["id"]
    if mark_loaded:
        add_loaded_detector_id(detector_id)
        det_ctx = DetectorContext(detector_id)
        register_detector_context(det_ctx)
        set_thread_detector_context(det_ctx)
    return detector_id


# ---------------------------------------------------------------------------
# GET /api/detectors/<name>/labels-detail
# ---------------------------------------------------------------------------


class TestLabelsDetail:
    def test_returns_404_for_unknown_model(self, client):
        res = client.get("/api/detectors/does-not-exist/labels-detail")
        assert res.status_code == 404

    def test_lists_saved_labels_independent_of_loaded_dataset(self, client):
        """Right-pane data source must surface labels that don't resolve into
        the active dataset — the whole point of the labelset-driven pane."""
        _seed_cross_dataset_model()

        res = client.get("/api/detectors/cross-ds-model/labels-detail")
        assert res.status_code == 200
        body = res.get_json()

        assert body["media_type"] == "audio"
        assert len(body["good"]) == 2
        assert len(body["bad"]) == 1

        ids = {e["id"] for e in body["good"] + body["bad"]}
        assert len(ids) == 3, "ids must be unique per element"

        for elem in body["good"] + body["bad"]:
            # Cross-dataset: nothing resolves into the active dataset.
            assert elem["cid"] is None
            assert elem["media_type"] == "audio"
            assert elem["name"]  # display string non-empty


# ---------------------------------------------------------------------------
# POST /api/detectors/<name>/labels/<id>/vote
# ---------------------------------------------------------------------------


class TestLabelElementVote:
    def test_flip_label(self, client):
        _seed_cross_dataset_model()
        detail = client.get("/api/detectors/cross-ds-model/labels-detail").get_json()
        target = detail["good"][0]

        res = client.post(
            f"/api/detectors/cross-ds-model/labels/{target['id']}/vote",
            json={"vote": "bad"},
        )
        assert res.status_code == 200
        assert res.get_json()["action"] == "flipped"

        after = client.get("/api/detectors/cross-ds-model/labels-detail").get_json()
        assert len(after["good"]) == 1
        assert len(after["bad"]) == 2
        assert any(e["id"] == target["id"] for e in after["bad"])

    def test_same_vote_removes_element(self, client):
        _seed_cross_dataset_model()
        detail = client.get("/api/detectors/cross-ds-model/labels-detail").get_json()
        target = detail["good"][0]

        res = client.post(
            f"/api/detectors/cross-ds-model/labels/{target['id']}/vote",
            json={"vote": "good"},
        )
        assert res.status_code == 200
        assert res.get_json()["action"] == "removed"

        after = client.get("/api/detectors/cross-ds-model/labels-detail").get_json()
        assert len(after["good"]) == 1
        assert len(after["bad"]) == 1
        assert not any(e["id"] == target["id"] for e in after["good"] + after["bad"])

    def test_unknown_id_404(self, client):
        _seed_cross_dataset_model()
        res = client.post(
            "/api/detectors/cross-ds-model/labels/deadbeef/vote",
            json={"vote": "good"},
        )
        assert res.status_code == 404

    def test_invalid_vote_value_400(self, client):
        _seed_cross_dataset_model()
        detail = client.get("/api/detectors/cross-ds-model/labels-detail").get_json()
        target = detail["good"][0]
        res = client.post(
            f"/api/detectors/cross-ds-model/labels/{target['id']}/vote",
            json={"vote": "maybe"},
        )
        assert res.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/detectors/<name>/labels/<id>/thumbnail
# ---------------------------------------------------------------------------


class TestLabelElementThumbnail:
    """The thumbnail endpoint is what the right-pane labelset list uses for
    every entry, so it must stay much smaller than ``/preview`` (which serves
    the full original file)."""

    def test_returns_404_for_unknown_model(self, client):
        res = client.get("/api/detectors/does-not-exist/labels/abc123/thumbnail")
        assert res.status_code == 404

    def test_returns_404_for_unknown_element(self, client):
        _seed_cross_dataset_model()
        res = client.get("/api/detectors/cross-ds-model/labels/deadbeef/thumbnail")
        assert res.status_code == 404

    def test_returns_404_when_file_unavailable_cross_dataset(self, client):
        """Cross-dataset elements with a fake importer can't be resolved on
        disk — the route should 404 cleanly, not 500."""
        _seed_cross_dataset_model()
        detail = client.get("/api/detectors/cross-ds-model/labels-detail").get_json()
        target = detail["good"][0]
        res = client.get(
            f"/api/detectors/cross-ds-model/labels/{target['id']}/thumbnail"
        )
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# Cross-dataset: voting in dataset B preserves dataset A's labels on disk
# ---------------------------------------------------------------------------


class TestCrossDatasetVoteDoesNotWipeLabels:
    """Regression test for the bug where opening a detector against a fresh
    dataset and casting a single vote would overwrite the on-disk labelset
    with just that vote, destroying everything trained on the prior dataset.
    """

    def test_vote_in_active_dataset_merges_with_saved_labelset(self, client):
        from vtsearch.models.label_sync import sync_labels_to_loaded_detector
        from vtsearch.models.detector_store import _detector_path, _read_detector
        from vtsearch.utils import good_votes

        _seed_cross_dataset_model()

        # Active-dataset votes (test medias use importer "test").
        good_votes[1] = None
        good_votes[2] = None

        sync_labels_to_loaded_detector()

        saved = _read_detector(_detector_path("cross-ds-model"))
        keys = {
            (el.get("origin", {}).get("importer", ""), el.get("origin_name", "")) for el in saved["labelset"]["labels"]
        }
        # Cross-dataset (ds_a) labels survived.
        assert ("ds_a", "alpha.wav") in keys
        assert ("ds_a", "beta.wav") in keys
        assert ("ds_a", "gamma.wav") in keys
        # Active-dataset votes were merged in.
        assert any(imp == "test" for imp, _ in keys)


# ---------------------------------------------------------------------------
# Cross-dataset: training uses ALL labels' vectors, not just current dataset
# ---------------------------------------------------------------------------


class TestCrossDatasetMLPTraining:
    """When a detector is loaded against a different dataset, the saved
    labels' origins are resolved + re-embedded, the resulting vectors are
    cached on the DetectorContext, and MLP training uses them.

    Without this, only labels that happen to resolve into the active
    dataset would contribute — defeating the point of the labelset-driven
    pane (the user's labels would be invisible to the model)."""

    def _seed_with_resolvable_labelset(self, tmp_path, monkeypatch):
        """Seed a model whose labelset elements are all resolvable.

        Stubs ``resolve_file_context`` to yield a unique temp WAV file
        per element, so that ``embed_file`` (already stubbed in conftest to
        fake_embed_audio) yields deterministic, distinct vectors.
        """
        from contextlib import contextmanager
        from pathlib import Path

        from vtsearch.utils.audio_generator import generate_wav

        files: dict[str, Path] = {}
        for name, freq in (("alpha.wav", 220), ("beta.wav", 330), ("gamma.wav", 440)):
            path = tmp_path / name
            path.write_bytes(generate_wav(freq, 0.1))
            files[name] = path

        @contextmanager
        def _fake_resolve_ctx(origin, origin_name="", filename=""):
            yield files.get(origin_name) or files.get(filename)

        # labelset_training imports resolve_file_context inside _embed_one,
        # so patching the resolver symbol is enough — the function-level
        # import picks the patched value.
        import vtsearch.models.labelset_training as lt_mod
        import vtsearch.models.resolver as resolver_mod

        monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_resolve_ctx)
        # Defensive: patch the binding inside labelset_training too in case
        # it ever moves to a top-level import.
        if hasattr(lt_mod, "resolve_file_context"):
            monkeypatch.setattr(lt_mod, "resolve_file_context", _fake_resolve_ctx)

        return _seed_cross_dataset_model(mark_loaded=False)

    def test_load_resolves_and_embeds_cross_dataset_labels(self, client, tmp_path, monkeypatch):
        from vtsearch.models.labelset_elements import stable_element_id
        from vtsearch.utils.state_core import get_detector_context

        detector_id = self._seed_with_resolvable_labelset(tmp_path, monkeypatch)

        # Trigger the load task via the public API (so the embed pass runs).
        _load_detector_and_wait_local(client, detector_id)

        det_ctx = get_detector_context(detector_id)
        assert det_ctx is not None, "Load task must register a DetectorContext"
        assert len(det_ctx.label_embeddings) == 3, (
            f"Expected all 3 cross-dataset labels embedded, got {list(det_ctx.label_embeddings)}"
        )
        assert det_ctx.model is not None, "Load must train an MLP from cross-dataset vectors"

        # Verify the cache is keyed by stable_element_id (so subsequent
        # lookups find the right vector).
        from vtsearch.datasets.labelset import LabelSet
        from vtsearch.models.detector_store import _detector_path, _read_detector

        saved = _read_detector(_detector_path("cross-ds-model"))
        ls = LabelSet.from_dict(saved["labelset"])
        for el in ls.elements:
            assert stable_element_id(el) in det_ctx.label_embeddings

    def test_learned_sort_uses_cross_dataset_labels_with_zero_active_votes(self, client, tmp_path, monkeypatch):
        """No active-dataset votes — yet learned-sort should still produce a
        ranked list, because training uses the cross-dataset labelset."""
        detector_id = self._seed_with_resolvable_labelset(tmp_path, monkeypatch)
        _load_detector_and_wait_local(client, detector_id)

        res = client.post("/api/learned-sort", json={"wait": True})
        assert res.status_code == 200, res.get_data(as_text=True)
        data = res.get_json()
        assert "results" in data and len(data["results"]) > 0
        assert "threshold" in data


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_detector_and_wait_local(client, detector_id, timeout=5.0):
    """Local copy of tests.load_detector_and_wait that doesn't reset state, so
    the labelset and registered detector context survive the load.
    """
    import time

    from vtsearch.utils.state_core import get_detector_context, set_thread_detector_context

    res = client.post("/api/detectors/registry/load", json={"detector_id": detector_id})
    assert res.status_code in (200, 202), res.get_data(as_text=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        tasks = client.get("/api/detectors/loading-tasks").get_json().get("tasks", [])
        if not [t for t in tasks if t.get("status") != "idle"]:
            break
        time.sleep(0.05)

    det_ctx = get_detector_context(detector_id)
    if det_ctx is not None:
        set_thread_detector_context(det_ctx)
    return res
