"""Tests for ``GET /api/find/evidence-coverage``.

The cross-user complement to the domain-shift report: how much of the
scored dataset the active detector is calling *without labeled evidence behind
the call*, measured from the detector's own labelset (re-embedded in memory at
load) — so it fires with no reference dataset loaded.  See
docs/plans/coverage-atlas.md §6.1 (phase v0).
"""

from __future__ import annotations

import pytest

from tests.helpers import setup_trainable_model_in_registry
from tests import load_detector_and_wait
from vtscore.detectors.evidence_baseline import EVIDENCE_BASELINE_KEY
from vtscore.detectors.registry import get_detector
from vtscore.detectors.store import _detector_path, _read_detector, _write_detector
from vtscore.state.core import get_active_context, get_active_detector_context
from vtsearch.state import snapshot_medias


class TestFindEvidenceCoverage:
    def _setup_find(self, client) -> str:
        """Register + load a detector, then run find-label so it enters find
        mode with frozen scores and populated label embeddings."""
        detector_id = setup_trainable_model_in_registry(
            "evidence-model",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        load_detector_and_wait(client, detector_id)
        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        return detector_id

    def test_unavailable_without_a_find_run(self, client):
        """No scored Find run → available False, section hidden, not a 4xx."""
        get_active_detector_context().find_scores.clear()
        resp = client.get("/api/find/evidence-coverage")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["available"] is False
        assert data["n_items"] == 0

    def test_reports_after_find_run(self, client):
        """After find-label the report is available and self-consistent."""
        self._setup_find(client)
        data = client.get("/api/find/evidence-coverage").get_json()
        assert data["available"] is True
        # Every scored item is covered.
        assert data["n_items"] == len(get_active_detector_context().find_scores)
        assert data["n_items"] > 0
        # The labelset carried both classes into the report.
        assert data["n_pos_labels"] >= 3
        assert data["n_neg_labels"] >= 3
        # Rates and p-values stay in range.
        assert 0.0 <= data["frac_unsupported"] <= 1.0
        assert 0.0 <= data["frac_low_trust"] <= 1.0
        assert 0.0 < data["median_support"] <= 1.0
        assert data["expected_unsupported"] == data["alpha"]

    def test_in_domain_data_reads_supported(self, client):
        """The training dataset *is* the active dataset here, so the detector is
        scoring data it has evidence for: the vacuum share is not a blowout and
        the headline verdict is not 'unsupported'."""
        self._setup_find(client)
        data = client.get("/api/find/evidence-coverage").get_json()
        assert data["unsupported"] is False
        assert data["frac_unsupported"] < 0.5

    def test_deterministic(self, client):
        """Two reads of the same frozen Find state return identical numbers."""
        self._setup_find(client)
        a = client.get("/api/find/evidence-coverage").get_json()
        b = client.get("/api/find/evidence-coverage").get_json()
        assert a == b


class TestTheTrainingBaseline:
    """#5089: the verdict ranks against the training dataset's own media, recorded on the detector there."""

    @pytest.fixture
    def detector_path(self, client):
        detector_id = TestFindEvidenceCoverage()._setup_find(client)
        entry = get_detector(detector_id)
        assert entry is not None
        return _detector_path(entry["name"])

    @staticmethod
    def _away(monkeypatch):
        """Make the training dataset another one, as on a dataset the detector meets after a handoff."""
        import vtsearch.domain_overlap as domain_overlap

        monkeypatch.setattr(domain_overlap, "infer_training_reference", lambda _labelset, _active: ("elsewhere", {}))

    def test_on_its_training_dataset_the_baseline_is_drawn_and_recorded(self, client, detector_path):
        data = client.get("/api/find/evidence-coverage").get_json()
        assert data["baseline"] == "training"
        assert data["baseline_dataset_id"] == get_active_context().dataset_id
        # Drawn on the scored media less the six labelled ones.
        assert data["n_baseline"] == data["n_items"] - 6
        assert 0.0 <= data["frac_beyond"] <= 1.0
        # Ranked against itself, the training dataset reads its own spread: no verdict fires.
        assert data["unsupported"] is False
        data_on_disk = _read_detector(detector_path)
        assert data_on_disk is not None
        stored = data_on_disk[EVIDENCE_BASELINE_KEY]
        assert stored["dataset_id"] == data["baseline_dataset_id"]
        assert stored["n_pos"] + stored["n_neg"] == data["n_baseline"]
        assert stored["k"] == data["k"] and stored["alpha"] == data["alpha"]

    def test_elsewhere_the_recorded_baseline_is_read(self, client, detector_path, monkeypatch):
        home = client.get("/api/find/evidence-coverage").get_json()
        self._away(monkeypatch)
        away = client.get("/api/find/evidence-coverage").get_json()
        assert away["baseline"] == "training"
        assert away["baseline_dataset_id"] == home["baseline_dataset_id"]
        assert away["frac_beyond"] == home["frac_beyond"]

    def test_a_stale_baseline_draws_no_verdict(self, client, detector_path, monkeypatch):
        client.get("/api/find/evidence-coverage")
        data = _read_detector(detector_path)
        assert data is not None
        data[EVIDENCE_BASELINE_KEY]["labels"] = "0" * 32  # as after a vote since
        _write_detector(detector_path, data)
        self._away(monkeypatch)
        away = client.get("/api/find/evidence-coverage").get_json()
        assert away["available"] is True
        assert away["baseline"] == "none" and away["n_baseline"] == 0
        assert away["frac_beyond"] is None and away["z_beyond"] is None
        assert away["baseline_dataset_id"] == ""
        assert away["unsupported"] is False

    def test_no_test_on_the_training_dataset_yet_draws_no_verdict(self, client, detector_path, monkeypatch):
        self._away(monkeypatch)
        away = client.get("/api/find/evidence-coverage").get_json()
        assert away["baseline"] == "none" and away["unsupported"] is False
        assert EVIDENCE_BASELINE_KEY not in (_read_detector(detector_path) or {})


class TestTheBaselineFromTheTrainView:
    """#5120: a detector trained in the Train view carries the baseline with no Tune on its training dataset."""

    @staticmethod
    def _dataset(tmp_path, name: str, medias: dict):
        """Register + load a dataset holding *medias*, so the training-dataset inference can see it."""
        from vtscore.datasets.registry import add_loaded_id, register_dataset
        from vtscore.state.core import DatasetContext, register_context
        from tests.fixtures.medias import DEFAULT_AUDIO_EMBEDDER

        entry = register_dataset(
            name=name,
            media_type="audio",
            num_items=len(medias),
            pkl_path=str(tmp_path / f"{name}.pkl"),
            embedder=DEFAULT_AUDIO_EMBEDDER,
        )
        ctx = DatasetContext(entry["id"])
        ctx.medias.update(medias)
        register_context(ctx)
        add_loaded_id(entry["id"])
        return entry, ctx

    @pytest.fixture
    def trained(self, client, tmp_path):
        """Votes on dataset A and a learned sort there, as the Train view leaves them; B holds none of A's media.

        Three Goods and four Bads: the label quota's trained tier, where a Find
        scores with the sort's head (#4643).
        """
        return self._train(client, tmp_path, bad_ids=[17, 18, 19, 20])

    def _train(self, client, tmp_path, *, bad_ids: list[int]) -> dict:
        from vtscore.state.core import set_thread_dataset_context
        from vtscore.utils.hashing import content_md5

        snap = snapshot_medias()
        a_entry, a_ctx = self._dataset(tmp_path, "train-a", {cid: dict(m) for cid, m in snap.items()})
        b_medias = {}
        for cid, media in snap.items():
            name = f"elsewhere_{cid}.wav"
            b_medias[cid] = {
                **media,
                "md5": content_md5(f"elsewhere/{cid}".encode()),
                "origin": {"importer": "test", "params": {"dataset": "elsewhere"}},
                "origin_name": name,
                "filename": name,
            }
        b_entry, b_ctx = self._dataset(tmp_path, "train-b", b_medias)

        set_thread_dataset_context(a_ctx)
        detector_id = setup_trainable_model_in_registry(
            "evidence-train-view", good_ids=[1, 2, 3], bad_ids=bad_ids, snap=a_ctx.medias
        )
        load_detector_and_wait(client, detector_id)
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 200, resp.get_json()
        assert resp.get_json()["status"] == "done"
        entry = get_detector(detector_id)
        assert entry is not None
        return {
            "detector_id": detector_id,
            "path": _detector_path(entry["name"]),
            "a": (a_entry, a_ctx),
            "b": (b_entry, b_ctx),
        }

    @staticmethod
    def _find_and_read(client, ctx, detector_id: str) -> dict:
        from vtscore.state.core import set_thread_dataset_context

        set_thread_dataset_context(ctx)
        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        return client.get("/api/find/evidence-coverage").get_json()

    @pytest.fixture
    def quiet_forever(self, monkeypatch):
        """Keep the sort's draw waiting, so only a read can land it."""
        import vtscore.detectors.evidence_baseline as evidence_baseline

        monkeypatch.setattr(evidence_baseline, "DRAW_QUIET_SECONDS", 3600.0)

    def test_the_sort_leaves_the_draw_waiting_rather_than_drawing_per_vote(self, quiet_forever, trained):
        from vtscore.detectors.evidence_baseline import wait_for_baseline_draws

        assert EVIDENCE_BASELINE_KEY not in (_read_detector(trained["path"]) or {})
        assert not wait_for_baseline_draws(timeout=0)

    def test_under_the_label_quota_the_sort_leaves_nothing(self, client, tmp_path, quiet_forever):
        """Three Bads: a Find there trains the Goods' centroid, whose line is not the sort's."""
        from vtscore.detectors.evidence_baseline import wait_for_baseline_draws

        trained = self._train(client, tmp_path, bad_ids=[18, 19, 20])
        assert wait_for_baseline_draws(timeout=0)
        _b_entry, b_ctx = trained["b"]
        data = self._find_and_read(client, b_ctx, trained["detector_id"])
        assert data["available"] is True
        assert data["baseline"] == "none"

    def test_another_dataset_reads_the_baseline_the_sort_left(self, client, quiet_forever, trained):
        from vtscore.detectors.evidence_baseline import baseline_stamp, stored_baseline
        from vtscore.detectors.learned_sort import resolve_active_labelset
        from tests.fixtures.medias import DEFAULT_AUDIO_EMBEDDER

        a_entry, a_ctx = trained["a"]
        _b_entry, b_ctx = trained["b"]
        data = self._find_and_read(client, b_ctx, trained["detector_id"])

        assert data["available"] is True
        assert data["baseline"] == "training"
        assert data["baseline_dataset_id"] == a_entry["id"]
        assert data["baseline_dataset_name"] == "train-a"
        # Drawn on A's twenty media less the seven labelled ones.
        assert data["n_baseline"] == len(a_ctx.medias) - 7
        labelset, _ = resolve_active_labelset(get_active_detector_context())
        assert labelset is not None
        dim = len(next(iter(a_ctx.medias.values()))["embeddings"][DEFAULT_AUDIO_EMBEDDER])
        stamp = baseline_stamp(labelset, DEFAULT_AUDIO_EMBEDDER, dim)
        assert stored_baseline(_read_detector(trained["path"]), stamp) is not None

    def test_it_is_the_baseline_a_tune_on_the_training_dataset_draws(self, client, quiet_forever, trained):
        """The sort and the route call the same pieces, so they agree on what the training baseline is."""
        from vtscore.detectors.evidence_baseline import flush_baseline_draw

        assert flush_baseline_draw(trained["detector_id"]) is True
        from_the_sort = (_read_detector(trained["path"]) or {})[EVIDENCE_BASELINE_KEY]
        _a_entry, a_ctx = trained["a"]
        data = self._find_and_read(client, a_ctx, trained["detector_id"])
        assert data["baseline"] == "training"
        assert (_read_detector(trained["path"]) or {})[EVIDENCE_BASELINE_KEY] == from_the_sort

    @pytest.fixture
    def quiet_now(self, monkeypatch):
        """Draw the moment the sort ends, as once the votes have stopped."""
        import vtscore.detectors.evidence_baseline as evidence_baseline

        monkeypatch.setattr(evidence_baseline, "DRAW_QUIET_SECONDS", 0.0)

    def test_once_the_votes_stop_it_is_recorded_and_the_session_survives(self, client, quiet_now, trained):
        """The background draw writes the detector file; the Train view's state in memory must not read it as an edit."""
        from vtscore.detectors.evidence_baseline import wait_for_baseline_draws

        assert wait_for_baseline_draws(timeout=30.0)
        a_entry, _a_ctx = trained["a"]
        stored = (_read_detector(trained["path"]) or {}).get(EVIDENCE_BASELINE_KEY)
        assert stored is not None and stored["dataset_id"] == a_entry["id"]

        det_ctx = get_active_detector_context()
        assert det_ctx.last_learned_scores and det_ctx.line_ranking is not None
        assert client.get("/api/votes").status_code == 200
        # No rehydrate: the scores and the ranking the sort left are still there.
        assert det_ctx.last_learned_scores and det_ctx.line_ranking is not None

    def test_unloading_the_detector_lands_the_waiting_draw(self, client, quiet_forever, trained):
        """An unloaded detector is something another process may read next, so its baseline lands now."""
        resp = client.post(f"/api/detectors/registry/{trained['detector_id']}/unload")
        assert resp.status_code == 200, resp.get_json()
        a_entry, _a_ctx = trained["a"]
        stored = (_read_detector(trained["path"]) or {}).get(EVIDENCE_BASELINE_KEY)
        assert stored is not None and stored["dataset_id"] == a_entry["id"]
