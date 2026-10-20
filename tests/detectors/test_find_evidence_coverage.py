"""Tests for ``GET /api/find/evidence-coverage``.

The cross-user complement to the atlas domain-shift report: how much of the
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
        import vtsearch.training_domain as training_domain

        monkeypatch.setattr(training_domain, "infer_training_reference", lambda _labelset, _active: ("elsewhere", {}))

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
