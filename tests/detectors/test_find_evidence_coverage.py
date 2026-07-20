"""Tests for ``GET /api/find/evidence-coverage``.

The cross-user complement to the atlas domain-shift report: how much of the
scored dataset the active detector is calling *without labeled evidence behind
the call*, measured from the detector's own labelset (re-embedded in memory at
load) — so it fires with no reference dataset loaded.  See
docs/plans/coverage-atlas.md §6.1 (phase v0).
"""

from __future__ import annotations

from helpers import setup_trainable_model_in_registry
from tests import load_detector_and_wait
from vtscore.state.core import get_active_detector_context
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
