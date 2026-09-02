"""API surface for vote surfacing provenance (issue #2850).

The three vote routes accept an optional ``provenance`` block, validate it
against the closed vocabulary in :mod:`vtscore.datasets.vote_provenance`, and
persist it into the detector's labelset.  Nothing reads it back to change
behaviour — this is the recording slice only.

The schemas use ``unknown = "exclude"``, so an unrecognised *key* is silently
dropped by design (the frontend is allowed to attach advisory keys).  An
unrecognised *value* for a key we do declare is a different matter and must be
rejected: a typo'd ``surfaced_by`` silently stored as nothing would look
exactly like a legacy vote to any later analysis.
"""

from __future__ import annotations

from vtscore.datasets.vote_provenance import METADATA_KEY
from vtsearch.state import bad_votes, good_votes

FULL = {
    "flow": "list_review",
    "select_mode": "hard",
    "sort_kind": "learned",
    "rank_at_vote": 12,
    "score_at_vote": 0.44,
}


def _recorded(media_id: int) -> dict | None:
    from vtscore.state import get_active_detector_context

    return get_active_detector_context().vote_provenance.get(media_id)


class TestSingleVoteRoute:
    """``POST /api/medias/<id>/vote``."""

    def test_provenance_is_recorded(self, client):
        resp = client.post("/api/medias/1/vote", json={"target": "good", "provenance": FULL})
        assert resp.status_code == 200
        assert 1 in good_votes
        assert _recorded(1) == {"v": 1, **FULL}

    def test_vote_without_provenance_still_works(self, client):
        """Every existing client keeps working, and records nothing."""
        resp = client.post("/api/medias/1/vote", json={"target": "good"})
        assert resp.status_code == 200
        assert _recorded(1) is None

    def test_unknown_flow_value_is_rejected(self, client):
        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "good", "provenance": {"flow": "toplist"}},
        )
        assert resp.status_code == 422
        assert 1 not in good_votes

    def test_unknown_select_mode_is_rejected(self, client):
        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "good", "provenance": {"select_mode": "margin"}},
        )
        assert resp.status_code == 422

    def test_negative_rank_is_rejected(self, client):
        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "good", "provenance": {"rank_at_vote": -3}},
        )
        assert resp.status_code == 422

    def test_idempotent_revote_does_not_overwrite(self, client):
        """A stale tab re-asserting the current target made no surfacing
        event, so it must not rewrite the first click's record."""
        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "provenance": {"flow": "autopilot", "phase": "hard"}},
        )
        resp = client.post(
            "/api/medias/1/vote",
            json={"target": "good", "provenance": {"flow": "list_review"}},
        )
        assert resp.status_code == 200
        assert _recorded(1)["flow"] == "autopilot"

    def test_a_flip_records_the_new_context(self, client):
        client.post(
            "/api/medias/1/vote",
            json={"target": "good", "provenance": {"flow": "autopilot", "phase": "hard"}},
        )
        client.post(
            "/api/medias/1/vote",
            json={"target": "bad", "provenance": {"flow": "list_review"}},
        )
        assert 1 in bad_votes
        assert _recorded(1)["flow"] == "list_review"

    def test_unvote_clears_the_record(self, client):
        client.post("/api/medias/1/vote", json={"target": "good", "provenance": FULL})
        client.post("/api/medias/1/vote", json={"target": "none"})
        assert _recorded(1) is None

    def test_provenance_reaches_the_exported_labelset(self, client):
        client.post("/api/medias/1/vote", json={"target": "good", "provenance": FULL})
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        labels = resp.get_json()["labels"]
        recorded = [el for el in labels if (el.get("metadata") or {}).get(METADATA_KEY)]
        assert len(recorded) == 1
        assert recorded[0]["metadata"][METADATA_KEY]["rank_at_vote"] == 12


class TestBulkVoteRoute:
    """``POST /api/medias/vote-bulk``."""

    def test_defaults_to_the_bulk_flow(self, client):
        """A batch action over a hand-selected set is its own surfacing flow,
        and the client never has to say so."""
        resp = client.post("/api/medias/vote-bulk", json={"ids": [1, 2], "target": "good"})
        assert resp.status_code == 200
        assert _recorded(1)["flow"] == "bulk"
        assert _recorded(2)["flow"] == "bulk"

    def test_explicit_provenance_overrides_the_default(self, client):
        resp = client.post(
            "/api/medias/vote-bulk",
            json={"ids": [1], "target": "good", "provenance": {"flow": "bulk", "sort_kind": "learned"}},
        )
        assert resp.status_code == 200
        assert _recorded(1)["sort_kind"] == "learned"

    def test_unknown_value_is_rejected(self, client):
        resp = client.post(
            "/api/medias/vote-bulk",
            json={"ids": [1], "target": "good", "provenance": {"flow": "nope"}},
        )
        assert resp.status_code == 422
        assert 1 not in good_votes


class TestDetectorLabelVoteRoute:
    """``POST /api/detectors/<name>/labels/<element_id>/vote``."""

    def _detector_with_one_label(self, client):
        from vtscore.datasets.labelset import LabeledElement, LabelSet
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.store import _detector_path, _write_detector

        from vtsearch.state import medias

        media = medias[1]
        el = LabeledElement(
            md5=media["md5"],
            label="good",
            origin=media.get("origin"),
            origin_name=media.get("origin_name", ""),
            filename=media.get("filename", ""),
        )
        path = _detector_path("prov-test")
        _write_detector(path, {"name": "prov-test", "labelset": LabelSet([el]).to_dict()})
        return "prov-test", stable_element_id(el), path

    def test_a_flip_defaults_to_labelset_review(self, client):
        """This surface reviews already-saved labels; it is not a draw off any
        ranking, so it gets its own flow rather than borrowing ``list_review``."""
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.store import _read_detector

        name, element_id, path = self._detector_with_one_label(client)
        try:
            resp = client.post(f"/api/detectors/{name}/labels/{element_id}/vote", json={"target": "bad"})
            assert resp.status_code == 200
            assert resp.get_json()["action"] == "flipped"

            stored = LabelSet.from_dict(_read_detector(path)["labelset"]).elements[0]
            assert stored.label == "bad"
            assert stored.metadata[METADATA_KEY]["flow"] == "labelset_review"
        finally:
            path.unlink(missing_ok=True)

    def test_an_unchanged_reassert_writes_nothing(self, client):
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.store import _read_detector

        name, element_id, path = self._detector_with_one_label(client)
        try:
            resp = client.post(f"/api/detectors/{name}/labels/{element_id}/vote", json={"target": "good"})
            assert resp.get_json()["action"] == "unchanged"
            stored = LabelSet.from_dict(_read_detector(path)["labelset"]).elements[0]
            assert stored.metadata is None
        finally:
            path.unlink(missing_ok=True)

    def test_unknown_value_is_rejected(self, client):
        name, element_id, path = self._detector_with_one_label(client)
        try:
            resp = client.post(
                f"/api/detectors/{name}/labels/{element_id}/vote",
                json={"target": "bad", "provenance": {"phase": "idle"}},
            )
            assert resp.status_code == 422
        finally:
            path.unlink(missing_ok=True)
