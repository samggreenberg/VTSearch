"""Find must not score with a cached head trained on labels that have since changed.

``DetectorContext.model`` is a cache of the head last trained for a detector,
and nothing that changes the detector's labels drops it: a vote, clearing the
votes, a dataset switch and the dashboard's saved-label review all leave it in
place.  Only a learned sort replaces it.  ``/api/find-label`` (and the legacy
``/api/find``) reused it whenever it was set, so after the labels changed Find
kept reporting the old detector's verdicts until the app restarted (issue
#4204).

Every writer of ``model`` now stamps it with the signature of the labelset it
was trained from, and Find reuses the head only while the saved labelset still
has that signature.  The tests pin both halves: a changed labelset retrains,
and an unchanged one still takes the cached head.
"""

from __future__ import annotations

import numpy as np

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import get_active_detector_context, get_detector_context
from vtsearch.state import snapshot_medias

GOOD = (1, 2, 3)
BAD = (17, 18, 19, 20)


def _vote(client, good, bad) -> None:
    for media_id in good:
        assert client.post(f"/api/medias/{media_id}/vote", json={"target": "good"}).status_code == 200
    for media_id in bad:
        assert client.post(f"/api/medias/{media_id}/vote", json={"target": "bad"}).status_code == 200


def _detector_with_votes(client, name: str) -> str:
    """A loaded detector whose labels came from Train votes, with no learned sort."""
    detector_id = setup_trainable_model_in_registry(name, good_ids=[], bad_ids=[], snap=snapshot_medias())
    load_detector_and_wait(client, detector_id)
    _vote(client, GOOD, BAD)
    return detector_id


def _find(client, detector_id: str) -> dict[int, float]:
    resp = client.post("/api/find-label", json={"detector_id": detector_id})
    assert resp.status_code == 200, resp.get_json()
    return {r["id"]: r["score"] for r in resp.get_json()["results"]}


def _mean(scores: dict[int, float], ids) -> float:
    return float(np.mean([scores[i] for i in ids]))


def _count_trains(monkeypatch) -> list[int]:
    """Count the labelset trains ``resolve_or_train_detector`` falls back to."""
    from vtscore.detectors import labelset_training

    calls: list[int] = []
    real = labelset_training.train_from_labelset

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(labelset_training, "train_from_labelset", _spy)
    return calls


class TestFindRetrainsAfterLabelsChange:
    """The reproduction from the issue: vote, Find, change the votes, Find again."""

    def test_find_after_votes_change_scores_the_new_labels(self, client):
        detector_id = _detector_with_votes(client, "stale-head-votes")

        before = _find(client, detector_id)
        assert _mean(before, GOOD) > _mean(before, BAD)

        # Back to Train, and relabel with a non-learned sort: clear, then vote
        # the two sides the other way round.  No learned sort runs.
        assert client.post("/api/find/end-session").status_code == 200
        assert client.post("/api/votes/clear").status_code == 200
        _vote(client, BAD, GOOD)

        after = _find(client, detector_id)
        assert after != before, "Find reused the head trained on the old labels"
        assert _mean(after, BAD) > _mean(after, GOOD)

    def test_a_labelset_edited_on_disk_is_retrained(self, client, monkeypatch):
        """Any writer of the detector file counts, not just the vote path.

        Stands in for the dashboard's saved-label review, which rewrites the
        labelset without going near ``det_ctx.model``.
        """
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.store import _detector_path, _read_detector, _write_detector

        detector_id = setup_trainable_model_in_registry(
            "stale-head-disk", good_ids=list(GOOD), bad_ids=list(BAD), snap=snapshot_medias()
        )
        load_detector_and_wait(client, detector_id)
        _find(client, detector_id)
        det_ctx = get_detector_context(detector_id)
        assert det_ctx is not None and det_ctx.model is not None

        path = _detector_path("stale-head-disk")
        data = _read_detector(path)
        assert data is not None
        flipped = LabelSet.from_clips_and_votes(
            snapshot_medias(), dict.fromkeys(BAD), dict.fromkeys(GOOD), expand_dupes=False
        )
        data["labelset"] = flipped.to_dict()
        _write_detector(path, data)

        trains = _count_trains(monkeypatch)
        after = _find(client, detector_id)
        assert trains == [1]
        assert _mean(after, BAD) > _mean(after, GOOD)


class TestUnchangedLabelsStillReuseTheCachedHead:
    """The cache is still a cache: Find retrains only when the labels moved."""

    def test_a_second_find_reuses_the_head_the_first_one_trained(self, client, monkeypatch):
        detector_id = _detector_with_votes(client, "reuse-find")
        first = _find(client, detector_id)

        trains = _count_trains(monkeypatch)
        second = _find(client, detector_id)
        assert trains == []
        assert second == first

    def test_find_reuses_the_head_a_learned_sort_trained(self, client, monkeypatch):
        """The learned sort's labelset signature matches the saved file's."""
        detector_id = _detector_with_votes(client, "reuse-learned")
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 200, resp.get_json()
        assert get_active_detector_context().model is not None

        trains = _count_trains(monkeypatch)
        _find(client, detector_id)
        assert trains == []

    def test_a_learned_sort_head_from_older_labels_is_not_reused(self, client, monkeypatch):
        """A vote that lands after the learned sort leaves its head stale."""
        detector_id = _detector_with_votes(client, "stale-learned")
        assert client.post("/api/learned-sort", json={"wait": True}).status_code == 200

        _vote(client, (4,), ())

        trains = _count_trains(monkeypatch)
        _find(client, detector_id)
        assert trains == [1]


class TestLegacyFindConfig:
    """``/api/find`` carries the live head only while its labels are current."""

    def test_live_head_is_dropped_once_the_labels_change(self, client):
        from vtscore.detectors.registry import get_detector
        from vtsearch.routes.detectors.find import _build_detector_config

        detector_id = _detector_with_votes(client, "legacy-find")
        _find(client, detector_id)
        entry = get_detector(detector_id)
        assert entry is not None
        assert "live_mlp" in _build_detector_config(entry)

        assert client.post("/api/find/end-session").status_code == 200
        _vote(client, (4,), ())

        config = _build_detector_config(entry)
        assert "live_mlp" not in config
        assert "detector_data" in config


class TestLabelsetSignature:
    """What counts as "the same labels" for the reuse check."""

    def _labelset(self, *, region=None, metadata=None, flip=False):
        from vtscore.datasets.labelset import LabeledElement, LabelSet

        origin = {"importer": "test", "params": {"id": 1}}
        return LabelSet(
            [
                LabeledElement(
                    md5="a",
                    label="bad" if flip else "good",
                    origin=origin,
                    origin_name="a.png",
                    region_box=region,
                    metadata=metadata,
                ),
                LabeledElement(md5="b", label="good" if flip else "bad", origin_name="b.png"),
            ]
        )

    def test_survives_a_round_trip_through_the_detector_file(self):
        import json

        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.model_loading import labelset_signature

        ls = self._labelset(region=(0.1, 0.2, 0.3, 0.4))
        round_tripped = LabelSet.from_dict(json.loads(json.dumps(ls.to_dict())))
        assert labelset_signature(round_tripped) == labelset_signature(ls)

    def test_a_flipped_label_changes_it(self):
        from vtscore.detectors.model_loading import labelset_signature

        assert labelset_signature(self._labelset(flip=True)) != labelset_signature(self._labelset())

    def test_a_redrawn_region_changes_it(self):
        """A Good label's box decides which patch it pools to."""
        from vtscore.detectors.model_loading import labelset_signature

        boxed = labelset_signature(self._labelset(region=(0.1, 0.2, 0.3, 0.4)))
        assert boxed != labelset_signature(self._labelset())
        assert boxed != labelset_signature(self._labelset(region=(0.5, 0.2, 0.9, 0.4)))

    def test_metadata_does_not_change_it(self):
        """Provenance rides in metadata and never reaches training."""
        from vtscore.detectors.model_loading import labelset_signature

        tagged = self._labelset(metadata={"vt:provenance": {"flow": "manual"}})
        assert labelset_signature(tagged) == labelset_signature(self._labelset())

    def test_no_labelset_matches_nothing(self):
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.model_loading import labelset_signature

        assert labelset_signature(None) is None
        assert labelset_signature(None) != labelset_signature(LabelSet([]))
