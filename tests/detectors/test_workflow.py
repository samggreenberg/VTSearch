"""Tests for ``vtsearch.detectors.workflow.apply_and_retrain``.

This is the Flask-aware retraining helper called from the
``/api/detectors/<name>/import-labels/<importer>`` endpoint after label
import.  It overrides the active detector context, resolves the new
label entries against the loaded dataset, applies them, and retrains
the MLP when both a good and a bad vote are present.
"""

from __future__ import annotations

import pytest

from vtsearch.detectors.workflow import apply_and_retrain
from vtsearch.state import medias
from vtsearch.state.core import DetectorContext


@pytest.fixture
def det_ctx():
    """A fresh DetectorContext that isn't tied to any registered detector.

    The autouse ``reset_state`` fixture in ``conftest.py`` already wipes
    the global registry, so spinning one up locally is safe.
    """
    return DetectorContext("test-det")


def _audio_entry(media_id: int, label: str) -> dict:
    """Build a label-import entry that matches an existing test audio media."""
    media = medias[media_id]
    return {
        "label": label,
        "origin": media["origin"],
        "origin_name": media["origin_name"],
        "filename": media.get("filename", ""),
        "md5": media.get("md5", ""),
    }


class TestEmptySnapshot:
    """When no dataset is loaded the workflow must return cleanly."""

    def test_returns_zero_and_false_when_no_medias(self, det_ctx):
        saved = dict(medias)
        medias.clear()
        try:
            entries = [{"label": "good", "md5": "deadbeef"}]
            resolved, trained = apply_and_retrain("test-det", det_ctx, entries, "Test")
            assert (resolved, trained) == (0, False)
            # No model should have been built.
            assert det_ctx.model is None
        finally:
            medias.update(saved)


class TestLabelResolution:
    def test_invalid_labels_are_skipped(self, det_ctx):
        # ``maybe`` is not a recognised label — counts as 0 resolved.
        entries = [_audio_entry(1, "maybe")]
        resolved, trained = apply_and_retrain("test-det", det_ctx, entries, "Test")
        assert resolved == 0
        assert trained is False

    def test_unresolvable_origin_does_not_count(self, det_ctx):
        # An entry whose origin/md5 don't match anything in the dataset.
        entries = [
            {
                "label": "good",
                "origin": {"importer": "other", "params": {"x": 1}},
                "origin_name": "ghost.wav",
                "md5": "0" * 32,
            }
        ]
        resolved, trained = apply_and_retrain("test-det", det_ctx, entries, "Test")
        assert resolved == 0
        assert trained is False

    def test_resolvable_entry_counts_once(self, det_ctx):
        entries = [_audio_entry(1, "good")]
        resolved, _ = apply_and_retrain("test-det", det_ctx, entries, "Test")
        assert resolved == 1


class TestTraining:
    """The MLP is only retrained when at least one good AND one bad vote exist."""

    def test_only_good_votes_does_not_train(self, det_ctx):
        entries = [_audio_entry(1, "good"), _audio_entry(2, "good")]
        _, trained = apply_and_retrain("test-det", det_ctx, entries, "Test")
        assert trained is False
        assert det_ctx.model is None

    def test_only_bad_votes_does_not_train(self, det_ctx):
        entries = [_audio_entry(1, "bad"), _audio_entry(2, "bad")]
        _, trained = apply_and_retrain("test-det", det_ctx, entries, "Test")
        assert trained is False
        assert det_ctx.model is None

    def test_one_good_one_bad_trains_mlp(self, det_ctx):
        entries = [
            _audio_entry(1, "good"),
            _audio_entry(2, "bad"),
        ]
        resolved, trained = apply_and_retrain("test-det", det_ctx, entries, "Test")
        assert resolved == 2
        assert trained is True
        assert det_ctx.model is not None
        # The threshold should be a finite float in [0, 1].
        assert isinstance(det_ctx.threshold, float)
        assert 0.0 <= det_ctx.threshold <= 1.0
        # training_medias cache should hold both voted media.
        assert set(det_ctx.training_medias) >= {1, 2}
        # embedder / media_type stamped from the snapshot.
        assert det_ctx.embedder == "clap"
        assert det_ctx.media_type == "audio"


class TestVoteApplicationIsContextScoped:
    """``apply_and_retrain`` swaps the active detector context inside its
    body; the votes on the *passed* ctx must reflect the entries."""

    def test_votes_are_applied_to_passed_context(self, det_ctx):
        entries = [
            _audio_entry(1, "good"),
            _audio_entry(2, "bad"),
        ]
        apply_and_retrain("test-det", det_ctx, entries, "Test")
        # The applied votes live on the supplied context.
        assert 1 in det_ctx.good_votes
        assert 2 in det_ctx.bad_votes

    def test_unrelated_empty_context_is_not_polluted(self, det_ctx):
        # Construct a second context, run the workflow on the first only,
        # and confirm the second remains pristine.
        other_ctx = DetectorContext("other-det")
        entries = [_audio_entry(1, "good"), _audio_entry(2, "bad")]
        apply_and_retrain("test-det", det_ctx, entries, "Test")
        assert not other_ctx.good_votes
        assert not other_ctx.bad_votes
        assert other_ctx.model is None
