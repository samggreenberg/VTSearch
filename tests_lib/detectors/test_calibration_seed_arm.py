"""The ``calibration_seed`` measurement arm (issue #3794).

The app pins its Train/Calibrate fold split to
:data:`~vtscore.training.thresholds.CALIBRATION_SPLIT_SEED` (42) — deliberately,
since issue #2934 fixed the unseeded global draw that made a detector's
threshold move run to run.  So the harness pins it too: the eval default arm
has to *be* the app, and a default run that reseeded the split would report a
detector nobody ships.

What the knob adds is the measurement that pin makes impossible otherwise.
Hold the cell seed — so the data cannot move, the same media are voted in the
same order over the same held-out split — and sweep this instead: the spread
across the sweep is the run-to-run noise the single pinned draw hides, and
therefore the error bar on every single-seed number in every other study.

These tests pin both halves: the default arm is byte-for-byte what it was, and
an explicit seed actually redraws the split.
"""

import numpy as np
import pytest

from vtscore.eval.step_trainers import _train_and_calibrate
from vtscore.eval.voting_columns import VOTING_COLUMNS
from vtscore.eval.voting_iterations import simulate_voting_iterations
from vtscore.training.thresholds import CALIBRATION_SPLIT_SEED

DIM = 16


def _clips(n=24, dim=DIM, seed=42):
    """A two-class pool with enough separation to train, and enough overlap
    that the calibration split has something to disagree about."""
    rng = np.random.default_rng(seed)
    medias = {}
    for i in range(n):
        cat = "target" if i < n // 2 else "other"
        emb = rng.standard_normal(dim).astype(np.float32) + (0.8 if cat == "target" else -0.8)
        medias[i + 1] = {"id": i + 1, "embedder": "e5", "embeddings": {"e5": emb}, "category": cat}
    return medias


def _run(**kwargs):
    return simulate_voting_iterations(_clips(), "target", seed=7, calibrate_count=2, **kwargs)


class TestDefaultArmIsTheApp:
    def test_unset_matches_the_apps_pinned_split_seed(self):
        """The knob's default is not merely *a* constant: it is the same 42 the
        app calibrates with, so an unqualified run is production's own draw."""
        assert _run() == _run(calibration_seed=CALIBRATION_SPLIT_SEED)

    def test_pinned_seed_is_forty_two(self):
        assert CALIBRATION_SPLIT_SEED == 42

    def test_column_records_the_resolved_seed(self):
        """Recorded as the resolved int rather than blanked on a default run:
        a pooled frame has to say which draw each row came from, and "" would
        make the default arm the one case an analyzer cannot read."""
        rows = _run()
        assert rows
        assert {r["calibration_seed"] for r in rows} == {CALIBRATION_SPLIT_SEED}

    def test_column_is_in_the_frame_schema(self):
        assert "calibration_seed" in VOTING_COLUMNS

    def test_explicit_seed_is_recorded_verbatim(self):
        rows = _run(calibration_seed=3)
        assert rows
        assert {r["calibration_seed"] for r in rows} == {3}


class TestArmRedrawsTheSplit:
    def test_a_different_seed_moves_the_threshold(self):
        """With the *votes* fixed, the only thing left to move is which of them
        land on the Calibrate side — so a threshold that moves here is the
        pinned draw's contribution, isolated."""
        clips = _clips()
        good = dict.fromkeys([1, 2, 3, 4, 5, 6])
        bad = dict.fromkeys([13, 14, 15, 16, 17, 18])

        def threshold_at(calibration_seed):
            _step, threshold, _n, _timings, _details = _train_and_calibrate(
                "app",
                good,
                bad,
                clips,
                "target",
                region_voting=False,
                input_dim=DIM,
                inclusion=0,
                calibrate_count=2,
                calibration_fraction=0.5,
                calibration_seed=calibration_seed,
            )
            return threshold

        pinned = threshold_at(CALIBRATION_SPLIT_SEED)
        assert threshold_at(CALIBRATION_SPLIT_SEED) == pinned, "a fixed seed must still be reproducible"
        assert any(threshold_at(s) != pinned for s in range(12)), (
            "no seed in 0..11 redrew the split: the knob is not reaching the fold RNG"
        )

    def test_sweeping_it_leaves_a_readable_spread(self):
        """What a calibration-noise arm actually collects: one number per draw,
        at one cell seed.  The assertion is only that the sweep is not constant
        — the magnitude is the study's finding, not a thing to pin in a test."""
        finals = {
            s: (_run(calibration_seed=s)[-1]["acq_threshold"], _run(calibration_seed=s)[-1]["cost"]) for s in range(6)
        }
        assert len(set(finals.values())) > 1


class TestKnobValidation:
    def test_a_non_int_seed_is_refused_at_second_zero(self):
        """Validated with the other pre-registered knobs, before anything
        expensive runs: a cell that dies forty minutes in on a typo has held a
        cluster slot for nothing."""
        with pytest.raises(ValueError, match="calibration_seed"):
            _run(calibration_seed="42")
