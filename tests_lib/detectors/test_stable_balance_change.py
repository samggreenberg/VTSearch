"""Stable measures detectors changing their minds, not the user changing the balance (#4745).

Stable counts the unlabeled items that cross the *served* line between two
successive detectors, each read at its own line.  Since #4452 that line is the
labels line at the detector's balance (F-beta's beta), so a balance change moves
it without any detector learning anything.  The re-sort a balance change
triggers retrains on unchanged votes, which records no stability entry, so the
next vote's detector used to be compared, at the new balance's line, against a
baseline drawn at the old one: every item between the two lines counted as a
flip, and the confident ones could hold Stable off green and send Autopilot
back to Boundary.

A balance change now restarts the comparison.  The line moving with the
*labels* under one balance still counts, because that movement is what the
user sees.
"""

from __future__ import annotations

import numpy as np
import pytest

import vtscore.detectors.labeling_progress as lp
from vtscore.embedding.media_vectors import EMBEDDINGS_KEY

DIM = 8

#: Two lines far apart on a head whose scores spread across (0, 1): dozens of
#: the pool's items sit between them, most of them clear of both by the
#: ambiguity band, so a comparison across the two counts confident flips.
STRICT_LINE = 0.8
LENIENT_LINE = 0.2


def _clips(n: int = 200, seed: int = 42) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    return {
        cid: {EMBEDDINGS_KEY: {"test": rng.standard_normal(DIM).astype(np.float32)}, "embedder": "test"}
        for cid in range(n)
    }


def _head():
    """One fixed linear head, so nothing but the line can differ between the two steps."""
    import torch
    import torch.nn as nn

    linear = nn.Linear(DIM, 1)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, 0] = 3.0
        linear.bias.zero_()
    model = nn.Sequential(linear)
    model.eval()
    return model


def _history(n_votes: int) -> list[tuple[int, str, float]]:
    return [(k, "good" if k % 2 == 0 else "bad", float(k)) for k in range(n_votes)]


def _stability(lines: dict[int, tuple[float, float | None]]) -> list[dict]:
    """The Stable entries for six votes, with the same head served at *lines*: ``{step: (threshold, beta)}``."""
    pytest.importorskip("torch")
    clips, history = _clips(), _history(6)
    lp.clear_progress_cache()
    model = _head()
    good: dict[int, None] = {}
    bad: dict[int, None] = {}
    for t, (mid, label, _) in enumerate(history):
        (good if label == "good" else bad)[mid] = None
        if t in lines:
            threshold, beta = lines[t]
            lp.inject_live_model(dict(good), dict(bad), model, threshold, beta=beta)
    return lp.calculate_prediction_stability_over_time(clips, history)


class TestBalanceChangeIsNotAFlip:
    def test_a_balance_change_between_two_detectors_counts_no_flips(self):
        """Beta 1/4 then beta 4, same head: the line moved, no detector changed its mind."""
        entries = _stability({4: (STRICT_LINE, 0.25), 5: (LENIENT_LINE, 4.0)})

        assert sum(e["num_flips"] for e in entries) == 0
        assert sum(e["num_confident_flips"] for e in entries) == 0

    def test_the_chain_resumes_after_the_balance_change(self):
        """The new balance's first detector is the baseline: the next one is compared against it."""
        entries = _stability({3: (STRICT_LINE, 0.25), 4: (LENIENT_LINE, 4.0), 5: (LENIENT_LINE, 4.0)})

        assert [e["time_index"] for e in entries] == [5]
        assert entries[0]["num_flips"] == 0


class TestTheLineMovingUnderOneBalanceStillCounts:
    @pytest.mark.parametrize("beta", [1.0, None], ids=["balance", "no-balance"])
    def test_a_line_move_at_one_balance_is_counted(self, beta):
        """A retrain that moves the line at the same balance is movement the user sees.

        Also the fixture's own check: the two lines have confident flips
        between them, so the balance-change tests above are not passing on a
        pool where nothing could cross.
        """
        entries = _stability({4: (STRICT_LINE, beta), 5: (LENIENT_LINE, beta)})

        assert [e["time_index"] for e in entries] == [5]
        assert entries[0]["num_confident_flips"] > 0
