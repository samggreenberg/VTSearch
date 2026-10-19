"""``scoring_progress``: a thread's scoring passes reported in media (#5101).

A Find's bar used to count detectors, so a one-detector run sat at 0/1 until it
finished.  Every pass a detector's train and its own scoring make goes through
:func:`~vtscore.detectors.training.score_rows_with_model`, which now reports
each ``ROW_CHUNK`` block to the calling thread's hook as the media finished so
far.  These tests pin that count on a patch-style stack (several rows per
media, so a chunk can end mid-media), and that the hook is scoped to its block
and to its thread.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch
import torch.nn as nn

import vtscore.embedding.matrix as matrix_mod
from vtscore.detectors.training import ScoringRows, score_rows_with_model, scoring_progress

DIM = 4
ROWS_PER_MEDIA = 3
N_MEDIA = 5


def _rows() -> ScoringRows:
    rng = np.random.default_rng(42)
    n_rows = N_MEDIA * ROWS_PER_MEDIA
    return ScoringRows(
        ids=list(range(100, 100 + N_MEDIA)),
        matrix=rng.standard_normal((n_rows, DIM)).astype(np.float32),
        media_index=np.repeat(np.arange(N_MEDIA), ROWS_PER_MEDIA),
        region_index=np.tile(np.arange(ROWS_PER_MEDIA), N_MEDIA),
    )


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(DIM, 1))


def test_reports_the_media_each_chunk_has_finished(monkeypatch):
    # 15 rows in chunks of 4: the chunks end at rows 4, 8, 12 and 15, which is
    # inside media 1, inside media 2, at the end of media 3, and at the end.
    monkeypatch.setattr(matrix_mod, "ROW_CHUNK", 4)
    calls: list[tuple[int, int]] = []
    with scoring_progress(lambda done, total: calls.append((done, total))):
        scores, _best = score_rows_with_model(_model(), _rows())
    assert len(scores) == N_MEDIA
    assert calls == [(0, 5), (1, 5), (2, 5), (4, 5), (5, 5)]


def test_one_chunk_is_a_start_and_an_end():
    calls: list[tuple[int, int]] = []
    with scoring_progress(lambda done, total: calls.append((done, total))):
        score_rows_with_model(_model(), _rows())
        score_rows_with_model(_model(), _rows())
    assert calls == [(0, 5), (5, 5), (0, 5), (5, 5)], "each pass starts again from zero"


def test_scores_are_the_same_with_and_without_the_hook(monkeypatch):
    monkeypatch.setattr(matrix_mod, "ROW_CHUNK", 4)
    bare = score_rows_with_model(_model(), _rows())
    with scoring_progress(lambda _done, _total: None):
        hooked = score_rows_with_model(_model(), _rows())
    assert hooked == bare


def test_reports_nothing_outside_the_block_and_restores_an_outer_hook():
    outer: list[tuple[int, int]] = []
    inner: list[tuple[int, int]] = []
    with scoring_progress(lambda done, total: outer.append((done, total))):
        with scoring_progress(lambda done, total: inner.append((done, total))):
            score_rows_with_model(_model(), _rows())
        with scoring_progress(None):
            score_rows_with_model(_model(), _rows())
        score_rows_with_model(_model(), _rows())
    score_rows_with_model(_model(), _rows())
    assert inner == [(0, 5), (5, 5)]
    assert outer == [(0, 5), (5, 5)], "only the pass back under the outer block reaches it"


def test_an_empty_stack_reports_nothing():
    calls: list[tuple[int, int]] = []
    empty = ScoringRows(
        ids=[], matrix=np.zeros((0, DIM), np.float32), media_index=np.zeros(0, int), region_index=np.zeros(0, int)
    )
    with scoring_progress(lambda done, total: calls.append((done, total))):
        assert score_rows_with_model(_model(), empty) == ([], [])
    assert calls == []


def test_the_hook_is_per_thread():
    """Detectors scored in parallel each count only their own passes."""
    here: list[tuple[int, int]] = []
    there: list[tuple[int, int]] = []
    hooked = threading.Event()
    scored_here = threading.Event()

    def other_thread() -> None:
        with scoring_progress(lambda done, total: there.append((done, total))):
            hooked.set()
            assert scored_here.wait(timeout=10)
            score_rows_with_model(_model(), _rows())

    worker = threading.Thread(target=other_thread)
    worker.start()
    assert hooked.wait(timeout=10)
    with scoring_progress(lambda done, total: here.append((done, total))):
        score_rows_with_model(_model(), _rows())
    scored_here.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert here == [(0, 5), (5, 5)]
    assert there == [(0, 5), (5, 5)]


def test_the_hook_is_cleared_when_the_block_raises():
    calls: list[tuple[int, int]] = []
    with pytest.raises(ValueError):
        with scoring_progress(lambda done, total: calls.append((done, total))):
            raise ValueError("boom")
    score_rows_with_model(_model(), _rows())
    assert calls == []
