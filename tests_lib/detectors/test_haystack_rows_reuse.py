"""The population estimator stacks the haystack once, not once per fold.

``_fused_threshold`` scores the haystack through every calibration fold model.
Each pass used to re-enter ``_score_all_media``, which rebuilds the
``(media, score row)`` matrix from the snapshot - free on the active dataset
(the builders cache there) and emphatically not free on a **cross-dataset**
snapshot, which is exactly where the estimator's new caller lives: cold Find
hands it a corpus loaded from a pkl, and those builders deliberately do not
populate the active context's cache with a foreign id set.

So the corpus was being restacked ``calibrate_count + 1`` times per (detector,
dataset) pair - on a patch dataset that is ~197 float16 rows per media rebuilt
three times over.  Only the head changes between the passes, never the rows, so
the rows are now built once and shared; a caller that already holds them (Find
scores the same corpus for its own verdicts) passes them in as ``haystack_rows``
and the corpus is stacked once for the whole operation.
"""

from __future__ import annotations

import numpy as np
import pytest

import vtscore.detectors.training as training

DIM = 8
EMB = "dinov3_patch"


def _snap(n: int = 40) -> dict[int, dict]:
    rng = np.random.default_rng(3516)
    snap: dict[int, dict] = {}
    for cid in range(1, n + 1):
        vec = rng.standard_normal(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        snap[cid] = {
            "id": cid,
            "media_type": "image",
            "embedder": EMB,
            "embeddings": {EMB: vec},
            "filename": f"m{cid}.png",
            "md5": f"m{cid}",
        }
    return snap


def _xy(snap: dict[int, dict], n: int = 12) -> tuple[list, list[float], set[int]]:
    ids = sorted(snap)[:n]
    X = [snap[cid]["embeddings"][EMB] for cid in ids]
    y = [1.0 if i % 2 == 0 else 0.0 for i in range(n)]
    return X, y, set(ids)


@pytest.fixture
def count_row_builds(monkeypatch):
    """Count ``scoring_rows_for_snap`` calls made from inside the training module."""
    real = training.scoring_rows_for_snap
    calls: list[int] = []

    def _counting(clips_dict, embedder_name=None):
        calls.append(1)
        return real(clips_dict, embedder_name)

    monkeypatch.setattr(training, "scoring_rows_for_snap", _counting)
    return calls


def test_one_row_build_per_training_call(count_row_builds):
    """Whatever ``calibrate_count`` is, the haystack is stacked once."""
    snap = _snap()
    X, y, voted = _xy(snap)

    _model, threshold = training.train_and_threshold(X, y, snap=snap, embedder_name=EMB, voted_ids=voted)

    assert np.isfinite(threshold)
    assert len(count_row_builds) == 1, (
        f"the haystack was stacked {len(count_row_builds)} times for one training call; "
        "the fold passes must reuse the rows the final model was scored over"
    )


def test_caller_supplied_rows_are_not_rebuilt(count_row_builds):
    """``haystack_rows`` means the corpus is stacked once for the whole operation."""
    snap = _snap()
    X, y, voted = _xy(snap)
    rows = training.scoring_rows_for_snap(snap, EMB)
    count_row_builds.clear()

    _model, _threshold = training.train_and_threshold(
        X, y, snap=snap, embedder_name=EMB, voted_ids=voted, haystack_rows=rows
    )

    assert count_row_builds == [], "haystack_rows was ignored and the corpus restacked"


def test_supplying_rows_changes_nothing_but_the_work():
    """The optimisation is invisible in the answer: same head, same cut."""
    snap = _snap()
    X, y, voted = _xy(snap)
    rows = training.scoring_rows_for_snap(snap, EMB)

    _m1, built_here = training.train_and_threshold(X, y, snap=snap, embedder_name=EMB, voted_ids=voted)
    _m2, passed_in = training.train_and_threshold(
        X, y, snap=snap, embedder_name=EMB, voted_ids=voted, haystack_rows=rows
    )

    assert built_here == passed_in
