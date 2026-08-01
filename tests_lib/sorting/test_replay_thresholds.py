"""Pure-core tests for the Stage-A threshold replay tool (issue #2790).

Exercise the cache-free layer — vote-set reconstruction, per-step replay over an
injected vector loader, and the variance decomposition — on synthetic traces, so
the science is verified with no npz cache, no models, and no GPU (the
:class:`CacheVectorLoader` glue is Grid-validated separately). The replay import
lives under ``scripts/sod``; the test adds that dir to ``sys.path`` the same way
the sweep venv does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sod"))

from replay_thresholds import (  # noqa: E402
    StepVectors,
    decompose_variance,
    reconstruct_vote_sets,
    replay_step_thresholds,
)


def _linear_trainer_factory():
    def factory():
        def trainer_fn(X, y, seed):  # noqa: ARG001
            def predict(Xc):
                return np.asarray(Xc, dtype=np.float64)[:, 0]

            return predict

        return trainer_fn

    return factory


class TestReconstructVoteSets:
    def test_accumulates_in_labeling_order(self):
        trace = [
            {"t": 0, "image_id": 10, "gt_label": "good"},
            {"t": 1, "image_id": 20, "gt_label": "bad"},
            {"t": 2, "image_id": 30, "gt_label": "good"},
        ]
        vs = reconstruct_vote_sets(trace)
        assert [t for t, _ in vs] == [0, 1, 2]
        assert vs[0][1] == ([10], [])
        assert vs[1][1] == ([10], [20])
        assert vs[2][1] == ([10, 30], [20])

    def test_accepts_numeric_and_bool_labels_and_dedups(self):
        trace = [
            {"t": 0, "image_id": 1, "gt_label": 1},
            {"t": 1, "image_id": 2, "gt_label": 0.0},
            {"t": 2, "image_id": 1, "gt_label": 1},  # duplicate id: ignored
        ]
        vs = reconstruct_vote_sets(trace)
        assert vs[-1][1] == ([1], [2])


class TestReplayAndDecompose:
    def _vote_vectors(self):
        rng = np.random.default_rng(0)
        # A fixed 1-D-separable embedding per image id, so the injected loader is
        # deterministic and the linear trainer separates the classes.
        coord = {i: float(v) for i, v in enumerate(rng.uniform(-2, 2, size=200))}

        def loader(good_ids, bad_ids):
            if not good_ids or not bad_ids:
                return None
            X = np.array([[coord[i] + 1.0] for i in good_ids] + [[coord[i] - 1.0] for i in bad_ids])
            y = np.array([1.0] * len(good_ids) + [0.0] * len(bad_ids))
            return StepVectors(X=X, y=y)

        return loader

    def _trace(self, n=30):
        # Alternate good/bad so both classes appear early; ids unique.
        return [{"t": t, "image_id": t, "gt_label": "good" if t % 2 == 0 else "bad"} for t in range(n)]

    def test_replay_emits_row_per_step_rule_seed_after_valid_split(self):
        vs = reconstruct_vote_sets(self._trace())
        rows = replay_step_thresholds(
            vs,
            self._vote_vectors(),
            _linear_trainer_factory(),
            rules=["argmin", "conformal"],
            fold_seeds=range(3),
            trainer_seeds=range(2),
            calibrate_count=2,
        )
        # Every emitted row carries the identifying keys and a finite threshold.
        assert rows, "expected replay rows once both classes are present"
        for r in rows:
            assert set(r) >= {"t", "rule", "fold_seed", "trainer_seed", "threshold", "threshold_smoothed"}
            assert np.isfinite(r["threshold"])
        assert {r["rule"] for r in rows} == {"argmin", "conformal"}

    def test_med3_smoothing_is_median_of_last_three(self):
        vs = reconstruct_vote_sets(self._trace())
        rows = replay_step_thresholds(
            vs,
            self._vote_vectors(),
            _linear_trainer_factory(),
            rules=["conformal"],
            fold_seeds=[0],
            trainer_seeds=[0],
            smooth="med3",
        )
        traj = [r["threshold"] for r in rows]
        smoothed = [r["threshold_smoothed"] for r in rows]
        # The 3rd+ smoothed value equals the median of the trailing 3 raw values.
        for i in range(2, len(traj)):
            assert smoothed[i] == float(np.median(traj[i - 2 : i + 1]))

    def test_decompose_variance_reports_sd_and_spike_rate(self):
        vs = reconstruct_vote_sets(self._trace(40))
        rows = replay_step_thresholds(
            vs,
            self._vote_vectors(),
            _linear_trainer_factory(),
            rules=["conformal"],
            fold_seeds=range(4),
            trainer_seeds=[0],
        )
        agg = decompose_variance(rows, warmup_t=20)
        assert agg, "expected aggregated cells past warmup"
        for a in agg:
            assert a["t"] >= 20  # warmup filter honored
            assert a["sd_threshold"] >= 0.0
            assert 0.0 <= a["spike_rate"] <= 1.0
            assert a["n_seeds"] == 4
