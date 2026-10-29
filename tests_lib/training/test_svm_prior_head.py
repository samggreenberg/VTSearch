"""#4776's linear SVM started from the Goods' centroid (``LINEAR_SVM_PRIOR_HEAD``), an eval arm."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vtscore.training.mlp import LINEAR_SVM_PRIOR_HEAD, train_model
from vtscore.training.svm_prior import CENTROID_PRIOR_SCALE, centroid_prior_direction

DIM = 32


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _few_labels(seed: int = 0, n_good: int = 3, n_bad: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A person's photos around one direction, and other people around their own."""
    rng = np.random.default_rng(seed)
    person = _unit(rng.standard_normal(DIM))
    goods = _unit(person + 0.5 * _unit(rng.standard_normal((n_good, DIM))))
    bads = _unit(rng.standard_normal((n_bad, DIM)))
    X = np.vstack([goods, bads])
    y = np.r_[np.ones(n_good), np.zeros(n_bad)]
    return X, y, person


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


class TestCentroidPriorDirection:
    def test_starts_at_the_centroid(self):
        """On three Goods and four Bads the head points closer to the Goods' centroid than the plain SVM does."""
        from vtscore.eval.arms_handover import _svm

        X, y, _person = _few_labels()
        centroid = _unit(_unit(X[y > 0.5]).mean(axis=0))
        prior, _ = centroid_prior_direction(X, y)
        plain, _ = _svm(X, y)
        assert _cos(prior, centroid) > _cos(plain, centroid)
        assert _cos(prior, centroid) > 0.95

    def test_a_cheaper_centroid_pulls_harder(self):
        X, y, _ = _few_labels(seed=1)
        centroid = _unit(_unit(X[y > 0.5]).mean(axis=0))
        near = [_cos(centroid_prior_direction(X, y, scale=s)[0], centroid) for s in (1.0, 3.0, CENTROID_PRIOR_SCALE)]
        assert near == sorted(near)

    def test_still_separates_the_labels(self):
        X, y, _ = _few_labels(seed=2, n_good=5, n_bad=40)
        w, b = centroid_prior_direction(X, y)
        scores = X @ w + b
        assert scores[y > 0.5].min() > scores[y < 0.5].max()

    def test_needs_both_classes(self):
        X, y, _ = _few_labels()
        with pytest.raises(ValueError, match="both Goods and Bads"):
            centroid_prior_direction(X[y > 0.5], y[y > 0.5])


class TestThroughTrainModel:
    def test_the_head_is_the_direction_as_a_linear_layer(self):
        X, y, _ = _few_labels(seed=3)
        model = train_model(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)).unsqueeze(1),
            DIM,
            hidden_dim=LINEAR_SVM_PRIOR_HEAD,
        )
        assert len(model) == 1 and isinstance(model[0], torch.nn.Linear)
        w, b = centroid_prior_direction(X, y)
        with torch.no_grad():
            got = (
                model(torch.from_numpy(X.astype(np.float32)).to(next(model.parameters()).device)).cpu().numpy().ravel()
            )
        assert np.allclose(got, X @ w + b, atol=1e-4)

    def test_sample_weights_hand_the_balance_to_the_caller(self):
        X, y, _ = _few_labels(seed=4, n_good=3, n_bad=12)
        weights = np.where(y > 0.5, 1.0, 0.25)
        balanced, _ = centroid_prior_direction(X, y)
        weighted, _ = centroid_prior_direction(X, y, sample_weight=weights)
        assert not np.allclose(balanced, weighted)


class TestHardViewGoods:
    """#5413: a hard-view face's Good weighs HARD_VIEW_GOOD_WEIGHT in the fit while some Good is clear."""

    def test_outside_the_block_or_without_a_hard_good_nothing_changes(self):
        from vtscore.training.svm_prior import hard_view_good_weights, hard_view_rows

        X, y, _ = _few_labels(seed=3, n_good=4, n_bad=20)
        plain = centroid_prior_direction(X, y)
        assert hard_view_good_weights(X, y) is None
        with hard_view_rows([X[-1]]):  # a Bad: never weighted
            assert hard_view_good_weights(X, y) is None
            w, b = centroid_prior_direction(X, y)
        np.testing.assert_array_equal(w, plain[0])
        assert b == plain[1]
        with hard_view_rows(list(X[:4])):  # every Good hard: nothing clear to lean on
            assert hard_view_good_weights(X, y) is None
        with hard_view_rows([]):
            assert hard_view_good_weights(X, y) is None

    def test_a_hard_good_weighs_the_constant_and_pulls_less(self, monkeypatch):
        from vtscore.training import svm_prior

        X, y, person = _few_labels(seed=4, n_good=4, n_bad=20)
        X = X.copy()
        X[0] = _unit(X[0] - 1.2 * person)  # the hard view: far from the person
        plain, _ = centroid_prior_direction(X, y)
        with svm_prior.hard_view_rows([X[0]]):
            np.testing.assert_array_equal(svm_prior.hard_view_good_weights(X, y), [0.25, 1.0, 1.0, 1.0])
            weighted, _ = centroid_prior_direction(X, y)
        assert _cos(weighted, person) > _cos(plain, person)
        monkeypatch.setattr(svm_prior, "HARD_VIEW_GOOD_WEIGHT", 1.0)
        with svm_prior.hard_view_rows([X[0]]):
            assert svm_prior.hard_view_good_weights(X, y) is None

    def test_weight_zero_fits_as_if_the_row_were_gone(self, monkeypatch):
        """The balance is shared within the Goods, so a weightless Good leaves the other rows' fit as it was."""
        from vtscore.training import svm_prior

        X, y, _ = _few_labels(seed=5, n_good=4, n_bad=20)
        monkeypatch.setattr(svm_prior, "HARD_VIEW_GOOD_WEIGHT", 0.0)
        with svm_prior.hard_view_rows([X[0]]):
            weighted, _ = centroid_prior_direction(X, y)
        n = len(y)
        keep = np.arange(n) > 0
        # Without the row, balanced weights are (n - 1) / (2 * 3) per Good; with it, n / 2 shared by three.
        sw = np.where(y[keep] > 0.5, n / 6.0, n / 40.0)
        dropped, _ = centroid_prior_direction(X[keep], y[keep], sample_weight=sw)
        assert _cos(weighted, dropped) > 0.99999

    def test_the_balance_keeps_each_class_mass(self):
        from vtscore.training.svm_prior import _weigh_goods

        pos = np.r_[np.ones(4, bool), np.zeros(12, bool)]
        out = _weigh_goods(pos, None, np.array([0.25, 1.0, 1.0, 1.0]))
        assert out[pos].sum() == pytest.approx(8.0) and out[~pos].sum() == pytest.approx(8.0)
        assert out[0] == pytest.approx(out[1] / 4)
        flood = np.r_[np.full(4, 2.0), np.full(12, 0.5)]
        out = _weigh_goods(pos, flood, np.array([0.25, 1.0, 1.0, 1.0]))
        assert out[pos].sum() == pytest.approx(8.0)
        np.testing.assert_array_equal(out[~pos], flood[~pos])

    def test_the_goods_centroid_takes_the_weights(self):
        from vtscore.detectors.centroid_head import goods_centroid

        X, _y, _ = _few_labels(seed=6, n_good=3)
        want = _unit(0.25 * _unit(X[0]) + _unit(X[1]) + _unit(X[2]))
        np.testing.assert_allclose(goods_centroid(list(X[:3]), [0.25, 1.0, 1.0]), want, atol=1e-6)
        np.testing.assert_allclose(goods_centroid(list(X[:3])), _unit(_unit(X[:3]).mean(axis=0)), atol=1e-6)

    def test_cached_calibration_folds_never_cross_the_hard_views(self):
        from vtscore.training.svm_prior import hard_view_rows
        from vtscore.training.thresholds import _calibration_cache_key

        X, y, _ = _few_labels(seed=7, n_good=4, n_bad=8)
        key = _calibration_cache_key(list(X), list(y), 2, 0.5, -3)
        with hard_view_rows([X[0]]):
            hard = _calibration_cache_key(list(X), list(y), 2, 0.5, -3)
        assert key != hard
