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
