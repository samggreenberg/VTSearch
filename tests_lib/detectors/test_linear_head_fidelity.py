"""The production detector head is a logistic regression, structurally and numerically.

``build_model(input_dim, hidden_dim=LINEAR_HEAD)`` is a single ``Linear(d, 1)``
emitting a logit; pushed through :func:`vtscore.training.mlp.train_model`'s
balanced BCE-with-logits loop it *is* ``LogisticRegression(class_weight=
"balanced")``.  These tests pin both halves of that claim:

* **fidelity** - on seeded synthetic 2-class data the shipped head must rank a
  held-out set in near-lockstep with scikit-learn's ``LogisticRegression``.  If
  a future change to the training loop (an added nonlinearity, a different loss,
  a dropped class weighting) quietly stops it from being logistic regression,
  the rank agreement drops and this test fails.
* **structure / round-trip** - one Linear layer, ``0.weight`` / ``0.bias`` as the
  only state-dict keys, and a clean trip through ``build_model_from_weights``
  and the portable ONNX exporter (whose 1-layer branch only the linear head
  exercises).

The fidelity tests raise ``TRAIN_EPOCHS`` and disable early-stop: the claim
under test is about the *objective* the loop optimises, so the loop has to be
run to convergence.  Production's 200-epoch budget (and this suite's 30) stop
well short of the optimum - an early-stopped linear model, still linear, but not
the ``LogisticRegression`` fixed point.  The synthetic features are deliberately
raw Gaussians rather than the unit-norm vectors a real embedder emits, because
un-normalised features make the problem well-conditioned enough for Adam to
actually reach that fixed point in a unit test's time budget.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from vtscore import config
from vtscore.detectors.portable_bundle import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    embedding_dim_from_weights,
    mlp_weights_to_onnx,
)
from vtscore.detectors.training import serialize_weights
from vtscore.training.mlp import LINEAR_HEAD, build_model, build_model_from_weights, train_model

DIM = 16


@pytest.fixture
def converged_training(monkeypatch):
    """Run ``train_model`` to convergence instead of the suite's 30-epoch budget."""
    monkeypatch.setattr(config, "TRAIN_EPOCHS", 2000, raising=False)
    monkeypatch.setattr(config, "TRAIN_PATIENCE", 0, raising=False)


def _two_class_data(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two overlapping Gaussian blobs: learnable, but not trivially separable.

    The overlap matters - on perfectly separable data the logistic MLE runs off
    to infinity, the two fits are then decided by their (different) penalties
    rather than by the shared objective, and the comparison proves nothing.
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(DIM).astype(np.float32)
    direction /= np.linalg.norm(direction)
    pos = rng.standard_normal((n_per_class, DIM)).astype(np.float32) + 1.2 * direction
    neg = rng.standard_normal((n_per_class, DIM)).astype(np.float32) - 1.2 * direction
    X = np.concatenate([pos, neg]).astype(np.float32)
    y = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)]).astype(np.float32)
    return X, y


def _head_scores(X_train: np.ndarray, y_train: np.ndarray, X_score: np.ndarray) -> np.ndarray:
    """Sigmoid scores of the shipped linear head, trained the production way."""
    model = train_model(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(1),
        DIM,
        seed=0,
        hidden_dim=LINEAR_HEAD,
    )
    with torch.no_grad():
        device = next(model.parameters()).device
        return torch.sigmoid(model(torch.from_numpy(X_score).to(device))).squeeze(1).cpu().numpy()


def _sklearn_scores(X_train: np.ndarray, y_train: np.ndarray, X_score: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=10000).fit(X_train, y_train)
    return clf.predict_proba(X_score)[:, 1]


def _rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr  # noqa: PLC0415

    return float(spearmanr(a, b).statistic)


class TestLogisticFidelity:
    def test_ranks_agree_with_sklearn_logistic_regression(self, converged_training):
        """Spearman >= 0.95 against ``LogisticRegression(class_weight='balanced')``."""
        X, y = _two_class_data(n_per_class=60, seed=12345)
        X_score, _ = _two_class_data(n_per_class=100, seed=999)

        rho = _rank_agreement(_head_scores(X, y, X_score), _sklearn_scores(X, y, X_score))
        assert rho >= 0.95, f"linear head ranks disagree with logistic regression (Spearman {rho:.4f})"

    def test_ranks_agree_when_positives_are_sparse(self, converged_training):
        """The loop's inverse-frequency weights match ``class_weight='balanced'``.

        12 positives against 60 negatives - the sparse-positive regime the linear
        head was adopted for (#2790).  An unweighted loss would tilt away from
        sklearn's balanced fit here even though it matches on balanced data.
        """
        X, y = _two_class_data(n_per_class=60, seed=12345)
        keep = np.concatenate([np.arange(12), np.arange(60, 120)])
        X, y = X[keep], y[keep]
        X_score, _ = _two_class_data(n_per_class=100, seed=999)

        rho = _rank_agreement(_head_scores(X, y, X_score), _sklearn_scores(X, y, X_score))
        assert rho >= 0.95, f"linear head ranks disagree under class imbalance (Spearman {rho:.4f})"


class TestProductionPathTrainsTheLinearHead:
    """``train_and_threshold`` (the Find path) hands back a one-layer head.

    Guards the #2790 swap end-to-end: a revert to ``_auto_hidden_dim`` here
    reinstates the MLP, and this fails rather than letting it back in silently.
    The vote/labelset path (``_train_and_score_xy``) and the load-time
    re-derivation (``train_detector_from_origins``) are pinned the same way from
    the app tier, where their snapshot fixtures live.
    """

    N_MEDIA = 12
    # Ids well clear of the active context's own: the embedding-matrix cache
    # keys on the sorted id list, so a colliding snap gets the wrong matrix.
    ID_BASE = 7000

    def _snap_and_labels(self, seed: int = 5):
        rng = np.random.default_rng(seed)

        def _unit(vec: np.ndarray) -> np.ndarray:
            return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)

        good_proto = _unit(rng.standard_normal(DIM))
        bad_proto = _unit(rng.standard_normal(DIM))
        snap = {
            cid: {
                "id": cid,
                "media_type": "audio",
                "embedder": "test",
                "embeddings": {"test": _unit(rng.standard_normal(DIM))},
            }
            for cid in range(self.ID_BASE + 1, self.ID_BASE + self.N_MEDIA + 1)
        }
        X_list, y_list = [], []
        for _ in range(4):
            X_list.append(_unit(good_proto + 0.1 * rng.standard_normal(DIM)))
            y_list.append(1.0)
            X_list.append(_unit(bad_proto + 0.1 * rng.standard_normal(DIM)))
            y_list.append(0.0)
        return snap, X_list, y_list

    def test_train_and_threshold_returns_a_linear_head(self):
        from vtscore.detectors.training import train_and_threshold  # noqa: PLC0415

        snap, X_list, y_list = self._snap_and_labels()
        model, _threshold = train_and_threshold(X_list, y_list, snap=snap)

        assert [type(layer) for layer in model] == [torch.nn.Linear]
        assert set(serialize_weights(model)) == {"0.weight", "0.bias"}


class TestLinearHeadStructure:
    def test_single_linear_layer(self):
        model = build_model(DIM, hidden_dim=LINEAR_HEAD)
        layers = list(model)
        assert len(layers) == 1
        assert isinstance(layers[0], torch.nn.Linear)
        assert layers[0].in_features == DIM
        assert layers[0].out_features == 1

    def test_dropout_argument_is_ignored(self):
        """A bare linear map has nothing to regularise - no Dropout is inserted."""
        model = build_model(DIM, hidden_dim=LINEAR_HEAD, dropout=0.5)
        assert not any(isinstance(layer, torch.nn.Dropout) for layer in model)

    def test_state_dict_keys(self):
        weights = serialize_weights(build_model(DIM, hidden_dim=LINEAR_HEAD))
        assert set(weights) == {"0.weight", "0.bias"}
        assert np.asarray(weights["0.weight"]).shape == (1, DIM)
        assert embedding_dim_from_weights(weights) == DIM

    def test_round_trips_through_build_model_from_weights(self):
        gen = torch.Generator().manual_seed(3)
        model = build_model(DIM, hidden_dim=LINEAR_HEAD, generator=gen).eval()
        rebuilt = build_model_from_weights(serialize_weights(model))

        assert len(list(rebuilt)) == 1
        rng = np.random.default_rng(4)
        x = torch.from_numpy(rng.standard_normal((5, DIM)).astype(np.float32))
        with torch.no_grad():
            np.testing.assert_allclose(rebuilt(x).numpy(), model(x).numpy(), atol=1e-6)

    def test_onnx_export_is_sigmoid_of_a_single_gemm(self):
        import onnx  # noqa: PLC0415

        gen = torch.Generator().manual_seed(5)
        weights = serialize_weights(build_model(DIM, hidden_dim=LINEAR_HEAD, generator=gen))
        exported = onnx.load_from_string(mlp_weights_to_onnx(weights))

        onnx.checker.check_model(exported)
        # No Relu: the linear head has no hidden activation to model.
        assert [node.op_type for node in exported.graph.node] == ["Gemm", "Sigmoid"]

    def test_onnx_scores_match_torch(self):
        ort = pytest.importorskip("onnxruntime")

        gen = torch.Generator().manual_seed(6)
        model = build_model(DIM, hidden_dim=LINEAR_HEAD, generator=gen).eval()
        weights = serialize_weights(model)

        rng = np.random.default_rng(7)
        x = rng.standard_normal((6, DIM)).astype(np.float32)
        with torch.no_grad():
            expected = torch.sigmoid(model(torch.from_numpy(x))).numpy().ravel()

        session = ort.InferenceSession(mlp_weights_to_onnx(weights))
        got = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: x})[0].ravel()
        np.testing.assert_allclose(got, expected, atol=1e-5)
