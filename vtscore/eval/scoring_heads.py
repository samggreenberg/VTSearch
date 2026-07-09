"""Scoring heads over region vectors for the small-object-detection sweep.

A head turns per-image region vectors into a per-image score. Both heads expose
a ``trainer_fn`` (``(X, y, seed) -> predict``) so the *same* cross-calibration
(:func:`vtscore.eval.xcal.cross_calibrated_threshold`) selects the threshold for
either — the threshold is chosen on held-out data, never from the model's own
0.5 default.

* :class:`MLPHead` — PRIMARY, works for any embedder. Trains the same small MLP
  the detector uses (:func:`vtscore.training.mlp.train_model`) on positive
  GT-box exemplar vectors vs negative whole-image vectors.
* :class:`CosineHead` — zero-shot baseline for text-capable embedders. Scores by
  cosine to a fixed query vector; "training" is a no-op, so cross-calibration
  just picks the cosine cutoff on held-out labels.

Per-image scoring is a max-pool over the image's region rows (an image scores by
its best-matching region), matching the detector's region-aware inference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

PredictFn = Callable[[np.ndarray], np.ndarray]
TrainerFn = Callable[[np.ndarray, np.ndarray, int], PredictFn]


def max_pool_over_images(score_fn: PredictFn, region_mats: Sequence[np.ndarray]) -> np.ndarray:
    """Per-image max of ``score_fn`` over each image's region rows.

    Concatenates every image's ``(R_i, D)`` region matrix, runs ``score_fn``
    once (a single MLP forward over all rows), then segment-maxes back to one
    score per image. Images with no regions score ``-inf``. Context-free
    equivalent of the detector's ``_segmented_max_pool`` scoring path.
    """
    counts = [int(m.shape[0]) for m in region_mats]
    out = np.full(len(region_mats), float("-inf"), dtype=np.float64)
    nonempty = [m for m in region_mats if m.shape[0] > 0]
    if not nonempty:
        return out
    big = np.concatenate(nonempty, axis=0)
    scores = np.asarray(score_fn(big), dtype=np.float64).reshape(-1)
    i = 0
    for j, c in enumerate(counts):
        if c > 0:
            out[j] = float(scores[i : i + c].max())
            i += c
    return out


def max_pool_with_argmax(score_fn: PredictFn, region_mats: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`max_pool_over_images` but also return the winning region index.

    Returns ``(scores, argmax)`` where ``argmax[j]`` is the index (within image
    ``j``'s region matrix) of the max-scoring region, or ``-1`` for an image with
    no regions. Used for localization (the IoU box is that region's box).
    """
    counts = [int(m.shape[0]) for m in region_mats]
    scores = np.full(len(region_mats), float("-inf"), dtype=np.float64)
    argmax = np.full(len(region_mats), -1, dtype=np.int64)
    nonempty = [m for m in region_mats if m.shape[0] > 0]
    if not nonempty:
        return scores, argmax
    big = np.concatenate(nonempty, axis=0)
    flat = np.asarray(score_fn(big), dtype=np.float64).reshape(-1)
    i = 0
    for j, c in enumerate(counts):
        if c > 0:
            block = flat[i : i + c]
            bi = int(block.argmax())
            scores[j] = float(block[bi])
            argmax[j] = bi
            i += c
    return scores, argmax


def _mlp_predict_factory(model) -> PredictFn:
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(np.asarray(x, dtype=np.float32), device=device)
            return torch.sigmoid(model(t)).squeeze(-1).cpu().numpy()

    return predict


class MLPHead:
    """Learned head: trains the detector MLP on exemplar/negative vectors."""

    name = "mlp"

    def __init__(self, input_dim: int) -> None:
        self.input_dim = int(input_dim)
        self._predict: PredictFn | None = None

    def _train(self, x: np.ndarray, y: np.ndarray, seed: int) -> PredictFn:
        import torch  # noqa: PLC0415

        from vtscore.training.mlp import train_model  # noqa: PLC0415

        xt = torch.tensor(np.asarray(x, dtype=np.float32))
        yt = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(1)
        model = train_model(xt, yt, self.input_dim, seed=seed)  # raises ValueError on single-class
        return _mlp_predict_factory(model)

    def trainer_fn(self) -> TrainerFn:
        return self._train

    def fit(self, x: np.ndarray, y: np.ndarray, seed: int) -> None:
        self._predict = self._train(x, y, seed)

    def score_rows(self, region_matrix: np.ndarray) -> np.ndarray:
        if self._predict is None:
            raise RuntimeError("MLPHead.fit must be called before scoring")
        return self._predict(region_matrix)


class CosineHead:
    """Zero-shot head: cosine to a fixed (L2-normalized) query vector."""

    name = "cosine"

    def __init__(self, query_vec: np.ndarray) -> None:
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        self.query = q / (np.linalg.norm(q) + 1e-12)
        self.input_dim = int(self.query.shape[0])

    def _train(self, x: np.ndarray, y: np.ndarray, seed: int) -> PredictFn:  # noqa: ARG002
        # No training: cosine to the fixed query. Cross-calibration still picks
        # the cutoff on the held-out split's cosine scores.
        return self.score_rows

    def trainer_fn(self) -> TrainerFn:
        return self._train

    def fit(self, x: np.ndarray, y: np.ndarray, seed: int) -> None:  # noqa: ARG002
        pass  # nothing to fit

    def score_rows(self, region_matrix: np.ndarray) -> np.ndarray:
        return np.asarray(region_matrix, dtype=np.float32) @ self.query
