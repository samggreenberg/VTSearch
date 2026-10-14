"""#4384's typed-query prior: the head started from the typed query as well as the Goods' centroid.

The app's since #4859 (``tests_lib/detectors/test_typed_query_prior.py`` pins the
app's entry points); here the fit's arithmetic, the eval's scale arm and #4384's
shadow.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests_lib.detectors.test_eval_query_tier import DIM, _run, _world
from vtscore.detectors.label_quota import TIER_TRAINED, label_quota
from vtscore.eval.arms_query_prior import QUERY_PRIOR_COLUMNS, VARIANTS, positive_ranks, query_prior_scale
from vtscore.training import svm_prior
from vtscore.training.mlp import LINEAR_SVM_PRIOR_HEAD, train_model
from vtscore.training.svm_prior import CENTROID_PRIOR_SCALE, centroid_prior_direction


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def _query(seed: int = 0) -> np.ndarray:
    """The typed query :func:`_world` draws its cosines from (the same draws, in the same order)."""
    rng = np.random.default_rng(seed)
    alpha = rng.standard_normal(DIM)
    rng.standard_normal(DIM)  # beta's centre
    q = alpha + 1.2 * rng.standard_normal(DIM)
    return q / np.linalg.norm(q)


def _labels(seed: int = 0, n_good: int = 3, n_bad: int = 4, dim: int = 32):
    """Goods around a class direction, Bads anywhere, and a query near the class."""
    rng = np.random.default_rng(seed)
    cls = _unit(rng.standard_normal(dim))
    goods = _unit(cls + 0.6 * _unit(rng.standard_normal((n_good, dim))))
    bads = _unit(rng.standard_normal((n_bad, dim)))
    query = _unit(cls + 0.4 * _unit(rng.standard_normal(dim)))
    return np.vstack([goods, bads]), np.r_[np.ones(n_good), np.zeros(n_bad)], query


def _scale() -> float:
    """The module's scale as it stands now (read fresh, so a type checker does not carry a narrowing over)."""
    return svm_prior.QUERY_PRIOR_SCALE


def _ap(ranks: str) -> float:
    r = sorted(int(v) for v in ranks.split())
    return float(np.mean([(i + 1) / (k + 1) for i, k in enumerate(r)]))


class TestTheFit:
    def test_unset_is_the_shipped_fit_exactly(self):
        """No query: the centroid-prior fit as #4785 shipped it, bit for bit."""
        from sklearn.svm import LinearSVC

        from vtscore import config
        from vtscore.detectors.centroid_head import goods_centroid

        X, y, _q = _labels()
        c = np.asarray(goods_centroid(list(X[y > 0.5])), dtype=np.float64)
        clf = LinearSVC(
            C=config.SVM_HEAD_C,
            class_weight="balanced",
            dual=False,  # pyright: ignore[reportArgumentType]  # the stubs type it as a string
            max_iter=5000,
            random_state=42,
        )
        clf.fit(np.hstack([X, CENTROID_PRIOR_SCALE * (X @ c)[:, None]]), y.astype(int))
        w = clf.coef_.ravel()
        got, b = centroid_prior_direction(X, y)
        assert np.array_equal(got, w[:-1] + CENTROID_PRIOR_SCALE * w[-1] * c)
        assert b == float(np.asarray(clf.intercept_).ravel()[0])

    def test_the_query_pulls_the_head_toward_it(self):
        """Alone, a cheaper query pulls harder.  (Beside the centroid it need not: the two make their
        difference cheap as well.)"""
        X, y, q = _labels(seed=1)
        near = [_cos(centroid_prior_direction(X, y, scale=0.0, query=(q, s))[0], q) for s in (0.0, 1.0, 3.0, 10.0)]
        assert near == sorted(near)
        assert near[-1] > 0.95

    def test_a_zero_centroid_scale_prices_the_query_alone(self):
        X, y, q = _labels(seed=2)
        alone, _ = centroid_prior_direction(X, y, scale=0.0, query=(q, 10.0))
        both, _ = centroid_prior_direction(X, y, query=(q, 10.0))
        assert not np.allclose(alone, both)
        centroid = _unit(X[y > 0.5].mean(axis=0))
        assert _cos(both, centroid) > _cos(alone, centroid)

    def test_a_query_of_another_size_adds_nothing(self):
        X, y, _q = _labels(seed=3)
        assert np.array_equal(
            centroid_prior_direction(X, y, query=(np.ones(X.shape[1] + 1), 10.0))[0], centroid_prior_direction(X, y)[0]
        )

    def test_still_separates_the_labels(self):
        X, y, q = _labels(seed=4, n_good=5, n_bad=40)
        w, b = centroid_prior_direction(X, y, query=(q, 10.0))
        scores = X @ w + b
        assert scores[y > 0.5].min() > scores[y < 0.5].max()


class TestTheArm:
    def test_the_scale_arm_sets_and_restores_the_scale(self):
        assert _scale() == 10.0
        with query_prior_scale(3.0):
            assert _scale() == 3.0
        with query_prior_scale(0.0):
            assert _scale() == 0.0
        assert _scale() == 10.0

    def test_none_changes_nothing(self):
        with query_prior_scale(None):
            assert _scale() == 10.0

    @pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
    def test_refuses_a_scale_that_is_not_a_number_at_least_zero(self, scale):
        with pytest.raises(ValueError, match=">= 0"), query_prior_scale(scale):
            pass

    def test_every_production_fit_in_the_block_takes_it(self):
        X, y, q = _labels(seed=5)
        with svm_prior.typed_query_prior(q):
            model = train_model(
                torch.from_numpy(X.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32)).unsqueeze(1),
                X.shape[1],
                hidden_dim=LINEAR_SVM_PRIOR_HEAD,
            )
        w, b = centroid_prior_direction(X, y, query=(q, 10.0))
        assert not np.allclose(w, centroid_prior_direction(X, y)[0])
        with torch.no_grad():
            got = model(torch.from_numpy(X.astype(np.float32)).to(next(model.parameters()).device)).cpu().numpy()
        assert np.allclose(got.ravel(), X @ w + b, atol=1e-4)


def test_positive_ranks_order_as_a_line_ranking():
    """Score descending, ties by id, and raw margins outside [0, 1] are ranked too."""
    assert positive_ranks([5, 3, 9, 1], np.array([0.2, 0.9, 0.9, 0.1]), {9, 1}) == "1 3"
    assert positive_ranks([5, 3, 9, 1], np.array([-2.0, 3.5, 3.5, -7.0]), {9, 1}) == "1 3"


@pytest.fixture(scope="module")
def plain_run():
    """Today's app: the typed query's prior on (the run is given its query)."""
    return _run(query_vector=_query())


@pytest.fixture(scope="module")
def shadow_run():
    sink: list[dict] = []
    rows, picks = _run(query_prior_sink=sink, query_vector=_query())
    return rows, picks, sink


@pytest.fixture(scope="module")
def before_run():
    """The head before #4859 (the centroid alone), the shadow on."""
    sink: list[dict] = []
    with query_prior_scale(0.0):
        rows, picks = _run(query_prior_sink=sink, query_vector=_query())
    return rows, picks, sink


def _trained_head_aps(rows: list[dict]) -> dict[int, float]:
    """The trained head's withheld-half AP at each click: the hold's tagged rows, then the untagged ones."""
    out: dict[int, float] = {}
    for r in rows:
        if r.get("schedule"):
            continue
        if r.get("gmm_variant") == "hold:head" or (not r.get("gmm_variant") and r["detector_tier"] == TIER_TRAINED):
            out[r["t"]] = r["average_precision"]
    return out


class TestTheShadow:
    def test_the_fixture_query_is_the_worlds(self):
        clips, seed_scores = _world()
        q = _query()
        assert all(seed_scores[c] == pytest.approx(float(clips[c]["embeddings"]["emb"] @ q), abs=1e-6) for c in clips)

    def test_the_session_is_the_apps(self, plain_run, shadow_run):
        rows, picks = plain_run
        s_rows, s_picks, _sink = shadow_run
        assert [(p["t"], p["picked_id"]) for p in s_picks] == [(p["t"], p["picked_id"]) for p in picks]
        key = ("t", "gmm_variant", "detector_tier", "fbeta", "threshold", "average_precision")
        assert [tuple(r.get(k) for k in key) for r in s_rows] == [tuple(r.get(k) for k in key) for r in rows]

    def test_a_row_at_every_click_a_trained_head_is_drawn(self, shadow_run):
        rows, _picks, sink = shadow_run
        assert sink and all(set(QUERY_PRIOR_COLUMNS) <= set(s) for s in sink)
        trained = sorted({r["t"] for r in rows if label_quota(r["n_good"], r["n_bad"]).tier == TIER_TRAINED})
        assert [s["t"] for s in sink] == trained
        assert all(s[f"ranks_{v}"] for s in sink for v in VARIANTS)

    def test_head_is_the_runs_own_head_with_the_prior_off(self, before_run):
        rows, _picks, sink = before_run
        head = _trained_head_aps(rows)
        assert {s["t"] for s in sink} <= set(head)
        for s in sink:
            assert _ap(s["ranks_head"]) == pytest.approx(head[s["t"]], abs=1e-3), s["t"]

    def test_query_is_the_query_tiers_set(self, shadow_run):
        rows, _picks, sink = shadow_run
        query_ap = {r["average_precision"] for r in rows if r["detector_tier"] == "query" and not r.get("schedule")}
        assert len(query_ap) == 1
        (want,) = query_ap
        assert all(_ap(s["ranks_query"]) == pytest.approx(want) for s in sink)

    def test_by_default_the_runs_head_is_cq10(self, shadow_run, before_run):
        rows, picks, sink = shadow_run
        head = _trained_head_aps(rows)
        for s in sink:
            assert _ap(s["ranks_cq10"]) == pytest.approx(head[s["t"]], abs=1e-3), s["t"]
        # The head started from the query asks its own Hard questions.
        assert [p["picked_id"] for p in picks] != [p["picked_id"] for p in before_run[1]]

    def test_without_a_query_vector_the_run_is_the_centroid_alone(self, before_run):
        rows, picks = _run()
        b_rows, b_picks, _sink = before_run
        assert [p["picked_id"] for p in picks] == [p["picked_id"] for p in b_picks]

    def test_off_without_a_query_vector(self):
        sink: list[dict] = []
        _run(query_prior_sink=sink)
        assert sink == []
