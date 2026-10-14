"""#4859: a typed-query detector's head starts from its typed query as well as from the Goods' centroid.

#4384 priced it on full sessions (the head's AP at its first fit 0.43 against the
centroid-only head's 0.34; the objective over votes 1-150 +0.013 / +0.013 / +0.018
at beta 1/4, 1 and 4).  These tests pin the fit inside
:func:`~vtscore.training.svm_prior.typed_query_prior`, the training entry points
that enter it (and where they do not), the learned sort's query, the caches that
must not cross it, and the harness default that follows the app.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import vtscore.detectors.labelset_training as lt
import vtscore.detectors.query_head as qh
from tests_lib.detectors.test_query_head import EMBEDDER, QUERY, _ctx, single_vector, world  # noqa: F401
from vtscore.detectors.model_loading import cached_head_is_current
from vtscore.detectors.query_head import is_quota_head, typed_query_prior_applies
from vtscore.eval.arms_query_prior import query_prior_scale
from vtscore.training import svm_prior
from vtscore.training.svm_prior import (
    QUERY_PRIOR_SCALE,
    active_query_prior,
    centroid_prior_direction,
    fitted_query_prior_text,
    typed_query_prior,
)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _labels(seed: int = 0, n_good: int = 3, n_bad: int = 4, dim: int = 16):
    rng = np.random.default_rng(seed)
    cls = _unit(rng.standard_normal(dim))
    goods = _unit(cls + 0.6 * _unit(rng.standard_normal((n_good, dim))))
    bads = _unit(rng.standard_normal((n_bad, dim)))
    return (
        np.vstack([goods, bads]),
        np.r_[np.ones(n_good), np.zeros(n_bad)],
        _unit(cls + 0.4 * rng.standard_normal(dim)),
    )


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.asarray(a).ravel() @ np.asarray(b).ravel() / np.linalg.norm(a) / np.linalg.norm(b))


def _head_weight(model) -> np.ndarray:
    return model[0].weight.detach().cpu().numpy().ravel().astype(np.float64)


class TestTheFit:
    def test_the_app_scale_is_ten(self):
        assert QUERY_PRIOR_SCALE == 10.0

    def test_the_context_is_the_explicit_query_at_the_app_scale(self):
        X, y, q = _labels()
        with typed_query_prior(q, "a thing"):
            inside, _ = centroid_prior_direction(X, y)
        explicit, _ = centroid_prior_direction(X, y, query=(_unit(q), QUERY_PRIOR_SCALE))
        assert np.allclose(inside, explicit)
        assert not np.allclose(inside, centroid_prior_direction(X, y)[0]), "outside the block: the centroid alone"

    def test_none_and_scale_zero_change_nothing(self):
        X, y, q = _labels(seed=1)
        before = centroid_prior_direction(X, y)[0]
        with typed_query_prior(None):
            assert active_query_prior() is None
            assert np.array_equal(centroid_prior_direction(X, y)[0], before)
        with query_prior_scale(0.0), typed_query_prior(q):
            assert active_query_prior() is None, "the eval's arm before #4859"
            assert np.array_equal(centroid_prior_direction(X, y)[0], before)
        assert svm_prior.QUERY_PRIOR_SCALE == QUERY_PRIOR_SCALE

    def test_heads_fitted_in_the_block_carry_the_query(self):
        import torch

        from vtscore.training.mlp import LINEAR_SVM_PRIOR_HEAD, train_model

        X, y, q = _labels(seed=2)
        Xt, yt = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
        with typed_query_prior(q, "a thing"):
            inside = train_model(Xt, yt, X.shape[1], hidden_dim=LINEAR_SVM_PRIOR_HEAD)
        outside = train_model(Xt, yt, X.shape[1], hidden_dim=LINEAR_SVM_PRIOR_HEAD)
        assert fitted_query_prior_text(inside) == "a thing"
        assert fitted_query_prior_text(outside) is None
        assert _cos(_head_weight(inside), centroid_prior_direction(X, y, query=(_unit(q), 10.0))[0]) > 0.99999

    def test_a_query_of_another_size_is_dropped_and_not_claimed(self):
        import torch

        from vtscore.training.mlp import LINEAR_SVM_PRIOR_HEAD, train_model

        X, y, _q = _labels(seed=3)
        with typed_query_prior(np.ones(X.shape[1] + 2), "a thing"):
            model = train_model(
                torch.from_numpy(X.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32)).unsqueeze(1),
                X.shape[1],
                hidden_dim=LINEAR_SVM_PRIOR_HEAD,
            )
        assert fitted_query_prior_text(model) is None
        assert _cos(_head_weight(model), centroid_prior_direction(X, y)[0]) > 0.99999

    def test_another_thread_does_not_see_the_query(self):
        _X, _y, q = _labels(seed=4)
        seen: list[object] = []
        with typed_query_prior(q):
            t = threading.Thread(target=lambda: seen.append(active_query_prior()))
            t.start()
            t.join()
            assert active_query_prior() is not None
        assert seen == [None]

    def test_cached_calibration_folds_never_cross_the_query(self):
        from vtscore.training.thresholds import _calibration_cache_key

        X, y, q = _labels(seed=5)
        key = _calibration_cache_key(list(X), list(y), 2, 0.5, -3)
        with typed_query_prior(q):
            with_q = _calibration_cache_key(list(X), list(y), 2, 0.5, -3)
        with typed_query_prior(-q):
            other_q = _calibration_cache_key(list(X), list(y), 2, 0.5, -3)
        assert len({key, with_q, other_q}) == 3


class TestWhereItApplies:
    def test_single_vector_text_spaces_only(self):
        assert typed_query_prior_applies("siglip")
        assert not typed_query_prior_applies("dinov3_patch"), "a patch space: the query has no direction there"
        assert not typed_query_prior_applies("not_an_embedder")
        assert not typed_query_prior_applies(None)

    def test_no_text_no_vector(self):
        assert qh.typed_query_prior_vector("", "siglip", "image") is None


@pytest.fixture
def applies(monkeypatch, single_vector):  # noqa: F811
    """The test embedder as a single-vector text space (the registry does not know it)."""
    monkeypatch.setattr(qh, "typed_query_prior_applies", lambda name: name == EMBEDDER)


def _past_the_hold(make):
    ls = make(list(range(1, 9)), list(range(40, 62)))
    assert len(ls.elements) == 30
    return ls


class TestTheApp:
    @pytest.mark.usefixtures("applies")
    def test_a_typed_query_detector_fits_from_its_query(self, world):  # noqa: F811
        snap, proto, make, _embedded = world
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, _past_the_hold(make), media_type="image", snap=snap, text_query=QUERY)
        assert ctx.model is not None and not is_quota_head(ctx.model)
        assert fitted_query_prior_text(ctx.model) == QUERY

    def test_without_a_query_or_off_its_space_it_fits_from_the_goods_alone(self, world, single_vector, monkeypatch):  # noqa: F811
        snap, _proto, make, _embedded = world
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, _past_the_hold(make), media_type="image", snap=snap)
        assert fitted_query_prior_text(ctx.model) is None
        monkeypatch.setattr(qh, "typed_query_prior_applies", lambda name: False)
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, _past_the_hold(make), media_type="image", snap=snap, text_query=QUERY)
        assert fitted_query_prior_text(ctx.model) is None

    @pytest.mark.usefixtures("applies")
    def test_the_prior_moves_the_head_toward_the_query(self, world):  # noqa: F811
        snap, proto, make, _embedded = world
        with_q, without = _ctx(), _ctx()
        ls = _past_the_hold(make)
        assert lt.train_from_labelset(with_q, ls, media_type="image", snap=snap, text_query=QUERY)
        assert lt.train_from_labelset(without, ls, media_type="image", snap=snap)
        assert _cos(_head_weight(with_q.model), proto) > _cos(_head_weight(without.model), proto)

    @pytest.mark.usefixtures("applies")
    def test_the_learned_sort_fits_from_the_query_it_is_handed(self, world):  # noqa: F811
        snap, _proto, make, _embedded = world
        ls = make([1, 2, 3], [40, 41, 42, 43])
        _r, _t, model = lt.labelset_train_and_score(_ctx(), ls, media_type="image", clips_dict=snap, text_query=QUERY)
        assert model is not None and not is_quota_head(model), "the Train view asks from the head"
        assert fitted_query_prior_text(model) == QUERY
        _r, _t, model = lt.labelset_train_and_score(_ctx(), ls, media_type="image", clips_dict=snap)
        assert fitted_query_prior_text(model) is None

    def test_the_learned_sort_reads_the_detectors_query_from_the_registry(self, monkeypatch):
        import vtscore.detectors.registry as registry
        from vtscore.detectors.learned_sort import resolve_active_text_query

        entries = {
            "d1": {"id": "d1", "text_query": " a chair "},
            "d2": {"id": "d2", "examples": [{"type": "text", "value": "a fork"}]},
        }
        monkeypatch.setattr(registry, "get_detector", lambda i: entries.get(i))
        for det_id, want in (("d1", "a chair"), ("d2", "a fork"), ("d3", ""), ("", "")):
            ctx = _ctx()
            ctx.detector_id = det_id
            assert resolve_active_text_query(ctx) == want

    @pytest.mark.usefixtures("applies")
    def test_a_cached_head_is_current_only_if_fitted_from_the_detectors_query(self, world):  # noqa: F811
        snap, _proto, make, _embedded = world
        ls = _past_the_hold(make)
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap)  # the raw labels, no query
        assert not cached_head_is_current(ctx, ls, QUERY), "a head fitted without the detector's query"
        assert cached_head_is_current(ctx, ls, ""), "a detector with no query takes it"
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap, text_query=QUERY)
        assert cached_head_is_current(ctx, ls, QUERY)
        assert not cached_head_is_current(ctx, ls, "another query")
        with query_prior_scale(0.0):
            ctx2 = _ctx()
            assert lt.train_from_labelset(ctx2, ls, media_type="image", snap=snap, text_query=QUERY)
            assert cached_head_is_current(ctx2, ls, QUERY), "with the prior off there is nothing to match"
