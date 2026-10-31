"""The 1-D similarity order a list of a dataset's items sorts look-alikes together by.

``vtscore.projection.similarity_order``: the fit (UMAP-1, PCA-1 below the
neighbour-graph floor, as-given below three points), the merge of an order
with the live ids, the container entry, and the per-dataset lifecycle (memory,
then container, then a fit that is cached and persisted).  A Face dataset's is
fit on its own, for the example picker (#5465); every dataset's comes with its
full Browse layout, and a list asking for one starts the fit that makes it
(#5464).
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import numpy as np
import pytest

from vtscore.concurrency.async_jobs import projection_jobs
from vtscore.datasets.container import append_similarity_order, read_similarity_order
from vtscore.projection import Projection, build_pyramid
from vtscore.projection import service as svc
from vtscore.projection import similarity_order as so
from vtscore.projection.similarity_order import SimilarityOrder, fit_similarity_order, ordered_ids
from vtscore.state.core import DatasetContext, register_context

# One test below runs a real umap-learn fit, whose first call per process
# numba-JIT-compiles UMAP's kernels; share the worker that already pays that
# (see the note in test_umap_projection.py).
pytestmark = pytest.mark.xdist_group("numba-umap-jit")


def _clusters(seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """Two tight, far-apart clusters, interleaved in id order."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.05, (15, 8)) + 5.0
    b = rng.normal(0.0, 0.05, (15, 8)) - 5.0
    mat = np.empty((30, 8), dtype=np.float32)
    mat[0::2], mat[1::2] = a, b
    return mat, list(range(100, 130))


class TestFit:
    def test_umap_puts_each_cluster_in_one_run(self):
        mat, ids = _clusters()
        # Near the shipped n_neighbors (15): far fewer splits a 15-point cluster
        # into neighbour-graph pieces, which UMAP places independently.
        order = fit_similarity_order(mat, ids, n_neighbors=14, min_dist=0.1, random_state=7)
        assert order.method == "umap"
        assert sorted(order.ids) == ids
        # Even ids are cluster a, odd ids cluster b: the order splits them
        # into two contiguous runs instead of the alternation they came in.
        clusters = [i % 2 for i in order.ids]
        assert clusters == sorted(clusters) or clusters == sorted(clusters, reverse=True)

    def test_few_points_order_by_pca(self):
        mat = np.array([[3.0, 0.0], [0.0, 0.0], [9.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        order = fit_similarity_order(mat, [1, 2, 3, 4], n_neighbors=15, min_dist=0.1)
        assert order.method == "pca"
        # PCA-1 runs along the one axis the points vary on (either direction).
        assert order.ids in ([2, 4, 1, 3], [3, 1, 4, 2])

    def test_two_points_keep_their_order(self):
        order = fit_similarity_order(np.ones((2, 4), dtype=np.float32), [8, 3], n_neighbors=15, min_dist=0.1)
        assert order == SimilarityOrder([8, 3], "trivial")

    def test_accepts_a_read_only_matrix(self):
        mat = np.arange(12, dtype=np.float32).reshape(4, 3)
        mat.setflags(write=False)
        assert sorted(fit_similarity_order(mat, [1, 2, 3, 4], n_neighbors=3, min_dist=0.1).ids) == [1, 2, 3, 4]

    def test_ids_must_label_every_row(self):
        with pytest.raises(ValueError, match="ids length"):
            fit_similarity_order(np.ones((3, 4), dtype=np.float32), [1, 2], n_neighbors=3, min_dist=0.1)


class TestOrderedIds:
    def test_places_ids_in_order_and_appends_the_rest(self):
        order = SimilarityOrder([5, 1, 9, 3], "umap")
        # 9 was removed since the fit; 7 is new and goes last, in given order.
        assert ordered_ids(order, [1, 3, 5, 7]) == [5, 1, 3, 7]

    def test_no_order_keeps_the_given_order(self):
        assert ordered_ids(None, [4, 2, 6]) == [4, 2, 6]

    def test_covers(self):
        order = SimilarityOrder([1, 2, 3], "umap")
        assert order.covers([1, 3]) and not order.covers([1, 4])


class TestContainerEntry:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "ds.pkl"
        append_similarity_order(path, [3, 1, 2], "umap")
        assert read_similarity_order(path) == ([3, 1, 2], "umap")
        # Replacing it leaves one entry with the new order.
        append_similarity_order(path, [2, 3, 1], "pca")
        assert read_similarity_order(path) == ([2, 3, 1], "pca")

    def test_absent_reads_as_none(self, tmp_path):
        assert read_similarity_order(tmp_path / "missing.pkl") is None


def _ctx(n: int = 6, media_type: str = "face") -> DatasetContext:
    ctx = DatasetContext("faces")
    rng = np.random.default_rng(0)
    for cid in range(1, n + 1):
        ctx.medias[cid] = {
            "id": cid,
            "media_type": media_type,
            "embedder": "face",
            "embeddings": {"face": rng.standard_normal(8).astype(np.float32)},
        }
    return ctx


@pytest.fixture
def container(tmp_path):
    """Point the order's persistence at a temp container."""
    path = str(tmp_path / "faces.pkl")
    with patch("vtscore.projection.store.pkl_path_for", return_value=path):
        yield path


class TestLifecycle:
    def test_a_face_dataset_is_fit_cached_and_persisted(self, container):
        ctx = _ctx()
        fits: list[int] = []
        real = so.fit_similarity_order

        def counting_fit(matrix, ids, **kwargs):
            fits.append(len(ids))
            return real(matrix, ids, **kwargs)

        with patch.object(so, "fit_similarity_order", side_effect=counting_fit):
            order = so.similarity_order_for(ctx)
            assert order is not None and sorted(order.ids) == [1, 2, 3, 4, 5, 6]
            assert ctx._similarity_order is order
            assert read_similarity_order(container) == (order.ids, order.method)
            # A second ask is served from memory, a fresh context from the container.
            assert so.similarity_order_for(ctx) is order
            fresh = _ctx()
            assert so.similarity_order_for(fresh) == order
        assert fits == [6]

    def test_other_media_types_are_not_fit_on_their_own(self, container):
        ctx = _ctx(media_type="image")
        assert so.similarity_order_for(ctx) is None
        assert read_similarity_order(container) is None

    def test_other_media_types_serve_a_stored_order(self, container):
        """One their Browse layout brought with it (#5464)."""
        append_similarity_order(container, [6, 5, 4, 3, 2, 1], "umap")
        with patch.object(so, "fit_similarity_order") as fit:
            order = so.similarity_order_for(_ctx(media_type="image"))
        fit.assert_not_called()
        assert order == SimilarityOrder([6, 5, 4, 3, 2, 1], "umap")

    def test_fit_false_reads_but_never_fits(self, container):
        ctx = _ctx()
        assert so.similarity_order_for(ctx, fit=False) is None
        append_similarity_order(container, [6, 5, 4, 3, 2, 1], "umap")
        assert so.similarity_order_for(ctx, fit=False) == SimilarityOrder([6, 5, 4, 3, 2, 1], "umap")

    def test_a_removed_item_keeps_the_stored_order(self, container):
        append_similarity_order(container, [6, 5, 4, 3, 2, 1], "umap")
        ctx = _ctx()
        del ctx.medias[4]
        with patch.object(so, "fit_similarity_order") as fit:
            order = so.similarity_order_for(ctx)
        fit.assert_not_called()
        assert ordered_ids(order, ctx.medias.keys()) == [6, 5, 3, 2, 1]

    def test_an_order_missing_an_item_is_refit(self, container):
        append_similarity_order(container, [3, 2, 1], "umap")
        ctx = _ctx(n=4)
        order = so.similarity_order_for(ctx)
        assert order is not None and sorted(order.ids) == [1, 2, 3, 4]
        stored = read_similarity_order(container)
        assert stored is not None and sorted(stored[0]) == [1, 2, 3, 4]

    def test_concurrent_asks_fit_once(self, container):
        ctx = _ctx()
        started = threading.Event()
        release = threading.Event()
        fits: list[int] = []

        def slow_fit(matrix, ids, **kwargs):
            fits.append(len(ids))
            started.set()
            release.wait(timeout=10)
            return SimilarityOrder(list(ids), "umap")

        results: list[SimilarityOrder | None] = []
        with patch.object(so, "fit_similarity_order", side_effect=slow_fit):
            first = threading.Thread(target=lambda: results.append(so.similarity_order_for(ctx)))
            first.start()
            assert started.wait(timeout=10)
            second = threading.Thread(target=lambda: results.append(so.similarity_order_for(ctx)))
            second.start()
            release.set()
            first.join(timeout=10)
            second.join(timeout=10)
        assert fits == [6]
        assert len(results) == 2 and results[0] == results[1]

    def test_a_dataset_without_vectors_has_no_order(self, container):
        ctx = _ctx()
        ctx.medias[2]["embeddings"] = {}
        assert so.similarity_order_for(ctx) is None


class _Tracker:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def check_cancelled(self) -> None:
        pass

    def update(self, status: str, message: str = "", **kwargs) -> None:
        self.messages.append(message)


class TestLoadStage:
    """The load pipeline's stage fits a Face dataset's order as it is built."""

    def test_a_face_build_fits_and_persists_the_order(self, container):
        from vtscore.datasets.stages.projection import _build_similarity_order_stage

        tracker = _Tracker()
        _build_similarity_order_stage(_ctx(), tracker)
        stored = read_similarity_order(container)
        assert stored is not None and sorted(stored[0]) == [1, 2, 3, 4, 5, 6]
        assert tracker.messages[0] == "Ordering 6 items by similarity…"
        assert tracker.messages[-1] == "Similarity order ready"

    def test_other_builds_skip_it(self, container):
        from vtscore.datasets.stages.projection import _build_similarity_order_stage

        tracker = _Tracker()
        _build_similarity_order_stage(_ctx(media_type="image"), tracker)
        assert tracker.messages == []
        assert read_similarity_order(container) is None


def _fake_layout_fit(matrix, ids, **kwargs):
    """Cheap stand-in for the 2-D UMAP fit: a seeded random layout."""
    coords = np.random.default_rng(len(ids)).standard_normal((len(ids), 2)).astype(np.float32)
    return Projection("pid", list(ids), coords, "fake")


def _reversed(matrix, ids, **kwargs):
    return SimilarityOrder(list(reversed(ids)), "umap")


def _no_signs():
    """Signposting is best-effort decoration the order does not touch."""
    return patch("vtscore.projection.service.prep_signposts_best_effort")


def _install_layout(ctx: DatasetContext) -> None:
    ids = sorted(ctx.medias)
    proj = Projection("pid", ids, np.zeros((len(ids), 2), dtype=np.float32), "fake")
    svc.install_layout(ctx, proj, build_pyramid(proj, bin_shape=svc.shape_for(ctx), n_levels=2))


def _await_jobs() -> None:
    """Block until the projection runner has nothing running or parked."""
    for _ in range(600):
        active = projection_jobs.active_jobs()
        if not active:
            return
        active[0].done_event.wait(timeout=0.05)
    raise TimeoutError("projection job did not finish")


class TestWithTheLayout:
    """Every full-dataset layout fit brings the order with it (#5464)."""

    def _fit_layout(self, ctx: DatasetContext, *, subset: bool = False) -> None:
        found = so._matrix_for(ctx)
        assert found is not None
        ids, matrix = found
        with patch("vtscore.projection.fit_projection", side_effect=_fake_layout_fit), _no_signs():
            svc.fit_and_install_layout(ctx, ids, matrix, svc.shape_for(ctx), subset=subset)

    def test_a_full_layout_fit_fits_and_persists_the_order(self, container):
        ctx = _ctx(media_type="image")
        with patch.object(so, "fit_similarity_order", side_effect=_reversed):
            self._fit_layout(ctx)
        assert ctx._projection is not None
        assert ctx._similarity_order == SimilarityOrder([6, 5, 4, 3, 2, 1], "umap")
        assert read_similarity_order(container) == ([6, 5, 4, 3, 2, 1], "umap")

    def test_a_subset_fit_leaves_the_order_alone(self, container):
        ctx = _ctx(media_type="image")
        with patch.object(so, "fit_similarity_order") as fit:
            self._fit_layout(ctx, subset=True)
        fit.assert_not_called()
        assert ctx._similarity_order is None

    def test_a_stored_order_is_kept_rather_than_refit(self, container):
        append_similarity_order(container, [2, 4, 6, 1, 3, 5], "umap")
        ctx = _ctx(media_type="image")
        with patch.object(so, "fit_similarity_order") as fit:
            self._fit_layout(ctx)
        fit.assert_not_called()
        assert ctx._similarity_order == SimilarityOrder([2, 4, 6, 1, 3, 5], "umap")

    def test_a_failed_order_fit_leaves_the_layout_standing(self, container):
        ctx = _ctx(media_type="image")
        with patch.object(so, "fit_similarity_order", side_effect=RuntimeError("umap exploded")):
            self._fit_layout(ctx)
        assert ctx._pyramids
        assert ctx._similarity_order is None


class TestRequest:
    """What a list sorting by similarity is told (#5464)."""

    def _loaded(self, media_type: str = "image", n: int = 6) -> DatasetContext:
        # Registered: a projection job whose dataset is not loaded fails unrun.
        ctx = _ctx(n=n, media_type=media_type)
        register_context(ctx)
        return ctx

    def test_a_stored_order_is_ready(self, container):
        append_similarity_order(container, [6, 5, 4, 3, 2, 1], "umap")
        answer = so.request_similarity_order(self._loaded())
        assert answer == {"status": "ready", "order": SimilarityOrder([6, 5, 4, 3, 2, 1], "umap")}
        assert projection_jobs.active_jobs() == []

    def test_a_face_dataset_is_fit_on_the_spot(self, container):
        with patch.object(so, "fit_similarity_order", side_effect=_reversed):
            answer = so.request_similarity_order(self._loaded("face"))
        assert answer["status"] == "ready"
        assert answer["order"].ids == [6, 5, 4, 3, 2, 1]

    def test_a_dataset_without_a_layout_builds_one_with_its_order(self, container):
        ctx = self._loaded()
        with (
            patch("vtscore.projection.fit_projection", side_effect=_fake_layout_fit) as layout_fit,
            patch.object(so, "fit_similarity_order", side_effect=_reversed),
            _no_signs(),
        ):
            first = so.request_similarity_order(ctx)
            assert first["status"] == "building"
            assert first["job_id"] == ctx._full_job_id == ctx._order_job_id
            _await_jobs()
            ready = so.request_similarity_order(ctx)
        assert layout_fit.call_count == 1
        assert ctx._projection is not None
        assert ready["status"] == "ready" and ready["order"].ids == [6, 5, 4, 3, 2, 1]

    def test_a_layout_without_an_order_fits_the_order_alone(self, container):
        ctx = self._loaded()
        _install_layout(ctx)
        with (
            patch("vtscore.projection.fit_projection") as layout_fit,
            patch.object(so, "fit_similarity_order", side_effect=_reversed),
        ):
            first = so.request_similarity_order(ctx)
            assert first["status"] == "building"
            job = projection_jobs.get(first["job_id"])
            assert job is not None and job.signature[1] == "similarity_order"
            _await_jobs()
            ready = so.request_similarity_order(ctx)
        layout_fit.assert_not_called()
        assert ready["status"] == "ready" and ready["order"].ids == [6, 5, 4, 3, 2, 1]

    def test_an_ask_while_the_fit_runs_reports_it_rather_than_starting_another(self, container):
        ctx = self._loaded()
        _install_layout(ctx)
        started = threading.Event()
        release = threading.Event()

        def slow_fit(matrix, ids, **kwargs):
            started.set()
            release.wait(timeout=10)
            return _reversed(matrix, ids)

        with patch.object(so, "fit_similarity_order", side_effect=slow_fit):
            first = so.request_similarity_order(ctx)
            assert started.wait(timeout=10)
            again = so.request_similarity_order(ctx)
            release.set()
            _await_jobs()
        assert again["status"] == "building" and again["job_id"] == first["job_id"]

    def test_a_failed_fit_is_reported_once_and_the_next_ask_starts_over(self, container):
        ctx = self._loaded()
        _install_layout(ctx)
        with patch.object(so, "fit_similarity_order", side_effect=RuntimeError("umap exploded")):
            first = so.request_similarity_order(ctx)
            _await_jobs()
            failed = so.request_similarity_order(ctx)
            retried = so.request_similarity_order(ctx)
            _await_jobs()
        assert failed == {"status": "error", "error": "umap exploded"}
        assert retried["status"] == "building" and retried["job_id"] != first["job_id"]

    def test_a_dataset_without_vectors_has_nothing_to_order(self, container):
        ctx = self._loaded()
        ctx.medias[2]["embeddings"] = {}
        with pytest.raises(svc.NothingToProject):
            so.request_similarity_order(ctx)
