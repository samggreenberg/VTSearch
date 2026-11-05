"""A Face dataset's faces merged into people (#5586).

``vtscore.datasets.people``: the cut (the face ball's beta 1 floor, in the one
space it is calibrated for), the fit (a neighbour graph, centroid linkage on
the unit sphere within each component, hard views attached last), the
partition's container entry, and the per-dataset lifecycle (memory, then
container, then a fit that is cached and persisted), the load stage and the
``request_people`` contract a view polls.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import numpy as np
import pytest

from vtscore.concurrency.async_jobs import projection_jobs
from vtscore.datasets import people as pp
from vtscore.datasets.container import append_people, read_people
from vtscore.datasets.people import NothingToPartition, PeoplePartition, fit_people
from vtscore.detectors.centroid_head import FACE_BALL_COMPAT, FACE_BALL_FLOORS
from vtscore.media.face.imagekit_bridge import COMPAT_METADATA_KEY, HARD_VIEW_QUALITY, QUALITY_METADATA_KEY
from vtscore.state.core import DatasetContext, register_context

CUT = FACE_BALL_FLOORS[1.0]
SPACE = FACE_BALL_COMPAT
#: A Face Quality under and over the calibrated space's hard-view cut.
HARD_Q = HARD_VIEW_QUALITY[SPACE] - 5.0
GOOD_Q = HARD_VIEW_QUALITY[SPACE] + 10.0


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _person(rng: np.random.Generator, centre: np.ndarray, n: int, noise: float) -> list[np.ndarray]:
    """*n* unit faces around *centre*: ``centre + noise * (random unit direction)``, normalised."""
    d = centre.size
    return [_unit(centre + noise * _unit(rng.standard_normal(d))) for _ in range(n)]


def _axis(d: int, i: int) -> np.ndarray:
    e = np.zeros(d)
    e[i] = 1.0
    return e


def _fit(rows: list[np.ndarray], ids: list[int] | None = None, **kwargs) -> PeoplePartition:
    ids = list(range(1, len(rows) + 1)) if ids is None else ids
    return fit_people(np.stack(rows).astype(np.float32), ids, cutoff=CUT, compat=SPACE, **kwargs)


def _as_sets(partition: PeoplePartition) -> set[frozenset[int]]:
    return {frozenset(p) for p in partition.people}


class TestCut:
    def test_the_calibrated_space_cuts_at_the_ball_floor(self):
        assert pp.people_cutoff(SPACE) == pytest.approx(FACE_BALL_FLOORS[1.0])

    @pytest.mark.parametrize("compat", ["oss-facenet:1", "", None])
    def test_any_other_space_has_no_cut(self, compat):
        assert pp.people_cutoff(compat) is None

    def test_the_uncalibrated_message_is_the_face_balls_wording(self):
        assert (
            pp.not_calibrated_message("ext-acme:3") == "People is not calibrated for this face embedder (ext-acme:3)."
        )
        assert pp.not_calibrated_message(None) == "People is not calibrated for this face embedder."


class TestFit:
    def test_two_people_with_a_chained_lookalike_stay_two(self):
        """A lookalike joins both people's faces in one neighbour component; the linkage splits it."""
        rng = np.random.default_rng(42)
        d = 64
        a = _person(rng, _axis(d, 0), 6, 0.1)
        b = _person(rng, _axis(d, 1), 6, 0.1)
        lookalike = _unit(_axis(d, 0) + _axis(d, 1))
        rows = a + b + [lookalike]
        # The chain is real: one component holds both people.
        assert len(set(pp._neighbour_components(np.stack(rows), CUT, 20, lambda *_: None))) == 1

        partition = _fit(rows)
        people = _as_sets(partition)
        person_a = next(p for p in people if 1 in p)
        person_b = next(p for p in people if 7 in p)
        assert person_a != person_b
        assert set(range(1, 7)) <= person_a and set(range(7, 13)) <= person_b
        assert 13 in person_a | person_b
        assert partition.unplaced == ()

    def test_spread_people_are_split_where_a_euclidean_centroid_linkage_merges_them(self):
        """Faces of one person ~0.5 apart, as buffalo_s's are: their raw means are far from unit length.

        scipy's centroid linkage measures the Euclidean distance between those
        raw means, and at the floor's Euclidean equivalent it merges two people
        whose centroids sit 0.2 apart.  The cosine between unit centroids does not.
        """
        from scipy.cluster.hierarchy import fcluster, linkage

        rng = np.random.default_rng(42)
        d = 512
        ua = _axis(d, 0)
        ub = 0.2 * _axis(d, 0) + np.sqrt(1 - 0.04) * _axis(d, 1)
        a = _person(rng, ua, 8, 1.0)
        b = _person(rng, ub, 8, 1.0)
        lookalike = _unit(ua + ub)
        rows = a + b + [lookalike]
        mat = np.stack(rows)
        assert len(set(pp._neighbour_components(mat, CUT, 20, lambda *_: None))) == 1
        euclid = fcluster(linkage(mat, method="centroid"), t=np.sqrt(2 - 2 * CUT), criterion="distance")
        assert len(set(euclid[:16])) == 1, "the Euclidean linkage no longer merges them; the guard is moot"

        people = _as_sets(_fit(rows))
        person_a = next(p for p in people if 1 in p)
        person_b = next(p for p in people if 9 in p)
        assert person_a != person_b
        assert set(range(1, 9)) <= person_a and set(range(9, 17)) <= person_b

    @pytest.mark.parametrize("seed", range(6))
    def test_the_linkage_merges_as_a_brute_force_centroid_linkage_does(self, seed):
        """The best-partner bookkeeping changes the cost, never the clusters."""
        rng = np.random.default_rng(seed)
        d = 24
        centres = [_unit(rng.standard_normal(d)) for _ in range(5)]
        rows = [f for c in centres for f in _person(rng, c, int(rng.integers(3, 12)), float(rng.uniform(0.4, 1.2)))]
        unit = np.stack(rows)

        clusters = [[i] for i in range(len(rows))]
        while len(clusters) > 1:
            cents = np.stack([_unit(unit[g].sum(axis=0)) for g in clusters])
            cos = (cents @ cents.T).astype(np.float32)
            np.fill_diagonal(cos, -np.inf)
            i, j = np.unravel_index(int(np.argmax(cos)), cos.shape)
            if cos[i, j] < CUT:
                break
            a, b = min(i, j), max(i, j)
            clusters[a] = clusters[a] + clusters.pop(b)
        expected = {frozenset(g) for g in clusters}

        assert {frozenset(g) for g in pp._centroid_linkage(unit, CUT)} == expected

    def test_hard_views_attach_or_stay_unplaced_and_a_lone_face_is_a_person(self):
        rng = np.random.default_rng(42)
        d = 64
        a = _person(rng, _axis(d, 0), 5, 0.1)
        b = _person(rng, _axis(d, 1), 3, 0.1)
        hard_like_a = _unit(_axis(d, 0) + 0.5 * _unit(rng.standard_normal(d)))
        hard_like_no_one = _axis(d, 5)
        lone = _axis(d, 9)
        rows = a + b + [hard_like_a, hard_like_no_one, lone]
        hard = [False] * 8 + [True, True, False]
        partition = _fit(rows, hard=hard)
        assert [set(p) for p in partition.people] == [{1, 2, 3, 4, 5, 9}, {6, 7, 8}, {11}]
        assert partition.unplaced == (10,)

    def test_hard_views_take_no_part_in_the_graph(self):
        """A hard view between two people would chain them; it is left out, then attached to one."""
        rng = np.random.default_rng(42)
        d = 64
        a = _person(rng, _axis(d, 0), 4, 0.1)
        b = _person(rng, _axis(d, 1), 4, 0.1)
        between = _unit(_axis(d, 0) + _axis(d, 1))
        with patch.object(pp, "_neighbour_components", wraps=pp._neighbour_components) as graph:
            partition = _fit(a + b + [between], hard=[False] * 8 + [True])
        assert graph.call_args.args[0].shape[0] == 8
        assert len(partition.people) == 2
        assert 9 in partition.person_of()

    def test_only_hard_views_leaves_everyone_unplaced(self):
        rng = np.random.default_rng(42)
        partition = _fit(_person(rng, _axis(16, 0), 3, 0.1), hard=[True, True, True])
        assert partition.people == () and partition.unplaced == (1, 2, 3)

    def test_the_representative_has_the_best_quality_ties_to_the_closest(self):
        d = 16
        rows = [
            _unit(_axis(d, 0) + 0.10 * _axis(d, 1)),
            _unit(_axis(d, 0) + 0.40 * _axis(d, 2)),
            _unit(_axis(d, 0) + 0.05 * _axis(d, 3)),
            _unit(_axis(d, 0) + 0.20 * _axis(d, 4)),
        ]
        partition = _fit(rows, quality=[10.0, 30.0, 30.0, None])
        (person,) = partition.people
        # 2 and 3 tie on quality; 3 sits closer to the centroid.
        assert person[0] == 3
        centroid = _unit(np.stack(rows).sum(axis=0))
        closeness = {i + 1: float(rows[i] @ centroid) for i in range(4)}
        rest = list(person[1:])
        assert rest == sorted(rest, key=lambda i: -closeness[i])

    def test_without_qualities_the_closest_face_represents(self):
        d = 16
        rows = [_unit(_axis(d, 0) + s * _axis(d, 1 + k)) for k, s in enumerate((0.3, 0.02, 0.2))]
        (person,) = _fit(rows).people
        assert person[0] == 2

    def test_people_are_listed_largest_first_then_by_their_lowest_id(self):
        rng = np.random.default_rng(42)
        d = 64
        rows = (
            _person(rng, _axis(d, 0), 2, 0.05)
            + _person(rng, _axis(d, 1), 4, 0.05)
            + _person(rng, _axis(d, 2), 2, 0.05)
            + [_axis(d, 3)]
        )
        sizes = [(len(p), min(p)) for p in _fit(rows).people]
        assert sizes == [(4, 3), (2, 1), (2, 7), (1, 9)]

    def test_the_same_faces_give_the_same_people(self):
        rng = np.random.default_rng(42)
        d = 64
        rows = (
            _person(rng, _axis(d, 0), 6, 0.3)
            + _person(rng, _axis(d, 1), 6, 0.3)
            + [_unit(_axis(d, 0) + _axis(d, 1)), _axis(d, 7)]
        )
        ids = [10 * (i + 1) for i in range(len(rows))]
        first = _fit(rows, ids)
        assert _fit(rows, ids) == first
        perm = rng.permutation(len(rows))
        shuffled = _fit([rows[i] for i in perm], [ids[i] for i in perm])
        assert _as_sets(shuffled) == _as_sets(first) and shuffled.unplaced == first.unplaced

    def test_stamps(self):
        partition = _fit([_axis(4, 0)])
        assert (partition.method, partition.linkage, partition.cutoff, partition.compat) == (
            pp.PEOPLE_METHOD,
            pp.PEOPLE_LINKAGE,
            CUT,
            SPACE,
        )
        assert partition.people == ((1,),)

    def test_ids_must_label_every_row(self):
        with pytest.raises(ValueError, match="ids length"):
            fit_people(np.ones((3, 4), dtype=np.float32), [1, 2], cutoff=CUT, compat=SPACE)


def _partition(people=((3, 1), (2,)), unplaced=(4,), cutoff=CUT, compat=SPACE) -> PeoplePartition:
    return PeoplePartition(people, unplaced, pp.PEOPLE_METHOD, pp.PEOPLE_LINKAGE, cutoff, compat)


def _stored(path) -> PeoplePartition:
    """The partition in the container at *path*, which must hold one."""
    payload = read_people(path)
    assert payload is not None, "no people stored"
    return PeoplePartition.from_payload(payload)


class TestPartition:
    def test_covers_placed_and_unplaced_faces(self):
        part = _partition()
        assert part.covers([1, 2, 3, 4]) and part.covers([2]) and not part.covers([1, 5])

    def test_cut_as(self):
        part = _partition()
        assert part.cut_as(CUT, SPACE)
        assert not part.cut_as(CUT + 0.01, SPACE)
        assert not part.cut_as(CUT, "oss-arcface-r50:1")
        other = PeoplePartition(part.people, part.unplaced, pp.PEOPLE_METHOD, "average", CUT, SPACE)
        assert not other.cut_as(CUT, SPACE)

    def test_restricted_to_drops_gone_faces_and_emptied_people(self):
        part = _partition(people=((3, 1), (2,), (5, 6)), unplaced=(4, 7))
        assert part.restricted_to([1, 5, 6, 7]) == _partition(people=((1,), (5, 6)), unplaced=(7,))

    def test_person_of(self):
        assert _partition().person_of() == {3: 0, 1: 0, 2: 1}

    def test_payload_round_trips(self):
        part = _partition()
        assert PeoplePartition.from_payload(part.to_payload()) == part


class TestContainerEntry:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "ds.pkl"
        append_people(path, _partition().to_payload())
        assert _stored(path) == _partition()
        # Replacing it leaves one entry with the new partition.
        append_people(path, _partition(people=((1, 2, 3),), unplaced=()).to_payload())
        assert _stored(path) == _partition(people=((1, 2, 3),), unplaced=())

    def test_absent_reads_as_none(self, tmp_path):
        assert read_people(tmp_path / "missing.pkl") is None


def _face_ctx(
    *,
    compat: str | None = SPACE,
    media_type: str = "face",
    qualities: list[float] | None = None,
    dataset_id: str = "faces",
) -> DatasetContext:
    """Two people of three faces and a lone face: ids 1-3, 4-6 and 7."""
    rng = np.random.default_rng(42)
    d = 32
    rows = _person(rng, _axis(d, 0), 3, 0.1) + _person(rng, _axis(d, 1), 3, 0.1) + [_axis(d, 2)]
    ctx = DatasetContext(dataset_id)
    for cid, vec in enumerate(rows, start=1):
        meta: dict = {QUALITY_METADATA_KEY: GOOD_Q if qualities is None else qualities[cid - 1]}
        if compat is not None:
            meta[COMPAT_METADATA_KEY] = compat
        ctx.medias[cid] = {
            "id": cid,
            "media_type": media_type,
            "embedder": "face",
            "embeddings": {"face": vec.astype(np.float32)},
            "custom_metadata": meta,
        }
    return ctx


@pytest.fixture
def container(tmp_path):
    """Point the partition's persistence at a temp container."""
    path = str(tmp_path / "faces.pkl")
    with patch("vtscore.projection.store.pkl_path_for", return_value=path):
        yield path


def _counting_fit(counts: list[int]):
    real = pp.fit_people

    def fit(matrix, ids, **kwargs):
        counts.append(len(ids))
        return real(matrix, ids, **kwargs)

    return fit


class TestCompat:
    def test_the_faces_shared_tag(self):
        assert pp.dataset_compat(_face_ctx()) == SPACE

    def test_no_tag_or_mixed_tags_are_no_space(self):
        assert pp.dataset_compat(_face_ctx(compat=None)) is None
        ctx = _face_ctx()
        ctx.medias[2]["custom_metadata"][COMPAT_METADATA_KEY] = "oss-arcface-r50:1"
        assert pp.dataset_compat(ctx) is None


class TestLifecycle:
    def test_a_face_dataset_is_fit_cached_and_persisted(self, container):
        ctx = _face_ctx()
        fits: list[int] = []
        with patch.object(pp, "fit_people", side_effect=_counting_fit(fits)):
            partition = pp.people_for(ctx)
            assert partition is not None
            assert _as_sets(partition) == {frozenset({1, 2, 3}), frozenset({4, 5, 6}), frozenset({7})}
            assert ctx._people is partition
            assert _stored(container) == partition
            # A second ask is served from memory, a fresh context from the container.
            assert pp.people_for(ctx) is partition
            assert pp.people_for(_face_ctx()) == partition
        assert fits == [7]

    def test_hard_views_come_from_the_faces_own_quality(self, container):
        ctx = _face_ctx(qualities=[GOOD_Q, GOOD_Q, HARD_Q, GOOD_Q, GOOD_Q, GOOD_Q, HARD_Q])
        partition = pp.people_for(ctx)
        assert partition is not None
        # Face 3 attaches to its person; the lone face 7, a hard view, attaches to no one.
        assert _as_sets(partition) == {frozenset({1, 2, 3}), frozenset({4, 5, 6})}
        assert partition.unplaced == (7,)
        # The hard view is never the representative.
        assert partition.person_of()[3] == partition.person_of()[1]
        assert partition.people[partition.person_of()[1]][0] != 3

    def test_fit_false_reads_but_never_fits(self, container):
        ctx = _face_ctx()
        assert pp.people_for(ctx, fit=False) is None
        append_people(container, _partition(people=((1, 2, 3, 4, 5, 6, 7),), unplaced=()).to_payload())
        assert pp.people_for(ctx, fit=False) == _partition(people=((1, 2, 3, 4, 5, 6, 7),), unplaced=())

    def test_a_removed_item_keeps_the_stored_partition(self, container):
        stored = _partition(people=((1, 2, 3), (4, 5, 6), (7,)), unplaced=())
        append_people(container, stored.to_payload())
        ctx = _face_ctx()
        del ctx.medias[5]
        with patch.object(pp, "fit_people") as fit:
            partition = pp.people_for(ctx)
        fit.assert_not_called()
        assert partition is not None and partition == stored
        assert partition.restricted_to(ctx.medias.keys()).people == ((1, 2, 3), (4, 6), (7,))

    def test_a_partition_missing_a_face_is_refit(self, container):
        append_people(container, _partition(people=((1, 2, 3), (4, 5, 6)), unplaced=()).to_payload())
        partition = pp.people_for(_face_ctx())
        assert partition is not None and partition.covers(range(1, 8))
        stored = read_people(container)
        assert stored is not None and PeoplePartition.from_payload(stored) == partition

    def test_a_partition_cut_another_way_is_refit(self, container):
        ctx = _face_ctx()
        first = pp.people_for(ctx)
        assert first is not None
        fits: list[int] = []
        with (
            patch.object(pp, "people_cutoff", return_value=CUT + 0.05),
            patch.object(pp, "fit_people", side_effect=_counting_fit(fits)),
        ):
            moved = pp.people_for(ctx)
            # A fresh context reads the container, which now holds the new cut.
            assert pp.people_for(_face_ctx()) == moved
        assert fits == [7]
        assert moved is not None and moved.cutoff == pytest.approx(CUT + 0.05)

    def test_a_stored_partition_of_another_space_is_refit(self, container):
        append_people(container, _partition(people=((1, 2, 3, 4, 5, 6, 7),), unplaced=(), compat="x:1").to_payload())
        partition = pp.people_for(_face_ctx())
        assert partition is not None and partition.compat == SPACE and len(partition.people) == 3

    def test_an_uncalibrated_space_gets_no_partition(self, container):
        with patch.object(pp, "fit_people") as fit:
            assert pp.people_for(_face_ctx(compat="ext-acme:3")) is None
        fit.assert_not_called()
        assert read_people(container) is None

    def test_other_media_types_have_no_people(self, container):
        assert pp.people_for(_face_ctx(media_type="image")) is None
        assert read_people(container) is None

    def test_concurrent_asks_fit_once(self, container):
        ctx = _face_ctx()
        started = threading.Event()
        release = threading.Event()
        fits: list[int] = []
        real = pp.fit_people

        def slow_fit(matrix, ids, **kwargs):
            fits.append(len(ids))
            started.set()
            release.wait(timeout=10)
            return real(matrix, ids, **kwargs)

        results: list[PeoplePartition | None] = []
        with patch.object(pp, "fit_people", side_effect=slow_fit):
            first = threading.Thread(target=lambda: results.append(pp.people_for(ctx)))
            first.start()
            assert started.wait(timeout=10)
            second = threading.Thread(target=lambda: results.append(pp.people_for(ctx)))
            second.start()
            release.set()
            first.join(timeout=10)
            second.join(timeout=10)
        assert fits == [7]
        assert len(results) == 2 and results[0] == results[1]


class _Tracker:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def check_cancelled(self) -> None:
        pass

    def update(self, status: str, message: str = "", **kwargs) -> None:
        self.messages.append(message)


class TestLoadStage:
    """The load pipeline's stage fits a Face dataset's people as it is built."""

    def test_a_face_build_fits_and_persists_the_people(self, container):
        from vtscore.datasets.stages.projection import _build_people_stage

        tracker = _Tracker()
        _build_people_stage(_face_ctx(), tracker)
        stored = read_people(container)
        assert stored is not None and PeoplePartition.from_payload(stored).covers(range(1, 8))
        assert tracker.messages[0] == "Finding people among 7 faces…"
        assert tracker.messages[-1] == "People ready"

    @pytest.mark.parametrize("kwargs", [{"media_type": "image"}, {"compat": "ext-acme:3"}])
    def test_other_builds_skip_it(self, container, kwargs):
        from vtscore.datasets.stages.projection import _build_people_stage

        tracker = _Tracker()
        _build_people_stage(_face_ctx(**kwargs), tracker)
        assert tracker.messages == []
        assert read_people(container) is None


def _await_jobs() -> None:
    """Block until the projection runner has nothing running or parked."""
    for _ in range(600):
        active = projection_jobs.active_jobs()
        if not active:
            return
        active[0].done_event.wait(timeout=0.05)
    raise TimeoutError("projection job did not finish")


class TestRequest:
    """What a view of a Face dataset's people is told."""

    def _loaded(self, **kwargs) -> DatasetContext:
        # Registered: a projection job whose dataset is not loaded fails unrun.
        ctx = _face_ctx(**kwargs)
        register_context(ctx)
        return ctx

    def test_a_stored_partition_is_ready(self, container):
        stored = _partition(people=((1, 2, 3), (4, 5, 6), (7,)), unplaced=())
        append_people(container, stored.to_payload())
        assert pp.request_people(self._loaded()) == {"status": "ready", "people": stored}
        assert projection_jobs.active_jobs() == []

    def test_the_first_ask_builds_and_a_later_one_is_ready(self, container):
        ctx = self._loaded()
        fits: list[int] = []
        with patch.object(pp, "fit_people", side_effect=_counting_fit(fits)):
            first = pp.request_people(ctx)
            assert first["status"] == "building"
            assert first["job_id"] == ctx._people_job_id
            job = projection_jobs.get(first["job_id"])
            assert job is not None and job.signature[1] == "people"
            _await_jobs()
            ready = pp.request_people(ctx)
            again = pp.request_people(ctx)
        assert fits == [7]
        assert ready["status"] == "ready" and ready["people"].covers(range(1, 8))
        assert again == ready and ctx._people_job_id is None

    def test_asks_while_the_fit_runs_report_it_rather_than_starting_another(self, container):
        ctx = self._loaded()
        started = threading.Event()
        release = threading.Event()
        fits: list[int] = []
        real = pp.fit_people

        def slow_fit(matrix, ids, **kwargs):
            fits.append(len(ids))
            started.set()
            release.wait(timeout=10)
            return real(matrix, ids, **kwargs)

        answers: list[dict] = []
        with patch.object(pp, "fit_people", side_effect=slow_fit):
            first = pp.request_people(ctx)
            assert started.wait(timeout=10)
            asks = [threading.Thread(target=lambda: answers.append(pp.request_people(ctx))) for _ in range(3)]
            for t in asks:
                t.start()
            for t in asks:
                t.join(timeout=10)
            forced = pp.request_people(ctx, force=True)
            release.set()
            _await_jobs()
        assert fits == [7]
        assert [a["job_id"] for a in answers] == [first["job_id"]] * 3
        assert forced["status"] == "building" and forced["job_id"] == first["job_id"]

    def test_force_fits_a_stored_partition_again(self, container):
        stale = _partition(people=((1, 2, 3, 4, 5, 6, 7),), unplaced=())
        append_people(container, stale.to_payload())
        ctx = self._loaded()
        building = pp.request_people(ctx, force=True)
        assert building["status"] == "building"
        _await_jobs()
        ready = pp.request_people(ctx)
        assert ready["status"] == "ready" and len(ready["people"].people) == 3
        assert _stored(container) == ready["people"]

    def test_a_failed_fit_is_reported_once_and_the_next_ask_starts_over(self, container):
        ctx = self._loaded()
        with patch.object(pp, "fit_people", side_effect=RuntimeError("clustering exploded")):
            first = pp.request_people(ctx)
            _await_jobs()
            failed = pp.request_people(ctx)
            retried = pp.request_people(ctx)
            _await_jobs()
        assert failed == {"status": "error", "error": "clustering exploded"}
        assert retried["status"] == "building" and retried["job_id"] != first["job_id"]

    def test_an_uncalibrated_space_is_told_so(self, container):
        answer = pp.request_people(self._loaded(compat="ext-acme:3"))
        assert answer == {
            "status": "not_calibrated",
            "compat": "ext-acme:3",
            "message": "People is not calibrated for this face embedder (ext-acme:3).",
        }
        assert projection_jobs.active_jobs() == []

    def test_a_dataset_of_another_type_has_no_people(self, container):
        with pytest.raises(NothingToPartition, match="Face datasets only"):
            pp.request_people(self._loaded(media_type="image"))

    def test_a_dataset_without_face_vectors_has_no_people(self, container):
        ctx = self._loaded()
        ctx.medias[2]["embeddings"] = {}
        with pytest.raises(NothingToPartition):
            pp.request_people(ctx)
