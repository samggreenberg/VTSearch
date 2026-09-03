"""The active-dataset cold train builds the app's detector, and is scored at its geometry.

``resolve_or_train_detector`` is the on-demand train that runs when a detector
has no live model and the dataset **is** loaded - the head behind
``/api/find-label``, ``/api/auto-detect`` and the portable export.  It used to
hand-roll its own training (image-level vectors, ``region_box`` discarded, one
negative per Bad label, md5-only in-dataset matching), so the same labelset
meant a different detector depending on which entry point trained it.  Worse,
its Find caller scored that whole-image head through the max-pooled
``scoring_rows_for_snap`` geometry - a head that had never been shown a patch as
a negative had its media score set by whichever distractor patch fired hardest
(issue #3544, the sibling of #3525 one path over).

It now delegates to ``train_from_labelset``, the app's own labelset training,
and every one of its callers scores the resulting MaxPatch head at the pooled
geometry.  The tests below pin both halves plus the diagnostic - the "why did
this detector produce nothing?" answer the routes show the user, which the
labelset path does not itself produce.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from vtsearch.settings import get_detectors_dir

DIM = 4
EMB = "dinov3_patch"


def _basis(i: int) -> np.ndarray:
    return np.eye(DIM, dtype=np.float32)[i]


@pytest.fixture(autouse=True)
def clean_detectors_dir():
    import shutil

    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)
    yield
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)


def _grid(*rows: np.ndarray) -> np.ndarray:
    """A ``(1, len(rows), DIM)`` patch grid: one row of cells."""
    grid = np.zeros((1, len(rows), DIM), dtype=np.float32)
    for i, vec in enumerate(rows):
        grid[0, i] = vec
    return grid


def _media(cid: int, vec: np.ndarray, grid: np.ndarray | None = None) -> dict:
    media = {
        "id": cid,
        "media_type": "image",
        "embedder": EMB,
        "embeddings": {EMB: vec},
        "filename": f"m{cid}.png",
        "md5": f"m{cid}",
        "origin_name": f"m{cid}.png",
        "origin": {"importer": "test", "params": {}},
    }
    if grid is not None:
        media["patch_grid"] = grid
    return media


def _patch_corpus() -> dict[int, dict]:
    """Six patch media: the odd ones carry the signal in a *patch*, not the image row.

    Every image-level vector is ``basis(0)``; ids 1-3 hide ``basis(1)`` in their
    second grid cell.  A head that only ever saw image-level vectors cannot
    separate the two halves at all, so anything that does is reading the grid.
    """
    corpus: dict[int, dict] = {}
    for cid in (1, 2, 3):
        corpus[cid] = _media(cid, _basis(0), grid=_grid(_basis(0), _basis(1)))
    for cid in (4, 5, 6):
        corpus[cid] = _media(cid, _basis(0), grid=_grid(_basis(0), _basis(2)))
    return corpus


def _activate(corpus: dict[int, dict]):
    from vtscore.state.core import DatasetContext, register_context, set_thread_dataset_context

    ctx = DatasetContext("ds-cold-head")
    ctx.medias.update(corpus)
    register_context(ctx)
    set_thread_dataset_context(ctx)
    return ctx


def _write_detector(name: str, corpus: dict[int, dict], *, region_boxes: dict[int, tuple] | None = None) -> str:
    """A patch detector on disk whose Good labels carry a region box."""
    from vtscore.detectors.registry import register_detector
    from vtscore.detectors.store import _detector_path
    from vtscore.detectors.store import _write_detector as write

    boxes = region_boxes or {}
    labels = []
    for cid, media in corpus.items():
        entry = {
            "md5": media["md5"],
            "label": "good" if cid in (1, 2, 3) else "bad",
            "origin": media["origin"],
            "origin_name": media["origin_name"],
            "filename": media["filename"],
        }
        if cid in boxes:
            entry["region_box"] = list(boxes[cid])
        labels.append(entry)
    data = {
        "name": name,
        "media_type": "image",
        "embedder_type": "patch_semantic",
        "labelset": {"labels": labels},
    }
    write(_detector_path(name), data)
    return register_detector(name=name, media_type="image", num_training=len(labels))["id"]


def _read(name: str) -> dict:
    from vtscore.detectors.store import _detector_path, _read_detector

    data = _read_detector(_detector_path(name))
    assert data is not None
    return data


class TestColdTrainIsTheAppsLabelsetTraining:
    """One labelset means one detector, whichever entry point built the head."""

    def test_it_delegates_to_train_from_labelset(self, monkeypatch):
        """Structural guard: no second training implementation lives here.

        The divergence #3544 names is invisible in the numbers - a hand-rolled
        build returns perfectly plausible scores - so what has to be pinned is
        that the app's own entry point is the one doing the work.
        """
        import vtscore.detectors.labelset_training as lt_mod
        from vtscore.detectors.model_loading import resolve_or_train_detector

        corpus = _patch_corpus()
        _activate(corpus)
        detector_id = _write_detector("cold-delegates", corpus)

        calls: list[tuple] = []
        real = lt_mod.train_from_labelset

        def _spy(det_ctx, labelset, **kwargs):
            calls.append((det_ctx, labelset, kwargs))
            return real(det_ctx, labelset, **kwargs)

        monkeypatch.setattr(lt_mod, "train_from_labelset", _spy)

        mlp, _threshold, diag = resolve_or_train_detector(detector_id, _read("cold-delegates"), "image", corpus)

        assert mlp is not None and diag is None
        assert len(calls) == 1, "the cold train did not go through train_from_labelset"
        _ctx, labelset, kwargs = calls[0]
        assert {e.label for e in labelset.elements} == {"good", "bad"}
        assert kwargs["snap"] is corpus

    def test_a_region_box_trains_on_the_patch_under_it(self, monkeypatch):
        """A Good label's ``region_box`` is pooled to its raw patch, not discarded.

        This is the headline divergence: the hand-rolled build read
        ``media_embedding`` for every md5-matched label, so a detector the user
        trained by drawing boxes became a whole-image detector the moment this
        path retrained it.  Here every image-level vector is ``basis(0)``, so a
        Good row that is ``basis(1)`` can only have come from the grid.
        """
        import vtscore.detectors.training as tr_mod
        from vtscore.detectors.model_loading import resolve_or_train_detector

        corpus = _patch_corpus()
        _activate(corpus)
        # Box the right-hand cell of each Good media - the one holding basis(1).
        boxes = {cid: (0.6, 0.0, 1.0, 1.0) for cid in (1, 2, 3)}
        detector_id = _write_detector("cold-region", corpus, region_boxes=boxes)

        seen: dict = {}
        real = tr_mod.train_and_threshold

        def _spy(X_list, y_list, **kwargs):
            seen["X"] = [np.asarray(x) for x in X_list]
            seen["y"] = list(y_list)
            seen["groups"] = kwargs.get("groups")
            return real(X_list, y_list, **kwargs)

        monkeypatch.setattr(tr_mod, "train_and_threshold", _spy)

        mlp, _threshold, _diag = resolve_or_train_detector(detector_id, _read("cold-region"), "image", corpus)
        assert mlp is not None

        good_rows = [x for x, y in zip(seen["X"], seen["y"], strict=True) if y == 1.0]
        assert len(good_rows) == 3
        for row in good_rows:
            assert np.allclose(row, _basis(1)), (
                "a boxed Good label trained on the image-level vector: the region box was discarded"
            )

    def test_a_bad_label_floods_its_patch_rows_as_negatives(self, monkeypatch):
        """One Bad label contributes its whole score-row stack, not one vector.

        Flooding is what stops the max-pool this head is scored under promoting
        a distractor patch into a media's score.  The hand-rolled build gave
        each Bad label exactly one negative row, which is precisely the train /
        score mismatch ``docs/ML.md`` names under region flooding.
        """
        import vtscore.detectors.training as tr_mod
        from vtscore.detectors.model_loading import resolve_or_train_detector

        corpus = _patch_corpus()
        _activate(corpus)
        detector_id = _write_detector("cold-flood", corpus)

        seen: dict = {}
        real = tr_mod.train_and_threshold

        def _spy(X_list, y_list, **kwargs):
            seen["y"] = list(y_list)
            seen["groups"] = list(kwargs.get("groups") or [])
            return real(X_list, y_list, **kwargs)

        monkeypatch.setattr(tr_mod, "train_and_threshold", _spy)
        resolve_or_train_detector(detector_id, _read("cold-flood"), "image", corpus)

        neg_bags = {g for g, y in zip(seen["groups"], seen["y"], strict=True) if y == 0.0}
        n_neg_rows = sum(1 for y in seen["y"] if y == 0.0)
        assert len(neg_bags) == 3, "the three Bad labels did not each become one bag"
        assert n_neg_rows > len(neg_bags), (
            f"{n_neg_rows} negative rows for {len(neg_bags)} Bad labels: the patch stack was not flooded"
        )

    def test_a_loaded_detector_keeps_the_head_it_just_trained(self):
        """The cold branch fills the fast path it is the counterpart of.

        ``train_from_labelset`` stores the head on the context it is handed, so
        a loaded detector whose model was invalidated gets it back rather than
        retraining on every call.
        """
        from vtscore.detectors.model_loading import resolve_or_train_detector
        from vtscore.state.core import DetectorContext, register_detector_context, unregister_detector_context

        corpus = _patch_corpus()
        _activate(corpus)
        detector_id = _write_detector("cold-caches", corpus)

        ctx = DetectorContext(detector_id, name="cold-caches", media_type="image", embedder_type="patch_semantic")
        register_detector_context(ctx)
        try:
            mlp, threshold, _diag = resolve_or_train_detector(detector_id, _read("cold-caches"), "image", corpus)
            assert mlp is not None
            assert ctx.model is mlp and ctx.threshold == threshold
            again, _t2, _d2 = resolve_or_train_detector(detector_id, _read("cold-caches"), "image", corpus)
            assert again is mlp, "the second call retrained instead of reusing the cached head"
        finally:
            unregister_detector_context(detector_id)


class TestColdHeadIsScoredAtItsGeometry:
    """Every caller pools the head over the rows it was trained and cut on."""

    def test_auto_detect_max_pools_the_patch_rows(self, client, monkeypatch):
        """A media whose *patch* fires is a hit, even when its image row does not.

        ``/api/auto-detect`` used to forward a bare image-level embedding matrix
        to its workers.  That was self-consistent only while this path built a
        whole-image head; against the app's MaxPatch head it is the same
        mismatch inverted, so the route now scores the shared
        ``scoring_rows_for_snap`` stack like every other scorer.
        """
        import vtsearch.routes.detectors.scoring as scoring_mod
        from vtsearch.settings import add_autofind_detector

        corpus = _patch_corpus()
        _activate(corpus)
        _write_detector("cold-geometry", corpus)
        add_autofind_detector("cold-geometry")

        # A head that fires only on basis(1) - which lives in a patch of ids
        # 1-3 and nowhere in any image-level vector.
        linear = nn.Linear(DIM, 1)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[0.0, 10.0, 0.0, 0.0]]))
            linear.bias.copy_(torch.tensor([-5.0]))
        head = nn.Sequential(linear).eval()
        monkeypatch.setattr(scoring_mod, "resolve_or_train_detector", lambda *a, **k: (head, 0.5, None))

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_json()
        result = resp.get_json()["results"]["cold-geometry"]
        assert {h["id"] for h in result["hits"]} == {1, 2, 3}, (
            "the hot patches never reached the head: auto-detect is scoring image-level vectors"
        )
        assert {h["id"] for h in result["negative_hits"]} == {4, 5, 6}


class TestColdTrainDiagnostic:
    """The labelset path only declines to train; the routes still owe an answer."""

    def test_find_label_reports_why_it_could_not_train(self, client):
        """An all-one-class labelset 400s with the counts and a hint.

        ``train_from_labelset`` returns a bare ``False`` here, so the report is
        rebuilt from what the training pass left on the context - which is what
        keeps the find-label UI's "why did this produce nothing?" answer alive.
        """
        corpus = _patch_corpus()
        _activate(corpus)
        detector_id = _write_detector("cold-one-class", corpus)
        data = _read("cold-one-class")
        for entry in data["labelset"]["labels"]:
            entry["label"] = "good"
        from vtscore.detectors.store import _detector_path
        from vtscore.detectors.store import _write_detector as write

        write(_detector_path("cold-one-class"), data)

        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 400, resp.get_json()
        diag = resp.get_json()["resolution_diagnostic"]
        assert diag["total_labels"] == 6
        assert diag["dataset_matched"] == 6, "every label is in the active dataset and must be counted as matched"
        assert diag["failed_resolution"] == 0
        assert diag["has_good"] is True and diag["has_bad"] is False
        assert "hint" in diag

    def test_unresolvable_labels_are_counted_and_sampled(self):
        """Labels that resolve nowhere are reported with up to three samples."""
        from vtscore.detectors.model_loading import resolve_or_train_detector

        corpus = _patch_corpus()
        _activate(corpus)
        detector_id = _write_detector("cold-unresolvable", corpus)
        data = _read("cold-unresolvable")
        for i, entry in enumerate(data["labelset"]["labels"]):
            # Nothing in the dataset matches, and the origin importer is fake.
            entry["md5"] = f"absent{i}"
            entry["origin_name"] = f"absent{i}.png"
            entry["filename"] = f"absent{i}.png"
            entry["origin"] = {"importer": "no_such_importer", "params": {}}

        mlp, threshold, diag = resolve_or_train_detector(detector_id, data, "image", corpus)

        assert mlp is None and threshold == 0.5
        assert diag is not None
        assert diag["total_labels"] == 6
        assert diag["dataset_matched"] == 0
        assert diag["needed_resolution"] == 6
        assert diag["resolved_from_origin"] == 0
        assert diag["failed_resolution"] == 6
        assert diag["media_type"] == "image"
        assert len(diag["sample_failures"]) == 3
        assert diag["sample_failures"][0]["origin"]["importer"] == "no_such_importer"
