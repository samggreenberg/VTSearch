"""Cross-dataset Find cuts and scores the way the rest of the app does.

Two properties, both about the corpus Find has *already loaded*:

**The cut is fitted on it.**  ``_score_with_cold_detector`` trains a head on the
fly from an unloaded detector's labelset.  Until #3516 it called
``train_and_threshold`` with no ``snap``, so - per that function's own contract,
*"without a snap there is no haystack to fuse and the cross-calibration cut
ships alone"* - every verdict in a cold Find was cut by the **pooled conformal**
rule, the weakest of the three #3115 measured, and the one whose
``CONFORMAL_BASE_BUDGET`` caps false negatives while only flooring false
positives (so it sits systematically below the oracle cut: over-admission by
design).  The haystack was in hand the whole time - ``temp_medias`` is the very
snapshot being scored.

**Each head is scored at the geometry it was trained and calibrated in.**  Both
scorers used to matmul the image-level embedding matrix, and both now read the
app's max-pooled score rows, because both now score a head the app's own
labelset training produced:

* the **live** head is the cached one - Bad votes flooded as patch negatives,
  threshold cut on the max-pooled distribution - so scoring its image-level row
  alone compares a systematically lower quantity against a cut made on a higher
  one, and the detector under-returns;
* the **cold** head used to be re-derived here from image-level label vectors
  with no flooding, which made it a *different detector* from the same labels
  (#3525).  It now comes from ``labelset_train_and_score``, so it is flooded and
  bag-calibrated like every other head the app builds, and the image-level pin
  that a whole-image head needed went with it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import app as app_module
from vtsearch.routes.detectors import find as find_mod

DIM = 4
EMB = "dinov3_patch"


def _basis(i: int) -> np.ndarray:
    return np.eye(DIM, dtype=np.float32)[i]


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


def _cold_corpus(n: int = 60) -> dict[int, dict]:
    """A plain (grid-less) corpus with a broad spread of scores.

    Ids 1-6 sit at the two poles and carry the detector's labels; the rest fill
    the space between them, so the population the threshold is fitted on is
    genuinely a distribution rather than two spikes.
    """
    rng = np.random.default_rng(3516)
    corpus: dict[int, dict] = {}
    for cid in range(1, 4):
        corpus[cid] = _media(cid, _basis(1))
    for cid in range(4, 7):
        corpus[cid] = _media(cid, _basis(0))
    for cid in range(7, n + 1):
        # A cloud tilted towards the Good pole, which is what makes the two cuts
        # land in different places (a symmetric cloud puts them both mid-gap).
        vec = _basis(1) * rng.uniform(0.2, 1.0) + _basis(0) * rng.uniform(0.0, 0.8)
        corpus[cid] = _media(cid, (vec / np.linalg.norm(vec)).astype(np.float32))
    return corpus


def _cold_config() -> dict:
    """A detector known only by its on-disk labelset - Find's cold path."""
    return {
        "name": "cold-det",
        "detector_id": "cold-det",
        "embedder_type": "patch_semantic",
        "detector_data": {
            "media_type": "image",
            "labelset": {
                "labels": [
                    {"md5": "m1", "label": "good"},
                    {"md5": "m2", "label": "good"},
                    {"md5": "m3", "label": "good"},
                    {"md5": "m4", "label": "bad"},
                    {"md5": "m5", "label": "bad"},
                    {"md5": "m6", "label": "bad"},
                ]
            },
        },
    }


def _run_find(corpus: dict[int, dict], dc: dict, monkeypatch) -> tuple[list[dict], list[dict]]:
    monkeypatch.setattr(find_mod, "_load_find_dataset_medias", lambda ds: corpus)
    with app_module.app.test_request_context("/api/find"):
        positives, negatives, _units, _added, _mt = find_mod._score_dataset(
            {"name": "corpus", "pkl_path": "ignored"}, [dc], 0, 0
        )
    return positives, negatives


class TestColdFindCutsOnTheCorpusItDecides:
    def test_the_estimator_is_fitted_on_the_corpus_being_scored(self, monkeypatch):
        """The haystack the population cut is fitted on *is* the Find snapshot.

        Before #3516 the cold path called ``train_and_threshold`` with no
        ``snap``, so - per that function's own contract - the pooled
        cross-calibration cut shipped alone.  What replaced it is the shared
        core's own fusion step, so this now watches ``_fused_threshold``: it
        must be handed the corpus's rows and ids, with the labelled media named
        so the estimator can drop them from the population it fits (#3308).
        """
        import vtscore.detectors.training as training_mod

        corpus = _cold_corpus()
        seen: list[dict] = []
        real_fused = training_mod._fused_threshold

        def _spy(xcal, folds, rows, scores, inclusion, blend_ctx, schedule, **kwargs):
            seen.append({"rows": rows, **kwargs})
            return real_fused(xcal, folds, rows, scores, inclusion, blend_ctx, schedule, **kwargs)

        monkeypatch.setattr(training_mod, "_fused_threshold", _spy)
        _run_find(corpus, _cold_config(), monkeypatch)

        assert seen, "the cold path never reached the population estimator"
        call = seen[0]
        assert call["rows"] is not None and len(call["rows"].ids) == len(corpus), (
            "the estimator was fitted without the haystack Find already holds (issue #3516)"
        )
        assert set(call["final_ids"]) == set(corpus)
        assert call["voted_ids"] == {1, 2, 3, 4, 5, 6}, (
            "the labelled media must be named so the estimator can drop them from the haystack it fits on (issue #3308)"
        )

    def test_the_two_cuts_move_verdicts(self, monkeypatch):
        """Guard the guard: on this corpus the substitution is user-visible.

        Without this the assertion above could pass on a fixture where the two
        cuts differ in the sixth decimal and nothing a user sees changes.  The
        counterfactual is the pre-fusion number - ``_fused_threshold``'s own
        ``xcal_threshold`` argument, which is exactly what used to ship - and
        the comparison is paired, since the head is identical either way
        (training is deterministic) and only the line moves.
        """
        import vtscore.detectors.training as training_mod

        corpus = _cold_corpus()
        anchored, _neg = _run_find(corpus, _cold_config(), monkeypatch)

        monkeypatch.setattr(
            training_mod,
            "_fused_threshold",
            lambda xcal, *args, **kwargs: xcal,
        )
        pooled, _neg2 = _run_find(corpus, _cold_config(), monkeypatch)

        assert len(pooled) > len(anchored), (
            f"the pooled cut admitted {len(pooled)} of {len(corpus)} and the anchored cut "
            f"{len(anchored)}; on this fixture the two rules must separate, or the "
            "assertion above is checking a difference nobody can see"
        )


class TestColdFindIsTheAppsLabelsetTraining:
    """One labelset means one detector, whichever path built the head (#3525)."""

    def test_cold_find_delegates_to_labelset_train_and_score(self, monkeypatch):
        """The head comes from the app's entry point, not a port living here.

        This is the whole point of #3525: a hand-rolled reimplementation
        silently trained a *different* detector from the same labels (image-level
        Good vectors with the ``region_box`` discarded, one negative per Bad vote
        instead of its flooded patch stack, per-row instead of per-bag
        calibration).  The guard is structural because that divergence is
        invisible in the numbers - both versions return plausible scores.
        """
        import vtscore.detectors.labelset_training as lt_mod

        calls: list[dict] = []
        real = lt_mod.labelset_train_and_score

        def _spy(det_ctx, labelset, **kwargs):
            calls.append({"det_ctx": det_ctx, "labelset": labelset, **kwargs})
            return real(det_ctx, labelset, **kwargs)

        monkeypatch.setattr(lt_mod, "labelset_train_and_score", _spy)
        corpus = _cold_corpus(n=20)
        _run_find(corpus, _cold_config(), monkeypatch)

        assert len(calls) == 1, "the cold path did not go through labelset_train_and_score"
        call = calls[0]
        assert call["clips_dict"] is corpus
        assert {e.label for e in call["labelset"].elements} == {"good", "bad"}
        assert call["rows"] is not None, (
            "the route's prebuilt rows were not passed through, so the shared core restacked the corpus"
        )

    def test_the_live_detector_context_is_not_populated(self, monkeypatch):
        """A cold Find must not write another dataset's vectors into a loaded detector.

        ``populate_label_embeddings`` caches its resolved vectors on whatever
        context it is handed and stamps that context's ``embedder``.  Handing it
        the live one would leave the loaded detector holding label embeddings
        built against whichever dataset Find happened to be scanning.  It would
        also persist that foreign space to the detector registry, which is why
        the throwaway context carries an empty ``detector_id``
        (``record_detector_embedder`` no-ops on one).
        """
        from vtscore.state.core import DetectorContext, register_detector_context, unregister_detector_context

        live = DetectorContext("cold-det", name="cold-det", embedder_type="patch_semantic")
        register_detector_context(live)
        try:
            _run_find(_cold_corpus(n=20), _cold_config(), monkeypatch)
        finally:
            unregister_detector_context("cold-det")

        assert not live.label_embeddings, (
            "cold Find populated the live detector context's label cache with a foreign dataset's vectors"
        )
        assert not live.embedder, "cold Find re-stamped the live context's embedder"

    def test_labels_resolve_once_across_every_dataset_in_the_run(self, monkeypatch):
        """A label that has to be fetched from its origin is fetched once, not per dataset.

        Delegating buys the app's real head at the cost of the app's real
        resolution: an element absent from the snapshot costs an importer fetch
        (plus a ``patch_forward`` on a patch detector).  Find scores every
        detector against every selected dataset, so the throwaway context is kept
        for the whole run and later datasets read the cache.
        """
        import vtscore.detectors.labelset_training as lt_mod

        resolved: list[str] = []
        monkeypatch.setattr(
            lt_mod,
            "_resolve_uncached_embedding",
            lambda elem, snap, **kw: resolved.append(elem.md5) or _basis(1),
        )
        monkeypatch.setattr(lt_mod, "_resolve_score_rows", lambda *a, **kw: None)

        # A corpus that resolves none of the labels, so every element takes the
        # origin path on the first dataset and the cache on the second.
        corpus = {cid: _media(cid, _basis(1)) for cid in range(100, 120)}
        dc = _cold_config()
        for i, entry in enumerate(dc["detector_data"]["labelset"]["labels"]):
            entry["label"] = "good" if i < 3 else "bad"

        monkeypatch.setattr(find_mod, "_load_find_dataset_medias", lambda ds: corpus)
        with app_module.app.test_request_context("/api/find"):
            for name in ("ds-a", "ds-b"):
                find_mod._score_dataset({"name": name, "pkl_path": "ignored"}, [dc], 0, 0)

        assert len(resolved) == 6, (
            f"{len(resolved)} origin resolutions for 6 labels over 2 datasets: the cold "
            "context is being rebuilt per dataset instead of held for the run"
        )


class TestFindScoresAtTheAppsGeometry:
    """A live detector's cut is only meaningful in the space it was fitted in."""

    def test_live_mlp_max_pools_the_patch_rows(self, monkeypatch):
        """A media whose *patch* fires is a hit, even when its image row does not.

        This is MaxPatch: the app scores a media at the max over its image-level
        row plus every raw patch.  Scoring the image-level row alone - what this
        route used to do - makes a cross-dataset Find systematically colder than
        the detector it is running, because the threshold on that detector was
        cut on the pooled distribution.
        """
        off_with_hot_patch = np.zeros((1, 2, DIM), dtype=np.float32)
        off_with_hot_patch[0, 0] = _basis(0)  # a cold patch
        off_with_hot_patch[0, 1] = _basis(1)  # the one that fires
        all_cold = np.zeros((1, 2, DIM), dtype=np.float32)
        all_cold[0, 0] = _basis(0)
        all_cold[0, 1] = _basis(0)

        corpus = {
            1: _media(1, _basis(0), grid=off_with_hot_patch),
            2: _media(2, _basis(0), grid=all_cold),
        }

        linear = nn.Linear(DIM, 1)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[0.0, 10.0, 0.0, 0.0]]))
            linear.bias.copy_(torch.tensor([-5.0]))
        dc = {
            "name": "patch-det",
            "detector_id": "patch-det",
            "live_mlp": nn.Sequential(linear).eval(),
            "threshold": 0.5,
            "live_embedder": EMB,
            "embedder_type": "patch_semantic",
        }

        positives, negatives = _run_find(corpus, dc, monkeypatch)

        assert {p["id"] for p in positives} == {1}, (
            "media 1's hot patch never reached the head: Find is scoring the "
            "image-level vector instead of the media's score rows"
        )
        assert {n["id"] for n in negatives} == {2}

    def test_cold_head_max_pools_like_every_other_head(self, monkeypatch):
        """The cold path scores the app's score rows, not one row per media.

        Its head is built by ``labelset_train_and_score``, which floods a Bad
        label's patch stack as negatives and collapses each calibration bag over
        the same rows the scorer max-pools - so image-level scoring would now be
        the mismatch (issue #3525).  The rows it scores are also the rows the
        threshold is fitted on, because the shared core is handed both.
        """
        grid = np.zeros((1, 2, DIM), dtype=np.float32)
        grid[0, 0] = _basis(0)
        grid[0, 1] = _basis(1)
        corpus = _cold_corpus(n=20)
        for media in corpus.values():
            media["patch_grid"] = grid

        seen: list[tuple[int, int]] = []
        real_rows = find_mod.scoring_rows_for_snap

        def _spy(clips_dict, embedder_name=None, *, region_pooling=None):
            rows = real_rows(clips_dict, embedder_name, region_pooling=region_pooling)
            seen.append((len(rows.ids), int(rows.matrix.shape[0])))
            return rows

        monkeypatch.setattr(find_mod, "scoring_rows_for_snap", _spy)
        _run_find(corpus, _cold_config(), monkeypatch)

        assert seen, "the cold path built no rows"
        n_media, n_rows = seen[0]
        assert n_media == len(corpus)
        # 1 image-level row + 2 patches per media.
        assert n_rows == n_media * 3, (
            f"the cold path stacked {n_rows} rows for {n_media} media: it is scoring "
            "image-level rows under a head that was trained with region flooding"
        )

    def test_cold_path_builds_the_corpus_rows_once_for_every_detector(self, monkeypatch):
        """N cold detectors over one dataset stack the corpus once, not N times.

        ``get_region_matrix_for_snap`` caches only against the *active* dataset
        context, and a Find snapshot never is one, so the route's own
        ``rows_cache`` is the only thing between this and an O(detectors)
        restack of a multi-million-row matrix.  Delegating the cold path must
        pass those rows through rather than let the shared core rebuild them.
        """
        grid = np.zeros((1, 2, DIM), dtype=np.float32)
        grid[0, 0] = _basis(0)
        grid[0, 1] = _basis(1)
        corpus = _cold_corpus(n=20)
        for media in corpus.values():
            media["patch_grid"] = grid

        builds: list[int] = []
        real_rows = find_mod.scoring_rows_for_snap

        def _spy(clips_dict, embedder_name=None, *, region_pooling=None):
            builds.append(len(clips_dict))
            return real_rows(clips_dict, embedder_name, region_pooling=region_pooling)

        monkeypatch.setattr(find_mod, "scoring_rows_for_snap", _spy)
        monkeypatch.setattr(find_mod, "_load_find_dataset_medias", lambda ds: corpus)

        configs = []
        for i in range(3):
            dc = _cold_config()
            dc["name"] = f"cold-det-{i}"
            dc["detector_id"] = f"cold-det-{i}"
            configs.append(dc)

        with app_module.app.test_request_context("/api/find"):
            find_mod._score_dataset({"name": "corpus", "pkl_path": "ignored"}, configs, 0, 0)

        assert len(builds) == 1, (
            f"the corpus was stacked {len(builds)} times for 3 detectors in one space; "
            "the cold path is rebuilding rows the route already holds"
        )

    def test_cold_path_does_not_write_the_foreign_embedder_to_the_registry(self, monkeypatch):
        """The throwaway context must not stamp this dataset's space on the detector.

        ``populate_label_embeddings`` ends by persisting the space it embedded
        in, so the preload predictor warms the right model next session.  Run
        against a dataset the detector is not loaded against, that would record
        a *foreign* dataset's embedder as the detector's own, so the throwaway
        context carries an empty ``detector_id`` and the write no-ops.
        """
        import vtscore.detectors.registry as registry_mod

        recorded: list[tuple[str, str]] = []
        monkeypatch.setattr(
            registry_mod,
            "record_detector_embedder",
            lambda det_id, emb: recorded.append((det_id, emb)),
        )
        _run_find(_cold_corpus(n=20), _cold_config(), monkeypatch)

        assert not [r for r in recorded if r[0]], (
            f"cold Find persisted an embedder against a real detector id: {recorded}"
        )


class TestEveryMediaGetsAVerdict:
    def test_media_with_no_vector_is_reported_not_dropped(self, monkeypatch):
        """One unembeddable media costs one ``N/A``, not the whole detector.

        The row builder drops such a media rather than failing (issue #3179), so
        it comes back in fewer rows than there are ids; without an explicit
        fill it would appear in neither the hit nor the miss table.
        """
        corpus = _cold_corpus(n=10)
        corpus[10]["embeddings"] = {}
        corpus[10]["embedding"] = None

        positives, negatives = _run_find(corpus, _cold_config(), monkeypatch)

        seen = {p["id"] for p in positives} | {n["id"] for n in negatives}
        assert seen == set(corpus), "a media with no usable vector fell out of both result tables"
        verdicts = {n["id"]: n["detector_verdicts"]["cold-det"]["verdict"] for n in negatives}
        assert verdicts[10] == "N/A"
        assert any(v != "N/A" for cid, v in verdicts.items() if cid != 10) or positives, (
            "one unscorable media must not sink the other nine"
        )
