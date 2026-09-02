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
scorers used to matmul the image-level embedding matrix, which is right for one
of them and wrong for the other:

* the **live** head came from the app's own training - Bad votes flooded as
  patch negatives, threshold cut on the max-pooled distribution - so scoring its
  image-level row alone compares a systematically lower quantity against a cut
  made on a higher one, and the detector under-returns;
* the **cold** head is re-derived here from image-level label vectors with no
  flooding, so it is a whole-image head and image-level rows are correct for it
  (``docs/ML.md``, region flooding).  What was wrong there was the *cut*, which
  the estimator would otherwise fit on the max-pool.
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
    def test_cold_threshold_is_not_the_pooled_conformal_cut(self, monkeypatch):
        """The shipped cut differs from the one a snap-less train would return.

        The counterfactual is computable exactly: training is deterministic
        (seeded fold splits, seeded torch init), so re-training the same X/y
        with no haystack reproduces the number cold Find used to ship.
        """
        from vtscore.detectors.training import train_and_threshold

        corpus = _cold_corpus()
        dc = _cold_config()

        captured: list[float] = []
        real_record = find_mod._record_verdicts

        def _spy(media_results, dc_name, all_ids, scores, threshold, fallback):
            if scores is not None:
                captured.append(threshold)
            return real_record(media_results, dc_name, all_ids, scores, threshold, fallback)

        monkeypatch.setattr(find_mod, "_record_verdicts", _spy)
        _run_find(corpus, dc, monkeypatch)
        assert captured, "the cold path scored nothing, so there is no threshold to check"
        shipped = captured[0]

        X_list, y_list, voted_ids = find_mod._collect_cold_training_data(dc["detector_data"], corpus, EMB)
        assert voted_ids == {1, 2, 3, 4, 5, 6}, (
            "the labelled media must be named so the estimator can drop them from the haystack it fits on (issue #3308)"
        )
        _model, pooled = train_and_threshold(X_list, y_list, embedder_name=EMB)

        assert shipped != pooled, (
            "cold Find still ships the pooled cross-calibration cut; the haystack "
            "it is scoring never reached the population estimator (issue #3516)"
        )

    def test_the_two_cuts_move_verdicts(self, monkeypatch):
        """Guard the guard: on this corpus the substitution is user-visible.

        Without this the assertion above could pass on a fixture where the two
        cuts differ in the sixth decimal and nothing a user sees changes.  The
        comparison is paired - the head is identical either way (training is
        deterministic), so the only thing that moves is the line.
        """
        import vtscore.detectors.training as training_mod

        corpus = _cold_corpus()
        anchored, _neg = _run_find(corpus, _cold_config(), monkeypatch)

        real_train = training_mod.train_and_threshold

        def _snapless(X_list, y_list, snap=None, **kwargs):
            kwargs.pop("haystack_rows", None)
            kwargs.pop("voted_ids", None)
            return real_train(X_list, y_list, **kwargs)

        monkeypatch.setattr(find_mod, "train_and_threshold", _snapless)
        pooled, _neg2 = _run_find(corpus, _cold_config(), monkeypatch)

        assert len(pooled) > len(anchored), (
            f"the pooled cut admitted {len(pooled)} of {len(corpus)} and the anchored cut "
            f"{len(anchored)}; on this fixture the two rules must separate, or the "
            "assertion above is checking a difference nobody can see"
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

    def test_cold_head_stays_on_image_level_rows(self, monkeypatch):
        """The cold path does **not** max-pool, and its cut is fitted the same way.

        Its head never saw a patch as a negative, so promoting a distractor
        patch to the media's score would over-fire.  The rows it scores are also
        the rows the threshold is fitted on, which is the whole point of handing
        them to ``train_and_threshold`` as ``haystack_rows``.
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
        assert n_rows == n_media, (
            f"the cold path stacked {n_rows} rows for {n_media} media: it is max-pooling "
            "patch rows into a head that was trained on image-level vectors alone"
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
