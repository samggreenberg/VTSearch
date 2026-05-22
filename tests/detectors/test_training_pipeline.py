"""Tests for :mod:`vtscore.detectors.training`.

The existing :mod:`tests/io/test_export_options.py` covers
``train_and_threshold`` happy path indirectly.  This file fills in:

* :func:`validate_good_bad_split` — the small ABC-level guard.
* :func:`collect_media_origins` — origin extraction from a media snapshot.
* :func:`train_detector_from_origins` — the load-time origin → file →
  embedding → MLP pipeline, with file resolution and embedding stubbed.
* :func:`train_and_score` — the empty-result and single-class branches
  (the happy path is already exercised by the sort routes).
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from vtscore.detectors.training import (
    collect_media_origins,
    train_and_score,
    train_detector_from_origins,
    validate_good_bad_split,
)


# ---------------------------------------------------------------------------
# validate_good_bad_split
# ---------------------------------------------------------------------------


class TestValidateGoodBadSplit:
    def test_returns_counts_when_valid(self):
        n_good, n_bad = validate_good_bad_split([1.0, 0.0, 1.0, 0.0, 1.0])
        assert (n_good, n_bad) == (3, 2)

    def test_zero_good_raises(self):
        with pytest.raises(ValueError, match="at least one good and one bad"):
            validate_good_bad_split([0.0, 0.0])

    def test_zero_bad_raises(self):
        with pytest.raises(ValueError, match="at least one good and one bad"):
            validate_good_bad_split([1.0, 1.0])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            validate_good_bad_split([])


# ---------------------------------------------------------------------------
# collect_media_origins
# ---------------------------------------------------------------------------


class TestCollectMediaOrigins:
    def _snap(self):
        return {
            1: {
                "id": 1,
                "origin": {"importer": "test", "params": {}},
                "origin_name": "a.wav",
                "filename": "a.wav",
                "md5": "aa",
            },
            2: {
                "id": 2,
                "origin": {"importer": "test", "params": {}},
                "origin_name": "b.wav",
                "filename": "b.wav",
                "md5": "bb",
            },
        }

    def test_collects_from_dict_keys(self):
        # Vote dicts in the codebase are keyed by media id with None values.
        origins = collect_media_origins({1: None, 2: None}, self._snap())
        assert len(origins) == 2
        assert {o["origin_name"] for o in origins} == {"a.wav", "b.wav"}

    def test_collects_from_list(self):
        origins = collect_media_origins([1, 2], self._snap())
        assert len(origins) == 2

    def test_skips_unknown_ids(self):
        origins = collect_media_origins([1, 999], self._snap())
        assert len(origins) == 1
        assert origins[0]["origin_name"] == "a.wav"

    def test_each_dict_has_full_origin_fields(self):
        origins = collect_media_origins([1], self._snap())
        assert origins[0] == {
            "origin": {"importer": "test", "params": {}},
            "origin_name": "a.wav",
            "filename": "a.wav",
            "md5": "aa",
        }

    def test_missing_fields_default_to_empty_string(self):
        snap = {1: {"id": 1, "origin": None}}
        origins = collect_media_origins([1], snap)
        assert origins[0] == {
            "origin": None,
            "origin_name": "",
            "filename": "",
            "md5": "",
        }


# ---------------------------------------------------------------------------
# train_and_score
# ---------------------------------------------------------------------------


class TestTrainAndScore:
    def test_empty_votes_returns_empty(self):
        clips = {1: {"id": 1, "embedding": np.ones(8, dtype=np.float32)}}
        results, threshold, model = train_and_score(clips, {}, {})
        assert results == []
        assert threshold == 0.5
        assert model is None

    def test_only_good_returns_empty(self):
        clips = {
            1: {"id": 1, "embedding": np.ones(8, dtype=np.float32)},
            2: {"id": 2, "embedding": np.zeros(8, dtype=np.float32)},
        }
        results, threshold, model = train_and_score(clips, {1: None, 2: None}, {})
        assert results == []
        assert model is None

    def test_only_bad_returns_empty(self):
        clips = {
            1: {"id": 1, "embedding": np.ones(8, dtype=np.float32)},
            2: {"id": 2, "embedding": np.zeros(8, dtype=np.float32)},
        }
        results, threshold, model = train_and_score(clips, {}, {1: None, 2: None})
        assert results == []
        assert model is None

    def test_happy_path_trains_and_scores_all(self):
        # Deterministic, well-separated embeddings so training is stable.
        rng = np.random.default_rng(42)
        clips = {}
        for i in range(1, 11):
            # Good cluster centered on +1, bad on -1.
            center = 1.0 if i <= 5 else -1.0
            clips[i] = {
                "id": i,
                "embedding": (center + rng.standard_normal(8) * 0.05).astype(np.float32),
            }
        good = {1: None, 2: None}
        bad = {6: None, 7: None}
        results, threshold, model = train_and_score(clips, good, bad)
        assert model is not None
        assert isinstance(threshold, float)
        # Every clip in the snap shows up exactly once in the ranked results.
        assert len(results) == len(clips)
        assert {r["id"] for r in results} == set(clips)
        # Results are sorted by score descending.
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# train_detector_from_origins
# ---------------------------------------------------------------------------


class TestTrainDetectorFromOrigins:
    @pytest.fixture
    def stubbed_resolver(self, monkeypatch):
        """Replace ``resolve_file_context`` + ``embed_file`` so the tests
        don't need real files or models.

        Each origin maps deterministically to a vector keyed by its
        ``origin_name`` so we can drive the train/threshold pipeline.
        """
        # vec table keyed by origin_name; defaults to a class-0 vec.
        rng = np.random.default_rng(7)
        vec_table: dict[str, np.ndarray] = {}

        def _vec_for(name: str, klass: int) -> np.ndarray:
            if name not in vec_table:
                base = np.full(8, 1.0 if klass else -1.0, dtype=np.float32)
                vec_table[name] = (base + rng.standard_normal(8) * 0.05).astype(np.float32)
            return vec_table[name]

        @contextmanager
        def _fake_ctx(origin, origin_name="", filename=""):
            # Yield a sentinel "path" — we only use it to drive embed_file.
            yield ("PATH:" + origin_name) if origin_name else None

        def _fake_embed(path, media_type, embedder_name=""):
            if path is None:
                return None
            name = path.split(":", 1)[1] if isinstance(path, str) else ""
            # Encode class in origin_name prefix: "good_X" or "bad_X".
            if name.startswith("good_"):
                return _vec_for(name, 1)
            if name.startswith("bad_"):
                return _vec_for(name, 0)
            return None

        import vtscore.detectors.resolver as resolver_mod

        monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_ctx)
        monkeypatch.setattr(resolver_mod, "embed_file", _fake_embed)

    def _origins(self, names: list[str]) -> list[dict]:
        return [
            {
                "origin": {"importer": "test", "params": {}},
                "origin_name": n,
                "filename": n,
                "md5": "",
            }
            for n in names
        ]

    def test_too_few_labels_returns_none(self, stubbed_resolver):
        weights, threshold = train_detector_from_origins(
            self._origins(["good_1"]),
            [],
            inclusion=0,
            media_type="audio",
            embedder_name="clap",
        )
        assert weights is None
        assert threshold == 0.5

    def test_only_good_returns_none(self, stubbed_resolver):
        weights, threshold = train_detector_from_origins(
            self._origins(["good_1", "good_2", "good_3"]),
            [],
            inclusion=0,
            media_type="audio",
            embedder_name="clap",
        )
        assert weights is None
        assert threshold == 0.5

    def test_only_bad_returns_none(self, stubbed_resolver):
        weights, threshold = train_detector_from_origins(
            [],
            self._origins(["bad_1", "bad_2"]),
            inclusion=0,
            media_type="audio",
            embedder_name="clap",
        )
        assert weights is None

    def test_happy_path_returns_weights(self, stubbed_resolver):
        weights, threshold = train_detector_from_origins(
            self._origins(["good_1", "good_2", "good_3"]),
            self._origins(["bad_1", "bad_2", "bad_3"]),
            inclusion=0,
            media_type="audio",
            embedder_name="clap",
        )
        assert weights is not None
        # weights is a dict of {layer_name: nested_list_of_floats}.
        assert isinstance(weights, dict)
        assert len(weights) > 0
        for key, layer in weights.items():
            assert isinstance(key, str)
            assert isinstance(layer, list)
        assert isinstance(threshold, float)

    def test_unresolvable_entries_are_skipped(self, stubbed_resolver):
        """If an origin resolves to ``None``, the entry is skipped silently.

        With three resolvable goods + three resolvable bads but two extra
        garbage entries, the trainer still succeeds.
        """
        good = self._origins(["good_1", "good_2", "good_3", "junk_x"])
        bad = self._origins(["bad_1", "bad_2", "bad_3", "junk_y"])
        weights, _ = train_detector_from_origins(
            good,
            bad,
            inclusion=0,
            media_type="audio",
            embedder_name="clap",
        )
        assert weights is not None

    def test_all_unresolvable_returns_none(self, monkeypatch):
        """If every origin fails to resolve, no model is produced."""

        @contextmanager
        def _none_ctx(origin, origin_name="", filename=""):
            yield None

        import vtscore.detectors.resolver as resolver_mod

        monkeypatch.setattr(resolver_mod, "resolve_file_context", _none_ctx)

        weights, threshold = train_detector_from_origins(
            [{"origin": {"importer": "x", "params": {}}, "origin_name": "a", "md5": ""}],
            [{"origin": {"importer": "x", "params": {}}, "origin_name": "b", "md5": ""}],
            inclusion=0,
            media_type="audio",
            embedder_name="clap",
        )
        assert weights is None
        assert threshold == 0.5

    def test_embedder_name_is_forwarded(self, monkeypatch):
        """The ``embedder_name`` argument must reach ``embed_file`` unchanged.

        Regression test for H3 (embedder drift on save → reload): the load-
        time retrainer used to call ``embed_file(file_path, media_type)``
        with no embedder, so a CLAP-trained detector re-derived from saved
        origins would silently re-embed audio with whatever the media
        type's default embedder happened to be.
        """
        seen_embedders: list[str] = []

        @contextmanager
        def _fake_ctx(origin, origin_name="", filename=""):
            yield ("PATH:" + origin_name) if origin_name else None

        def _fake_embed(path, media_type, embedder_name=""):
            seen_embedders.append(embedder_name)
            if path is None:
                return None
            # One-hot-ish vector keyed by class so the MLP has something to learn.
            base = np.full(8, 1.0 if "good" in path else -1.0, dtype=np.float32)
            return base

        import vtscore.detectors.resolver as resolver_mod

        monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_ctx)
        monkeypatch.setattr(resolver_mod, "embed_file", _fake_embed)

        weights, _ = train_detector_from_origins(
            self._origins(["good_1", "good_2", "good_3"]),
            self._origins(["bad_1", "bad_2", "bad_3"]),
            inclusion=0,
            media_type="audio",
            embedder_name="clap",
        )
        assert weights is not None
        assert seen_embedders, "embed_file should have been called"
        assert set(seen_embedders) == {"clap"}, (
            f"every embed_file call must receive embedder_name='clap'; got {seen_embedders}"
        )
