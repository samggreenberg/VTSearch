"""Executable mirror of the ``vtscore`` front-door documentation.

Every test here runs the code a reader would copy out of
``vtscore/docs/quickstart.md`` (and its siblings ``concepts.md``,
``integration.md`` and ``tutorials/train-and-score.md``) against a synthetic
audio dataset.  The doc snippets went stale silently because nothing executed
them: ``media["embedding"]`` outlived its own removal, ``LabelSet``'s
constructor was renamed, and ``CoreConfig`` grew required fields, all while the
quickstart kept claiming otherwise.

The rule this file encodes: **a snippet the docs present as runnable has a test
here that runs it.**  When an API in the walkthrough changes, this suite breaks
in the same commit, and the doc gets updated instead of rotting.  Keep the code
below a faithful transcription of the prose - same call shapes, same argument
names - rather than the tersest way to reach the same assertion.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from vtscore.config import CoreConfig
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.datasets.loader import load_dataset_from_folder
from vtscore.datasets.origin import Origin
from vtscore.datasets.stages.embedding import embed_missing
from vtscore.detectors.store import load_detector, save_detector
from vtscore.detectors.training import train_detector_from_origins
from vtscore.embedding.helpers import embed_text_query
from vtscore.embedding.media_vectors import media_embedding
from vtscore.media import audio  # noqa: F401 - registers the audio MediaType + embedders
from vtscore.state import DetectorContext, register_detector_context
from vtscore.training import calculate_cross_calibration_threshold, train_model
from vtscore.training.mlp import LINEAR_SVM_HEAD

from tests_lib.helpers import make_wav_file


# The labels the quickstart hard-codes by filename: three "good", three "bad".
GOOD_FILES = ["bark/poodle.wav", "bark/labrador.wav", "bark/beagle.wav"]
BAD_FILES = ["music/track-01.wav", "speech/news-clip.wav", "speech/interview.wav"]
LABELS = {**{f: "good" for f in GOOD_FILES}, **{f: "bad" for f in BAD_FILES}}


def _make_corpus(root: Path) -> Path:
    """Write the quickstart's little bark/music/speech folder tree."""
    root.mkdir(parents=True, exist_ok=True)
    for i, rel in enumerate([*GOOD_FILES, *BAD_FILES]):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        make_wav_file(path.parent, path.name, frequency=220.0 + 40.0 * i)
    return root


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return _make_corpus(tmp_path / "audio-corpus")


def _load_folder(folder: Path) -> dict[int, dict[str, Any]]:
    """Quickstart step 2: load a folder of audio files, then embed them.

    Two stages, because the loader deliberately does not embed: it emits media
    dicts with an empty ``embeddings`` map and ``embed_missing`` fills them in.
    """
    origin = Origin(
        importer="server_folder",
        params={"path": str(folder), "media_type": "audio"},
    ).to_dict()

    medias: dict[int, dict[str, Any]] = {}
    load_dataset_from_folder(
        folder_path=folder,
        media_type="audio",
        medias=medias,
        recursive=True,
        origin=origin,
    )
    embed_missing(medias)
    return medias


@pytest.fixture
def loaded_medias(corpus: Path) -> dict[int, dict[str, Any]]:
    medias = _load_folder(corpus)
    assert len(medias) == len(LABELS)
    return medias


@pytest.fixture
def detectors_dir(tmp_path: Path, monkeypatch) -> Path:
    """Point ``CoreConfig.detectors_dir`` at a scratch directory."""
    import vtscore.config as config_mod

    target = tmp_path / "detectors"
    base = config_mod.CoreConfig.from_settings()
    monkeypatch.setattr(
        config_mod,
        "_core_config_builder",
        lambda _settings_path=None: replace_detectors_dir(base, target),
        raising=False,
    )
    return target


def replace_detectors_dir(config: CoreConfig, detectors_dir: Path) -> CoreConfig:
    """Return *config* with a different ``detectors_dir`` (it is frozen)."""
    import dataclasses

    return dataclasses.replace(config, detectors_dir=detectors_dir)


# ---------------------------------------------------------------------------
# quickstart.md §1 - Set up CoreConfig
# ---------------------------------------------------------------------------


class TestCoreConfigSetup:
    """The config builder the docs tell a library-only consumer to install."""

    def test_builder_snippet_constructs_and_registers(self, tmp_path: Path, monkeypatch):
        import vtscore.config as config_mod

        DATA = tmp_path / "vtscore-quickstart"
        DATA.mkdir(exist_ok=True)

        def _build(settings_path=None) -> CoreConfig:
            return CoreConfig(
                data_dir=DATA,
                saved_datasets_dir=DATA / "datasets",
                detectors_dir=DATA / "detectors",
                max_concurrent_dataset_downloads=1,
                max_concurrent_dataset_embeddings=1,
                autofind_detectors=(),
                dataset_max_age_days=None,
                calibrate_count=2,
                calibration_fraction=0.5,
                enrich_descriptions=False,
                autopilot_goal_diversity=8,
                inclusion=0,
            )

        monkeypatch.setattr(config_mod, "_core_config_builder", _build, raising=False)

        config = CoreConfig.from_settings()
        assert config.data_dir == DATA
        assert config.detectors_dir == DATA / "detectors"

    def test_from_settings_takes_a_settings_path(self, monkeypatch):
        """The builder is called with the ``settings_path`` argument, so a
        zero-argument builder (as the docs used to show) would raise."""
        import vtscore.config as config_mod

        seen: list[Any] = []

        def _build(settings_path=None) -> CoreConfig:
            seen.append(settings_path)
            return _minimal_config()

        monkeypatch.setattr(config_mod, "_core_config_builder", _build, raising=False)
        CoreConfig.from_settings("/tmp/run-settings.json")
        assert seen == ["/tmp/run-settings.json"]


def _minimal_config() -> CoreConfig:
    return CoreConfig(
        data_dir=Path("/tmp/vtscore-doc-test"),
        saved_datasets_dir=Path("/tmp/vtscore-doc-test/datasets"),
        detectors_dir=Path("/tmp/vtscore-doc-test/detectors"),
        max_concurrent_dataset_downloads=1,
        max_concurrent_dataset_embeddings=1,
        autofind_detectors=(),
        dataset_max_age_days=None,
        calibrate_count=2,
        calibration_fraction=0.5,
        enrich_descriptions=False,
        autopilot_goal_diversity=8,
        inclusion=0,
    )


# ---------------------------------------------------------------------------
# quickstart.md §2 / concepts.md §1 - the media dict
# ---------------------------------------------------------------------------


class TestMediaShape:
    def test_media_registry_is_auto_discovered(self):
        """The docs say the registry populates itself on `import vtscore.media`."""
        from vtscore.media import all_types

        registered = {t.type_id for t in all_types()}
        assert {"audio", "image", "text", "video", "document", "face"} <= registered

    def test_loader_does_not_embed_and_embed_missing_does(self, corpus):
        """The two-stage split the load walkthrough now documents."""
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(
            folder_path=corpus,
            media_type="audio",
            medias=medias,
            recursive=True,
        )
        assert medias, "the loader should still have produced media dicts"
        assert all(m["embeddings"] == {} for m in medias.values()), "loaders must not embed"
        assert all(m["origin"] is None for m in medias.values()), "no origin unless one is passed"

        embed_missing(medias)
        assert all(media_embedding(m) is not None for m in medias.values())

    def test_unknown_media_type_raises_value_error(self, corpus):
        with pytest.raises(ValueError, match="Invalid media type"):
            load_dataset_from_folder(folder_path=corpus, media_type="hologram", medias={})

    def test_documented_keys_are_present(self, loaded_medias):
        media = next(iter(loaded_medias.values()))
        for key in (
            "id",
            "media_type",
            "embedder",
            "file_size",
            "md5",
            "embeddings",
            "filename",
            "origin",
            "origin_name",
            "media_path",
        ):
            assert key in media, f"documented media key missing: {key}"

    def test_vectors_live_in_the_per_embedder_dict(self, loaded_medias):
        media = next(iter(loaded_medias.values()))
        assert "embedding" not in media, "the singular key was removed; docs must not show it"
        assert isinstance(media["embeddings"], dict)

        vec = media_embedding(media)
        assert vec is not None
        assert vec.dtype == np.float32
        assert vec.ndim == 1
        # The same vector, addressed by embedder name.
        assert media_embedding(media, media["embedder"]) is vec

    def test_origin_round_trips_through_the_documented_shape(self, loaded_medias, corpus):
        media = next(iter(loaded_medias.values()))
        assert media["origin"]["importer"] == "server_folder"
        assert media["origin"]["params"]["path"] == str(corpus)


# ---------------------------------------------------------------------------
# quickstart.md §3 / tutorials - train a head from labels
# ---------------------------------------------------------------------------


def _build_xy(medias: dict[int, dict[str, Any]]) -> tuple[list[np.ndarray], list[float]]:
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for m in medias.values():
        label = LABELS.get(m["filename"])
        if label is None:
            continue
        X_list.append(media_embedding(m))
        y_list.append(1.0 if label == "good" else 0.0)
    return X_list, y_list


class TestTrainFromLabels:
    def test_train_model_and_threshold(self, loaded_medias):
        X_list, y_list = _build_xy(loaded_medias)
        assert len(X_list) == len(LABELS)

        X = torch.from_numpy(np.stack(X_list).astype(np.float32))
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

        model = train_model(X, y, input_dim=X.shape[1], hidden_dim=LINEAR_SVM_HEAD)
        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim=X.shape[1],
            inclusion_value=0,
            hidden_dim=LINEAR_SVM_HEAD,
        )
        assert isinstance(threshold, float)

        # The linear SVM head is a single Linear(D, 1) - one weight matrix, one bias.
        assert len(list(model.parameters())) == 2

    def test_scoring_a_fresh_folder(self, loaded_medias, tmp_path):
        X_list, y_list = _build_xy(loaded_medias)
        X = torch.from_numpy(np.stack(X_list).astype(np.float32))
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
        model = train_model(X, y, input_dim=X.shape[1], hidden_dim=LINEAR_SVM_HEAD)

        target_folder = tmp_path / "new-audio"
        target_folder.mkdir()
        for i in range(4):
            make_wav_file(target_folder, f"clip_{i}.wav", frequency=300.0 + 25.0 * i)

        target_medias = _load_folder(target_folder)

        ids = list(target_medias.keys())
        embeddings = np.stack([media_embedding(target_medias[i]) for i in ids]).astype(np.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(embeddings)).squeeze(1).numpy()
        scores = 1.0 / (1.0 + np.exp(-logits))

        hits = sorted(
            [(target_medias[i]["media_path"], float(s)) for i, s in zip(ids, scores)],
            key=lambda h: h[1],
            reverse=True,
        )
        assert len(hits) == len(ids)
        assert all(0.0 <= s <= 1.0 for _, s in hits)


# ---------------------------------------------------------------------------
# quickstart.md §5 / tutorials §5-6 - persist and reload a detector
# ---------------------------------------------------------------------------


class TestPersistAndReload:
    def test_labelset_round_trips_through_save_detector(self, loaded_medias, detectors_dir):
        elements = []
        for m in loaded_medias.values():
            label = LABELS.get(m["filename"])
            if label is None:
                continue
            elements.append(
                LabeledElement(
                    md5=m["md5"],
                    label=label,
                    origin_name=m["origin_name"],
                    origin=m["origin"],
                    filename=m["filename"],
                )
            )

        labelset = LabelSet(elements)
        assert len(labelset) == len(LABELS)
        assert labelset.elements[0].label in ("good", "bad")

        path = save_detector("barks", labelset, media_type="audio")
        assert path.parent == detectors_dir

        data = load_detector("barks")
        assert data is not None
        assert data["name"] == "barks"
        assert data["media_type"] == "audio"

        # No embeddings, no weights - just origins and labels.
        serialised = data["labelset"]["labels"]
        assert len(serialised) == len(LABELS)
        assert set(serialised[0]) <= {
            "md5",
            "label",
            "origin",
            "origin_name",
            "filename",
            "category",
            "metadata",
            "region_box",
        }
        assert "embedding" not in serialised[0]
        assert "embeddings" not in serialised[0]

        restored = LabelSet.from_dict(data["labelset"])
        assert len(restored) == len(LABELS)

    def test_retrain_from_saved_origins(self, loaded_medias, detectors_dir):
        elements = [
            LabeledElement(
                md5=m["md5"],
                label=LABELS[m["filename"]],
                origin_name=m["origin_name"],
                origin=m["origin"],
                filename=m["filename"],
            )
            for m in loaded_medias.values()
            if m["filename"] in LABELS
        ]
        save_detector("barks", LabelSet(elements), media_type="audio")

        ctx = DetectorContext(
            detector_id="barks",
            name="barks",
            media_type="audio",
            embedder=next(iter(loaded_medias.values()))["embedder"],
        )
        register_detector_context(ctx)

        data = load_detector("barks")
        assert data is not None
        saved = LabelSet.from_dict(data["labelset"])
        good_origins = [
            {"origin": e.origin, "origin_name": e.origin_name, "filename": e.filename, "md5": e.md5}
            for e in saved.elements
            if e.label == "good"
        ]
        bad_origins = [
            {"origin": e.origin, "origin_name": e.origin_name, "filename": e.filename, "md5": e.md5}
            for e in saved.elements
            if e.label == "bad"
        ]
        assert good_origins and bad_origins

        weights, threshold = train_detector_from_origins(
            good_origins,
            bad_origins,
            inclusion=0,
            media_type="audio",
            embedder_name=ctx.embedder,
        )
        assert weights is not None, "every origin should have re-resolved to its file"
        assert isinstance(threshold, float)
        ctx.threshold = threshold

        # ``weights`` is a state-dict-shaped dict of plain lists; it is what a
        # caller hands to ``build_model_from_weights`` to get a live head back.
        from vtscore.training.mlp import build_model_from_weights

        model = build_model_from_weights(weights)
        vec = media_embedding(next(iter(loaded_medias.values())))
        with torch.no_grad():
            score = torch.sigmoid(model(torch.from_numpy(vec[None, :]))).item()
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# concepts.md §4 - LabelSet / LabeledElement
# ---------------------------------------------------------------------------


class TestLabelSetConcepts:
    def _elem(self, name: str, label: str = "good") -> LabeledElement:
        return LabeledElement(
            md5=f"md5-{name}",
            label=label,
            origin_name=name,
            origin={"importer": "server_folder", "params": {"path": "/data/audio"}},
            filename=name,
            metadata={"reviewer": "alice"},
            region_box=None,
        )

    def test_constructor_len_iter_and_elements(self):
        a, b = self._elem("a.wav"), self._elem("b.wav", "bad")
        labelset = LabelSet([a, b], detector_meta={"media_type": "audio", "threshold": 0.41})

        assert len(labelset) == 2
        assert labelset.elements == [a, b]
        assert list(labelset) == [a, b]
        assert labelset.detector_meta == {"media_type": "audio", "threshold": 0.41}

    def test_no_labels_keyword_and_no_json_helpers(self):
        # The pre-rename spelling the docs used to show.  Called through a
        # kwargs dict so the type checker doesn't flag the deliberate misuse.
        bad_kwargs: dict[str, Any] = {"labels": [self._elem("a.wav")]}
        with pytest.raises(TypeError):
            LabelSet(**bad_kwargs)
        assert not hasattr(LabelSet, "to_json")
        assert not hasattr(LabelSet, "from_json")

    def test_dict_round_trip_and_merge(self):
        one = LabelSet([self._elem("a.wav")])
        two = LabelSet([self._elem("b.wav", "bad")])

        merged = one.merge(two)
        assert isinstance(merged, LabelSet)
        assert len(merged) == 2
        assert len(one) == 1, "merge returns a new LabelSet rather than mutating"

        payload = merged.to_dict()
        assert set(payload) == {"labels"}
        restored = LabelSet.from_dict(payload)
        assert [e.origin_name for e in restored.elements] == ["a.wav", "b.wav"]
        assert restored.elements[0].metadata == {"reviewer": "alice"}


# ---------------------------------------------------------------------------
# packages/training.md - the split calibration API
# ---------------------------------------------------------------------------


class TestCalibrationFoldsSnippet:
    def test_folds_then_threshold(self, loaded_medias):
        from vtscore.training import calibration_folds_cached, threshold_from_folds

        X_list, y_list = _build_xy(loaded_medias)
        folds = calibration_folds_cached(
            X_list,
            y_list,
            X_list[0].shape[0],
            calibrate_count=2,
            calibration_fraction=0.5,
            hidden_dim=LINEAR_SVM_HEAD,
        )
        assert folds._fields == ("orderings", "fallback", "models")
        assert isinstance(threshold_from_folds(folds, inclusion_value=0), float)


# ---------------------------------------------------------------------------
# tutorials/train-and-score.md §8 - eval
# ---------------------------------------------------------------------------


class TestEvalSnippet:
    def test_eval_learned_sort_reads_ground_truth_off_the_medias(self, loaded_medias):
        from vtscore.eval.config import EvalQuery
        from vtscore.eval.runner import eval_learned_sort

        for media in loaded_medias.values():
            media["category"] = (
                "dog" if media["filename"] in LABELS and LABELS[media["filename"]] == "good" else "other"
            )

        metrics_list = eval_learned_sort(
            loaded_medias,
            [EvalQuery("a dog barking", "dog")],
            train_fraction=0.5,
            seed=42,
        )
        assert len(metrics_list) == 1
        metrics = metrics_list[0]
        assert metrics.target_category == "dog"
        for field in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= getattr(metrics, field) <= 1.0
        assert metrics.num_train + metrics.num_test == len(loaded_medias)

    def test_eval_learned_sort_is_not_re_exported_from_the_package(self):
        import vtscore.eval as eval_pkg

        assert not hasattr(eval_pkg, "eval_learned_sort")


# ---------------------------------------------------------------------------
# packages/concurrency.md + packages/security.md - namespace packages
# ---------------------------------------------------------------------------


class TestNamespacePackageImports:
    def test_documented_symbols_import_from_their_defining_modules(self):
        from vtscore.concurrency.async_jobs import JobManager  # noqa: F401
        from vtscore.concurrency.memory_budget import cap_workers_by_memory  # noqa: F401
        from vtscore.concurrency.progress import (  # noqa: F401
            LoadingTasksTracker,
            ProgressTracker,
            resolve_progress_callback,
            update_progress,
        )
        from vtscore.security.path_validation import validate_server_filepath  # noqa: F401
        from vtscore.security.pickle import safe_pickle_load  # noqa: F401
        from vtscore.security.url_validation import validate_url  # noqa: F401

    def test_the_packages_export_nothing_themselves(self):
        """Both are PEP 420 namespace packages - the docs must not show
        `from vtscore.concurrency import X`."""
        import vtscore.concurrency
        import vtscore.security

        for pkg in (vtscore.concurrency, vtscore.security):
            assert getattr(pkg, "__file__", None) is None, "no __init__.py: implicit namespace package"

        # Which is why ``from vtscore.concurrency import JobManager`` (and its
        # security twin) raise ImportError: the package holds no such name.
        for module, symbol in (("vtscore.concurrency", "JobManager"), ("vtscore.security", "validate_url")):
            assert not hasattr(importlib.import_module(module), symbol), (
                f"{module} now exports {symbol}; the package docs may stop routing around it"
            )


# ---------------------------------------------------------------------------
# quickstart.md §6 - text query without training
# ---------------------------------------------------------------------------


class TestTextQuery:
    def test_embed_text_query_and_rank(self, loaded_medias):
        query_emb = embed_text_query("dog barking", "audio")
        assert query_emb is not None

        embeddings = np.stack([media_embedding(m) for m in loaded_medias.values()]).astype(np.float32)
        cosines = (embeddings @ query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9)
        ranked = sorted(zip(loaded_medias.values(), cosines), key=lambda t: t[1], reverse=True)
        assert len(ranked) == len(loaded_medias)
        assert all(-1.01 <= float(c) <= 1.01 for _, c in ranked)

    def test_cache_clear_is_importable_where_documented(self):
        from vtscore.embedding import clear_text_query_cache

        clear_text_query_cache()


# ---------------------------------------------------------------------------
# integration.md - the hooks an embedding application installs
# ---------------------------------------------------------------------------


class TestIntegrationHooks:
    def test_context_resolver_hooks_are_importable_and_installable(self, monkeypatch):
        from contextvars import ContextVar

        import vtscore.state.core as state_core
        from vtscore.state import (
            DatasetContext,
            get_context,
            get_detector_context,
            register_context,
            register_dataset_context_resolver,
            register_detector_context_resolver,
        )

        monkeypatch.setattr(state_core, "_dataset_context_resolver", state_core._default_context_resolver)
        monkeypatch.setattr(state_core, "_detector_context_resolver", state_core._default_context_resolver)

        current_dataset_id: ContextVar[str | None] = ContextVar("current_dataset_id", default=None)
        current_detector_id: ContextVar[str | None] = ContextVar("current_detector_id", default=None)

        def _resolve_dataset_context():
            did = current_dataset_id.get()
            return get_context(did) if did else None

        def _resolve_detector_context():
            did = current_detector_id.get()
            return get_detector_context(did) if did else None

        register_dataset_context_resolver(_resolve_dataset_context)
        register_detector_context_resolver(_resolve_detector_context)

        ctx = DatasetContext("doc-dataset")
        register_context(ctx)
        token = current_dataset_id.set("doc-dataset")
        try:
            from vtscore.state import get_active_context

            assert get_active_context() is ctx
        finally:
            current_dataset_id.reset(token)

    def test_setting_persister_hook(self):
        from vtscore.state import register_setting_persister

        seen: dict[str, Any] = {}
        register_setting_persister("inclusion", lambda v: seen.__setitem__("inclusion", v))
        assert "inclusion" not in seen or seen["inclusion"] is not None

    def test_thread_progress_helpers_are_where_documented(self):
        from vtscore.concurrency.progress import (
            clear_thread_progress,
            get_thread_progress,
            set_thread_progress,
        )

        calls: list[tuple] = []
        set_thread_progress(lambda *args: calls.append(args))
        try:
            assert get_thread_progress() is not None
        finally:
            clear_thread_progress()
        assert get_thread_progress() is None
